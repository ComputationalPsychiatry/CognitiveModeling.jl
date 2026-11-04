## Sampling actions ##
"""
    sample_action(rng, action_distributions)

Sample each action in the `NamedTuple` that a model returned. A model without actions keeps
[`NoAction`](@ref).
"""
function sample_action(rng, action_distributions::NamedTuple)
    return map(action_distribution -> rand(rng, action_distribution), action_distributions)
end

#Models without actions have nothing to sample
function sample_action(rng, ::NoAction)
    return NoAction()
end

## Saving outputs ##
"""
    initialize_storage(first_value, n_steps)

Create the vector for saving one kind of output, using the first value to set its type.
"""
function initialize_storage(first_value, n_steps)

    #Create the vector and put the value in it
    storage = Vector{typeof(first_value)}(undef, n_steps)
    storage[1] = first_value

    return storage
end

"""
    store_output!(storage, value, t)

Save `value` as element `t` of `storage`, and return the vector.

If the value doesn't fit the type of `storage`, the outputs so far are moved to a wider
vector which can hold both types, and that vector is returned instead.
"""
function store_output!(storage::Vector{T}, value, t) where {T}

    #If it has the type of the vector, which the compiler can usually see, store it
    if value isa T
        storage[t] = value
        return storage
    end

    #Otherwise, move the outputs so far to a wider vector which can hold both types
    wider_storage = Vector{Union{T, typeof(value)}}(undef, length(storage))
    copyto!(wider_storage, 1, storage, 1, t - 1)
    wider_storage[t] = value

    return wider_storage
end

## Running timesteps ##
"""
    get_timestep_size(timestep_sizes, t)

Return the size of timestep `t`. A vector has one size for each timestep, and anything else
is the size of all of them.
"""
function get_timestep_size(timestep_sizes::AbstractVector, t)
    return timestep_sizes[t]
end

function get_timestep_size(timestep_size, t)
    return timestep_size
end

"""
    collect_timestep_outputs(environment_output, cognitive_output, action, model,
                             environment)

Collect the outputs of a timestep in a `NamedTuple`, with each state both as it is, which
the next timestep starts from, and as it is saved.

This is its own function so that if an output can have two types, like an observation or
[`NoObservation`](@ref), the compiler can split the call and get a concrete type for each.
"""
function collect_timestep_outputs(environment_output, cognitive_output, action, model, environment)
    return (;
        observation = environment_output.observation,
        action,
        cognitive_state = cognitive_output.new_state,
        environment_state = environment_output.new_state,
        saved_cognitive_state = prepare_state_for_saving(model, cognitive_output.new_state),
        saved_environment_state = prepare_state_for_saving(environment, environment_output.new_state),
    )
end

"""
    check_output(output, model)

Return `output` if it has the type that an `apply_model` method of `model` must return,
and throw an error otherwise.

This makes a mistake give an informative error instead of a missing field later on.
"""
function check_output(output::ModelOutput, ::AbstractCognitiveModel)
    return output
end

function check_output(output::EnvironmentOutput, ::AbstractEnvironmentModel)
    return output
end

function check_output(output, model::AbstractCognitiveModel)
    error("The apply_model method of a model of type $(typeof(model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).")
end

function check_output(output, environment::AbstractEnvironmentModel)
    error("The apply_model method of an environment model of type $(typeof(environment)) must return a CognitiveModeling.EnvironmentOutput, but it returned a value of type $(typeof(output)).")
end

"""
    timestep(::EnvironmentStarts, model, environment, cognitive_parameters,
             environment_parameters, rng, timestep_size, previous)

Run a timestep where the environment moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::EnvironmentStarts, model, environment, cognitive_parameters, environment_parameters, rng, timestep_size, previous)

    #Update the environment
    environment_output = check_output(@inline(apply_model(environment, EnvironmentInput(previous.action, previous.environment_state, previous.observation, environment_parameters, timestep_size, rng))), environment)

    #Run timestep of cognitive model
    cognitive_output = check_output(@inline(apply_model(model, ModelInput(environment_output.observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    return collect_timestep_outputs(environment_output, cognitive_output, action, model, environment)
end

"""
    timestep(::AgentStarts, model, environment, cognitive_parameters,
             environment_parameters, rng, timestep_size, previous)

Run a timestep where the agent moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::AgentStarts, model, environment, cognitive_parameters, environment_parameters, rng, timestep_size, previous)

    #Run timestep of cognitive model
    cognitive_output = check_output(@inline(apply_model(model, ModelInput(previous.observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    #Update the environment
    environment_output = check_output(@inline(apply_model(environment, EnvironmentInput(action, previous.environment_state, previous.observation, environment_parameters, timestep_size, rng))), environment)

    return collect_timestep_outputs(environment_output, cognitive_output, action, model, environment)
end

"""
    run_remaining_timesteps(observations, actions, cognitive_states, environment_states,
                            outputs, who_starts, model, environment, cognitive_parameters,
                            environment_parameters, rng, timestep_sizes, n_steps)

Run the timesteps after the first, and return the vectors with the outputs of all timesteps,
followed by the outputs of the last timestep.

This is its own function so that it's compiled for the types of the first outputs.
"""
function run_remaining_timesteps(observations, actions, cognitive_states, environment_states, outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, rng, timestep_sizes, n_steps)

    #Loop through timesteps
    for t in 2:n_steps

        #Run the timestep, starting from the outputs of the one before
        outputs = timestep(who_starts, model, environment, cognitive_parameters, environment_parameters, rng, get_timestep_size(timestep_sizes, t), outputs)

        #Store trial outputs
        observations = store_output!(observations, outputs.observation, t)
        actions = store_output!(actions, outputs.action, t)
        cognitive_states = store_output!(cognitive_states, outputs.saved_cognitive_state, t)
        environment_states = store_output!(environment_states, outputs.saved_environment_state, t)
    end

    #Return stored outputs, and the last ones for continuing from
    return observations, actions, cognitive_states, environment_states, outputs
end

"""
    run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters,
                  environment_parameters, rng, timestep_sizes, n_steps)

Run all timesteps starting from `previous_outputs`, and return the vectors with the outputs
of all timesteps, followed by the outputs of the last timestep.

[`simulate!`](@ref) continues from the last outputs instead of from what was saved, because
[`prepare_state_for_saving`](@ref) can return anything, and because the next simulation can
change a state in place.
"""
function run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, rng, timestep_sizes, n_steps)

    #The first timestep sets the types of the storage, so there must be at least one
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, but it is $n_steps."))

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_steps
        throw(DimensionMismatch("timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and n_steps is $n_steps."))
    end

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs = timestep(who_starts, model, environment, cognitive_parameters, environment_parameters, rng, get_timestep_size(timestep_sizes, 1), previous_outputs)

    #Initialize storage for outputs, and run the remaining timesteps
    return run_remaining_timesteps(
        initialize_storage(outputs.observation, n_steps),
        initialize_storage(outputs.action, n_steps),
        initialize_storage(outputs.saved_cognitive_state, n_steps),
        initialize_storage(outputs.saved_environment_state, n_steps),
        outputs,
        who_starts,
        model,
        environment,
        cognitive_parameters,
        environment_parameters,
        rng,
        timestep_sizes,
        n_steps,
    )
end

"""
    simulate(model, environment; <keyword arguments>)

Simulate `model` interacting with `environment`, and return the vectors `observations`,
`actions`, `cognitive_states` and `environment_states`, with one element for each timestep.

Each element is what happened on that timestep, and the states are those after it.

# Arguments
- `model::AbstractCognitiveModel`: the cognitive model.
- `environment::AbstractEnvironmentModel`: the environment model.
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep. It
  has no default, because a wrong choice shifts the model by a timestep without any error.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1.
- `timestep_sizes=1`: how much time each timestep covers, which both `model` and
  `environment` receive. A vector has one size for each timestep, so its length must be
  `n_steps`. Anything else is the size of every timestep.
- `cognitive_parameters::NamedTuple`: the parameters of `model`.
- `environment_parameters::NamedTuple`: the parameters of `environment`.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample actions from,
  which `model` and `environment` also receive in their inputs.
- `previous_state=initialize_state(model, cognitive_parameters)`: the state of `model`
  before the first timestep. A state that `model` changes in place will be changed.
- `previous_action=NoAction()`: the action before the first timestep.
- `previous_environment_state=initialize_state(environment, environment_parameters)`: the
  state of `environment` before the first timestep.
- `previous_observation=NoObservation()`: the observation before the first timestep.

See also [`simulate!`](@ref), which continues from where the last simulation ended.

# Examples
```jldoctest
using CognitiveModeling
using CognitiveModeling: EnvironmentInput, EnvironmentOutput
using Distributions: BernoulliLogit

#A cognitive model which learns the expected reward
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = BernoulliLogit(Vₜ - 0.5)), (; expected_reward = Vₜ))
end

#An environment which gives a reward with a fixed probability, whatever the agent does
struct BernoulliRewards <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(environment::BernoulliRewards, parameters)
    return (;)
end

function CognitiveModeling.apply_model(environment::BernoulliRewards, input::EnvironmentInput)
    reward = float(rand(input.rng) < input.parameters.reward_probability)
    return EnvironmentOutput((; reward))
end

observations, actions, cognitive_states, environment_states = simulate(
    RescorlaWagner(),
    BernoulliRewards();
    who_starts = EnvironmentStarts(),
    n_steps = 3,
    cognitive_parameters = (; learning_rate = 0.5),
    environment_parameters = (; reward_probability = 1.0),
)

cognitive_states

# output

3-element Vector{@NamedTuple{expected_reward::Float64}}:
 (expected_reward = 0.75,)
 (expected_reward = 0.875,)
 (expected_reward = 0.9375,)
```

# Extended help
The return type is concrete as long as the types of the outputs don't change between
timesteps. If they do, like an observation that is sometimes [`NoObservation`](@ref), the
vector for that output holds a `Union` of the types, and the simulation is a little slower.

Nothing is copied when outputs are saved. A model or environment which changes an object in
its state in place, instead of returning a new one, must add a method of
[`prepare_state_for_saving`](@ref), or all the saved states show the final values, with no
error. Observations and actions have no such function, so they must be new
objects on each timestep.

Short simulations can be compiled with Reactant, if the action distributions work with
traced numbers, which those from Distributions.jl don't. Compile a function that takes a
`Reactant.ReactantRNG()` as a positional argument and passes it on as `rng`, because `@jit`
doesn't trace keyword arguments and the default generator repeats its numbers on every run.
Compile times grow quickly with `n_steps`, so this is for a few hundred timesteps at most.
"""
function simulate(
        model::AbstractCognitiveModel,
        environment::AbstractEnvironmentModel;
        who_starts::Union{AgentStarts, EnvironmentStarts},
        n_steps::Int,
        timestep_sizes = 1,
        cognitive_parameters::NamedTuple,
        environment_parameters::NamedTuple,
        rng::AbstractRNG = default_rng(),
        previous_state = initialize_state(model, cognitive_parameters),
        previous_action = NoAction(),
        previous_environment_state = initialize_state(environment, environment_parameters),
        previous_observation = NoObservation(),
    )

    #Collect what was left from before the first timestep
    previous_outputs = (;
        observation = previous_observation,
        action = previous_action,
        cognitive_state = previous_state,
        environment_state = previous_environment_state,
    )

    #Run the timesteps
    observations, actions, cognitive_states, environment_states, _ = run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, rng, timestep_sizes, n_steps)

    #Return stored outputs
    return observations, actions, cognitive_states, environment_states
end

"""
    simulate!(agent::CognitiveAgent, environment::Environment; <keyword arguments>)

Simulate `agent` interacting with `environment` from where they are, update them both, and
return the same vectors as [`simulate`](@ref). If `agent.save_history` is true, the
observations, actions and states are also added to `agent.history`.

Use the same `who_starts` on each call, so that each simulation continues the last one.

Reading the agent's untyped fields costs a dynamic dispatch on each call, which makes the
return type unknown to the compiler. The loop inside is still fully typed. Use `simulate` if
the return type matters, or if most calls are for a single timestep.

# Arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1.
- `timestep_sizes=1`: how much time each timestep covers. A vector has one size for each
  timestep of this call, so its length must be `n_steps`.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.

# Examples
```jldoctest
using CognitiveModeling
using CognitiveModeling: EnvironmentInput, EnvironmentOutput
using Distributions: Bernoulli

struct RandomChoices <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RandomChoices, parameters)
    return (;)
end

function CognitiveModeling.apply_model(model::RandomChoices, input::ModelInput)
    return ModelOutput((; choice = Bernoulli(0.5)))
end

struct TimestepCounter <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(model::TimestepCounter, parameters)
    return (; n_timesteps = 0)
end

function CognitiveModeling.apply_model(model::TimestepCounter, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    return EnvironmentOutput((; n_timesteps), (; n_timesteps))
end

agent = CognitiveAgent(RandomChoices(), (;))
environment = Environment(TimestepCounter(), (;))

simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)

environment.state

# output

(n_timesteps = 5,)
```
"""
function simulate!(
        agent::CognitiveAgent,
        environment::Environment;
        who_starts::Union{AgentStarts, EnvironmentStarts},
        n_steps::Int,
        timestep_sizes = 1,
        rng::AbstractRNG = default_rng(),
    )

    #Collect where the agent and the environment are
    previous_outputs = (;
        observation = environment.previous_observation,
        action = agent.previous_action,
        cognitive_state = agent.state,
        environment_state = environment.state,
    )

    #Run the timesteps
    observations, actions, cognitive_states, environment_states, last_outputs = run_timesteps(previous_outputs, who_starts, agent.model, environment.model, agent.parameters, environment.parameters, rng, timestep_sizes, n_steps)

    #Update agent and environment
    agent.state = last_outputs.cognitive_state
    agent.previous_action = last_outputs.action
    environment.state = last_outputs.environment_state
    environment.previous_observation = last_outputs.observation

    #Add to the agent's history
    if agent.save_history
        append!(agent.history.observations, observations)
        append!(agent.history.actions, actions)
        append!(agent.history.states, cognitive_states)
    end

    #Return stored outputs
    return observations, actions, cognitive_states, environment_states
end

"""
    run_cognitive_model(model, observation, previous_state, previous_action, parameters,
                        timestep_size, rng)

Run a timestep of `model` without an environment, and return its output and the action
sampled from it.

This is its own function so that it's compiled for the types in the agent's untyped fields,
which makes [`simulate!`](@ref) on a single observation several times faster.
"""
function run_cognitive_model(model, observation, previous_state, previous_action, parameters, timestep_size, rng)

    #Run timestep of cognitive model
    output = check_output(@inline(apply_model(model, ModelInput(observation, previous_state, previous_action, parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, output.action_distribution)

    return output, action
end

"""
    simulate!(agent::CognitiveAgent, observation; timestep_size=1, rng=default_rng())

Run a single timestep of `agent` on `observation`, update it, and return the sampled action.
If `agent.save_history` is true, the observation, action and state are also added to
`agent.history`.

The action is in a `NamedTuple` with the name the model gave it, as in `action.choice`, as
for [`simulate`](@ref).

# Arguments
- `agent::CognitiveAgent`: the agent to update.
- `observation`: what the agent observes on this timestep, or [`NoObservation`](@ref).
- `timestep_size=1`: how much time this timestep covers.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

agent = CognitiveAgent(RescorlaWagner(), (; learning_rate = 0.5))

simulate!(agent, (; reward = 1.0))
simulate!(agent, (; reward = 1.0))

agent.history.states

# output

3-element Vector{Any}:
 (expected_reward = 0.5,)
 (expected_reward = 0.75,)
 (expected_reward = 0.875,)
```
"""
function simulate!(agent::CognitiveAgent, observation; timestep_size = 1, rng::AbstractRNG = default_rng())

    #Run the timestep
    output, action = run_cognitive_model(agent.model, observation, agent.state, agent.previous_action, agent.parameters, timestep_size, rng)

    #Update the agent
    agent.state = output.new_state
    agent.previous_action = action

    #Add to the agent's history
    if agent.save_history
        push!(agent.history.observations, observation)
        push!(agent.history.actions, action)
        push!(agent.history.states, prepare_state_for_saving(agent.model, output.new_state))
    end

    return action
end
