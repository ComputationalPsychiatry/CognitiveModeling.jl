## Sampling actions ##
"""
    sample_action(rng, action_distribution)

Sample an action from what a model returned, using `rand`. Each action in a `NamedTuple` is
sampled on its own, so a `NamedTuple` inside it gives a `NamedTuple` of samples, and an
entry that is [`NoAction`](@ref) is kept.
"""
function sample_action(rng, action_distribution)
    return rand(rng, action_distribution)
end

#Samples each named action on its own
function sample_action(rng, action_distributions::NamedTuple)
    return map(action_distribution -> sample_action(rng, action_distribution), action_distributions)
end

#An action that isn't taken has nothing to sample
function sample_action(rng, ::NoAction)
    return NoAction()
end

"""
    initial_action(model)

Return the action before the first timestep: a `NamedTuple` with the action names from
[`attribute_names`](@ref), each [`NoAction`](@ref).
"""
function initial_action(model::AbstractCognitiveModel)
    return no_actions(check_attribute_names(attribute_names(model), model))
end

#Builds the NamedTuple from the names in the type, so that it is known at compile time
function no_actions(::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}) where {Tparameters, Tstates, Tobservations, Tactions}
    return NamedTuple{Tactions}(map(_ -> NoAction(), Tactions))
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

"""
    observation_selection(save_history)
    action_selection(save_history)
    state_selection(save_history)

Return the selection of one kind in a [`SaveHistory`](@ref) as a `Val`, so that the names are
known at compile time wherever something is selected with it.
"""
observation_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tobservations)

#The same for the actions and the states
action_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tactions)
state_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tstates)

"""
    select_names(x, selection::Val)

Return what a [`SaveHistory`](@ref) selects of `x`: `x` itself for [`SaveEverything`](@ref),
an empty `NamedTuple` for no names, and the named entries of a `NamedTuple` for a tuple of
names. [`NoObservation`](@ref) is returned as it is.
"""
select_names(x, ::Val{SaveEverything()}) = x
select_names(x, ::Val{()}) = (;)

#Names can only be selected from a NamedTuple
function select_names(x, ::Val{names}) where {names}
    if x isa NamedTuple
        return NamedTuple{names}(x)
    elseif x isa NoObservation
        return x
    else
        error("A SaveHistory can only select names from a NamedTuple, but the value to select $names from has type $(typeof(x)). Use SaveEverything() or () for this kind of attribute.")
    end
end

"""
    save_state(model, state, selection::Val)

Return what is saved of `state`: what `selection` selects of the output of
[`prepare_state_for_saving`](@ref). When nothing is selected, the state isn't prepared at
all, so that no copy is made of a state which isn't saved.
"""
save_state(model, state, selection::Val) = select_names(prepare_state_for_saving(model, state), selection)
save_state(model, state, ::Val{()}) = (;)

"""
    saves_nothing(save_history)

Return whether `save_history` selects nothing of any kind, which is known at compile time.
"""
saves_nothing(::SaveHistory{(), (), ()}) = true
saves_nothing(::SaveHistory) = false

"""
    states_only(save_history)

Return a [`SaveHistory`](@ref) which selects only the states that `save_history` selects,
which is what [`simulate`](@ref) keeps of an environment's selection.
"""
states_only(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = SaveHistory{(), (), Tstates}()

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
                             environment, save_history, save_environment_history)

Collect the outputs of a timestep in a `NamedTuple`, each both as it is, which the next
timestep starts from, and as it is saved, which `save_history` and
`save_environment_history` select.

This is its own function so that if an output can have two types, like an observation or
[`NoObservation`](@ref), the compiler can split the call and get a concrete type for each.
"""
function collect_timestep_outputs(environment_output, cognitive_output, action, model, environment, save_history, save_environment_history)
    return (;
        observation = environment_output.observation,
        action,
        cognitive_state = cognitive_output.new_state,
        environment_state = environment_output.new_state,
        saved_observation = select_names(environment_output.observation, observation_selection(save_history)),
        saved_action = select_names(action, action_selection(save_history)),
        saved_cognitive_state = save_state(model, cognitive_output.new_state, state_selection(save_history)),
        saved_environment_state = save_state(environment, environment_output.new_state, state_selection(save_environment_history)),
        saved_environment_observation = select_names(environment_output.observation, observation_selection(save_environment_history)),
        saved_environment_action = select_names(action, action_selection(save_environment_history)),
    )
end

"""
    check_model_output(output, model)

Return `output` if it is the `ModelOutput` that an `apply_model` method of `model` must
return, and throw an error otherwise.

This makes a mistake give an informative error instead of a missing field later on.
"""
function check_model_output(output::ModelOutput, ::AbstractCognitiveModel)
    return output
end

#Untyped, so that there is always a method, whatever the model is inferred as
function check_model_output(output, model)
    error("The apply_model method of a model of type $(typeof(model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).")
end

"""
    check_environment_output(output, environment)

Return `output` if it is the `EnvironmentOutput` that an `apply_model` method of
`environment` must return, and throw an error otherwise.
"""
function check_environment_output(output::EnvironmentOutput, ::AbstractEnvironmentModel)
    return output
end

#Untyped, so that there is always a method, whatever the environment is inferred as
function check_environment_output(output, environment)
    error("The apply_model method of an environment model of type $(typeof(environment)) must return a CognitiveModeling.EnvironmentOutput, but it returned a value of type $(typeof(output)).")
end

"""
    timestep(::EnvironmentStarts, model, environment, cognitive_parameters,
             environment_parameters, save_history, save_environment_history, rng,
             timestep_size, previous)

Run a timestep where the environment moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::EnvironmentStarts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, timestep_size, previous)

    #Update the environment
    environment_output = check_environment_output(@inline(apply_model(environment, EnvironmentInput(previous.action, previous.environment_state, previous.observation, environment_parameters, timestep_size, rng))), environment)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(@inline(apply_model(model, ModelInput(environment_output.observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    return collect_timestep_outputs(environment_output, cognitive_output, action, model, environment, save_history, save_environment_history)
end

"""
    timestep(::AgentStarts, model, environment, cognitive_parameters,
             environment_parameters, save_history, save_environment_history, rng,
             timestep_size, previous)

Run a timestep where the agent moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::AgentStarts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, timestep_size, previous)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(@inline(apply_model(model, ModelInput(previous.observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    #Update the environment
    environment_output = check_environment_output(@inline(apply_model(environment, EnvironmentInput(action, previous.environment_state, previous.observation, environment_parameters, timestep_size, rng))), environment)

    return collect_timestep_outputs(environment_output, cognitive_output, action, model, environment, save_history, save_environment_history)
end

"""
    run_remaining_timesteps(storage, outputs, who_starts, model, environment,
                            cognitive_parameters, environment_parameters, save_history,
                            save_environment_history, rng, timestep_sizes, n_steps)

Run the timesteps after the first, and return the `NamedTuple` of vectors with the saved
outputs of all timesteps, followed by the outputs of the last timestep.

This is its own function so that it's compiled for the types of the first outputs.
"""
function run_remaining_timesteps(storage, outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, timestep_sizes, n_steps)

    #Loop through timesteps
    for t in 2:n_steps

        #Run the timestep, starting from the outputs of the one before
        outputs = timestep(who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, get_timestep_size(timestep_sizes, t), outputs)

        #Store trial outputs
        storage = (;
            observations = store_output!(storage.observations, outputs.saved_observation, t),
            actions = store_output!(storage.actions, outputs.saved_action, t),
            cognitive_states = store_output!(storage.cognitive_states, outputs.saved_cognitive_state, t),
            environment_states = store_output!(storage.environment_states, outputs.saved_environment_state, t),
            environment_observations = store_output!(storage.environment_observations, outputs.saved_environment_observation, t),
            environment_actions = store_output!(storage.environment_actions, outputs.saved_environment_action, t),
        )
    end

    #Return stored outputs, and the last ones for continuing from
    return storage, outputs
end

"""
    run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters,
                  environment_parameters, save_history, save_environment_history, rng,
                  timestep_sizes, n_steps)

Run all timesteps starting from `previous_outputs`, and return a `NamedTuple` of vectors with
the saved outputs of all timesteps, followed by the outputs of the last timestep.

The vectors are `observations`, `actions`, `cognitive_states` and `environment_states`, as
`save_history` and `save_environment_history` select them, and `environment_observations`
and `environment_actions` with what `save_environment_history` selects of the observations
and actions, for the environment's history.

[`simulate!`](@ref) continues from the last outputs instead of from what was saved, because
[`prepare_state_for_saving`](@ref) can return anything, and because the next simulation can
change a state in place.
"""
function run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, timestep_sizes, n_steps)

    #The first timestep sets the types of the storage, so there must be at least one
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, but it is $n_steps."))

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_steps
        throw(DimensionMismatch("timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and n_steps is $n_steps."))
    end

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs = timestep(who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, get_timestep_size(timestep_sizes, 1), previous_outputs)

    #Initialize storage for the saved outputs
    storage = (;
        observations = initialize_storage(outputs.saved_observation, n_steps),
        actions = initialize_storage(outputs.saved_action, n_steps),
        cognitive_states = initialize_storage(outputs.saved_cognitive_state, n_steps),
        environment_states = initialize_storage(outputs.saved_environment_state, n_steps),
        environment_observations = initialize_storage(outputs.saved_environment_observation, n_steps),
        environment_actions = initialize_storage(outputs.saved_environment_action, n_steps),
    )

    #Run the remaining timesteps
    return run_remaining_timesteps(storage, outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, rng, timestep_sizes, n_steps)
end

"""
    simulate(model, environment; <keyword arguments>)

Simulate `model` interacting with `environment`, and return the vectors `observations`,
`actions`, `cognitive_states` and `environment_states`, with one element for each timestep.

Each element is what happened on that timestep, and the states are those after it.
`save_history` and `save_environment_history` select what is kept of each element, which is
nothing by default: give `SaveHistory(SaveEverything())` to keep everything.

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
- `previous_action=initial_action(model)`: the action before the first timestep. By default
  every action is [`NoAction`](@ref), with the names from [`attribute_names`](@ref).
- `previous_environment_state=initialize_state(environment, environment_parameters)`: the
  state of `environment` before the first timestep.
- `previous_observation=NoObservation()`: the observation before the first timestep.
- `save_history::SaveHistory=SaveHistory()`: which observations, actions and states of
  `model` are saved, by name. Nothing is saved by default, and
  `SaveHistory(SaveEverything())` saves everything. See [`SaveHistory`](@ref).
- `save_environment_history::SaveHistory=SaveHistory()`: which states of `environment` are
  saved. Only its `states` is used here, since the observations and actions are returned
  once, as `save_history` selects them.

See also [`simulate!`](@ref), which continues from where the last simulation ended.

# Examples
```jldoctest
using CognitiveModeling
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

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
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
    save_history = SaveHistory(SaveEverything()),
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
        previous_action = initial_action(model),
        previous_environment_state = initialize_state(environment, environment_parameters),
        previous_observation = NoObservation(),
        save_history::SaveHistory = SaveHistory(),
        save_environment_history::SaveHistory = SaveHistory(),
    )

    #Collect what was left from before the first timestep
    previous_outputs = (;
        observation = previous_observation,
        action = previous_action,
        cognitive_state = previous_state,
        environment_state = previous_environment_state,
    )

    #Run the timesteps, keeping only the states of the environment's selection, since the observations and actions are returned once
    storage, _ = run_timesteps(previous_outputs, who_starts, model, environment, cognitive_parameters, environment_parameters, save_history, states_only(save_environment_history), rng, timestep_sizes, n_steps)

    #Return stored outputs
    return storage.observations, storage.actions, storage.cognitive_states, storage.environment_states
end

"""
    simulate!(agent::CognitiveAgent, environment::Environment; <keyword arguments>)

Simulate `agent` interacting with `environment` from where they are, update them both, and
return the same vectors as [`simulate`](@ref), with what `agent.save_history` selects of the
observations, actions and states, and what `environment.save_history` selects of the
environment's states. What the two select is also added to `agent.history` and
`environment.history`, unless a selection is empty.

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
using Distributions: Bernoulli

struct RandomChoices <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RandomChoices, parameters)
    return (;)
end

function CognitiveModeling.apply_model(model::RandomChoices, input::ModelInput)
    return ModelOutput((; choice = Bernoulli(0.5)))
end

function CognitiveModeling.attribute_names(model::RandomChoices)
    return ModelAttributes(; parameters = (), states = (), observations = :n_timesteps, actions = :choice)
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
    storage, last_outputs = run_timesteps(previous_outputs, who_starts, agent.model, environment.model, agent.parameters, environment.parameters, agent.save_history, environment.save_history, rng, timestep_sizes, n_steps)

    #Update agent and environment
    agent.state = last_outputs.cognitive_state
    agent.previous_action = last_outputs.action
    environment.state = last_outputs.environment_state
    environment.previous_observation = last_outputs.observation

    #Add to the agent's history
    if !saves_nothing(agent.save_history)
        append!(agent.history.observations, storage.observations)
        append!(agent.history.actions, storage.actions)
        append!(agent.history.states, storage.cognitive_states)
    end

    #Add to the environment's history
    if !saves_nothing(environment.save_history)
        append!(environment.history.observations, storage.environment_observations)
        append!(environment.history.actions, storage.environment_actions)
        append!(environment.history.states, storage.environment_states)
    end

    #Return stored outputs
    return storage.observations, storage.actions, storage.cognitive_states, storage.environment_states
end

"""
    run_cognitive_model(model, observation, previous_state, previous_action, parameters,
                        timestep_size, rng, save_history)

Run a timestep of `model` without an environment, and return its new state, the action
sampled from its output, and what `save_history` selects of the observation, the action and
the state, in a `NamedTuple`.

This is its own function so that it's compiled for the types in the agent's untyped fields,
which makes [`simulate!`](@ref) on a single observation several times faster.
"""
function run_cognitive_model(model, observation, previous_state, previous_action, parameters, timestep_size, rng, save_history)

    #Run timestep of cognitive model
    output = check_model_output(@inline(apply_model(model, ModelInput(observation, previous_state, previous_action, parameters, timestep_size, rng))), model)

    #Sample action
    action = sample_action(rng, output.action_distribution)

    return (;
        new_state = output.new_state,
        action,
        saved_observation = select_names(observation, observation_selection(save_history)),
        saved_action = select_names(action, action_selection(save_history)),
        saved_state = save_state(model, output.new_state, state_selection(save_history)),
    )
end

"""
    simulate!(agent::CognitiveAgent, observation; timestep_size=1, rng=default_rng())

Run a single timestep of `agent` on `observation`, update it, and return the sampled action.
What `agent.save_history` selects of the observation, the action and the state is added to
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

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
end

agent = CognitiveAgent(RescorlaWagner(), (; learning_rate = 0.5); save_history = SaveHistory(SaveEverything()))

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
    outputs = run_cognitive_model(agent.model, observation, agent.state, agent.previous_action, agent.parameters, timestep_size, rng, agent.save_history)

    #Update the agent
    agent.state = outputs.new_state
    agent.previous_action = outputs.action

    #Add to the agent's history
    if !saves_nothing(agent.save_history)
        push!(agent.history.observations, outputs.saved_observation)
        push!(agent.history.actions, outputs.saved_action)
        push!(agent.history.states, outputs.saved_state)
    end

    return outputs.action
end
