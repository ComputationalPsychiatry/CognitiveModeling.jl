"""
    AbstractCognitiveModel

Supertype for cognitive models.

# Implementation
Create a subtype and add a method of [`apply_model`](@ref) for it. A model can also add
methods of [`initialize_state`](@ref), [`default_parameters`](@ref),
[`get_parameters`](@ref), [`get_states`](@ref), [`get_observations`](@ref) and
[`get_actions`](@ref).
"""
abstract type AbstractCognitiveModel end

#Q: Is this better than using "nothing" for trials where there was no input or observation? Give pros and cons
#Types for cases where there is no observation or action

"""
    NoObservation()

Stand-in for the observation on a timestep where a model observes nothing, such as the first
timestep when the agent starts.
"""
struct NoObservation end

"""
    NoAction()

Stand-in for an action when there is none, such as the previous action on the first
timestep. A model without actions gives this as its action distribution.
"""
struct NoAction end

"""
    ModelInput(observation, previous_state, previous_action, parameters, timestep_size=1,
               rng=default_rng())

Everything a model needs to run a single timestep, which [`apply_model`](@ref) receives.

Each argument is kept in a field of the same name, such as `input.observation`.

# Arguments
- `observation`: what the model observes on this timestep, or [`NoObservation`](@ref).
- `previous_state`: the state of the model after the previous timestep. On the first
  timestep, this is the output of [`initialize_state`](@ref).
- `previous_action`: the action from the previous timestep, or [`NoAction`](@ref). A single
  action is in a `NamedTuple`, as in `input.previous_action.action`.
- `parameters`: the parameters of the model.
- `timestep_size=1`: how much time this timestep covers.
- `rng=default_rng()`: the random number generator to sample from if anything in the model
  is random, so that a simulation can be reproduced.

# Examples
```jldoctest
julia> using CognitiveModeling: ModelInput

julia> observation = (; reward = 1.0);

julia> state = (; expected_reward = 0.5);

julia> parameters = (; learning_rate = 0.1);

julia> ModelInput(observation, state, NoAction(), parameters).timestep_size
1

julia> ModelInput(observation, state, NoAction(), parameters, 0.5).timestep_size
0.5
```
"""
struct ModelInput{Tobservation, Tstate, Taction, Tparameters, Ttimestep_size, Trng}
    observation::Tobservation
    previous_state::Tstate
    previous_action::Taction
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
end

#Sets the timestep size to 1 and the generator to the default one if they aren't specified. An integer keeps Float32 models in Float32
function ModelInput(observation, previous_state, previous_action, parameters, timestep_size = 1)
    return ModelInput(observation, previous_state, previous_action, parameters, timestep_size, default_rng())
end

"""
    ModelOutput(action_distribution, new_state=(;))

The outputs of a model for a single timestep, which [`apply_model`](@ref) returns.

Each argument is kept in a field of the same name, such as `output.new_state`.

# Arguments
- `action_distribution`: what the action on this timestep is sampled from. It can be
  anything that works with `rand`, such as a distribution from Distributions.jl, and is put
  in a `NamedTuple` as `(; action = action_distribution)`. Several actions can be given as a
  `NamedTuple`, which names them and samples each of them, and a model without actions gives
  [`NoAction`](@ref).
- `new_state=(;)`: the state of the model after this timestep, which is the
  `previous_state` of the next one.

# Examples
```jldoctest
julia> using CognitiveModeling: ModelOutput

julia> using Distributions: Bernoulli

julia> ModelOutput(Bernoulli(0.7), (; expected_reward = 0.5)).new_state
(expected_reward = 0.5,)

julia> ModelOutput(Bernoulli(0.7)).new_state
NamedTuple()

julia> keys(ModelOutput(Bernoulli(0.7)).action_distribution)
(:action,)

julia> keys(ModelOutput((; choice = Bernoulli(0.7))).action_distribution)
(:choice,)
```
"""
struct ModelOutput{Taction, Tstate}
    action_distribution::Taction
    new_state::Tstate

    #Keeps action distributions that are already in a NamedTuple, and models without actions
    function ModelOutput(action_distribution::Union{NamedTuple, NoAction}, new_state)
        return new{typeof(action_distribution), typeof(new_state)}(action_distribution, new_state)
    end
end

#Puts a single action distribution in a NamedTuple
function ModelOutput(action_distribution, new_state)
    return ModelOutput((; action = action_distribution), new_state)
end

#Sets the new state to an empty NamedTuple if it isn't specified
function ModelOutput(action_distribution)
    return ModelOutput(action_distribution, (;))
end
