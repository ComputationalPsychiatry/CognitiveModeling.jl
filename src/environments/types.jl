"""
    AbstractEnvironmentModel

Supertype for environment models, which give a cognitive model its observations.

# Implementation
Create a subtype and add a method of `apply_model` for it, which takes an
[`EnvironmentInput`](@ref) and returns an [`EnvironmentOutput`](@ref). An environment model
with a state also adds a method of `initialize_state`.

See also [`simulate`](@ref), which has an example.
"""
abstract type AbstractEnvironmentModel end

"""
    EnvironmentInput(action, previous_state, previous_observation, parameters,
                     timestep_size=1, rng=default_rng())

Everything an environment model needs to run a single timestep.

Each argument is kept in a field of the same name, such as `input.action`.

# Arguments
- `action`: the action that the environment responds to, or [`NoAction`](@ref) if there is
  none yet. A single action is in a `NamedTuple`, as in `input.action.action`.
- `previous_state`: the state of the environment after the previous timestep.
- `previous_observation`: the observation from the previous timestep, or
  [`NoObservation`](@ref).
- `parameters`: the parameters of the environment model.
- `timestep_size=1`: how much time this timestep covers.
- `rng=default_rng()`: the random number generator to sample from if anything in the
  environment is random, so that a simulation can be reproduced.
"""
struct EnvironmentInput{Taction, Tstate, Tobservation, Tparameters, Ttimestep_size, Trng}
    action::Taction
    previous_state::Tstate
    previous_observation::Tobservation
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
end

#Sets the timestep size to 1 and the generator to the default one if they aren't specified. An integer keeps Float32 models in Float32
function EnvironmentInput(action, previous_state, previous_observation, parameters, timestep_size = 1)
    return EnvironmentInput(action, previous_state, previous_observation, parameters, timestep_size, default_rng())
end

"""
    EnvironmentOutput(observation, new_state=(;))

The outputs of an environment model for a single timestep.

Each argument is kept in a field of the same name, such as `output.observation`.

# Arguments
- `observation`: what the cognitive model observes, or [`NoObservation`](@ref) if it
  observes nothing.
- `new_state=(;)`: the state of the environment after this timestep, which is the
  `previous_state` of the next one.
"""
struct EnvironmentOutput{Tobservation, Tstate}
    observation::Tobservation
    new_state::Tstate
end

#Sets the new state to an empty NamedTuple if it isn't specified
function EnvironmentOutput(observation)
    return EnvironmentOutput(observation, (;))
end

"""
    Environment(model, parameters)
    Environment(model, parameters, state, previous_observation)

An environment model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

The state defaults to the output of `initialize_state`, and the previous observation to
[`NoObservation`](@ref).

See also [`CognitiveAgent`](@ref).
"""
mutable struct Environment{Tmodel <: AbstractEnvironmentModel, Tparameters}
    model::Tmodel
    parameters::Tparameters
    state::Any
    previous_observation::Any
end

#Starts the environment from its initial state, with no previous observation
function Environment(model::AbstractEnvironmentModel, parameters)
    return Environment(model, parameters, initialize_state(model, parameters), NoObservation())
end

"""
    apply_model(environment, input::EnvironmentInput)::EnvironmentOutput

Run a single timestep of `environment`: update its state and give the next observation.

# Implementation
Every environment model must add a method of this function, which throws an error otherwise.
Anything random should be sampled from `input.rng`, so that a simulation can be reproduced.

See also [`simulate`](@ref), which has an example.
"""
function apply_model(environment::AbstractEnvironmentModel, input)
    error("There is no apply_model method for an environment model of type $(typeof(environment)) with an input of type $(typeof(input)). Each environment model must define CognitiveModeling.apply_model(environment::$(typeof(environment)), input::CognitiveModeling.EnvironmentInput).")
end

"""
    initialize_state(environment::AbstractEnvironmentModel, parameters)

Create the state that `environment` has before its first timestep, which can depend on
`parameters`.

# Implementation
The default is an empty `NamedTuple`, so only environment models with a state need to add a
method.
"""
function initialize_state(model::AbstractEnvironmentModel, parameters)
    return (;)
end
