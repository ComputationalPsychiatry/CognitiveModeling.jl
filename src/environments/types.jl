"""
    AbstractEnvironmentModel

Supertype for environment models, which give a cognitive model its observations.

# Implementation
Create a subtype and add methods of [`apply_model`](@ref) and [`initialize_state`](@ref) for
it. The method of `apply_model` takes an [`EnvironmentInput`](@ref) and returns an
[`EnvironmentOutput`](@ref). An environment model can also add a method of
[`prepare_state_for_saving`](@ref).

See also [`simulate`](@ref), which has an example.
"""
abstract type AbstractEnvironmentModel end

"""
    EnvironmentInput(action, previous_state, previous_observation, parameters,
                     timestep_size=1, rng=default_rng())

Everything an environment model needs to run a single timestep.

Each argument is kept in a field of the same name, such as `input.action`.

# Arguments
- `action`: the actions that the environment responds to, in a `NamedTuple` with the names
  the cognitive model gave them, as in `input.action.choice`, or [`NoAction`](@ref) if there
  are none yet.
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
