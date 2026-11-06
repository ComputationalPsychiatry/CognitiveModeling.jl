"""
    AbstractEnvironmentModel

Supertype for environment models, which give a cognitive model its observations.

# Implementation
Create a subtype and add methods of [`apply_model`](@ref), [`initialize_state`](@ref) and
[`attribute_names`](@ref) for it. The method of `apply_model` takes an
[`EnvironmentInput`](@ref) and returns an [`EnvironmentOutput`](@ref). An environment model
can also add methods of [`prepare_state_for_saving`](@ref) and [`default_parameters`](@ref).

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
  the cognitive model gave them, as in `input.action.choice`. On the first timestep when the
  environment starts, each of them is [`BeforeFirstTimestep`](@ref), and an action that
  isn't taken is [`NoAction`](@ref).
- `previous_state`: the state of the environment after the previous timestep.
- `previous_observation`: the observation from the previous timestep, in a `NamedTuple` with
  the names the environment gave them. On the first timestep, each of them is
  [`BeforeFirstTimestep`](@ref).
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
- `observation::NamedTuple`: what the cognitive model observes, in a `NamedTuple` with one
  entry for each observation, such as `(; reward = 1.0)`. An observation that isn't made on
  this timestep is [`NoObservation`](@ref), and an environment model without observations
  gives an empty `NamedTuple`, `(;)`.
- `new_state=(;)`: the state of the environment after this timestep, which is the
  `previous_state` of the next one.
"""
struct EnvironmentOutput{Tobservation, Tstate}
    observation::Tobservation
    new_state::Tstate

    #Only takes named observations, so that the environment model names every observation
    function EnvironmentOutput(observation::NamedTuple, new_state = (;))
        return new{typeof(observation), typeof(new_state)}(observation, new_state)
    end
end
