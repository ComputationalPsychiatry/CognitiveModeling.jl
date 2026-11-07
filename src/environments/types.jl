"""
    AbstractEnvironmentModel

Supertype for environment models, which give a cognitive model its observations.

# Implementation
Create a subtype and add methods of [`apply_model`](@ref), [`initialize_state`](@ref) and
[`attribute_names`](@ref) for it. The method of `apply_model` takes an
[`EnvironmentInput`](@ref) and returns an [`EnvironmentOutput`](@ref). An environment model
can also add methods of [`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
[`stopping_criterion`](@ref).

See also [`simulate`](@ref), which has an example.
"""
abstract type AbstractEnvironmentModel <: AbstractModel end

"""
    EnvironmentInput(action, previous_state, previous_observation, parameters,
                     timestep_size=1, rng=default_rng(), who_starts=UnknownWhoStarts())

Everything an environment model needs to run a single timestep.

Each argument is kept in a field of the same name, such as `input.action`.

# Arguments
- `action`: the actions that the environment responds to, in a `NamedTuple` with the names
  the cognitive model gave them, as in `input.action.choice`. On the first timestep when the
  environment starts, each of them is [`BeforeFirstTimestep`](@ref), and an action that
  isn't taken is [`NoAction`](@ref).
- `previous_state`: the state of the environment after the previous timestep.
- `previous_observation`: the observations that were sampled on the previous timestep, in a
  `NamedTuple` with the names the environment gave them. On the first timestep, each of them
  is [`BeforeFirstTimestep`](@ref).
- `parameters`: the parameters of the environment model.
- `timestep_size=1`: how much time this timestep covers.
- `rng=default_rng()`: the random number generator to sample from if anything in the
  environment is random, so that a simulation can be reproduced.
- `who_starts=UnknownWhoStarts()`: who moves first on the timestep, as for
  [`ModelInput`](@ref): [`EnvironmentStarts`](@ref), [`AgentStarts`](@ref) or
  [`UnknownWhoStarts`](@ref).
"""
struct EnvironmentInput{Taction, Tstate, Tobservation, Tparameters, Ttimestep_size, Trng, Twho_starts}
    action::Taction
    previous_state::Tstate
    previous_observation::Tobservation
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
    who_starts::Twho_starts
end

#Sets the timestep size to 1, the generator to the default one and who starts to unknown if they aren't specified. An integer keeps Float32 models in Float32
function EnvironmentInput(action, previous_state, previous_observation, parameters, timestep_size = 1, rng::AbstractRNG = default_rng(), who_starts = UnknownWhoStarts())
    return EnvironmentInput{typeof(action), typeof(previous_state), typeof(previous_observation), typeof(parameters), typeof(timestep_size), typeof(rng), typeof(who_starts)}(action, previous_state, previous_observation, parameters, timestep_size, rng, who_starts)
end

"""
    EnvironmentOutput(observation_distribution, new_state=(;))

The outputs of an environment model for a single timestep, which [`apply_model`](@ref)
returns.

Each argument is kept in a field of the same name, such as `output.new_state`.

# Arguments
- `observation_distribution::NamedTuple`: what the observations on this timestep are sampled
  from, in a `NamedTuple` with one entry for each observation, such as
  `(; reward = Bernoulli(0.8))`. Each entry can be anything that works with `rand`, such as
  a distribution from Distributions.jl, or a [`Deterministic`](@ref) for an observation that
  isn't random, as in `(; reward = Deterministic(1.0))`, and the sampled observations keep
  the names. An observation that isn't made on this timestep is [`NoObservation`](@ref),
  and an environment model without observations gives an empty `NamedTuple`, `(;)`.
- `new_state=(;)`: the state of the environment after this timestep, which is the
  `previous_state` of the next one. The observations are sampled after it is made, so it
  can't depend on them: the environment receives them as `input.previous_observation` on
  the next timestep.
"""
struct EnvironmentOutput{Tobservation, Tstate}
    observation_distribution::Tobservation
    new_state::Tstate

    #Only takes named observation distributions, so that the environment model names every observation
    function EnvironmentOutput(observation_distribution::NamedTuple, new_state = (;))
        return new{typeof(observation_distribution), typeof(new_state)}(observation_distribution, new_state)
    end
end
