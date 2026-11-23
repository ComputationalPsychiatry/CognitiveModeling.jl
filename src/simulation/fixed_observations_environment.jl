## An environment which gives fixed observations ##

"""
    FixedObservationsEnvironment(observations)

An environment model which gives the observations in `observations` in order, an entry of
each on every timestep.

# Arguments
- `observations::NamedTuple`: a vector for each observation, with an entry for each
  timestep, such as `(; reward = [1.0, 0.0, 1.0])`. The vectors must have the same length.
  Each entry is given in a [`Deterministic`](@ref), so nothing is sampled. An entry that is
  [`NoValue`](@ref) is given as it is, for a timestep without that observation.

It takes no actions, so a simulation with a cognitive model that gives actions is given
`action_map = NameMap()`, which passes none of them on. Its [`attribute_names`](@ref) are
the observations as outputs, no inputs, no parameters, and the state
`n_observations_given`, the number of timesteps it has given its observations for. Its
[`stopping_criterion`](@ref) is met when every entry is given, so a simulation stops there
by itself, and `n_steps` is the most timesteps it can take. A timestep after the last entry
throws an error.

# Examples
```jldoctest
using CognitiveModeling

#A Rescorla-Wagner model which chooses when it expects more than half a reward
struct LearnsRewards <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsRewards, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.timestep_model(model::LearnsRewards, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput((; choice = Deterministic(Vₜ > 0.5)), (; expected_reward = Vₜ))
end
function CognitiveModeling.attribute_names(model::LearnsRewards)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice)
end

#The rewards of three timesteps
environment = FixedObservationsEnvironment((; reward = [1.0, 0.0, 1.0]))

#The simulation stops after the three rewards, before it has simulated all of n_steps. The environment takes no actions, so the map passes none on
simulation = simulate_session(
    LearnsRewards(),
    environment;
    who_starts = EnvironmentStarts(),
    n_steps = 10,
    cognitive_parameters = (; learning_rate = 0.5),
    environment_parameters = (;),
    action_map = NameMap(),
    cognitive_save_history = SaveHistory(; states = :expected_reward, save_initial_timestep_values = false),
)

simulation.states

# output

3-element Vector{@NamedTuple{expected_reward::Float64}}:
 (expected_reward = 0.75,)
 (expected_reward = 0.375,)
 (expected_reward = 0.6875,)
```
"""
struct FixedObservationsEnvironment{Tobservations<:NamedTuple} <: AbstractEnvironmentModel
    observations::Tobservations

    #Takes a vector of the same length for each observation
    function FixedObservationsEnvironment(observations::NamedTuple)

        #There must be an observation to give, as a vector with an entry for each timestep
        if isempty(observations) ||
            !all(column -> column isa AbstractVector, values(observations))
            throw(
                ArgumentError(
                    "The observations of a FixedObservationsEnvironment must be a NamedTuple with a vector for each observation, such as (; reward = [1.0, 0.0, 1.0]), but they are $(repr(observations)).",
                ),
            )
        end

        #Every observation has an entry for every timestep
        n_entries = map(length, values(observations))
        if !all(==(first(n_entries)), n_entries)
            throw(
                DimensionMismatch(
                    "The observations of a FixedObservationsEnvironment must have an entry each for every timestep, but $(keys(observations)) have $n_entries entries.",
                ),
            )
        end

        #A name is only used once among the attributes, which ModelAttributes checks, so no observation can have the name of the state
        environment = new{typeof(observations)}(observations)
        attribute_names(environment)
        return environment
    end
end

#The observations are its outputs, it takes no actions, and it counts in its state how many timesteps it has given observations for
function attribute_names(
    ::FixedObservationsEnvironment{<:NamedTuple{observation_names}}
) where {observation_names}
    return ModelAttributes(;
        parameters=(), states=:n_observations_given, inputs=(), outputs=observation_names
    )
end

#Starts before the first entry
function initialize_state(::FixedObservationsEnvironment, parameters)
    return (; n_observations_given=0)
end

#Gives the next entry of each observation
function timestep_model(environment::FixedObservationsEnvironment, context::ModelContext)
    n_observations_given = context.previous_state.n_observations_given + 1

    #Every entry is given once, and the environment doesn't start again from the first
    if n_observations_given > n_fixed_timesteps(environment)
        error(
            "The FixedObservationsEnvironment has given the observations of all its $(n_fixed_timesteps(environment)) timesteps, so it has none for another timestep. A simulation stops by itself after the last one, while a single timestep of simulate_timestep doesn't check that. A new call of simulate_session starts from the first again, and so does reset! of an InstantiatedModel.",
        )
    end

    #Give the entries of this timestep, each as fixed_observation wraps it
    observations = map(
        fixed_observation, select_row(environment.observations, n_observations_given)
    )
    return ModelOutput(observations, (; n_observations_given))
end

#A simulation stops when the last entry is given
function stopping_criterion(environment::FixedObservationsEnvironment, state, parameters)
    return state.n_observations_given >= n_fixed_timesteps(environment)
end

"""
    n_fixed_timesteps(environment)

Return the number of timesteps that `environment`, a
[`FixedObservationsEnvironment`](@ref), has observations for. Used by its timestep and by
its [`stopping_criterion`](@ref).
"""
function n_fixed_timesteps(environment::FixedObservationsEnvironment)
    return length(first(environment.observations))
end

"""
    fixed_observation(value)

Return an entry of a [`FixedObservationsEnvironment`](@ref) as the environment gives it: in
a [`Deterministic`](@ref), so that nothing is sampled, or as it is when it marks that there
is no value, such as [`NoValue`](@ref). Used by the timestep of a
`FixedObservationsEnvironment`.
"""
fixed_observation(value) = Deterministic(value)
fixed_observation(value::AbstractNoValue) = value
