### SCORING THE RECORDED VALUES ###

"""
    scorable_distributions(distributions)

Return the vector of the distributions of one output to condition on as it is, or an empty
vector of `Normal`s for an output the model gave on no timestep and the data has on none,
whose vector never got a type. Either scores an empty vector of values to condition on as
zero, where a `Vector{Union{}}` can't be made into a product distribution at all. Used by
[`turing_run_all_timesteps`](@ref) before the values to condition on are scored.
"""
scorable_distributions(distributions::AbstractVector) = distributions
scorable_distributions(::Vector{Union{}}) = Normal{Float64}[]

"""
    initialize_distributions_for_conditioning(session_info, timestep_values)
    store_distributions_for_conditioning(session_info, distributions_for_conditioning,
                                         timestep_values)

Keep the distributions that the values to condition on of a session are scored against: the
cognitive model's for the actions, and the environment's for the observations when the
environment runs, in a `NamedTuple` with `actions` and `observations`, each a `NamedTuple`
with a vector for each name to condition on, as
[`initialize_output_distributions_for_conditioning`](@ref) and
[`store_output_distributions_for_conditioning`](@ref) make them from the values of a
timestep. A [`Deterministic`](@ref) is kept as the `Dirac` which scores it, see
[`scorable_distribution`](@ref). Used by [`turing_run_all_timesteps`](@ref).
"""
function initialize_distributions_for_conditioning(
    session_info::SessionInfo, timestep_values
)
    return (;
        actions=initialize_output_distributions_for_conditioning(
            scorable_distribution_of_each(
                select_names(
                    timestep_values.action_distribution,
                    action_names_to_condition_on(session_info),
                ),
            ),
        ),
        observations=initialize_output_distributions_for_conditioning(
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, timestep_values),
                    observation_names_to_condition_on(session_info),
                ),
            ),
        ),
    )
end

function store_distributions_for_conditioning(
    session_info::SessionInfo, distributions_for_conditioning, timestep_values
)
    return (;
        actions=store_output_distributions_for_conditioning(
            distributions_for_conditioning.actions,
            scorable_distribution_of_each(
                select_names(
                    timestep_values.action_distribution,
                    action_names_to_condition_on(session_info),
                ),
            ),
            timestep_values.actions_to_condition_on_at_timestep,
        ),
        observations=store_output_distributions_for_conditioning(
            distributions_for_conditioning.observations,
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, timestep_values),
                    observation_names_to_condition_on(session_info),
                ),
            ),
            observations_to_condition_on_at_timestep_of(session_info, timestep_values),
        ),
    )
end

"""
    observation_distribution_of(session_info, timestep_values)
    observations_to_condition_on_at_timestep_of(session_info, timestep_values)

Return the distributions the environment gave for the observations of a timestep, and which
observations to condition on at it, from the environment's part of the values of the
timestep, or an empty `NamedTuple` for a session without an environment, whose observations
aren't scored. Used by [`store_distributions_for_conditioning`](@ref).
"""
observation_distribution_of(::SessionInfo{<:Any,NoEnvironment}, timestep_values) = (;)
function observation_distribution_of(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observation_distribution
end
function observations_to_condition_on_at_timestep_of(
    ::SessionInfo{<:Any,NoEnvironment}, timestep_values
)
    return (;)
end
function observations_to_condition_on_at_timestep_of(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observations_to_condition_on_at_timestep
end

"""
    scorable_distribution(distribution)
    scorable_distribution_of_each(distributions)

Return what a value to condition on is scored against: `distribution` as it is, or the
`Dirac` at the value of a [`Deterministic`](@ref), which gives a log density of 0 to that
value and `-Inf` to any other. Only a number or a `Bool` can be scored as a point mass, so a
`Deterministic` with another value is an error, which asks for a distribution. A marker for
no output is kept, since it is only there on timesteps where the data has no value to
condition on, which [`find_outputs_to_condition_on`](@ref) checks. The second form maps over
a `NamedTuple` of distributions. Used by [`store_distributions_for_conditioning`](@ref).
"""
scorable_distribution(distribution) = distribution
scorable_distribution(deterministic::Deterministic{<:Real}) = Dirac(deterministic.value)
function scorable_distribution(deterministic::Deterministic)
    return throw(
        ArgumentError(
            "A value in the data is scored against what the model gave for it, which is Deterministic($(repr(deterministic.value))), of type $(typeof(deterministic.value)). Only a number or a Bool can be scored as a point mass. Give a distribution for an output that has values in the data, or leave it out of the data.",
        ),
    )
end
function scorable_distribution_of_each(distributions::NamedTuple)
    return map(scorable_distribution, distributions)
end

"""
    check_no_data_after(session_info, t)

Throw an informative error if the data of a session has a value on a timestep after `t`, the
last one that ran, since a session which ends while its data continues can't score the
values in the data after it. By default the criterion is only checked after the last
timestep with data, so this only happens with `check_stopping_criterion = true`. Used by
[`turing_run_all_timesteps`](@ref) when a stopping criterion is met.
"""
function check_no_data_after(session_info::SessionInfo, t)
    last_with_data = max(
        last_timestep_with_data(session_info.observations),
        last_timestep_with_data(session_info.actions),
    )
    if last_with_data > t
        throw(
            ArgumentError(
                "A stopping criterion is met before timestep $(t + 1), but the data of the session has values until timestep $last_with_data. A session can't end while its data continues. Give check_stopping_criterion = nothing to check the criterion only after the last timestep with data.",
            ),
        )
    end
    return nothing
end
