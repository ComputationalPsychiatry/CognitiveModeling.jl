## Saving what a fitted session returns ##

"""
    select_timestep_values_to_save(session_info, timestep_values)

Return what a session saves of the values of one timestep, or of where it starts, as the
save histories of `session_info` select it, arranged as the session returns it: the
cognitive model's `states`, `observations`, `actions` and `action_distributions`, and with
an environment its `states`, `observations`, `actions` and `observation_distributions` under
`environment`, as [`map_saved`](@ref) arranges them. `timestep_values` has the cognitive
model's `state`, `observation`, `action` and `action_distribution`, and under `environment`
the environment's `state`, `observation`, `action` and `observation_distribution`, each in
its own names. Used by fitted sessions, on every timestep.
"""
function select_timestep_values_to_save(
    session_info::SessionInfo{<:Any,NoEnvironment}, timestep_values
)
    return select_cognitive_timestep_values_to_save(
        session_info.cognitive_model, timestep_values, session_info.cognitive_save_history
    )
end

function select_timestep_values_to_save(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return merge(
        select_cognitive_timestep_values_to_save(
            session_info.cognitive_model,
            timestep_values,
            session_info.cognitive_save_history,
        ),
        build_named_entry(
            Val(:environment),
            select_environment_timestep_values_to_save(
                session_info.environment,
                timestep_values.environment,
                session_info.environment_save_history,
            ),
        ),
    )
end

"""
    select_cognitive_timestep_values_to_save(cognitive_model, timestep_values, save_history)
    select_environment_timestep_values_to_save(environment, environment_timestep_values,
                                               save_history)

Return what `save_history` selects of one model's part of the values of a timestep: the
`states`, `observations`, `actions` and `action_distributions` of the cognitive model, or
the `states`, `observations`, `actions` and `observation_distributions` of the environment,
from the values of the environment alone. The observation and the action can have one of
two types, so they go in through [`build_named_entry`](@ref). Used by
[`select_timestep_values_to_save`](@ref).
"""
function select_cognitive_timestep_values_to_save(
    cognitive_model, timestep_values, save_history
)
    return merge(
        (;
            states=save_state(
                cognitive_model, timestep_values.state, state_selection(save_history)
            ),
        ),
        build_named_entry(
            Val(:observations),
            select_names(timestep_values.observation, input_selection(save_history)),
        ),
        build_named_entry(
            Val(:actions),
            select_names(timestep_values.action, output_selection(save_history)),
        ),
        (;
            action_distributions=select_names(
                timestep_values.action_distribution,
                output_distribution_selection(save_history),
            ),
        ),
    )
end

function select_environment_timestep_values_to_save(
    environment, environment_timestep_values, save_history
)
    return merge(
        (;
            states=save_state(
                environment,
                environment_timestep_values.state,
                state_selection(save_history),
            ),
        ),
        build_named_entry(
            Val(:observations),
            select_names(
                environment_timestep_values.observation, output_selection(save_history)
            ),
        ),
        build_named_entry(
            Val(:actions),
            select_names(environment_timestep_values.action, input_selection(save_history)),
        ),
        (;
            observation_distributions=select_names(
                environment_timestep_values.observation_distribution,
                output_distribution_selection(save_history),
            ),
        ),
    )
end

"""
    select_initial_timestep_values_to_save(save_initial_timestep_values::Val, session_info,
                                           previous)

Return what a session saves from before the first timestep, as timestep 0, when
`save_initial_timestep_values` is `Val(true)`, and `nothing` otherwise. `previous` has the
values that the first timestep starts from, arranged as those of a timestep. The
distributions of timestep 0 are [`BeforeFirstTimestep`](@ref) for every output of each
model, since none was given yet. Used by fitted sessions, before their first timestep runs.
"""
function select_initial_timestep_values_to_save(
    ::Val{true}, session_info::SessionInfo{<:Any,NoEnvironment}, previous
)
    cognitive_model = session_info.cognitive_model

    #No action distribution was given before the first timestep
    return select_timestep_values_to_save(
        session_info,
        merge(
            previous, (; action_distribution=output_before_first_timestep(cognitive_model))
        ),
    )
end

function select_initial_timestep_values_to_save(
    ::Val{true}, session_info::SessionInfo{<:Any,<:AbstractTimestepModel}, previous
)
    cognitive_model = session_info.cognitive_model

    #No observation distribution was given before the first timestep
    environment_timestep_values = merge(
        previous.environment,
        (;
            observation_distribution=output_before_first_timestep(session_info.environment),
        ),
    )

    #Nor an action distribution
    return select_timestep_values_to_save(
        session_info,
        merge(
            previous,
            (;
                action_distribution=output_before_first_timestep(cognitive_model),
                environment=environment_timestep_values,
            ),
        ),
    )
end
function select_initial_timestep_values_to_save(
    ::Val{false}, session_info::SessionInfo, previous
)
    return nothing
end

"""
    with_empty_environment_entry(saved_trajectories)

Return `saved_trajectories`, what a session without an environment saved, with an
`environment` entry whose vectors hold an empty `NamedTuple` for each saved timestep, for a
session which has an environment but didn't include it, so that every session with an
environment returns the same entries, in the same order, since the model's return otherwise
has two types. Used by fitted sessions.
"""
function with_empty_environment_entry(saved_trajectories::NamedTuple)
    empty_vector = fill((;), length(saved_trajectories.states))
    return (;
        saved_trajectories.states,
        saved_trajectories.observations,
        saved_trajectories.actions,
        saved_trajectories.action_distributions,
        environment=(;
            states=empty_vector,
            observations=empty_vector,
            actions=empty_vector,
            observation_distributions=empty_vector,
        ),
        saved_trajectories.timestep_sizes,
    )
end

"""
    timestep_sizes_until(timestep_sizes, n_saved)

Return the vector of the sizes of the timesteps that a session returns, cut to its first
`n_saved` elements, for a session that ended before its data did when a
[`stopping_criterion`](@ref) was met. An empty vector of sizes, for a session that saves
nothing, is kept. Used by fitted sessions.
"""
function timestep_sizes_until(timestep_sizes::AbstractVector, n_saved)
    return isempty(timestep_sizes) ? timestep_sizes : timestep_sizes[1:n_saved]
end
