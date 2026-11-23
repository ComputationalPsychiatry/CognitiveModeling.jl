## Sessions with other data, for conditioning and deconditioning them ##

"""
    with_data(session_info, observations, actions;
              timestep_sizes=session_info.timestep_sizes, missing_observation_values=(),
              missing_action_values=())

Return a copy of `session_info`, a [`SessionInfo`](@ref), with `observations` and `actions`
as its data, prepared again with its settings, so that what is scored and whether the
environment runs follow from the new data. The copy has the sizes of the timesteps of
`session_info`, unless others are given, as for data with other timesteps. No value in the data is replaced,
unless values that stand for a missing one are given. Used by [`decondition_sessions`](@ref)
and the methods of [`condition_sessions`](@ref).
"""
function with_data(
    session_info::SessionInfo,
    observations,
    actions;
    timestep_sizes=session_info.timestep_sizes,
    missing_observation_values=(),
    missing_action_values=(),
)
    return prepare_session(
        session_info.cognitive_model,
        observations,
        actions;
        session_settings(session_info)...,
        timestep_sizes,
        missing_observation_values,
        missing_action_values,
    )
end

"""
    session_settings(session_info)

Return the settings that `session_info`, a [`SessionInfo`](@ref), was prepared with, as the
keyword arguments of `SessionInfo` that take them: who starts, the environment and the maps,
whether the environment is included, the sizes of the timesteps, the save histories, the
checks, and the observation and the action before the first timestep. Used by
[`with_data`](@ref) and [`condition_sessions`](@ref).
"""
function session_settings(session_info::SessionInfo)
    return (;
        session_info.who_starts,
        session_info.environment,
        session_info.observation_map,
        session_info.action_map,
        session_info.include_environment,
        session_info.timestep_sizes,
        session_info.cognitive_save_history,
        session_info.environment_save_history,
        session_info.check_parameter_domains,
        session_info.previous_observation,
        session_info.previous_action,
    )
end

#The timesteps after the last one with data are where a session samples every action and observation, and where a stopping criterion can end it without the parameters turning invalid, so that their number, and with it the number of variables, would change with the parameters, which a sampler that uses gradients can't handle. Without them, the session is the one that a table with the timesteps of a stopped simulation gives
"""
    cut_after_last_timestep_with_data(observations, actions, timestep_sizes)

Return `observations` and `actions`, the columns of the data of a session, and
`timestep_sizes`, the sizes of its timesteps, without the timesteps after the last one with
data, from [`last_timestep_with_data`](@ref), in a `NamedTuple`. Data without any value is
returned as it is, with every timestep. Used by the methods of [`condition_sessions`](@ref).
"""
function cut_after_last_timestep_with_data(
    observations::NamedTuple, actions::NamedTuple, timestep_sizes
)
    #The last timestep with data, after which the session ends
    n_timesteps = max(
        last_timestep_with_data(observations), last_timestep_with_data(actions)
    )

    #Data without any value has no timestep to end after
    n_timesteps == 0 && return (; observations, actions, timestep_sizes)

    return (;
        observations=map(column -> column[1:n_timesteps], observations),
        actions=map(column -> column[1:n_timesteps], actions),
        timestep_sizes=cut_timestep_sizes(timestep_sizes, n_timesteps),
    )
end

"""
    cut_timestep_sizes(timestep_sizes, n_timesteps)

Return the sizes of the first `n_timesteps` timesteps: those of a vector with a size for
each timestep, and a single size for every timestep as it is. Used by
[`cut_after_last_timestep_with_data`](@ref).
"""
function cut_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps)
    return timestep_sizes[1:n_timesteps]
end
cut_timestep_sizes(timestep_size, n_timesteps) = timestep_size

"""
    with_common_element_types(columns_of_sessions)

Return the `NamedTuple`s of columns in `columns_of_sessions`, the observations or the
actions of each of several sessions, with the column of each name given the same element
type in every session: the types of its entries in all of them. Used by the methods of
[`condition_sessions`](@ref), so that the sessions have one type, as
[`MultipleSessionsInfo`](@ref) needs.
"""
function with_common_element_types(
    columns_of_sessions::AbstractVector{<:NamedTuple{names}}
) where {names}
    #The types of the entries of each name, in every session
    element_types = map(names) do name
        element_type = Union{}
        for columns in columns_of_sessions, entry in getproperty(columns, name)
            element_type = Union{element_type,typeof(entry)}
        end
        return element_type
    end

    #Each column gets the element type of its name
    return map(columns_of_sessions) do columns
        return NamedTuple{names}(
            map(
                (name, element_type) ->
                    convert(Vector{element_type}, getproperty(columns, name)),
                names,
                element_types,
            ),
        )
    end
end
