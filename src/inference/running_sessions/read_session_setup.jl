## Reading what a session is set up with ##

"""
    has_environment(session_info)
    has_environment(sessions_info)

Return whether a [`SessionInfo`](@ref) has an environment model, or every session of a
[`MultipleSessionsInfo`](@ref) has, which is decided by the type. A session can have an
environment which doesn't run, see [`SessionInfo`](@ref). Used by
[`check_environment_parameters`](@ref).
"""
has_environment(::SessionInfo{<:Any,NoEnvironment}) = false
has_environment(::SessionInfo{<:Any,<:AbstractTimestepModel}) = true
function has_environment(sessions_info::MultipleSessionsInfo)
    return has_environment(first(sessions_info.sessions))
end

"""
    check_environment_parameters(session_info, environment_parameters)
    check_environment_parameters(sessions_info, environment_parameters)

Throw an informative error if environment parameters are given for a session, or for
sessions, without an environment model, which is decided by the types. Used by the models
of several sessions.
"""
function check_environment_parameters(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, environment_parameters::NamedTuple
)
    return nothing
end
function check_environment_parameters(
    ::SessionInfo{<:Any,NoEnvironment}, environment_parameters::NamedTuple{()}
)
    return nothing
end
function check_environment_parameters(
    ::SessionInfo{<:Any,NoEnvironment}, environment_parameters::NamedTuple
)
    return throw(
        ArgumentError(
            "Environment parameters were given, $environment_parameters, but the session has no environment.",
        ),
    )
end
function check_environment_parameters(
    sessions_info::MultipleSessionsInfo, environment_parameters::NamedTuple
)
    return check_environment_parameters(
        first(sessions_info.sessions), environment_parameters
    )
end

"""
    sessions_save_anything(sessions_info)

Return whether the save history of either model of any session in `sessions_info` selects
anything. Used by the full models of the inference backends, to warn when the sessions save
trajectories that nothing in the model uses.
"""
function sessions_save_anything(sessions_info::MultipleSessionsInfo)
    for session in sessions_info.sessions
        if !saves_nothing(session.cognitive_save_history) ||
            !saves_nothing(session.environment_save_history)
            return true
        end
    end
    return false
end

"""
    last_timestep_with_data(columns)

Return the last timestep on which any column in `columns`, a `NamedTuple` of vectors, has a
value that isn't [`MissingValue`](@ref), and 0 when none has. Any other marker counts as
data, since it says that the timestep happened. Used by
[`first_timestep_to_check_stopping`](@ref).
"""
function last_timestep_with_data(columns::NamedTuple)
    last_timestep = 0
    for column in columns
        t = findlast(value -> !(value isa MissingValue), column)
        t === nothing || (last_timestep = max(last_timestep, t))
    end
    return last_timestep
end
