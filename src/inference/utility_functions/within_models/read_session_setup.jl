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

#Decided by the element types of the columns, which are in the type of the session, so that it is known when the code is compiled, and the sessions of a model have names of one type
"""
    simulates_values(session_info)
    simulates_values(sessions_info)

Return whether the data of a [`SessionInfo`](@ref), or of every session of a
[`MultipleSessionsInfo`](@ref), can hold [`SimulatedMissingValue`](@ref), as that of a
session from [`decondition_sessions`](@ref) does, so that its model simulates the values
instead of being fitted to them. Used by the inference backends' models of several sessions,
which name the variables of such sessions apart from those of fitted ones, and by
[`condition_sessions`](@ref).
"""
function simulates_values(session_info::SessionInfo)
    return can_hold_simulated_value(session_info.observations, session_info.actions)
end
function simulates_values(sessions_info::MultipleSessionsInfo)
    return simulates_values(first(sessions_info.sessions))
end

"""
    can_hold_simulated_value(observations, actions)

Return whether a column of `observations` or `actions`, the data of a session, can hold
[`SimulatedMissingValue`](@ref), which the element types of the columns say. Used by
[`simulates_values`](@ref) and [`marker_of_unmapped_outputs`](@ref).
"""
function can_hold_simulated_value(observations::NamedTuple, actions::NamedTuple)
    return any_column_can_hold_simulated_value((
        values(observations)..., values(actions)...
    ))
end

#The columns are taken one at a time, which the compiler follows through the types
"""
    any_column_can_hold_simulated_value(columns)

Return whether any of `columns`, a tuple of vectors, has an element type which
[`SimulatedMissingValue`](@ref) belongs to. Used by [`can_hold_simulated_value`](@ref).
"""
any_column_can_hold_simulated_value(::Tuple{}) = false
function any_column_can_hold_simulated_value(columns::Tuple)
    return SimulatedMissingValue <: eltype(first(columns)) ||
           any_column_can_hold_simulated_value(Base.tail(columns))
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
value that isn't [`ImputedMissingValue`](@ref) or [`SimulatedMissingValue`](@ref), and 0
when none has. Any other marker counts as data, since it says that the timestep happened.
Used by [`first_timestep_to_check_stopping`](@ref).
"""
function last_timestep_with_data(columns::NamedTuple)
    last_timestep = 0
    for column in columns
        t = findlast(value -> !(value isa LatentValueMarker), column)
        t === nothing || (last_timestep = max(last_timestep, t))
    end
    return last_timestep
end

#Decided by the type of check_stopping_criterion: with nothing, the criterion is only checked after the last timestep with data, so the data never continues, which keeps the flag that a session returns of the type the models give it
"""
    data_continues_after(session_info, t)

Return whether the data of a session has a value on a timestep after `t`, the last one that
ran, so that a [`stopping_criterion`](@ref) met there ends the session while its data
continues, which the parameters then can't give. With `check_stopping_criterion = nothing`,
the criterion is only checked after the last timestep with data, so this is `false` when the
code is compiled. Used by fitted sessions when a stopping criterion is met.
"""
function data_continues_after(session_info::SessionInfo, t)
    return data_continues_after(session_info.check_stopping_criterion, session_info, t)
end
data_continues_after(::Nothing, session_info::SessionInfo, t) = false
function data_continues_after(::Bool, session_info::SessionInfo, t)
    return max(
        last_timestep_with_data(session_info.observations),
        last_timestep_with_data(session_info.actions),
    ) > t
end

"""
    session_parameters_valid(session_info, parameters_valid, criterion_allows_data)

Return whether the parameters of a session stayed valid, from `parameters_valid`, as the
models gave it on the last timestep that ran, and `criterion_allows_data`, which is `false`
when a [`stopping_criterion`](@ref) ended the session while its data continues. With
`check_stopping_criterion` given as a `Bool`, the criterion can do that, so the result is a
`Bool`. With `nothing` it can't, and the result is `parameters_valid` as the models gave it,
[`ValidByDefault`](@ref) for models that can't set it. Used by
fitted sessions, so that what a session returns has one type.
"""
function session_parameters_valid(
    session_info::SessionInfo, parameters_valid, criterion_allows_data::Bool
)
    return session_parameters_valid(
        session_info.check_stopping_criterion, parameters_valid, criterion_allows_data
    )
end
function session_parameters_valid(::Nothing, parameters_valid, criterion_allows_data)
    return parameters_valid
end
function session_parameters_valid(::Bool, parameters_valid, criterion_allows_data)
    return parameters_are_valid(parameters_valid) & criterion_allows_data
end
