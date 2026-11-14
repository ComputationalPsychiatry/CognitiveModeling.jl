## Including the environment in a session, or leaving it out ##

"""
    has_missing_entries(columns)

Return whether any entry of `columns`, the observations or the actions in the data of a
session with one vector for each name, is [`MissingValue`](@ref). Used by
[`SessionInfo`](@ref), to decide whether the environment runs, and by
[`check_environment_fits`](@ref).
"""
function has_missing_entries(columns::NamedTuple)
    return any(column -> any(value -> value isa MissingValue, column), values(columns))
end

"""
    decide_whether_environment_is_included(include_environment, environment, observations,
                                           actions, environment_save_history,
                                           check_stopping_criterion)

Return whether a session includes its environment, as the `include_environment` of
[`SessionInfo`](@ref) says: always with `true`, which needs an environment, never with
`false`, and with `nothing` only when an observation or an action is
[`MissingValue`](@ref), the environment's history is saved, or its stopping criterion is
checked on every timestep. A session without an environment never includes one. With
`false`, it is an error when an observation is `MissingValue()`, the environment's history
is saved, or its stopping criterion is checked on every timestep, since those need the
environment. Used by
[`prepare_session`](@ref).
"""
function decide_whether_environment_is_included(
    include_environment::Union{Nothing,Bool},
    environment,
    observations,
    actions,
    environment_save_history::SaveHistory,
    check_stopping_criterion,
)
    #A session without an environment has none to include
    if environment isa NoEnvironment
        include_environment === true && throw(
            ArgumentError(
                "include_environment = true needs an environment model, but the session has none.",
            ),
        )
        return false
    end
    include_environment === true && return true

    #What can't be done without the environment
    needs_environment = if has_missing_entries(observations)
        "the data has observations that are MissingValue(), which only the environment can impute"
    elseif !saves_nothing(environment_save_history)
        "environment_save_history saves its history"
    elseif check_stopping_criterion === true
        "check_stopping_criterion = true checks its stopping criterion on every timestep"
    else
        nothing
    end
    if include_environment === false
        needs_environment === nothing || throw(
            ArgumentError(
                "include_environment = false leaves the environment out of the session, but $needs_environment.",
            ),
        )
        return false
    end

    #By default it is included when it is needed, or when an action is to be imputed, which the observations it scores inform
    return needs_environment !== nothing || has_missing_entries(actions)
end

"""
    remake_session_without_environment(session_info)

Return the session with `NoEnvironment()` as its environment and everything else the same,
so that it runs as a session without an environment. Used by fitted sessions whose
environment doesn't run, since every observation is in the data.
"""
function remake_session_without_environment(session_info::SessionInfo)
    return SessionInfo(
        session_info.cognitive_model,
        NoEnvironment(),
        nothing,
        nothing,
        session_info.include_environment,
        false,
        session_info.observations,
        session_info.actions,
        session_info.observations,
        session_info.actions_to_condition_on,
        session_info.observations_to_condition_on,
        session_info.n_timesteps,
        session_info.who_starts,
        session_info.timestep_sizes,
        session_info.saved_timestep_sizes,
        session_info.cognitive_save_history,
        session_info.environment_save_history,
        session_info.check_stopping_criterion,
        session_info.first_timestep_to_check_stopping,
        session_info.previous_observation,
        session_info.previous_action,
        session_info.previous_observation,
    )
end

"""
    remake_session_including_environment(session_info)

Return the session including its environment when it has one, and everything else the
same. A session that was prepared with `include_environment = false` is an error. Used by
[`include_environments_if_estimated`](@ref).
"""
function remake_session_including_environment(session_info::SessionInfo)
    if has_environment(session_info) && session_info.include_environment === false
        throw(
            ArgumentError(
                "The environment's parameters are estimated, which every session must include its environment for, but a session was prepared with include_environment = false.",
            ),
        )
    end
    return SessionInfo(
        session_info.cognitive_model,
        session_info.environment,
        session_info.observation_map,
        session_info.action_map,
        session_info.include_environment,
        has_environment(session_info),
        session_info.observations,
        session_info.actions,
        session_info.environment_observations,
        session_info.actions_to_condition_on,
        session_info.observations_to_condition_on,
        session_info.n_timesteps,
        session_info.who_starts,
        session_info.timestep_sizes,
        session_info.saved_timestep_sizes,
        session_info.cognitive_save_history,
        session_info.environment_save_history,
        session_info.check_stopping_criterion,
        session_info.first_timestep_to_check_stopping,
        session_info.previous_observation,
        session_info.previous_action,
        session_info.previous_environment_observation,
    )
end

"""
    include_environments_if_estimated(environment_parameters_model, sessions_info)

Return the sessions as a full model runs them with `environment_parameters_model`: as they
are when it estimates nothing, and otherwise with every session that has an environment
including it, so that the observations in the data inform the parameters. Used by
[`prepare_full_model`](@ref).
"""
function include_environments_if_estimated(
    environment_parameters_model, sessions_info::MultipleSessionsInfo
)
    estimates_parameters(environment_parameters_model) || return sessions_info
    return MultipleSessionsInfo(
        map(remake_session_including_environment, sessions_info.sessions),
        sessions_info.session_ids;
        table=sessions_info.table,
        session_rows=sessions_info.session_rows,
    )
end
