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

"""
    session_column_values(sessions_info, column)

Return the values of `column` in the table that the sessions in `sessions_info` were split
from, as one vector for each session, of its rows in the table. It is an error when the
sessions were given already prepared, without a table, and when the table has no such
column. Useful for a session parameter model that reads a column, such as one to group the
sessions by, or the covariate of a regression. Used by [`group_sessions`](@ref).
"""
function session_column_values(sessions_info::MultipleSessionsInfo, column::Symbol)
    table = sessions_info.table
    if table === nothing
        throw(
            ArgumentError(
                "The sessions were given already prepared, without the table they came from, so they have no column $column. Split them from a table with MultipleSessionsInfo, or give it the table and the rows of each session as table and session_rows.",
            ),
        )
    end
    if !haskey(table, column)
        throw(
            ArgumentError(
                "The table that the sessions were split from has no column $column. Its columns are $(keys(table)).",
            ),
        )
    end
    values = getproperty(table, column)
    return [values[rows] for rows in sessions_info.session_rows]
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
    session_id_to_symbol(session_id)

Return the symbol name of a session, a `Symbol` made from its identifier: the value of a
single session column, as in `S1`, and the name and the value of each of several columns,
joined by `_` within a column and by `__` between columns, as in `id_S1__block_2`. The
chains of a fit name the entries of a session after it, such as `S1.learning_rate`, and
[`FixedParametersBySession`](@ref) can find the values of a session by it. After
`using CognitiveModeling: session_id_to_symbol`, the learning rate of a session is
`chain[@varname(\$(session_id_to_symbol(session_id)).learning_rate)]`.

# Examples
```jldoctest
julia> CognitiveModeling.session_id_to_symbol("S1")
:S1

julia> CognitiveModeling.session_id_to_symbol((; id = "S1", block = 2))
:id_S1__block_2
```
"""
function session_id_to_symbol(session_id)

    #Asserted, since the compiler can't infer that Symbol gives a Symbol for an identifier of any type
    return Symbol(session_id)::Symbol
end
function session_id_to_symbol(session_id::NamedTuple)
    return Symbol(
        join((string(name, "_", value) for (name, value) in pairs(session_id)), "__")
    )
end

"""
    check_unique_session_symbol_names(session_ids)

Throw an informative error if the identifiers of two sessions give the same symbol name, as
[`session_id_to_symbol`](@ref) gives it: when they are the same identifier, or give the same
text, such as `1` and `"1"`. Used by [`MultipleSessionsInfo`](@ref).
"""
function check_unique_session_symbol_names(session_ids::AbstractVector)
    symbol_names = map(session_id_to_symbol, session_ids)
    allunique(symbol_names) && return nothing
    for (i, symbol_name) in enumerate(symbol_names)
        first_i = findfirst(==(symbol_name), symbol_names)
        first_i == i && continue
        first_id = session_ids[first_i]
        session_id = session_ids[i]
        if isequal(first_id, session_id)
            throw(
                ArgumentError(
                    "The session identifier $(repr(session_id)) is given for more than one session. Every session must have an identifier of its own.",
                ),
            )
        end
        throw(
            ArgumentError(
                "The session identifiers $(repr(first_id)) and $(repr(session_id)) both give the symbol name $symbol_name, which the chains of a fit and FixedParametersBySession find a session by. The identifier of every session must give a symbol name of its own.",
            ),
        )
    end
    return nothing
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
    action_names_to_condition_on(session_info)
    observation_names_to_condition_on(session_info)

Return the names of the actions, or of the observations, that have values to condition on in
the data of a session, which are scored against the distributions the cognitive model, or
the environment model, gave for them, in a `Val`. A session without an environment scores no
observations, so it has no such names. Used by fitted sessions, to select the distributions
they score.
"""
function action_names_to_condition_on(session_info::SessionInfo)
    return Val(keys(session_info.actions_to_condition_on))
end
observation_names_to_condition_on(::SessionInfo{<:Any,NoEnvironment}) = Val(())
function observation_names_to_condition_on(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel}
)
    return Val(keys(session_info.observations_to_condition_on))
end

"""
    session_index(sessions_info, session_id)

Return the index `i` of the session with the identifier `session_id` in a
[`MultipleSessionsInfo`](@ref), which is how the models of several sessions number it: its
parameters are element `i` of each vector a session parameter model gives, and its saved
trajectories are element `i` of what the sessions return. The identifier is the value in the
session column, or a `NamedTuple` of the values when there are several, as in
`sessions_info.session_ids`. Used by users reading the results of a fit, not by the
models.
"""
function session_index(sessions_info::MultipleSessionsInfo, session_id)
    i = findfirst(==(session_id), sessions_info.session_ids)
    i === nothing && throw(
        ArgumentError(
            "There is no session with the identifier $(repr(session_id)). The identifiers are $(sessions_info.session_ids).",
        ),
    )
    return i
end

"""
    n_sessions(sessions_info)

Return how many sessions a [`MultipleSessionsInfo`](@ref) has, which a session parameter
model gives the parameters of. Used by the session parameter models of the inference
backends, such as those of [`IndependentSessionParameters`](@ref).
"""
n_sessions(sessions_info::MultipleSessionsInfo) = length(sessions_info.sessions)
