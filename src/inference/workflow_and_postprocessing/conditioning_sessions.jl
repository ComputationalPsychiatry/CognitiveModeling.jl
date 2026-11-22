## Sessions whose data is simulated instead of fitted to, and the other way round ##

"""
    decondition_sessions(session_info; check_stopping_criterion=nothing)
    decondition_sessions(sessions_info; check_stopping_criterion=nothing)
    decondition_sessions(model; check_stopping_criterion=nothing)

Return a copy of a [`SessionInfo`](@ref), or of every session in a
[`MultipleSessionsInfo`](@ref), where every action is [`SimulatedMissingValue`](@ref), and
every observation too when the session has an environment model that it can include, so that
an inference backend's model of the copy simulates them instead of fitting to them. Given an
inference backend's model of fitted sessions, return a copy of the model whose sessions are
deconditioned, which is otherwise the same. An inference backend names the variables of such
sessions apart from those of fitted ones, so that everything within them is drawn again.
This can be used for predictive checks: evaluating the model of the copy with the samples of
a fit to the sessions, or sampling it from its prior, simulates each session again, with the
same number of timesteps at most, the same settings and the same environment, and
[`condition_sessions`](@ref) puts simulated values back in the data. The observations of a
session without an environment stay those in the data, as for an experiment whose
observations are fixed in advance. Every action is simulated, whatever it was in the data,
also one that was [`UnimputedMissingValue`](@ref) or [`NoValue`](@ref). The copy checks the
[`stopping_criterion`](@ref) of the models as `check_stopping_criterion` says, as for
[`SessionInfo`](@ref), whatever the session was prepared with: by default after the last
timestep with data, which in a session with an environment is before every timestep, so that
the copy can end before the data did, as a simulation does.

The models of the sessions take their data as arguments, and not as observed variables,
since the data does more than give values to score: the models receive it on each timestep,
it decides how many timesteps a session has, what is imputed and whether the environment
runs, and it is checked against the models before a fit. An inference backend's own way of
deconditioning, which frees observed variables, therefore leaves the data as it is, and
this function turns the data into variables that the model samples, so that the backend's
own functions can simulate it.

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> session_info = SessionInfo(Counter(), [1.0, 0.0, 1.0], [true, false, true]; who_starts = EnvironmentStarts());

julia> decondition_sessions(session_info).actions
(choice = [SimulatedMissingValue(), SimulatedMissingValue(), SimulatedMissingValue()],)
```
"""
function decondition_sessions(
    session_info::SessionInfo; check_stopping_criterion::Union{Nothing,Bool}=nothing
)

    #Every action is simulated, and every observation too when there is an environment to give them
    actions = map(
        column -> fill(SimulatedMissingValue(), length(column)), session_info.actions
    )
    observations =
        if has_environment(session_info) && session_info.include_environment !== false
            map(
                column -> fill(SimulatedMissingValue(), length(column)),
                session_info.observations,
            )
        else
            session_info.observations
        end

    return with_data(session_info, observations, actions; check_stopping_criterion)
end

function decondition_sessions(sessions_info::MultipleSessionsInfo; keywords...)
    return with_sessions(
        sessions_info,
        map(session -> decondition_sessions(session; keywords...), sessions_info.sessions),
    )
end

"""
    condition_sessions(model, values; check_stopping_criterion=nothing)
    condition_sessions(model, table; check_stopping_criterion=nothing)
    condition_sessions(model, observations, actions; check_stopping_criterion=nothing,
                       timestep_sizes=...)

Return a copy of `model`, an inference backend's model of fitted sessions, whose sessions
have `values` as data, and which is otherwise the same. Each action or observation that is
[`SimulatedMissingValue`](@ref) or [`ImputedMissingValue`](@ref) in the data of the
sessions, so that the model samples it as a variable, is the value that `values` has for
that variable, if it has one. A value of any other variable, such as a parameter, is an
error. This can be used for parameter recovery: a draw from the prior of the model that
[`decondition_sessions`](@ref) gives has parameters and data simulated with them, and the
copy is fitted to that data as to data in a table.

Each session of the copy ends after its last timestep with data: the timesteps after it,
where every action and observation would be sampled, are left out, as when a simulation that
a model stopped is fitted from a table with the timesteps it had. A session that gets no
value at all keeps its timesteps. The copy checks the [`stopping_criterion`](@ref) of the
models as `check_stopping_criterion` says, as for [`SessionInfo`](@ref), from the data it
has, which by default is after its last timestep, so never.

An inference backend's own way of conditioning gives the variables values while they stay
variables, which the model looks up by name on every evaluation. This function puts the
values in the data instead, so that the copy is the model that the same data in a table
gives: as fast to evaluate, checked in the same way, with an error for a value that has no
place in the data, and with each session ending after its last value.

The second form gives the sessions of `model`, several of them, the data in `table`, read
with the column settings that the sessions were made with, from the `column_settings` of
their [`MultipleSessionsInfo`](@ref), and with their other settings. The table must have the
same sessions, with the same data identifiers in the same order, so that what the model was
prepared with for each session still fits, and a session can have other timesteps. The third
form gives a single session `observations` and `actions`, in the forms that
[`SessionInfo`](@ref) takes them, with the session's sizes of timesteps unless others are
given as `timestep_sizes`. In both, the values that stand for a missing one, by default
`missing`, become [`ImputedMissingValue`](@ref), as when the sessions were made. The same
forms take a `MultipleSessionsInfo` or a `SessionInfo` in place of `model`, and return the
sessions with the data. A table which isn't a `NamedTuple` of columns needs Tables.jl to be
loaded.

Each inference backend has a method for its own models, which needs its package to be
loaded. Its documentation describes how the variables of the sessions are named, and in
which collections the values can be given.
"""
function condition_sessions end

#Reads the table with the settings that read the sessions' own, and checks that it has the same sessions, since the parameter models of a full model are prepared for them
function condition_sessions(
    sessions_info::MultipleSessionsInfo,
    table;
    check_stopping_criterion::Union{Nothing,Bool}=nothing,
)
    #The settings that read the sessions' table
    column_settings = sessions_info.column_settings
    if column_settings === nothing
        throw(
            ArgumentError(
                "The sessions were given already prepared, without the column settings that read them from a table, so another table can't be read in the same way. Make the sessions from a table with MultipleSessionsInfo(cognitive_model, table; observation_columns, action_columns, session_columns), or give MultipleSessionsInfo their column_settings.",
            ),
        )
    end

    #Read the table with them, and with the other settings of the sessions
    first_session = first(sessions_info.sessions)
    conditioned = MultipleSessionsInfo(
        first_session.cognitive_model,
        table;
        session_settings(first_session)...,
        column_settings...,
        check_stopping_criterion,
    )

    #The table must have the same sessions, in the same order
    if !isequal(conditioned.session_ids, sessions_info.session_ids)
        throw(
            ArgumentError(
                "The table has the sessions $(first(conditioned.session_ids, 5)), but the model has $(first(sessions_info.session_ids, 5)), and condition_sessions gives the same sessions other data, in the same order, since what the model was prepared with for each session must still fit. For other sessions, make the model again with them.",
            ),
        )
    end
    return conditioned
end

#Prepares the session again with the new data, where the values that stand for a missing one are marked as when a session is made
function condition_sessions(
    session_info::SessionInfo,
    observations,
    actions;
    check_stopping_criterion::Union{Nothing,Bool}=nothing,
    timestep_sizes=session_info.timestep_sizes,
)
    return with_data(
        session_info,
        observations,
        actions;
        check_stopping_criterion,
        timestep_sizes,
        missing_observation_values=(missing,),
        missing_action_values=(missing,),
    )
end

"""
    with_data(session_info, observations, actions; check_stopping_criterion,
              timestep_sizes=session_info.timestep_sizes, missing_observation_values=(),
              missing_action_values=())

Return a copy of `session_info`, a [`SessionInfo`](@ref), with `observations` and `actions`
as its data, prepared again with its settings, so that what is scored and whether the
environment runs follow from the new data. The copy checks the stopping criterion as
`check_stopping_criterion` says, and has the sizes of the timesteps of `session_info`,
unless others are given, as for data with other timesteps. No value in the data is replaced,
unless values that stand for a missing one are given. Used by [`decondition_sessions`](@ref)
and the methods of [`condition_sessions`](@ref).
"""
function with_data(
    session_info::SessionInfo,
    observations,
    actions;
    check_stopping_criterion::Union{Nothing,Bool},
    timestep_sizes=session_info.timestep_sizes,
    missing_observation_values=(),
    missing_action_values=(),
)
    return prepare_session(
        session_info.cognitive_model,
        observations,
        actions;
        session_settings(session_info)...,
        check_stopping_criterion,
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
        session_info.check_stopping_criterion,
        session_info.check_parameter_domains,
        session_info.previous_observation,
        session_info.previous_action,
    )
end

#The timesteps after the last one with data are where a session checks the stopping criterion of its models by default, and samples every action and observation, so that their number, and with it the number of variables, would change with the parameters, which a sampler that uses gradients can't handle. Without them, the session is the one that a table with the timesteps of a stopped simulation gives
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
