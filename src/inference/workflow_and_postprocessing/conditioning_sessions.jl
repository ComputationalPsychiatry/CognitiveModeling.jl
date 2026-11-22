## Sessions whose data is simulated instead of fitted to, and the other way round ##

"""
    decondition_sessions(session_info; check_stopping_criterion=nothing)
    decondition_sessions(sessions_info; check_stopping_criterion=nothing)
    decondition_sessions(model; check_stopping_criterion=nothing)

Return a copy of a [`SessionInfo`](@ref), or of every session in a
[`MultipleSessionsInfo`](@ref), where every action is [`SimulatedValue`](@ref), and every
observation too when the session has an environment model that it can include, so that an
inference backend's model of the copy simulates them instead of fitting to them. Given an
inference backend's model of fitted sessions, return a copy of the model whose sessions are
deconditioned, which is otherwise the same. This can be used for predictive checks:
evaluating the model of the copy with the samples of a fit to the sessions, or sampling it
from its prior, simulates each session again, with the same number of timesteps at most, the
same settings and the same environment, and [`condition_sessions`](@ref) puts simulated
values back in the data. The observations of a session without an environment stay those in
the data, as for an experiment whose observations are fixed in advance. Every action is
simulated, whatever it was in the data, also one that was [`UnimputedMissingValue`](@ref) or
[`NoValue`](@ref). The copy checks the [`stopping_criterion`](@ref) of the models as
`check_stopping_criterion` says, as for [`SessionInfo`](@ref), whatever the session was
prepared with: by default after the last timestep with data, which in a session with an
environment is before every timestep, so that the copy can end before the data did, as a
simulation does.

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> session_info = SessionInfo(Counter(), [1.0, 0.0, 1.0], [true, false, true]; who_starts = EnvironmentStarts());

julia> decondition_sessions(session_info).actions
(choice = [SimulatedValue(), SimulatedValue(), SimulatedValue()],)
```
"""
function decondition_sessions(
    session_info::SessionInfo; check_stopping_criterion::Union{Nothing,Bool}=nothing
)

    #Every action is simulated, and every observation too when there is an environment to give them
    actions = map(column -> fill(SimulatedValue(), length(column)), session_info.actions)
    observations =
        if has_environment(session_info) && session_info.include_environment !== false
            map(column -> fill(SimulatedValue(), length(column)), session_info.observations)
        else
            session_info.observations
        end

    return with_data(session_info, observations, actions; check_stopping_criterion)
end

function decondition_sessions(sessions_info::MultipleSessionsInfo; keywords...)
    return MultipleSessionsInfo(
        map(session -> decondition_sessions(session; keywords...), sessions_info.sessions),
        sessions_info.session_ids;
        table=sessions_info.table,
        session_rows=sessions_info.session_rows,
    )
end

"""
    condition_sessions(model, values; check_stopping_criterion=nothing)

Return a copy of `model`, an inference backend's model of fitted sessions, whose sessions
have `values` as data, and which is otherwise the same. Each action or observation that is
[`SimulatedValue`](@ref) or [`MissingValue`](@ref) in the data of the sessions, so that the
model samples it as a variable, is the value that `values` has for that variable, if it has
one. A value of any other variable, such as a parameter, is an error. This can be used for
parameter recovery: a draw from the prior of the model that [`decondition_sessions`](@ref)
gives has parameters and data simulated with them, and the copy is fitted to that data as to
data in a table. The copy checks the [`stopping_criterion`](@ref) of the models as
`check_stopping_criterion` says, as for [`SessionInfo`](@ref), from the data it has.

Each inference backend has a method for its own models, which needs its package to be
loaded. Its documentation describes how the variables of the sessions are named, and in
which collections the values can be given.
"""
function condition_sessions end

"""
    with_data(session_info, observations, actions; check_stopping_criterion)

Return a copy of `session_info`, a [`SessionInfo`](@ref), with `observations` and `actions`
as its data, prepared again with its settings, so that what is scored and whether the
environment runs follow from the new data, in which no value is replaced. The copy checks
the stopping criterion as `check_stopping_criterion` says. Used by
[`decondition_sessions`](@ref) and the methods of [`condition_sessions`](@ref).
"""
function with_data(
    session_info::SessionInfo,
    observations::NamedTuple,
    actions::NamedTuple;
    check_stopping_criterion::Union{Nothing,Bool},
)
    return prepare_session(
        session_info.cognitive_model,
        observations,
        actions;
        missing_observation_values=(),
        missing_action_values=(),
        who_starts=session_info.who_starts,
        environment=session_info.environment,
        observation_map=session_info.observation_map,
        action_map=session_info.action_map,
        include_environment=session_info.include_environment,
        timestep_sizes=session_info.timestep_sizes,
        cognitive_save_history=session_info.cognitive_save_history,
        environment_save_history=session_info.environment_save_history,
        check_stopping_criterion,
        check_parameter_domains=session_info.check_parameter_domains,
        previous_observation=session_info.previous_observation,
        previous_action=session_info.previous_action,
    )
end

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
