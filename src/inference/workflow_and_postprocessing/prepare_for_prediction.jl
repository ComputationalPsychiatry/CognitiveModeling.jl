## Sessions whose data is simulated instead of fitted to ##

"""
    prepare_for_prediction(session_info; check_stopping_criterion=nothing)
    prepare_for_prediction(sessions_info; check_stopping_criterion=nothing)

Return a copy of a [`SessionInfo`](@ref), or of every session in a
[`MultipleSessionsInfo`](@ref), where every action is [`MissingValue`](@ref), and every
observation too when the session has an environment model that it can include, so that an
inference backend's model of the copy simulates them instead of fitting to them. This can be
used for predictive checks: evaluating the model of the copy with the samples of a fit to
the sessions, or sampling it from its prior, simulates each session again, with the same
number of timesteps at most, the same settings and the same environment. The observations of
a session without an environment stay those in the data, as for an experiment whose
observations are fixed in advance. Every action is simulated, whatever it was in the data,
also one that was [`UnimputedMissingValue`](@ref) or [`NoValue`](@ref). The copy checks the
[`stopping_criterion`](@ref) of the models as `check_stopping_criterion` says, as for
[`SessionInfo`](@ref), whatever the session was prepared with: by default after the last
timestep with data, which in a session with an environment is before every timestep, so
that the copy can end before the data did, as a simulation does.

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> session_info = SessionInfo(Counter(), [1.0, 0.0, 1.0], [true, false, true]; who_starts = EnvironmentStarts());

julia> prepare_for_prediction(session_info).actions
(choice = [MissingValue(), MissingValue(), MissingValue()],)
```
"""
function prepare_for_prediction(
    session_info::SessionInfo; check_stopping_criterion::Union{Nothing,Bool}=nothing
)

    #Every action is missing, and every observation too when there is an environment to give them
    actions = map(column -> fill(MissingValue(), length(column)), session_info.actions)
    observations =
        if has_environment(session_info) && session_info.include_environment !== false
            map(column -> fill(MissingValue(), length(column)), session_info.observations)
        else
            session_info.observations
        end

    #The copy is prepared again with the settings of the session, so that what is scored and whether the environment runs follow from the new data, in which nothing more is replaced
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
        previous_observation=session_info.previous_observation,
        previous_action=session_info.previous_action,
    )
end

function prepare_for_prediction(sessions_info::MultipleSessionsInfo; keywords...)
    return MultipleSessionsInfo(
        map(
            session -> prepare_for_prediction(session; keywords...), sessions_info.sessions
        ),
        sessions_info.session_ids;
        table=sessions_info.table,
        session_rows=sessions_info.session_rows,
    )
end
