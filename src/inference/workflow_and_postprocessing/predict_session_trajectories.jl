## The trajectories of fitted sessions simulated again ##

"""
    predict_session_trajectories([rng,] model, chain;
                                 cognitive_save_history=SaveHistory(),
                                 environment_save_history=SaveHistory())

Simulate the sessions of `model`, an inference backend's model of fitted sessions, again
with the parameters of every sample in `chain`, and return the trajectories they saved for
every sample. This can be used for prior and posterior predictive checks.

The sessions are deconditioned with [`decondition_sessions`](@ref), so that every action is
simulated, and every observation too when the sessions have an environment, and they save
what the two save histories select. The model of those sessions then runs once for each
sample, with the sample's parameters, and any start that a model samples, and with every
variable within the timesteps drawn again, as data is in a predictive check: an action
imputed in the fit is simulated like the others. The trajectories the sessions return are
kept. The sessions have at most the number of timesteps of their data, and fewer when a
[`stopping_criterion`](@ref) of a model is met, since no timestep of a predictive session
has data. For a prior predictive check, give a chain sampled from the prior.

Each inference backend has methods for its own models and chains, which need its package to
be loaded. Which models they take, and the form of the chain they return, are described in
the documentation of the inference backend.
"""
function predict_session_trajectories end

## Sessions whose data is simulated instead of fitted to ##

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
same settings and the same environment. The observations of
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
        check_parameter_domains=session_info.check_parameter_domains,
        previous_observation=session_info.previous_observation,
        previous_action=session_info.previous_action,
    )
end

function decondition_sessions(sessions_info::MultipleSessionsInfo; keywords...)
    return MultipleSessionsInfo(
        map(session -> decondition_sessions(session; keywords...), sessions_info.sessions),
        sessions_info.session_ids;
        table=sessions_info.table,
        session_rows=sessions_info.session_rows,
    )
end
