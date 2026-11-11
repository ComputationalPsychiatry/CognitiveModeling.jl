## Type for data where a value wasn't recorded ##

"""
    MissingValue()

Stand-in for an action or an observation that happened on a timestep but isn't in the data,
as one entry of a column or a `NamedTuple`, such as `(; choice = MissingValue())`. Whether
it is an action or an observation is given by where it is.

Fitting imputes a missing action by sampling it from the model's action distribution on that
timestep, or passes it on as [`NoAction`](@ref) when it is told not to impute. It imputes a
missing observation from what an environment model gives for the observation on that
timestep: as a latent variable with that distribution, or as the value of a
[`Deterministic`](@ref). [`NoAction`](@ref) and [`NoObservation`](@ref) are different: they
say that nothing was done or observed on the timestep. Unlike `missing`, arithmetic on it is
an error. A [`SessionInfo`](@ref) replaces `missing` in the data by it.
"""
struct MissingValue end

## Types for how a session is set up, which are the same whatever it is fitted with ##

"""
    NoEnvironment()

The environment of a session that has none, which is the default of [`SessionInfo`](@ref).
Every observation then comes from the data, and none can be [`MissingValue`](@ref).
"""
struct NoEnvironment end

"""
    SessionInfo(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Everything about one behavioural session that is fixed within a generative model to be
fitted: the data, the models and the settings, checked and prepared once. An inference
backend's model of the session takes it together with the parameters, which are what
changes between evaluations. [`MultipleSessionsInfo`](@ref) makes one for each session in a
table.

# Arguments
- `cognitive_model::AbstractModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `observations`: the observations in the data, as a `NamedTuple` with one vector for each
  name and one entry for each timestep, such as `(; reward = [1.0, 0.0, 1.0])`. The data
  must have every observation the model lists in [`attribute_names`](@ref), and can have
  more. When the model has only one observation in its attributes, a single vector,
  `[1.0, 0.0, 1.0]`, is the data of that observation, whatever its name. An observation that
  wasn't recorded is [`MissingValue`](@ref), which `environment` fills in if it imputed, and
  one that wasn't made is [`NoObservation`](@ref).
- `actions`: the actions in the data, in the same forms, with exactly the actions the model
  lists. An action that was taken but not recorded is `MissingValue()`, and is imputed
  unless `impute_missing_actions` is `false`. An action that wasn't taken is
  [`NoAction`](@ref), and isn't scored.

# Keyword arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: whether the model acts on the
  observation of the same timestep, `EnvironmentStarts()`, or on the observation of the
  timestep before, `AgentStarts()`, as in [`simulate`](@ref). It has no default, because a
  wrong choice shifts the model by a timestep without any error.
- `environment::Union{NoEnvironment, AbstractModel}=NoEnvironment()`: an environment
  model, usually an [`AbstractEnvironmentModel`](@ref), which fills in the observations that
  are `MissingValue()`, and against whose distributions the observations in the data are
  scored. Its outputs must be the inputs of `cognitive_model` and the other way round,
  directly or through `observation_map` and `action_map`. Without it, the data can't have
  missing observations. It runs in a session where an observation or an action is missing,
  where `environment_save_history` saves anything, or where `check_stopping_criterion` is
  `true`: the scored observations then inform what is missing, such as which arm of a
  bandit a missing choice was, from the reward that followed. It also runs in every session
  of an inference backend's full model that samples the environment's parameters, so that
  the observations inform them. Otherwise the session runs without it, since the scores of
  its observations would be the same for every value of the parameters, and has none of the
  environment's variables. The environment is kept in the session all the same.
- `observation_map::Union{Nothing, NameMap}=nothing` and
  `action_map::Union{Nothing, NameMap}=nothing`: which observations of `environment` the
  cognitive model receives, and which of its actions the environment receives, and under
  which names, as [`NameMap`](@ref)s, as for `simulate`. The data is in the cognitive
  model's names, and is scored against the environment's distributions as the map passes
  them on. An output of the environment that the map doesn't pass on isn't in the data, so
  it is imputed on every timestep. Each model's history is in its own names.
- `timestep_sizes=1`: how much time each timestep covers, as in `simulate`.
- `impute_missing_actions::Bool=true`: whether an action that is `MissingValue()` is
  imputed, as a latent variable. When it is `false`, nothing is sampled for it, and it is
  replaced by `NoAction()` here: `cognitive_model` receives `NoAction()` as that previous
  action on the next timestep, and `environment` as the action it responds to.
- `cognitive_save_history::SaveHistory=SaveHistory()`: which states, inputs, outputs and
  output distributions of `cognitive_model` a fitted session returns, by name, and whether
  timestep 0 is, as a [`SaveHistory`](@ref) in the model's own names. Nothing is selected by
  default. What is returned names them by their role, as `states`, `observations`,
  `actions` and `action_distributions`, as [`simulate`](@ref) does.
- `environment_save_history::SaveHistory=SaveHistory()`: the same for `environment`, whose
  inputs are the actions and whose outputs are the observations, returned under
  `environment`. Selecting anything makes the environment run in the session, also when
  nothing is missing, which adds its scores of the observations to the log density; with
  fixed environment parameters they are the same for every value of the cognitive
  parameters, so the posterior is unchanged. Whether timestep 0 is saved follows
  `cognitive_save_history`.
- `check_stopping_criterion::Union{Nothing, Bool}=nothing`: when
  [`stopping_criterion`](@ref) of the two models is checked before a timestep, so that the
  session ends there, with what it saved up to it. By default it is checked only on the
  timesteps after the last one with data, where a value is neither `MissingValue()` nor a
  marker for nothing happening, so that a fit never checks it and a session prepared by
  [`prepare_for_prediction`](@ref), whose data is all missing, ends as a simulation does.
  With `true` it is checked before every timestep, which makes the environment run, and a
  model whose criterion is met while the data continues is an error, since the recorded
  values after that timestep couldn't be scored: this checks that the models agree with the
  length of the data. With `false` it is never checked.
- `missing_observations::Union{MissingValue, NoObservation}=MissingValue()` and
  `missing_actions::Union{MissingValue, NoAction}=MissingValue()`: what a `missing` in the
  data becomes, since a table marks every gap the same way: a value that wasn't recorded, or
  nothing happening on that timestep. The two markers can also be put in the data directly.
- `previous_observation` and `previous_action`: what the first timestep receives as the
  observation and the action before it, in the cognitive model's names, with
  [`BeforeFirstTimestep`](@ref) for each name by default, as in `simulate`. The environment
  starts from the same observation in its own names, with `BeforeFirstTimestep()` for an
  output that `observation_map` doesn't pass on. The states where the session starts are
  not given here, since they come from [`initialize_state`](@ref) with the parameters that
  are fitted.

Each argument is kept in a field of the same name, with the two replacements made in the
data, except `impute_missing_actions`, whose work is done here. The other fields are
`runs_environment`, whether the environment runs, `environment_observations`, the data of
the environment's outputs in its own names, `previous_environment_observation`, the
observation the environment starts from in its own names, `recorded_actions` and
`recorded_observations`, the recorded values of each name which are scored, the observations
in the environment's names, `n_timesteps`, `saved_timestep_sizes`, the sizes a fitted
session returns with what it saves, and `first_timestep_to_check_stopping`, the first
timestep before which the stopping criterion is checked, as `check_stopping_criterion`
decides it.

    SessionInfo(session_info; cognitive_save_history, environment_save_history)

Update a session to have other save histories, which returns a new `SessionInfo`. This is
for after a fit: a session is fitted fastest when it saves nothing, and its states, or
anything else it can save, are read afterwards from the updated session, by evaluating its
model with each sample of the fit. The two models have the same variables, so every sample
fits both. The documentation of each inference backend describes how.

In the REPL, a session shows a summary on one line each: its models, its number of
timesteps, the names of the observations and the actions in its data, which are the
cognitive model's names, and who starts.

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> session_info = SessionInfo(Counter(), (; reward = [1.0, 0.0, missing]), (; choice = [true, missing, false]); who_starts = EnvironmentStarts(), missing_observations = NoObservation())
SessionInfo
Cognitive model: Counter
Environment: none
Timesteps: 3
Observations: reward
Actions: choice
Who starts: environment

julia> session_info.observations.reward[3], session_info.actions.choice[2]
(NoObservation(), MissingValue())

julia> session_info.recorded_actions
(choice = Bool[1, 0],)
```
"""
struct SessionInfo{
    Tcognitive_model,
    Tenvironment,
    Tobservations,
    Tactions,
    Trecorded_actions,
    Trecorded_observations,
    Twho_starts,
    Ttimestep_sizes,
    Tsaved_timestep_sizes,
    Tcognitive_save_history,
    Tenvironment_save_history,
    Tprevious_observation,
    Tprevious_action,
    Tobservation_map,
    Taction_map,
    Tenvironment_observations,
    Tprevious_environment_observation,
}
    cognitive_model::Tcognitive_model
    environment::Tenvironment
    observation_map::Tobservation_map
    action_map::Taction_map
    runs_environment::Bool
    observations::Tobservations
    actions::Tactions
    environment_observations::Tenvironment_observations
    recorded_actions::Trecorded_actions
    recorded_observations::Trecorded_observations
    n_timesteps::Int
    who_starts::Twho_starts
    timestep_sizes::Ttimestep_sizes
    saved_timestep_sizes::Tsaved_timestep_sizes
    cognitive_save_history::Tcognitive_save_history
    environment_save_history::Tenvironment_save_history
    check_stopping_criterion::Union{Nothing,Bool}
    first_timestep_to_check_stopping::Int
    previous_observation::Tprevious_observation
    previous_action::Tprevious_action
    previous_environment_observation::Tprevious_environment_observation
end

#Prepares the session
function SessionInfo(cognitive_model::AbstractModel, observations, actions; keywords...)
    return prepare_session(cognitive_model, observations, actions; keywords...)
end

#The same session with other save histories, prepared again, since whether the environment runs can follow from them
function SessionInfo(
    session_info::SessionInfo;
    cognitive_save_history::SaveHistory=session_info.cognitive_save_history,
    environment_save_history::SaveHistory=session_info.environment_save_history,
)
    return prepare_session(
        session_info.cognitive_model,
        session_info.observations,
        session_info.actions;
        who_starts=session_info.who_starts,
        environment=session_info.environment,
        observation_map=session_info.observation_map,
        action_map=session_info.action_map,
        timestep_sizes=session_info.timestep_sizes,
        cognitive_save_history,
        environment_save_history,
        check_stopping_criterion=session_info.check_stopping_criterion,
        previous_observation=session_info.previous_observation,
        previous_action=session_info.previous_action,
    )
end

#A session without an environment has no environment history whose names could be checked
check_savehistory_selection_names(::SaveHistory, ::NoEnvironment) = nothing

#A session without an environment has no environment whose stopping criterion could be met
stopping_criterion(::NoEnvironment, state, parameters) = false
check_stopping_criterion_not_met(::NoEnvironment, state, parameters) = nothing

"""
    prepare_session(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Return the [`SessionInfo`](@ref) of one session, with the checks of the data and the names,
the replacements of `missing` in the data, whether the environment runs, and what is scored
and returned prepared, as its constructor describes. Used by `SessionInfo` and
[`MultipleSessionsInfo`](@ref).
"""
function prepare_session(
    cognitive_model::AbstractModel,
    observations,
    actions;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    environment::Union{NoEnvironment,AbstractModel}=NoEnvironment(),
    observation_map::Union{Nothing,NameMap}=nothing,
    action_map::Union{Nothing,NameMap}=nothing,
    timestep_sizes=1,
    impute_missing_actions::Bool=true,
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
    check_stopping_criterion::Union{Nothing,Bool}=nothing,
    missing_observations::Union{MissingValue,NoObservation}=MissingValue(),
    missing_actions::Union{MissingValue,NoAction}=MissingValue(),
    previous_observation=input_before_first_timestep(cognitive_model),
    previous_action=output_before_first_timestep(cognitive_model),
)

    #A single vector is the data of the observation or the action, when the model has only one in its attributes
    observations = named_data(observations, cognitive_model, :observations)
    actions = named_data(actions, cognitive_model, :actions)

    #Replace missing in the data by the marker for a value that wasn't recorded, or for nothing happening
    observations = replace_missing(observations, missing_observations)
    actions = replace_missing(actions, missing_actions)

    #Check the data against the names the cognitive model lists, and that the environment fits
    n_timesteps = check_data(cognitive_model, observations, actions)
    check_environment_fits(
        cognitive_model, environment, observations, observation_map, action_map
    )

    #Check that the save histories select names that the models list
    check_savehistory_selection_names(cognitive_save_history, cognitive_model)
    check_savehistory_selection_names(environment_save_history, environment)

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_timesteps
        throw(
            DimensionMismatch(
                "timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and the data has $n_timesteps timesteps.",
            ),
        )
    end

    #Missing actions that aren't imputed are passed on as actions that weren't taken
    if !impute_missing_actions
        actions = without_missing_actions(actions)
    end

    #The environment runs only when something is missing, an observation it fills in or an action that the observations it scores inform, when its history is saved, or when its stopping criterion is checked on every timestep
    runs_environment =
        environment isa AbstractModel && (
            has_missing_entries(observations) ||
            has_missing_entries(actions) ||
            !saves_nothing(environment_save_history) ||
            check_stopping_criterion === true
        )

    #The environment's data is in its own names, as the observation map passes it on, with outputs that the map doesn't pass on imputed on every timestep, and it starts from the previous observation in the same way
    environment_observations = environment_view(
        observations, environment, observation_map, fill(MissingValue(), n_timesteps)
    )
    previous_environment_observation = environment_view(
        previous_observation, environment, observation_map, BeforeFirstTimestep()
    )

    #The recorded actions and observations are scored at the end, the observations in the environment's names, and the sizes of the timesteps are returned with what is saved
    recorded_actions = recorded_values(actions)
    recorded_observations = recorded_values(environment_observations)
    saved_sizes = saved_timestep_sizes(timestep_sizes, n_timesteps, cognitive_save_history)

    #The stopping criterion is checked from the timestep after the last one with data, unless it is always or never checked
    first_timestep_to_check = first_timestep_to_check_stopping(
        check_stopping_criterion, observations, actions, n_timesteps
    )

    return SessionInfo(
        cognitive_model,
        environment,
        observation_map,
        action_map,
        runs_environment,
        observations,
        actions,
        environment_observations,
        recorded_actions,
        recorded_observations,
        n_timesteps,
        who_starts,
        timestep_sizes,
        saved_sizes,
        cognitive_save_history,
        environment_save_history,
        check_stopping_criterion,
        first_timestep_to_check,
        previous_observation,
        previous_action,
        previous_environment_observation,
    )
end

"""
    MultipleSessionsInfo(cognitive_model, table; observation_columns, action_columns,
                         session_columns, <keyword arguments>)
    MultipleSessionsInfo(sessions, session_ids)

Multiple behavioural sessions in a generative model that are fitted together, as a vector of
[`SessionInfo`](@ref) with an identifier for each, which an inference backend's models of
several sessions take.

The first form splits a table into sessions. `table` is a `NamedTuple` with one vector for
each column, or any other table that Tables.jl can read, such as a `DataFrame` or a
`CSV.File`, whose columns are read when Tables.jl is loaded, as it is with DataFrames.jl and
CSV.jl.
A `missing` in a column becomes [`MissingValue`](@ref), or the marker for nothing happening,
as the keyword arguments `missing_observations` and `missing_actions` of `SessionInfo` say.
Every session must come out with the same type, which they do when they share the models and
the settings, since the timesteps of different sessions are then compiled once. As for
[`SessionInfo`](@ref), an environment runs in a session where an observation or an action
is missing, where `environment_save_history` saves anything or where
`check_stopping_criterion` is `true`, and in every session of an inference backend's full
model that samples the environment's parameters.

    MultipleSessionsInfo(sessions_info; cognitive_save_history, environment_save_history)

The same sessions with other save histories for every one, see [`SessionInfo`](@ref).

# Arguments
- `cognitive_model::AbstractModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `table`: the data of every session, in columns with one entry for each timestep, such as
  `(; id = ["S1", "S1", "S2", "S2"], reward = [1.0, 0.0, 1.0, 1.0], choice = [true, false, true, true])`,
  or a `DataFrame` with those columns.

# Keyword arguments
- `observation_columns`: the columns with the observations. When the model has only one
  observation in its attributes, a single name, such as `:outcome`, is its column, whatever
  the observation is called. Several observations are given as the names the model lists in
  [`attribute_names`](@ref) when the columns are named the same, such as `(:reward, :cue)`,
  or as a `NamedTuple` from the model's name to the column's, such as `(; reward = :outcome,
  cue = :stimulus)`, so that one model fits tables with other names. A single name that the
  model lists as one of its actions is an error, since the two keyword arguments may then be
  swapped. An observation or action whose value is composed from several columns is
  specified by how to build it from the columns: a `NamedTuple` of column names for a
  `NamedTuple` in each row, `(; ssm_action = (; choice = :choicecol, rt = :rtcol))`, a
  `Tuple` of column names for a `Tuple`, `(; ssm_action = (:choicecol, :rtcol))`, or a
  `Tuple` of column names and a function of their values,
  `(; ssm_action = (:choicecol, :rtcol) => (choice, rt) -> [choice, rt])`. A `missing` in
  any of the columns makes the value `missing`.
- `action_columns`: the columns with the actions, in the same forms.
- `session_columns`: the column or columns which separate sessions, such as `:id` or
  `(:id, :block)`. A session is every row with one combination of their values, in the
  order of the first row of each. The identifier of a session is its value in a single
  column, or a `NamedTuple` of its values in several.
- `timestep_sizes=1`: how much time each timestep covers, as one size for all, or the name
  of a column with the size of each timestep.
- The other keyword arguments of [`SessionInfo`](@ref), which every session gets.

The second form takes sessions that are already prepared. Each argument is kept in a field
of the same name.

In the REPL, the sessions show a summary on one line each: how many there are, and the
models and the names in the data of the first.

# Examples
```jldoctest multiple_sessions_info
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> table = (; id = ["S1", "S1", "S2", "S2", "S2"], outcome = [1.0, 0.0, 1.0, 1.0, 0.0], choice = [true, false, true, missing, false]);

julia> sessions_info = MultipleSessionsInfo(Counter(), table; observation_columns = (; reward = :outcome), action_columns = :choice, session_columns = :id, who_starts = EnvironmentStarts())
MultipleSessionsInfo
Sessions: 2
Cognitive model: Counter
Environment: none
Observations: reward
Actions: choice

julia> sessions_info.session_ids
2-element Vector{String}:
 "S1"
 "S2"

julia> sessions_info.sessions[2].actions
(choice = Union{MissingValue, Bool}[true, MissingValue(), false],)
```

A `DataFrame` is given in the same way:

```jldoctest multiple_sessions_info
julia> using DataFrames

julia> data = DataFrame(id = ["S1", "S1", "S2", "S2"], reward = [1.0, 0.0, 1.0, 1.0], choice = [true, missing, false, true]);

julia> sessions_info = MultipleSessionsInfo(Counter(), data; observation_columns = :reward, action_columns = :choice, session_columns = :id, who_starts = EnvironmentStarts());

julia> sessions_info.sessions[1].actions
(choice = Union{MissingValue, Bool}[true, MissingValue()],)
```
"""
struct MultipleSessionsInfo{
    Tsessions<:AbstractVector{<:SessionInfo},Tsession_ids<:AbstractVector
}
    sessions::Tsessions
    session_ids::Tsession_ids

    #Only takes sessions of one type, with an identifier for each
    function MultipleSessionsInfo(
        sessions::AbstractVector{<:SessionInfo}, session_ids::AbstractVector
    )
        isempty(sessions) && throw(ArgumentError("There must be at least one session."))
        if length(sessions) != length(session_ids)
            throw(
                DimensionMismatch(
                    "There are $(length(sessions)) sessions and $(length(session_ids)) session identifiers, but there must be one identifier for each session.",
                ),
            )
        end
        check_unique_session_names(session_ids)
        if !isconcretetype(eltype(sessions))
            throw(
                ArgumentError(
                    "The sessions must all have the same type, but the vector has element type $(eltype(sessions)). They have one type when they share the cognitive model, the environment and the settings, and their columns have the same element types.",
                ),
            )
        end
        return new{typeof(sessions),typeof(session_ids)}(sessions, session_ids)
    end
end

#Splits a table into sessions by the values of the session columns, and prepares each of them in the same way
function MultipleSessionsInfo(
    cognitive_model::AbstractModel,
    table;
    observation_columns,
    action_columns,
    session_columns,
    timestep_sizes=1,
    keywords...,
)

    #A table of another kind, such as a DataFrame, is read as a NamedTuple of its columns
    table = column_table(table)

    #The columns of each kind, from the names the model uses to the names in the table
    observation_column_map = column_map(observation_columns, cognitive_model, :observations)
    action_column_map = column_map(action_columns, cognitive_model, :actions)
    id_columns = as_tuple_of_names(session_columns)
    check_table(
        table,
        (
            Iterators.flatten(map(columns_used, values(observation_column_map)))...,
            Iterators.flatten(map(columns_used, values(action_column_map)))...,
            id_columns...,
            (timestep_sizes isa Symbol ? (timestep_sizes,) : ())...,
        ),
    )

    #The rows of each session, and its identifier
    sessions_rows = rows_per_session(table, id_columns)

    #Each session is prepared from its rows of the table
    sessions = map(sessions_rows) do (_, rows)
        return prepare_session(
            cognitive_model,
            select_rows(table, observation_column_map, rows),
            select_rows(table, action_column_map, rows);
            timestep_sizes=select_timestep_sizes(table, timestep_sizes, rows),
            keywords...,
        )
    end
    session_ids = map(first, sessions_rows)

    return MultipleSessionsInfo(sessions, session_ids)
end

#The same sessions with other save histories
function MultipleSessionsInfo(sessions_info::MultipleSessionsInfo; keywords...)
    return MultipleSessionsInfo(
        map(session -> SessionInfo(session; keywords...), sessions_info.sessions),
        sessions_info.session_ids,
    )
end

"""
    prepare_for_prediction(session_info; check_stopping_criterion=nothing)
    prepare_for_prediction(sessions_info; check_stopping_criterion=nothing)

Return a copy of a [`SessionInfo`](@ref), or of every session in a
[`MultipleSessionsInfo`](@ref), where every action is [`MissingValue`](@ref), and every
observation too when the session has an environment model, so that an inference backend's
model of the copy simulates them instead of fitting to them. This can be used for
predictive checks: evaluating the model of the copy with the samples of a fit to the
sessions, or sampling it from its prior, simulates each session again, with the same number
of timesteps at most, the same settings and the same environment. The observations of a
session without an environment stay those in the data, as for an experiment whose
observations are fixed in advance. Every missing action is imputed, whatever
`impute_missing_actions` the session was prepared with. The copy checks the
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
    observations = if has_environment(session_info)
        map(column -> fill(MissingValue(), length(column)), session_info.observations)
    else
        session_info.observations
    end

    #The copy is prepared again with the settings of the session, so that what is scored and whether the environment runs follow from the new data
    return prepare_session(
        session_info.cognitive_model,
        observations,
        actions;
        who_starts=session_info.who_starts,
        environment=session_info.environment,
        observation_map=session_info.observation_map,
        action_map=session_info.action_map,
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
        sessions_info.session_ids,
    )
end

## Types for the parts of a full model ##

"""
    AbstractSessionParametersModel

Supertype for the descriptions of how the parameters of every session are given or sampled
in a full model, the model that an inference backend fits to several sessions, for the
cognitive model and for the environment model alike. The package comes with
[`NoParameters`](@ref), [`FixedSessionParameters`](@ref),
[`FixedParametersBySession`](@ref), [`IdenticalSessionParameters`](@ref) and
[`IndependentSessionParameters`](@ref).

# Implementation
Create a subtype, and for each inference backend it is used with, add a method of that
backend's function for parameters models, as the backend's documentation describes.
"""
abstract type AbstractSessionParametersModel end

"""
    NoParameters()

The parameters model of a full model which gives no parameters, so that each parameter
takes the value the model gives it in [`default_parameters`](@ref). It is the default for
the parameters of the environment.
"""
struct NoParameters <: AbstractSessionParametersModel end

"""
    FixedSessionParameters(parameters)

The parameters model of a full model which gives every session the same values, such as
`(; scale = 2.0)`, with nothing sampled. The values are kept in the field `parameters`.
"""
struct FixedSessionParameters{Tparameters<:NamedTuple} <: AbstractSessionParametersModel
    parameters::Tparameters
end

"""
    FixedParametersBySession(parameters_by_session)

The parameters model of a full model which gives each session values of its own, with
nothing sampled. `parameters_by_session` holds the values of each session, either in a
`NamedTuple` with an entry for each session, named by the session's name as
[`CognitiveModeling.session_name`](@ref) gives it, such as
`(; S1 = (; scale = 2.0), S2 = (; scale = 3.0))`, or in a dictionary from each session's
identifier to its values, such as `Dict("S1" => (; scale = 2.0), "S2" => (; scale = 3.0))`.
An identifier from several session columns is the `NamedTuple` of their values, so its
entry is named like `id_S1__block_2`, and its key is like `(; id = "S1", block = 2)`. Every
session must have values for the same parameters. The values are kept in the field
`parameters_by_session`.
"""
struct FixedParametersBySession{Tparameters_by_session<:Union{NamedTuple,AbstractDict}} <:
       AbstractSessionParametersModel
    parameters_by_session::Tparameters_by_session
end

"""
    IdenticalSessionParameters(priors)

The parameters model of a full model which samples each parameter once, from its prior,
and gives every session that value. `priors` is a `NamedTuple` with the name of each
parameter that is fitted and the distribution it is sampled from, such as
`(; learning_rate = Beta(2, 2))`. The priors are kept in the field `priors`.
"""
struct IdenticalSessionParameters{Tpriors<:NamedTuple} <: AbstractSessionParametersModel
    priors::Tpriors
end

"""
    IndependentSessionParameters(priors)

The parameters model of a full model which samples each parameter for every session from
the same prior, with nothing shared between the sessions. `priors` is a `NamedTuple` with
the name of each parameter that is fitted and the distribution it is sampled from, such as
`(; learning_rate = Beta(2, 2), inverse_temperature = LogNormal(0, 1))`. The priors are kept
in the field `priors`.
"""
struct IndependentSessionParameters{Tpriors<:NamedTuple} <: AbstractSessionParametersModel
    priors::Tpriors
end

"""
    AbstractDownstreamModel

Supertype for the descriptions of what runs after the sessions in a full model, with what
they saved, such as a regression of a measurement on a state. [`NoDownstreamModel`](@ref) is
the default.

# Implementation
Create a subtype, and for each inference backend it is used with, add a method of that
backend's function for downstream models, as the backend's documentation describes.
"""
abstract type AbstractDownstreamModel end

"""
    NoDownstreamModel()

The `downstream_model` of a full model that has no third part, so that nothing runs after
the sessions. This is the default.
"""
struct NoDownstreamModel <: AbstractDownstreamModel end

## Types for how the models of several sessions run ##

"""
    NoParallelisation()

The default `parallelisation` of an inference backend's models of several sessions: the
sessions run one after the other. See [`ParallelSessions`](@ref) for the other choice.
"""
struct NoParallelisation end

"""
    ParallelSessions()

Given as `parallelisation` to an inference backend's model of several sessions, it spreads
the sessions over the threads Julia was started with, as `Threads.@threads` does. What the
model then needs, which automatic differentiation packages work with it and how much faster
it is are described in the documentation of each inference backend.
"""
struct ParallelSessions end
