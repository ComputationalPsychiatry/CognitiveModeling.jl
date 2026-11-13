## Types for data where a value wasn't recorded ##

"""
    MissingValue()

Stand-in for an action or an observation that happened on a timestep but isn't in the data,
and which fitting imputes, as one entry of a column or a `NamedTuple`, such as
`(; choice = MissingValue())`. Whether it is an action or an observation is given by where
it is.

Fitting imputes a missing action by sampling it from the model's action distribution on that
timestep. It imputes a missing observation from what an environment model gives for the
observation on that timestep: as a latent variable with that distribution, or as the value
of a [`Deterministic`](@ref). A value that isn't in the data and isn't to be imputed is
[`UnimputedMissingValue`](@ref) instead. [`NoValue`](@ref) is different: it says that
nothing was done or observed on the timestep. Unlike `missing`, arithmetic on it is an
error. By default, a [`SessionInfo`](@ref) replaces `missing` in the
data by it.
"""
struct MissingValue end

"""
    UnimputedMissingValue()

Stand-in for an action or an observation that happened on a timestep but isn't in the data,
and which fitting doesn't impute, as one entry of a column or a `NamedTuple`, such as
`(; reward = UnimputedMissingValue())`.

Nothing is sampled or scored for it, and the models receive it as it is: a cognitive model
as its observation, or as its previous action on the next timestep, and an environment model
as the action it responds to. A model which is fitted to such data must handle it, as it
does the other subtypes of [`AbstractNoValue`](@ref). Useful for a session without an
environment model whose data lacks some observations, and for missing actions which
shouldn't become latent variables. A [`SessionInfo`](@ref) puts it in place of the missing
observations with `impute_missing_observations = false`, and of the missing actions with
`impute_missing_actions = false`.
"""
struct UnimputedMissingValue <: AbstractNoValue end

## Types for how a session is set up, which are the same whatever it is fitted with ##

"""
    NoEnvironment()

The environment of a session that has none, which is the default of [`SessionInfo`](@ref).
Every observation then comes from the data, and none can be [`MissingValue`](@ref), since
nothing imputes it.
"""
struct NoEnvironment end

#A session without an environment has no environment history whose names could be checked
check_savehistory_selection_names(::SaveHistory, ::NoEnvironment) = nothing
#A session without an environment has no environment whose stopping criterion could be met
stopping_criterion(::NoEnvironment, state, parameters) = false
check_stopping_criterion_not_met(::NoEnvironment, state, parameters) = nothing

"""
    SessionInfo(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Everything about one behavioral session that is fixed within a generative model to be
fitted: the data, the models and the settings, checked and prepared once. An inference
backend's model of the session takes it together with the parameters, which are what
changes between evaluations. [`MultipleSessionsInfo`](@ref) makes one for each session in a
table.

# Arguments
- `cognitive_model::AbstractTimestepModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `observations`: the observations in the data, as a `NamedTuple` with one vector for each
  name and one entry for each timestep, such as `(; reward = [1.0, 0.0, 1.0])`. The data
  must have every observation in the model's attributes, from [`attribute_names`](@ref), and
  can have more. When the model has only one kind of observation in its attributes, a single
  vector, `[1.0, 0.0, 1.0]`, is the data of that observation, whatever its name. An
  observation that wasn't recorded is [`MissingValue`](@ref), which `environment` imputes
  unless `impute_missing_observations` is `false`, and one that wasn't made is
  [`NoValue`](@ref).
- `actions`: the actions in the data, in the same forms, with exactly the actions in the
  model's attributes. An action that was taken but not recorded is `MissingValue()`, which
  is imputed unless `impute_missing_actions` is `false`, and an action that wasn't taken is
  `NoValue()`. Neither is scored.

# Keyword arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: whether the model acts on the
  observation of the same timestep, `EnvironmentStarts()`, or on the observation of the
  timestep before, `AgentStarts()`, as in [`simulate`](@ref). It has no default, because a
  wrong choice shifts the model by a timestep without any error.
- `environment::Union{NoEnvironment, AbstractTimestepModel}=NoEnvironment()`: an environment
  model, usually an [`AbstractEnvironmentModel`](@ref), which imputes the observations that
  are `MissingValue()`, and against whose distributions the observations in the data are
  scored. Its outputs must be the inputs of `cognitive_model` and the other way round,
  directly or through `observation_map` and `action_map`. Without it, the data can't have
  observations that are `MissingValue()`. Whether the session includes it is decided by
  `include_environment`, and it is kept in the session either way.
- `include_environment::Union{Nothing, Bool}=nothing`: whether the session includes
  `environment`, so that it runs on every timestep. By default, with `nothing`, it is
  included where an observation or an action is `MissingValue()`, where
  `environment_save_history` saves anything, or where `check_stopping_criterion` is `true`:
  the scored observations then inform what is imputed, such as which arm of a bandit a
  missing choice was, from the reward that followed. It is also included in every session of
  an inference backend's full model that estimates the environment's parameters, so that the
  observations inform them. Otherwise it is left out, since the scores of its observations
  would be the same for every value of the parameters, and the session has none of its
  variables. With `true` it is always included, such as for a model of one's own which
  estimates its parameters. With `false` it never is: an observation that is
  `MissingValue()`, a selection in `environment_save_history` and
  `check_stopping_criterion = true` are then errors, and an action that is `MissingValue()`
  is imputed without the observations informing it.
- `observation_map::Union{Nothing, NameMap}=nothing` and
  `action_map::Union{Nothing, NameMap}=nothing`: which observations of `environment` the
  cognitive model receives, and which of its actions the environment receives, and under
  which names, as [`NameMap`](@ref)s, as for `simulate`. The data is in the cognitive
  model's names, and is scored against the environment's distributions as the map passes
  them on. An output of the environment that the map doesn't pass on isn't in the data, so
  it is imputed on every timestep. Each model's history is in its own names.
- `timestep_sizes=1`: how much time each timestep covers, as in `simulate`.
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
- `check_stopping_criterion::Union{Nothing, Bool}=nothing`: when the
  [`stopping_criterion`](@ref) of the two models is checked before a timestep, so that the
  session ends there, with what it saved up to it. With `nothing`, it is checked on the
  timesteps after the last one with data. With `true`, it is checked before every timestep,
  and a criterion that is met while the data continues is an error. With `false`, it is
  never checked. The extended help says why.
- `missing_observation_values=(missing,)` and `missing_action_values=(missing,)`: the
  values in the data which stand for a missing observation or action, as one value, such as
  `-1`, or several in a tuple or a vector, such as `(missing, -1)`. Each is replaced by
  `MissingValue()`. A value is compared to them with `isequal`, and `()` keeps every value
  as it is, `missing` too. `MissingValue()` can also be put in the data directly.
- `impute_missing_observations::Bool=true` and `impute_missing_actions::Bool=true`: whether
  the observations and the actions that are `MissingValue()` are imputed, as latent
  variables. With `false`, nothing is sampled for them, and they are replaced by
  [`UnimputedMissingValue`](@ref) here, which the models receive as it is.
- `previous_observation` and `previous_action`: what the first timestep receives as the
  observation and the action before it, in the cognitive model's names, with
  [`BeforeFirstTimestep`](@ref) for each name by default, as in `simulate`. The environment
  starts from the same observation in its own names, with `BeforeFirstTimestep()` for an
  output that `observation_map` doesn't pass on. The states where the session starts are
  not given here, since they come from [`initialize_state`](@ref) with the parameters that
  are fitted.

Each argument is kept in a field of the same name, except the values that stand for a
missing one and whether they are imputed, whose work is done in the data here. The other
fields are `includes_environment`, whether the session
includes the environment, as `include_environment` decides it, `environment_observations`,
the data of the environment's outputs in its own names, `previous_environment_observation`,
the observation the environment starts from in its own names, `actions_to_condition_on` and
`observations_to_condition_on`, the values in the data of each name, which fitting
conditions on, with the observations in the environment's names, `n_timesteps`,
`saved_timestep_sizes`, the sizes a fitted session returns with what it saves, and
`first_timestep_to_check_stopping`, the first timestep before which the stopping criterion
is checked, as `check_stopping_criterion` decides it.

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

julia> session_info = SessionInfo(Counter(), (; reward = [1.0, 0.0, missing]), (; choice = [true, missing, false]); who_starts = EnvironmentStarts(), impute_missing_observations = false)
SessionInfo
Cognitive model: Counter
Environment: none
Timesteps: 3
Observations: reward
Actions: choice
Who starts: environment

julia> session_info.observations.reward[3], session_info.actions.choice[2]
(UnimputedMissingValue(), MissingValue())

julia> session_info.actions_to_condition_on
(choice = Bool[1, 0],)
```

# Extended help
When the stopping criterion is checked follows from the data. The last timestep with data is
the last one where an observation or an action isn't `MissingValue()`. With
`check_stopping_criterion = nothing`, the criterion is only checked after it, so that a fit,
where every timestep has data, never checks it, and a session prepared by
[`prepare_for_prediction`](@ref), whose data is all `MissingValue()`, ends as a simulation
does. With `true`, the session includes its environment, whose criterion is checked too. A
criterion that is met while the data continues is then an error, since the values in the
data after that timestep couldn't be scored, which checks that the models agree with the
length of the data.
"""
struct SessionInfo{
    Tcognitive_model,
    Tenvironment,
    Tobservations,
    Tactions,
    Tactions_to_condition_on,
    Tobservations_to_condition_on,
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
    include_environment::Union{Nothing,Bool}
    includes_environment::Bool
    observations::Tobservations
    actions::Tactions
    environment_observations::Tenvironment_observations
    actions_to_condition_on::Tactions_to_condition_on
    observations_to_condition_on::Tobservations_to_condition_on
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
function SessionInfo(
    cognitive_model::AbstractTimestepModel, observations, actions; keywords...
)
    return prepare_session(cognitive_model, observations, actions; keywords...)
end

#The same session with other save histories, prepared again, since whether the environment runs can follow from them. Its data has its markers already, so nothing more is replaced in it
function SessionInfo(
    session_info::SessionInfo;
    cognitive_save_history::SaveHistory=session_info.cognitive_save_history,
    environment_save_history::SaveHistory=session_info.environment_save_history,
)
    return prepare_session(
        session_info.cognitive_model,
        session_info.observations,
        session_info.actions;
        missing_observation_values=(),
        missing_action_values=(),
        who_starts=session_info.who_starts,
        environment=session_info.environment,
        observation_map=session_info.observation_map,
        action_map=session_info.action_map,
        include_environment=session_info.include_environment,
        timestep_sizes=session_info.timestep_sizes,
        cognitive_save_history,
        environment_save_history,
        check_stopping_criterion=session_info.check_stopping_criterion,
        previous_observation=session_info.previous_observation,
        previous_action=session_info.previous_action,
    )
end

"""
    prepare_session(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Return the [`SessionInfo`](@ref) of one session, with the checks of the data and the names,
the missing values in the data marked, whether the environment runs, and what is scored and
returned prepared, as its constructor describes. Used by `SessionInfo` and
[`MultipleSessionsInfo`](@ref).
"""
function prepare_session(
    cognitive_model::AbstractTimestepModel,
    observations,
    actions;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    environment::Union{NoEnvironment,AbstractTimestepModel}=NoEnvironment(),
    observation_map::Union{Nothing,NameMap}=nothing,
    action_map::Union{Nothing,NameMap}=nothing,
    include_environment::Union{Nothing,Bool}=nothing,
    timestep_sizes=1,
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
    check_stopping_criterion::Union{Nothing,Bool}=nothing,
    missing_observation_values=(missing,),
    missing_action_values=(missing,),
    impute_missing_observations::Bool=true,
    impute_missing_actions::Bool=true,
    previous_observation=input_before_first_timestep(cognitive_model),
    previous_action=output_before_first_timestep(cognitive_model),
)

    #A single vector is the data of the observation or the action, when the model has only one kind of them in its attributes
    observations = named_data(observations, cognitive_model, :observations)
    actions = named_data(actions, cognitive_model, :actions)

    #Replace the values in the data which stand for a missing value by the marker for one, which is the marker for a missing value that isn't imputed when imputation isn't asked for
    observations = mark_missing_values(
        observations, as_tuple(missing_observation_values), impute_missing_observations
    )
    actions = mark_missing_values(
        actions, as_tuple(missing_action_values), impute_missing_actions
    )

    #Check the data against the names in the cognitive model's attributes, and that the environment fits
    n_timesteps = check_data_fits(cognitive_model, observations, actions)
    check_environment_fits(
        cognitive_model, environment, observations, observation_map, action_map
    )

    #Check that the save histories select names in the models' attributes
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

    #Whether the session includes its environment: as it is told, and otherwise only when something is to be imputed, an observation it fills in or an action that the observations it scores inform, when its history is saved, or when its stopping criterion is checked on every timestep
    includes_environment = decide_whether_environment_is_included(
        include_environment,
        environment,
        observations,
        actions,
        environment_save_history,
        check_stopping_criterion,
    )

    #The environment's data is in its own names, as the observation map passes it on, with outputs that the map doesn't pass on imputed on every timestep, and it starts from the previous observation in the same way
    environment_observations = map_to_environment_outputs(
        observations, environment, observation_map, fill(MissingValue(), n_timesteps)
    )
    previous_environment_observation = map_to_environment_outputs(
        previous_observation, environment, observation_map, BeforeFirstTimestep()
    )

    #The actions and observations in the data are scored at the end, the observations in the environment's names, and the sizes of the timesteps are returned with what is saved
    actions_to_condition_on = collect_values_to_condition_on(actions)
    observations_to_condition_on = collect_values_to_condition_on(environment_observations)
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
        include_environment,
        includes_environment,
        observations,
        actions,
        environment_observations,
        actions_to_condition_on,
        observations_to_condition_on,
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
    MultipleSessionsInfo(sessions, session_ids; table=nothing, session_rows=nothing)

Multiple behavioral sessions in a generative model that are fitted together, as a vector of
[`SessionInfo`](@ref) with an identifier for each, which an inference backend's models of
several sessions take.

The first form splits a table into sessions. `table` is a `NamedTuple` with one vector for
each column, or any other table that Tables.jl can read, such as a `DataFrame` or a
`CSV.File`, whose columns are read when Tables.jl is loaded, as it is with DataFrames.jl and
CSV.jl. The values of a column that stand for a missing observation or action become
[`MissingValue`](@ref), as for [`SessionInfo`](@ref): by default those that are `missing`.
As for [`SessionInfo`](@ref), a session includes its environment where an observation or an
action is `MissingValue()`, where `environment_save_history` saves anything or where
`check_stopping_criterion` is `true`, and in an inference backend's full model that
estimates the environment's parameters, unless `include_environment` says otherwise. The
table is kept in the field `table`, without being copied, with the rows of each session in
`session_rows`, so that a session parameter model can read the columns it needs, such as
those that a [`GroupedHierarchicalSessionParameters`](@ref) groups the sessions by.

    MultipleSessionsInfo(sessions_info; cognitive_save_history, environment_save_history)

The same sessions with other save histories for every one, see [`SessionInfo`](@ref).

# Arguments
- `cognitive_model::AbstractTimestepModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `table`: the data of every session, in columns with one entry for each timestep, such as
  `(; id = ["S1", "S1", "S2", "S2"], reward = [1.0, 0.0, 1.0, 1.0], choice = [true, false, true, true])`,
  or a `DataFrame` with those columns.

# Keyword arguments
- `observation_columns`: the columns with the observations. When the model has only one
  kind of observation in its attributes, a single name, such as `:outcome`, is its column,
  whatever the observation is called. Several observations are given in a tuple, each as the
  name in the model's attributes when the column is named the same, or as
  a pair from the column's name to the model's, as in a [`NameMap`](@ref), such as
  `(:outcome => :reward, :cue)`, so that one model fits tables with other names. A single
  name among the actions in the model's attributes is an error, since the two keyword
  arguments may then be swapped. An observation or action whose value is composed from
  several columns is specified by how to build it from the columns, followed by the model's
  name: a `Tuple` of column names for a `Tuple` in each row,
  `(:choicecol, :rtcol) => :ssm_action`, a `NamedTuple` of column names for a `NamedTuple`,
  `(; choice = :choicecol, rt = :rtcol) => :ssm_action`, or a `Tuple` of column names and a
  function of their values, which needs parentheses around it,
  `(:choicecol, :rtcol) => ((choice, rt) -> [choice, rt]) => :ssm_action`. When the model
  has only one kind of them in its attributes, the name can be left out of the last two, as
  in `(; choice = :choicecol, rt = :rtcol)`. A missing value in any of the columns makes the
  whole value missing.
- `action_columns`: the columns with the actions, in the same forms.
- `session_columns`: the column or columns which separate sessions, such as `:id` or
  `(:id, :block)`. A session is every row with one combination of their values, in the
  order of the first row of each. The identifier of a session is its value in a single
  column, or a `NamedTuple` of its values in several.
- `timestep_sizes=1`: how much time each timestep covers, as one size for all, or the name
  of a column with the size of each timestep.
- `missing_observation_values=(missing,)`, `missing_action_values=(missing,)`,
  `impute_missing_observations=true` and `impute_missing_actions=true`: the values in the
  columns which stand for a missing observation or action, and whether those are imputed, as
  for [`SessionInfo`](@ref). They are marked in the whole table before it is split, so that
  the columns of every session have the same element types.
- The other keyword arguments of [`SessionInfo`](@ref), which every session gets.

The second form takes sessions that are already prepared. They have no table, unless it is
given as `table`, with the rows of each session in it as `session_rows`, a vector of
vectors of row numbers. Each argument is kept in a field of the same name.

Each session is identified in three ways: by its order, its index in `sessions`, which the
models of several sessions number it by and [`CognitiveModeling.session_index`](@ref)
finds; by its identifier, the value of its session column, or a `NamedTuple` of the values
when there are several, as in `session_ids`; and by its symbol name, a `Symbol` made from
the identifier by [`CognitiveModeling.session_id_to_symbol`](@ref), such as `S1` or
`id_S1__block_2`, which the chains of a fit name its entries after.

In the REPL, the sessions show a summary on one line each: how many there are, and the
models and the names in the data of the first.

# Examples
```jldoctest multiple_sessions_info
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, inputs = :reward, outputs = :choice);

julia> table = (; id = ["S1", "S1", "S2", "S2", "S2"], outcome = [1.0, 0.0, 1.0, 1.0, 0.0], choice = [true, false, true, missing, false]);

julia> sessions_info = MultipleSessionsInfo(Counter(), table; observation_columns = :outcome => :reward, action_columns = :choice, session_columns = :id, who_starts = EnvironmentStarts())
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
    table::Union{Nothing,NamedTuple}
    session_rows::Union{Nothing,Vector{Vector{Int}}}

    #Only takes sessions of one type, with an identifier for each, and the rows of each in the table when there is one
    function MultipleSessionsInfo(
        sessions::AbstractVector{<:SessionInfo},
        session_ids::AbstractVector;
        table::Union{Nothing,NamedTuple}=nothing,
        session_rows::Union{Nothing,AbstractVector{<:AbstractVector{Int}}}=nothing,
    )
        isempty(sessions) && throw(ArgumentError("There must be at least one session."))
        if length(sessions) != length(session_ids)
            throw(
                DimensionMismatch(
                    "There are $(length(sessions)) sessions and $(length(session_ids)) session identifiers, but there must be one identifier for each session.",
                ),
            )
        end
        if (table === nothing) != (session_rows === nothing)
            throw(
                ArgumentError(
                    "A table must come with the rows of each session in it, as session_rows, and the rows with the table.",
                ),
            )
        end
        if session_rows !== nothing && length(session_rows) != length(sessions)
            throw(
                DimensionMismatch(
                    "There are $(length(sessions)) sessions and rows for $(length(session_rows)), but there must be rows for each session.",
                ),
            )
        end
        check_unique_session_symbol_names(session_ids)
        if !isconcretetype(eltype(sessions))
            throw(
                ArgumentError(
                    "The sessions must all have the same type, but the vector has element type $(eltype(sessions)). They have one type when they share the cognitive model, the environment and the settings, and their columns have the same element types.",
                ),
            )
        end
        session_rows = session_rows === nothing ? nothing : map(Vector{Int}, session_rows)
        return new{typeof(sessions),typeof(session_ids)}(
            sessions, session_ids, table, session_rows
        )
    end
end

#Splits a table into sessions by the values of the session columns, and prepares each of them in the same way
function MultipleSessionsInfo(
    cognitive_model::AbstractTimestepModel,
    table;
    observation_columns,
    action_columns,
    session_columns,
    timestep_sizes=1,
    missing_observation_values=(missing,),
    missing_action_values=(missing,),
    impute_missing_observations::Bool=true,
    impute_missing_actions::Bool=true,
    keywords...,
)

    #A table of another kind, such as a DataFrame, is read as a NamedTuple of its columns
    table = column_table(table)

    #The columns of each kind, from the names the model uses to the names in the table
    observation_column_map = column_map(observation_columns, cognitive_model, :observations)
    action_column_map = column_map(action_columns, cognitive_model, :actions)
    id_columns = as_tuple(session_columns)::Tuple{Vararg{Symbol}}
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

    #The columns of the observations and of the actions are built once for the whole table, with the missing values replaced by their marker, so that every session gets the same element types
    observation_columns_of_table = build_columns(
        table,
        observation_column_map,
        as_tuple(missing_observation_values),
        impute_missing_observations,
    )
    action_columns_of_table = build_columns(
        table, action_column_map, as_tuple(missing_action_values), impute_missing_actions
    )

    #Each session is prepared from its rows of those columns, in which the missing values are marked already
    sessions = map(sessions_rows) do (_, rows)
        return prepare_session(
            cognitive_model,
            select_rows(observation_columns_of_table, rows),
            select_rows(action_columns_of_table, rows);
            timestep_sizes=select_timestep_sizes(table, timestep_sizes, rows),
            missing_observation_values=(),
            missing_action_values=(),
            keywords...,
        )
    end
    session_ids = map(first, sessions_rows)

    #The table is kept with the rows of each session, so that a session parameter model can read the columns it needs
    return MultipleSessionsInfo(
        sessions, session_ids; table, session_rows=map(last, sessions_rows)
    )
end

#The same sessions with other save histories
function MultipleSessionsInfo(sessions_info::MultipleSessionsInfo; keywords...)
    return MultipleSessionsInfo(
        map(session -> SessionInfo(session; keywords...), sessions_info.sessions),
        sessions_info.session_ids;
        table=sessions_info.table,
        session_rows=sessions_info.session_rows,
    )
end

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

## Types for what gives the parameters of the sessions in a full model ##

"""
    AbstractSessionParametersModel

Supertype for session parameter models: the part of a full model, the model that an
inference backend fits to several sessions, which gives or estimates the parameters of every
session, for the cognitive model and for the environment model alike. The package comes with
[`NoParameters`](@ref), [`FixedSessionParameters`](@ref),
[`FixedParametersBySession`](@ref), [`IdenticalSessionParameters`](@ref),
[`IndependentSessionParameters`](@ref) and [`GroupedHierarchicalSessionParameters`](@ref),
and [`CompositeSessionParameters`](@ref) combines several, each giving some of the
parameters.

# Implementation
Create a subtype, and for each inference backend it is used with, add a method of that
backend's function for session parameter models, as the backend's documentation describes. A
method of [`CognitiveModeling.parameter_names`](@ref) lets the parameters it gives be
checked before a fit, and a method of [`CognitiveModeling.prepare_parameters_model`](@ref)
lets it work out once, before the backend's model is made, what doesn't change between
evaluations. A model which only gives values, and estimates nothing, adds a method of
[`CognitiveModeling.estimates_parameters`](@ref).
"""
abstract type AbstractSessionParametersModel end

## Types for what runs after the sessions in a full model ##

"""
    AbstractDownstreamModel

Supertype for downstream models: the final submodel of a full model, which is run after
the sessions with the trajectories they saved, such as for correlations with other kinds of
data, or a regression of a measurement on a state. [`NoDownstreamModel`](@ref) is the
default.

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
    SerialSessions()

The default `parallelization` of an inference backend's models of several sessions: the
sessions run one after the other. See [`ParallelSessions`](@ref) for the other choice.
"""
struct SerialSessions end

"""
    ParallelSessions()

Given as `parallelization` to an inference backend's model of several sessions, it spreads
the sessions over the threads Julia was started with, as `Threads.@threads` does. What the
model then needs, which automatic differentiation packages work with it and how much faster
it is are described in the documentation of each inference backend.
"""
struct ParallelSessions end
