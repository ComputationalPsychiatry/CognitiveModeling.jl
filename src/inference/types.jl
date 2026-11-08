#Type for data where a value wasn't recorded

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

#Types for how a session is set up, which are the same whatever it is fitted with

"""
    NoEnvironment()

The environment of a session that has none, which is the default of [`SessionInfo`](@ref).
Every observation then comes from the data, and none can be [`MissingValue`](@ref).
"""
struct NoEnvironment end

"""
    SessionInfo(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Everything about one session that is fixed while a cognitive model is fitted to it: the
data, the models and the settings, checked and prepared once. A Turing.jl model of the
session, [`turing_behavioural_session`](@ref), takes it together with the parameters, which
are what changes between evaluations. [`MultipleSessionsInfo`](@ref) makes one for each
session in a table.

# Arguments
- `cognitive_model::AbstractCognitiveModel`: the cognitive model.
- `observations::NamedTuple`: the observations in the data, with one vector for each name
  and one entry for each timestep, such as `(; reward = [1.0, 0.0, 1.0])`. The data must
  have every observation the model lists in [`attribute_names`](@ref), and can have more. An
  observation that wasn't recorded is [`MissingValue`](@ref), which `environment` fills in,
  and one that wasn't made is [`NoObservation`](@ref).
- `actions::NamedTuple`: the actions in the data, in the same form, with exactly the actions
  the model lists. An action that was taken but not recorded is `MissingValue()`, and is
  imputed unless `impute_missing_actions` is `false`. An action that wasn't taken is
  [`NoAction`](@ref), and isn't scored.

# Keyword arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: whether the model acts on the
  observation of the same timestep, `EnvironmentStarts()`, or on the observation of the
  timestep before, `AgentStarts()`, as in [`simulate`](@ref). It has no default, because a
  wrong choice shifts the model by a timestep without any error.
- `environment::Union{NoEnvironment, AbstractEnvironmentModel}=NoEnvironment()`: an
  environment model which fills in the observations that are `MissingValue()`. It must list
  the same observations and the same actions as `cognitive_model`, since a session takes no
  maps of names. Without it, the data can't have missing observations. With it, some
  observation should be missing, and a warning says so otherwise, since filling them in is
  all an environment does in a session: a missing action is imputed from the cognitive
  model's own distribution for it, and nothing in the log density depends on an environment
  whose observations are all in the data, not even its parameters. Such an environment only
  costs time.
- `timestep_sizes=1`: how much time each timestep covers, as in `simulate`.
- `impute_missing_actions::Bool=true`: whether an action that is `MissingValue()` is
  imputed, as a latent variable. When it is `false`, nothing is sampled for it, and it is
  replaced by `NoAction()` here: `cognitive_model` receives `NoAction()` as that previous
  action on the next timestep, and `environment` as the action it responds to.
- `save_history::SaveHistory=SaveHistory()`: which observations, actions, action
  distributions and states a fitted session returns, by name, and whether timestep 0 is.
  Nothing is selected by default.
- `missing_observations::Union{MissingValue, NoObservation}=MissingValue()` and
  `missing_actions::Union{MissingValue, NoAction}=MissingValue()`: what a `missing` in the
  data becomes, since a table marks every gap the same way: a value that wasn't recorded, or
  nothing happening on that timestep. The two markers can also be put in the data directly.
- `previous_observation` and `previous_action`: what the first timestep receives as the
  observation and the action before it, with [`BeforeFirstTimestep`](@ref) for each name by
  default, as in `simulate`. The states where the session starts are not given here, since
  they come from [`initialize_state`](@ref) with the parameters that are fitted.

Each argument is kept in a field of the same name, with the two replacements made in the
data, except `impute_missing_actions`, whose work is done here. There are three fields more:
`recorded`, the recorded actions of each name which are scored, `n_timesteps`, and
`saved_timestep_sizes`, the sizes a fitted session returns with what it saves.

    SessionInfo(session_info; save_history)

The same session with another `save_history`, for reading the states of a fitted session
from a chain: a session is fitted fastest when it saves nothing, and a chain from that model
gives the states through `returned` with the model of a copy that saves them, since the two
have the same variables. See [`turing_full_model`](@ref).

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, observations = :reward, actions = :choice);

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

julia> session_info.recorded
(choice = Bool[1, 0],)
```
"""
struct SessionInfo{
    Tcognitive_model,
    Tenvironment,
    Tobservations,
    Tactions,
    Trecorded,
    Twho_starts,
    Ttimestep_sizes,
    Tsaved_timestep_sizes,
    Tsave_history,
    Tprevious_observation,
    Tprevious_action,
}
    cognitive_model::Tcognitive_model
    environment::Tenvironment
    observations::Tobservations
    actions::Tactions
    recorded::Trecorded
    n_timesteps::Int
    who_starts::Twho_starts
    timestep_sizes::Ttimestep_sizes
    saved_timestep_sizes::Tsaved_timestep_sizes
    save_history::Tsave_history
    previous_observation::Tprevious_observation
    previous_action::Tprevious_action
end

#Prepares the session, and warns if an environment has nothing to fill in
function SessionInfo(
    cognitive_model::AbstractCognitiveModel, observations, actions; keywords...
)
    session_info = prepare_session(cognitive_model, observations, actions; keywords...)
    warn_if_environment_is_unused(session_info)
    return session_info
end

#The same session with another save history, with the sizes it returns made again
function SessionInfo(session_info::SessionInfo; save_history::SaveHistory)
    saved_sizes = saved_timestep_sizes(
        session_info.timestep_sizes, session_info.n_timesteps, save_history
    )
    return SessionInfo(
        session_info.cognitive_model,
        session_info.environment,
        session_info.observations,
        session_info.actions,
        session_info.recorded,
        session_info.n_timesteps,
        session_info.who_starts,
        session_info.timestep_sizes,
        saved_sizes,
        save_history,
        session_info.previous_observation,
        session_info.previous_action,
    )
end

"""
    prepare_session(cognitive_model, observations, actions; who_starts, <keyword arguments>)

Return the [`SessionInfo`](@ref) of one session, with the checks of the data and the names,
the replacements of `missing` in the data, and what is scored and returned prepared, as its
constructor describes. It leaves out the warning about an environment with nothing to fill
in, which [`MultipleSessionsInfo`](@ref) gives over all its sessions instead of each.
"""
function prepare_session(
    cognitive_model::AbstractCognitiveModel,
    observations,
    actions;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    environment::Union{NoEnvironment,AbstractEnvironmentModel}=NoEnvironment(),
    timestep_sizes=1,
    impute_missing_actions::Bool=true,
    save_history::SaveHistory=SaveHistory(),
    missing_observations::Union{MissingValue,NoObservation}=MissingValue(),
    missing_actions::Union{MissingValue,NoAction}=MissingValue(),
    previous_observation=initial_observation(
        environment isa NoEnvironment ? cognitive_model : environment
    ),
    previous_action=initial_action(cognitive_model),
)

    #Replace missing in the data by the marker for a value that wasn't recorded, or for nothing happening
    observations = replace_missing(observations, missing_observations)
    actions = replace_missing(actions, missing_actions)

    #Check the data against the names the cognitive model lists, and that the environment fits
    n_timesteps = check_data(cognitive_model, observations, actions)
    check_environment_fits(cognitive_model, environment, observations)

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

    #The recorded actions are scored at the end, and the sizes of the timesteps are returned with what is saved
    recorded = recorded_actions(actions)
    saved_sizes = saved_timestep_sizes(timestep_sizes, n_timesteps, save_history)

    return SessionInfo(
        cognitive_model,
        environment,
        observations,
        actions,
        recorded,
        n_timesteps,
        who_starts,
        timestep_sizes,
        saved_sizes,
        save_history,
        previous_observation,
        previous_action,
    )
end

"""
    MultipleSessionsInfo(cognitive_model, table; observation_columns, action_columns,
                         session_columns, <keyword arguments>)
    MultipleSessionsInfo(sessions, session_ids)

The sessions that a cognitive model is fitted to together, as a vector of
[`SessionInfo`](@ref) with an identifier for each, which
[`turing_multiple_behavioural_sessions`](@ref) and [`turing_full_model`](@ref) take.

The first form splits a table into sessions. `table` is a `NamedTuple` with one vector for
each column, which is what `Tables.columntable` gives for a `DataFrame` or any other table.
A `missing` in a column becomes [`MissingValue`](@ref), or the marker for nothing happening,
as the keyword arguments `missing_observations` and `missing_actions` of `SessionInfo` say.
Every session must come out with the same type, which they do when they share the models and
the settings, since the timesteps of different sessions are then compiled once. An
environment should have an observation to fill in in some session, not in every one, and the
warning comes when there is none in any.

    MultipleSessionsInfo(sessions_info; save_history)

The same sessions with another `save_history` for every one, see [`SessionInfo`](@ref).

# Arguments
- `cognitive_model::AbstractCognitiveModel`: the cognitive model.
- `table::NamedTuple`: the data of every session, with one vector for each column and one
  entry for each timestep, such as `(; id = ["S1", "S1", "S2", "S2"], reward = [1.0, 0.0,
  1.0, 1.0], choice = [true, false, true, true])`.

# Keyword arguments
- `observation_columns`: the columns with the observations, as the names the model lists in
  [`attribute_names`](@ref) when the columns are named the same, such as `:reward` or
  `(:reward, :cue)`, or as a `NamedTuple` from the model's name to the column's, such as
  `(; reward = :outcome)`, so that one model fits tables with other names. An attribute whose
  value is made of several columns is specified by how to build it from them: a `NamedTuple`
  of column names for a `NamedTuple` in each row, `(; ssm_action = (; choice = :choicecol,
  rt = :rtcol))`, a `Tuple` of column names for a `Tuple`, `(; ssm_action = (:choicecol,
  :rtcol))`, or a `Tuple` of column names and a function of their values, `(; ssm_action =
  (:choicecol, :rtcol) => (choice, rt) -> [choice, rt])`. A `missing` in any of the columns
  makes the value `missing`.
- `action_columns`: the columns with the actions, in the same forms.
- `session_columns`: the column or columns which identify a session, such as `:id` or
  `(:id, :block)`. A session is every row with one combination of their values, in the
  order of the first row of each. The identifier of a session is its value in a single
  column, or a `NamedTuple` of its values in several.
- `timestep_sizes=1`: how much time each timestep covers, as one size for all, or the name
  of a column with the size of each timestep.
- The other keyword arguments of [`SessionInfo`](@ref), which every session gets.

The second form takes sessions that are already prepared. Each argument is kept in a field
of the same name.

# Examples
```jldoctest
julia> struct Counter <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Counter) = ModelAttributes(; parameters = :probability, states = :n_timesteps, observations = :reward, actions = :choice);

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
    cognitive_model::AbstractCognitiveModel,
    table;
    observation_columns,
    action_columns,
    session_columns,
    timestep_sizes=1,
    keywords...,
)

    #The columns of each kind, from the names the model uses to the names in the table
    observation_column_map = column_map(observation_columns)
    action_column_map = column_map(action_columns)
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

    #Each session is prepared from its rows of the table, and an environment should have something to fill in in some session
    sessions = map(sessions_rows) do (_, rows)
        return prepare_session(
            cognitive_model,
            select_rows(table, observation_column_map, rows),
            select_rows(table, action_column_map, rows);
            timestep_sizes=select_timestep_sizes(table, timestep_sizes, rows),
            keywords...,
        )
    end
    warn_if_environment_is_unused(sessions)
    session_ids = map(first, sessions_rows)

    return MultipleSessionsInfo(sessions, session_ids)
end

#The same sessions with another save history
function MultipleSessionsInfo(
    sessions_info::MultipleSessionsInfo; save_history::SaveHistory
)
    return MultipleSessionsInfo(
        map(session -> SessionInfo(session; save_history), sessions_info.sessions),
        sessions_info.session_ids,
    )
end

#Types for the parts of a full model

"""
    AbstractPopulationModel

Supertype for the descriptions of how the cognitive parameters of several sessions are
sampled, the first part of [`turing_full_model`](@ref). [`IndependentSessions`](@ref) is the
one that comes with the package.

# Implementation
Create a subtype and add a method of [`turing_population_model`](@ref) for it.
"""
abstract type AbstractPopulationModel end

"""
    IndependentSessions(priors)

The simplest population model for [`turing_full_model`](@ref): every session has its own
parameters, each sampled from the same prior, with nothing shared between the sessions.
`priors` is a `NamedTuple` with the name of each parameter that is fitted and the
distribution it is sampled from, such as `(; learning_rate = Beta(2, 2), inverse_temperature
= LogNormal(0, 1))`. The priors are kept in the field `priors`.

In the Turing.jl model, each parameter is a vector with one entry for each session, named
after the parameter, such as `learning_rate.session`, so the learning rate of the third
session is `learning_rate.session[3]`. [`turing_full_model`](@ref) can put them under
`population_model` too.
"""
struct IndependentSessions{Tpriors<:NamedTuple} <: AbstractPopulationModel
    priors::Tpriors
end

"""
    AbstractDownstreamModel

Supertype for the descriptions of what runs after the sessions in a
[`turing_full_model`](@ref), with what they saved, such as a regression of a measurement on
a state. [`NoDownstreamModel`](@ref) is the default.

# Implementation
Create a subtype and add a method of [`turing_downstream_model`](@ref) for it.
"""
abstract type AbstractDownstreamModel end

"""
    NoDownstreamModel()

The `downstream_model` of a [`turing_full_model`](@ref) that has no third part: nothing is
sampled after the sessions, and the `downstream_values` the model returns are `nothing`.
This is the default.
"""
struct NoDownstreamModel <: AbstractDownstreamModel end

#Types for how the Turing.jl models run

"""
    NoParallelisation()

The default `parallelisation` of [`turing_multiple_behavioural_sessions`](@ref) and
[`turing_full_model`](@ref): the sessions run one after the other. See
[`ParallelSessions`](@ref) for the other choice.
"""
struct NoParallelisation end

"""
    ParallelSessions()

Given as `parallelisation` to [`turing_multiple_behavioural_sessions`](@ref) or
[`turing_full_model`](@ref), it spreads the sessions over the threads Julia was started with,
as `Threads.@threads` does. The model that is evaluated, which is the outermost one, must
then be marked with DynamicPPL's `setthreadsafe(model, true)`, and the sessions throw an
error if it isn't, since the log density is silently wrong otherwise. The gradient of a
threaded model is right with ForwardDiff and ReverseDiff, and Mooncake and Enzyme throw.
With 4 threads and 20 sessions of 200 timesteps, the log density took 26 µs instead of 39,
and a ForwardDiff gradient 244 µs instead of 471.
"""
struct ParallelSessions end
