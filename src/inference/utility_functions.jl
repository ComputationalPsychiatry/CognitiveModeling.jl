## Preparing the data of a session ##

"""
    is_recorded(action)

Return whether a single action in the data was taken and recorded, so that it is scored:
neither [`MissingValue`](@ref) nor [`NoAction`](@ref). Used by [`recorded_actions`](@ref)
and [`recorded_on_timestep`](@ref).
"""
is_recorded(action) = !(action isa MissingValue || action isa NoAction)

"""
    recorded_actions(actions)

Return the recorded values of each action in `actions`, a `NamedTuple` with one vector per
action, as a `NamedTuple` of vectors in the order of the timesteps. Fitting scores these
against the model's action distributions on the same timesteps. Each vector has the element
type of its column without [`MissingValue`](@ref) and [`NoAction`](@ref), so that a session
without recorded values for an action gets an empty vector of the same type as other
sessions, which scores nothing. An action whose column can only hold those markers is left
out, which is decided by its type, so that it is the same for every session of a table. Used
by [`SessionInfo`](@ref), which keeps the result in its field `recorded`.
"""
function recorded_actions(actions::NamedTuple{names}) where {names}
    recorded = map(
        column -> Vector{Base.typesplit(eltype(column), Union{MissingValue,NoAction})}(
            filter(is_recorded, column)
        ),
        actions,
    )
    names_with_a_type = Tuple(
        name for name in names if eltype(getproperty(recorded, name)) !== Union{}
    )
    return NamedTuple{names_with_a_type}(recorded)
end

"""
    without_missing_actions(actions)

Return `actions`, a `NamedTuple` with one vector per action, with every
[`MissingValue`](@ref) replaced by [`NoAction`](@ref), for a session where a missing action
is passed on as an action that wasn't taken instead of being imputed. As for
[`replace_missing`](@ref), a column gets `NoAction` in its element type when the element
type allows a `MissingValue`, whether it holds one or not, and is kept as it is otherwise.
Used by [`SessionInfo`](@ref) when `impute_missing_actions` is `false`.
"""
function without_missing_actions(actions::NamedTuple)
    return map(without_missing_actions, actions)
end

function without_missing_actions(column::AbstractVector)
    MissingValue <: eltype(column) || return column
    replaced = Vector{Union{Base.typesplit(eltype(column), MissingValue),NoAction}}(
        undef, length(column)
    )
    for i in eachindex(column)
        replaced[i] = column[i] isa MissingValue ? NoAction() : column[i]
    end
    return replaced
end

"""
    narrow_column(column)

Return `column` with the types it holds as its element type, when its element type is `Any`
or another abstract type that isn't a `Union`, and `column` itself otherwise. A vector
written as `[true, MissingValue(), true]` has the element type `Any`, since only `missing`
is promoted to a `Union`, and an `Any` in the data makes every timestep run with its types
unknown. Used by [`replace_missing`](@ref) and [`select_rows`](@ref).
"""
function narrow_column(column::AbstractVector)
    T = eltype(column)
    (isconcretetype(T) || T isa Union) && return column
    narrowed_type = Union{}
    for value in column
        narrowed_type = Union{narrowed_type,typeof(value)}
    end
    return Vector{narrowed_type}(column)
end

"""
    replace_missing(columns, replacement)

Return `columns`, a `NamedTuple` with one vector per name, with every `missing` replaced by
`replacement`: [`MissingValue`](@ref) for a value that wasn't recorded, or
[`NoAction`](@ref) or [`NoObservation`](@ref) for nothing happening on that timestep. A
column whose element type allows `missing` gets the replacement in its element type whether
it holds a `missing` or not, so that the sessions of one table all have the same type, and a
column whose element type doesn't is kept as it is, after [`narrow_column`](@ref). Anything
that isn't a `NamedTuple` of vectors is returned as it is, for [`check_data`](@ref) to
report. Used by [`SessionInfo`](@ref) on the observations and the actions.
"""
function replace_missing(columns::NamedTuple, replacement)
    return map(column -> replace_missing(column, replacement), columns)
end

function replace_missing(column::AbstractVector, replacement)
    column = narrow_column(column)
    Missing <: eltype(column) || return column
    replaced = Vector{Union{nonmissingtype(eltype(column)),typeof(replacement)}}(
        undef, length(column)
    )
    for i in eachindex(column)
        replaced[i] = ismissing(column[i]) ? replacement : column[i]
    end
    return replaced
end

replace_missing(x, replacement) = x

"""
    entry(::Val{name}, value)

Return `(; name = value)`, built in a call of its own, so that a value which can have two
types gives one of two `NamedTuple`s with a concrete field each, as `(; a = x)` written out
doesn't: Julia infers that as a `NamedTuple` whose field type is the `Union`, which nothing
is compiled for, while a call with a `Union`-typed argument is split by type. A `NamedTuple`
which holds such a value is built with `merge` of these, as [`row`](@ref) does. Used by
[`row`](@ref), by [`saved_session_outputs`](@ref) and by the Turing.jl models of a timestep.
"""
entry(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of
vectors with one entry for each timestep, built with [`entry`](@ref) so that a column which
holds two types, such as an action that is [`MissingValue`](@ref) on some timesteps, gives
one of two `NamedTuple`s with concrete fields rather than one without a definite type. It is
unrolled over the names, which it takes from the type. Used by the Turing.jl models of a
timestep and of several sessions, which reads the parameters of one session with it.
"""
row(columns::NamedTuple{()}, t) = (;)
function row(columns::NamedTuple{names}, t) where {names}
    return merge(entry(Val(first(names)), first(columns)[t]), row(Base.tail(columns), t))
end

"""
    has_missing_value(data)

Return whether any entry of `data`, the `NamedTuple` of the actions or the observations of
one timestep, is [`MissingValue`](@ref). It is decided at compile time when no entry can be.
Used by the Turing.jl models of a timestep, to skip imputing when nothing is missing.
"""
has_missing_value(data::NamedTuple) = any(value -> value isa MissingValue, Tuple(data))

"""
    check_data(cognitive_model, observations, actions)

Throw an informative error if the data of a session isn't a `NamedTuple` of vectors for the
observations and one for the actions, doesn't have the observations and the actions that
[`attribute_names`](@ref) lists, or has columns of different lengths, and return the number
of timesteps otherwise. Used by [`SessionInfo`](@ref).
"""
function check_data(
    cognitive_model::AbstractCognitiveModel, observations::NamedTuple, actions::NamedTuple
)
    attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    if !has_every_name(attributes.observations, keys(observations))
        throw(
            ArgumentError(
                "The observations in the data are named $(keys(observations)), but attribute_names lists the observations of a cognitive model of type $(typeof(cognitive_model)) as $(attributes.observations). The data must have every observation the cognitive model lists.",
            ),
        )
    end
    if !same_names(keys(actions), attributes.actions)
        throw(
            ArgumentError(
                "The actions in the data are named $(keys(actions)), but attribute_names lists the actions of a cognitive model of type $(typeof(cognitive_model)) as $(attributes.actions).",
            ),
        )
    end
    lengths = (map(length, values(observations))..., map(length, values(actions))...)
    isempty(lengths) && throw(
        ArgumentError(
            "The data has no observations and no actions, so the number of timesteps is unknown.",
        ),
    )
    allequal(lengths) || throw(
        ArgumentError(
            "Every observation and action in the data must have one value for each timestep, but their lengths are $(map(length, observations)) and $(map(length, actions)).",
        ),
    )
    return first(lengths)
end

#The same for the data that isn't a NamedTuple of vectors
function check_data(cognitive_model::AbstractCognitiveModel, observations, actions)
    return throw(
        ArgumentError(
            "The observations and the actions must each be a NamedTuple with one vector for each name, such as (; reward = [1.0, 0.0]), but they have types $(typeof(observations)) and $(typeof(actions)).",
        ),
    )
end

"""
    check_environment_fits(cognitive_model, environment, observations)

Throw an informative error if a session can't run with `environment`: without one, the
observations in the data can't have a [`MissingValue`](@ref), since nothing fills it in, and
with one, it must list the same observations and actions as `cognitive_model`. Used by
[`SessionInfo`](@ref).
"""
function check_environment_fits(
    cognitive_model::AbstractCognitiveModel, ::NoEnvironment, observations::NamedTuple
)
    if any(column -> any(value -> value isa MissingValue, column), values(observations))
        throw(
            ArgumentError(
                "The observations have entries that are MissingValue(), but no environment was given to fill them in.",
            ),
        )
    end
    return nothing
end

function check_environment_fits(
    cognitive_model::AbstractCognitiveModel,
    environment::AbstractEnvironmentModel,
    observations::NamedTuple,
)
    check_names_fit(cognitive_model, environment)
    return nothing
end

## Splitting a table into sessions ##

"""
    column_map(names)

Return the columns of one kind of attribute as a `NamedTuple` from the name the model uses to
the specification of the column: a `NamedTuple` as it is, and a single name, a tuple of names
or a vector of names as the same names in both places. A specification is the name of a
column, or how to build one from several, see [`columns_used`](@ref). Used by
[`MultipleSessionsInfo`](@ref) when it splits a table.
"""
column_map(names::NamedTuple) = names
column_map(names) = NamedTuple{as_tuple_of_names(names)}(as_tuple_of_names(names))

"""
    columns_used(specification)

Return the names of the columns of a table that one specification of an attribute's column
uses, as a tuple. A specification is one of:
- a `Symbol`, the column itself;
- a `NamedTuple` of `Symbol`s, such as `(; choice = :choicecol, rt = :rtcol)`, for a column
  of `NamedTuple`s with those names built from the columns, one for each row;
- a `Tuple` of `Symbol`s, such as `(:choicecol, :rtcol)`, for a column of `Tuple`s;
- a `Pair` of a `Tuple` of `Symbol`s and a function, such as
  `(:choicecol, :rtcol) => (choice, rt) -> [choice, rt]`, for a column with what the
  function gives for the values of each row.
A `missing` in any column used makes the value of that row `missing`. Used by
[`MultipleSessionsInfo`](@ref) when it checks a table, and by [`build_column`](@ref).
"""
columns_used(specification::Symbol) = (specification,)
function columns_used(
    specification::NamedTuple{names,<:Tuple{Vararg{Symbol}}}
) where {names}
    return values(specification)
end
columns_used(specification::Tuple{Vararg{Symbol}}) = specification
columns_used(specification::Pair{<:Tuple{Vararg{Symbol}}}) = first(specification)
function columns_used(specification)
    return throw(
        ArgumentError(
            "A column is specified by its name, a NamedTuple or a Tuple of names for a column built from several, or a Tuple of names => a function, but this specification is $(repr(specification)).",
        ),
    )
end

"""
    build_column(table, specification, rows)

Return the column of an attribute at the rows `rows` of `table`, as [`columns_used`](@ref)
describes the specification: the column itself, or one built row by row from several, where
a row with a `missing` in any of them gives `missing`. The column is narrowed as a whole
first, see [`narrow_column`](@ref), so that every session gets the same element type. Used by
[`select_rows`](@ref).
"""
function build_column(table::NamedTuple, specification::Symbol, rows)
    return narrow_column(getproperty(table, specification))[rows]
end
function build_column(
    table::NamedTuple, specification::NamedTuple{names,<:Tuple{Vararg{Symbol}}}, rows
) where {names}
    return build_column(
        table, values(specification) => (values...) -> NamedTuple{names}(values), rows
    )
end
function build_column(table::NamedTuple, specification::Tuple{Vararg{Symbol}}, rows)
    return build_column(table, specification => tuple, rows)
end
function build_column(table::NamedTuple, specification::Pair{<:Tuple{Vararg{Symbol}}}, rows)
    columns = map(name -> getproperty(table, name), first(specification))
    combine = last(specification)
    built = map(eachindex(first(columns))) do i
        values = map(column -> column[i], columns)
        return any(ismissing, values) ? missing : combine(values...)
    end
    return narrow_column(built)[rows]
end

"""
    check_table(table, columns)

Throw an informative error if `table` isn't a `NamedTuple` of vectors of one length, or
doesn't have every column in `columns`. Used by [`MultipleSessionsInfo`](@ref) when it
splits a table.
"""
function check_table(table::NamedTuple, columns::Tuple{Vararg{Symbol}})
    for column in columns
        if !haskey(table, column)
            throw(
                ArgumentError(
                    "The table has no column named $column. Its columns are $(keys(table))."
                ),
            )
        end
    end
    isempty(table) && throw(ArgumentError("The table has no columns."))
    all(column -> column isa AbstractVector, values(table)) || throw(
        ArgumentError(
            "Every column of the table must be a vector, but the columns have types $(map(typeof, values(table))).",
        ),
    )
    allequal(map(length, values(table))) || throw(
        ArgumentError(
            "Every column of the table must have the same length, but their lengths are $(map(length, table)).",
        ),
    )
    return nothing
end

function check_table(table, columns::Tuple{Vararg{Symbol}})
    return throw(
        ArgumentError(
            "The table must be a NamedTuple with one vector for each column, which Tables.columntable gives for a DataFrame, but it has type $(typeof(table)).",
        ),
    )
end

"""
    rows_per_session(table, id_columns)

Return a vector with one pair for each session in `table`: its identifier, and the indices
of its rows. A session is every row with one combination of the values in `id_columns`, in
the order of the first row of each. The identifier is the value itself for a single column,
and a `NamedTuple` of the values for several. Used by [`MultipleSessionsInfo`](@ref) when it
splits a table.
"""
function rows_per_session(table::NamedTuple, id_columns::Tuple{Vararg{Symbol}})
    n_rows = length(first(table))
    ids = [session_id(table, id_columns, i) for i in 1:n_rows]
    return [id => findall(==(id), ids) for id in unique(ids)]
end

"""
    session_id(table, id_columns, i)

Return the identifier of the session that row `i` of `table` belongs to. Used by
[`rows_per_session`](@ref).
"""
function session_id(table::NamedTuple, id_columns::Tuple{Symbol}, i)
    return getproperty(table, only(id_columns))[i]
end
function session_id(table::NamedTuple, id_columns::Tuple{Vararg{Symbol}}, i)
    return NamedTuple{id_columns}(map(column -> getproperty(table, column)[i], id_columns))
end

"""
    select_rows(table, columns, rows)

Return the columns in `columns`, a `NamedTuple` from the name the model uses to the
specification of the column in `table`, at the rows `rows`, as a `NamedTuple` with the
model's names, each built with [`build_column`](@ref). Used by
[`MultipleSessionsInfo`](@ref) when it splits a table.
"""
function select_rows(table::NamedTuple, columns::NamedTuple, rows)
    return map(specification -> build_column(table, specification, rows), columns)
end

"""
    select_timestep_sizes(table, timestep_sizes, rows)

Return the sizes of the timesteps of one session: the column named `timestep_sizes` of
`table` at the rows `rows`, or `timestep_sizes` itself when it is a size for all. Used by
[`MultipleSessionsInfo`](@ref) when it splits a table.
"""
function select_timestep_sizes(table::NamedTuple, timestep_sizes::Symbol, rows)
    return getproperty(table, timestep_sizes)[rows]
end
select_timestep_sizes(table::NamedTuple, timestep_sizes, rows) = timestep_sizes

## Reading what a session is set up with ##

"""
    has_environment(session_info)
    has_environment(sessions_info)

Return whether a [`SessionInfo`](@ref) has an environment model, or every session of a
[`MultipleSessionsInfo`](@ref) has, which is decided by the type. Used by
[`warn_if_environment_is_unused`](@ref) and [`check_environment_parameters`](@ref).
"""
has_environment(::SessionInfo{<:Any,NoEnvironment}) = false
has_environment(::SessionInfo{<:Any,<:AbstractEnvironmentModel}) = true
function has_environment(sessions_info::MultipleSessionsInfo)
    return has_environment(first(sessions_info.sessions))
end

"""
    has_missing_observations(session_info)

Return whether any observation in the data of a session is [`MissingValue`](@ref), which an
environment model fills in. Used by [`warn_if_environment_is_unused`](@ref).
"""
function has_missing_observations(session_info::SessionInfo)
    return any(
        column -> any(value -> value isa MissingValue, column),
        values(session_info.observations),
    )
end

"""
    warn_if_environment_is_unused(session_info)
    warn_if_environment_is_unused(sessions)

Warn if a session, or every session in a vector of them, has an environment model but no
observation that is [`MissingValue`](@ref), since filling those in is all an environment
does in a session, so it only costs time. The environment is kept as given, since leaving it
out would change the variables of a model whose environment samples. Used by the
constructors of [`SessionInfo`](@ref) and [`MultipleSessionsInfo`](@ref).
"""
warn_if_environment_is_unused(::SessionInfo{<:Any,NoEnvironment}) = nothing
function warn_if_environment_is_unused(
    session_info::SessionInfo{<:Any,<:AbstractEnvironmentModel}
)
    if !has_missing_observations(session_info)
        @warn "An environment model of type $(typeof(session_info.environment)) was given, but no observation in the data is MissingValue(), so it has nothing to fill in and only costs time. Leave the environment out, or mark the observations it should give as MissingValue()."
    end
    return nothing
end

function warn_if_environment_is_unused(sessions::AbstractVector{<:SessionInfo})
    if has_environment(first(sessions)) && !any(has_missing_observations, sessions)
        @warn "An environment model of type $(typeof(first(sessions).environment)) was given, but no observation in the data of any session is MissingValue(), so it has nothing to fill in and only costs time. Leave the environment out, or mark the observations it should give as MissingValue()."
    end
    return nothing
end

"""
    check_environment_parameters(session_info, environment_parameters)
    check_environment_parameters(sessions_info, environment_parameters)

Throw an informative error if environment parameters are given for a session, or for
sessions, without an environment model, which is decided by the types. Used by the Turing.jl
model of several sessions.
"""
function check_environment_parameters(
    ::SessionInfo{<:Any,<:AbstractEnvironmentModel}, environment_parameters::NamedTuple
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
    recorded_action_names(session_info)

Return the names of the actions that have recorded values in the data of a session, which
are scored against the model's distributions, in a `Val`, so that the timesteps are compiled
knowing them. Used by the Turing.jl model of a session, to select the distributions it
scores.
"""
recorded_action_names(session_info::SessionInfo) = Val(keys(session_info.recorded))

"""
    session_index(sessions_info, session_id)

Return the index `i` of the session with the identifier `session_id` in a
[`MultipleSessionsInfo`](@ref), so that its variables are under `session[i]`, its saved
values are element `i` of what [`turing_multiple_behavioural_sessions`](@ref) returns, and its
parameters are element `i` of each vector a population model gives. The identifier is the
value in the session column, or a `NamedTuple` of the values when there are several, as in
`sessions_info.session_ids`. Used by users reading a chain, not by the models.
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

Return how many sessions a [`MultipleSessionsInfo`](@ref) has, which a population model
samples the parameters of. Used by [`turing_population_model`](@ref) methods, such as the
one for [`IndependentSessions`](@ref).
"""
n_sessions(sessions_info::MultipleSessionsInfo) = length(sessions_info.sessions)

## Scoring the recorded actions ##

"""
    recorded_on_timestep(data_action, action_distribution, cognitive_model)

Return whether each action of one timestep has a recorded value in the data, as a
`NamedTuple` of `Bool`s with the names of `data_action`. A recorded action is scored against
the distribution `cognitive_model` gave for it in `action_distribution`, so it is an error
when that is [`NoAction`](@ref). Used by the Turing.jl models of a timestep.
"""
function recorded_on_timestep(
    data_action::NamedTuple{names}, action_distribution::NamedTuple, cognitive_model
) where {names}
    return NamedTuple{names}(
        recorded_on_timestep(
            names, Tuple(data_action), action_distribution, cognitive_model
        ),
    )
end

function recorded_on_timestep(
    names::Tuple{}, data_values::Tuple{}, action_distribution, cognitive_model
)
    return ()
end
function recorded_on_timestep(
    names::Tuple, data_values::Tuple, action_distribution, cognitive_model
)
    name = first(names)
    value = first(data_values)
    recorded = is_recorded(value)
    if recorded && getproperty(action_distribution, name) isa NoAction
        throw(
            ArgumentError(
                "The cognitive model of type $(typeof(cognitive_model)) took no action $name on a timestep where the data has $value for it.",
            ),
        )
    end
    return (
        recorded,
        recorded_on_timestep(
            Base.tail(names), Base.tail(data_values), action_distribution, cognitive_model
        )...,
    )
end

"""
    initialize_distributions(recorded_action_distributions)

Return an empty vector for the distributions of each action that is recorded in a session,
typed by the distribution the model gave on the first timestep, so that the vectors keep one
type through the session when every timestep gives the same type. An action the model
doesn't take on the first timestep gets a `Vector{Union{}}()`, which the first distribution
kept sets the type of, as for [`push_output!`](@ref). Used by the Turing.jl model of a
session, after its first timestep.
"""
function initialize_distributions(recorded_action_distributions::NamedTuple)
    return map(
        distribution -> if distribution isa AbstractNoValue
            Vector{Union{}}()
        else
            Vector{typeof(distribution)}()
        end, recorded_action_distributions
    )
end

"""
    store_distributions(distributions, recorded_action_distributions, recorded_on_timestep)

Add the distributions of the actions that are recorded on one timestep, where
`recorded_on_timestep` says which, to the vectors in `distributions`, one for each action
that is recorded in the session, and return them. A distribution of another type than its
vector widens the vector, as for [`push_output!`](@ref). Used by the Turing.jl model of a
session, on every timestep.
"""
function store_distributions(
    distributions::NamedTuple{names},
    recorded_action_distributions::NamedTuple,
    recorded_on_timestep::NamedTuple,
) where {names}
    return NamedTuple{names}(
        map(
            name -> if getproperty(recorded_on_timestep, name)
                push_output!(
                    getproperty(distributions, name),
                    getproperty(recorded_action_distributions, name),
                )
            else
                getproperty(distributions, name)
            end,
            names,
        ),
    )
end

## Saving what a fitted session returns ##

"""
    saved_session_outputs(cognitive_model, outputs, save_history)

Return what a session saves of the outputs of one timestep, or of where it starts, as
`save_history` selects it: the observation, the action, the action distribution and the
state, in a `NamedTuple` with those names. The observation and the action can have one of
two types, so they go in through [`entry`](@ref). Used by the Turing.jl model of a session,
on every timestep.
"""
function saved_session_outputs(cognitive_model, outputs, save_history)
    return merge(
        entry(
            Val(:observation),
            select_names(outputs.observation, observation_selection(save_history)),
        ),
        entry(Val(:action), select_names(outputs.action, action_selection(save_history))),
        (;
            action_distribution=select_names(
                outputs.action_distribution, action_distribution_selection(save_history)
            ),
            state=save_state(
                cognitive_model, outputs.cognitive_state, state_selection(save_history)
            ),
        ),
    )
end

"""
    initial_session_outputs(save_initial_values::Val, cognitive_model, previous, save_history)

Return what a session saves from before the first timestep, as timestep 0, when
`save_history` saves the initial values, and `nothing` otherwise. The action distribution of
timestep 0 is [`BeforeFirstTimestep`](@ref) for every action, since none was given yet. Used
by the Turing.jl model of a session, before its first timestep runs.
"""
function initial_session_outputs(::Val{true}, cognitive_model, previous, save_history)
    return saved_session_outputs(
        cognitive_model,
        merge(previous, (; action_distribution=initial_action(cognitive_model))),
        save_history,
    )
end
initial_session_outputs(::Val{false}, cognitive_model, previous, save_history) = nothing

"""
    initialize_session_storage(initial_saved, saved, n_timesteps)

Return the vectors for what a session saves over `n_timesteps` timesteps, typed by `saved`,
what the first timestep saved, as [`initialize_storage`](@ref) makes them. They start with
`initial_saved`, what timestep 0 saved, when it isn't `nothing`. Used by the Turing.jl model
of a session, after its first timestep.
"""
function initialize_session_storage(::Nothing, saved, n_timesteps)
    return (;
        observations=initialize_storage(saved.observation, n_timesteps),
        actions=initialize_storage(saved.action, n_timesteps),
        action_distributions=initialize_storage(saved.action_distribution, n_timesteps),
        states=initialize_storage(saved.state, n_timesteps),
    )
end

function initialize_session_storage(initial_saved::NamedTuple, saved, n_timesteps)
    return (;
        observations=initialize_storage(
            initial_saved.observation, saved.observation, n_timesteps
        ),
        actions=initialize_storage(initial_saved.action, saved.action, n_timesteps),
        action_distributions=initialize_storage(
            initial_saved.action_distribution, saved.action_distribution, n_timesteps
        ),
        states=initialize_storage(initial_saved.state, saved.state, n_timesteps),
    )
end

"""
    store_session_outputs!(storage, saved, i)

Store `saved`, what one timestep saved, as element `i` of each vector in `storage`, and
return the storage, which [`store_output!`](@ref) widens when a value has a new type. Used
by the Turing.jl model of a session, on every timestep after the first.
"""
function store_session_outputs!(storage, saved, i)
    return (;
        observations=store_output!(storage.observations, saved.observation, i),
        actions=store_output!(storage.actions, saved.action, i),
        action_distributions=store_output!(
            storage.action_distributions, saved.action_distribution, i
        ),
        states=store_output!(storage.states, saved.state, i),
    )
end

## Display ##

#Prints the name of a model, or none
print_model_name(io::IO, model::AbstractModel) = print(io, nameof(typeof(model)))
print_model_name(io::IO, ::NoEnvironment) = print(io, "none")

#Prints the names of one kind of attribute in the data, or none
function print_data_names(io::IO, kind, names)
    print(io, "\n", kind, ": ")
    isempty(names) ? print(io, "none") : join(io, names, ", ")
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", session_info::SessionInfo)
    show(io::IO, ::MIME"text/plain", sessions_info::MultipleSessionsInfo)

Print a summary of a session, or of several, on one line each: the models, the timesteps or
the sessions, the names in the data, and who starts. This is what the REPL and notebooks
show, since the full type and the data would fill the screen.
"""
function Base.show(io::IO, ::MIME"text/plain", session_info::SessionInfo)
    print(io, "SessionInfo\nCognitive model: ")
    print_model_name(io, session_info.cognitive_model)
    print(io, "\nEnvironment: ")
    print_model_name(io, session_info.environment)
    print(io, "\nTimesteps: ", session_info.n_timesteps)
    print_data_names(io, "Observations", keys(session_info.observations))
    print_data_names(io, "Actions", keys(session_info.actions))
    print(
        io,
        "\nWho starts: ",
        session_info.who_starts isa AgentStarts ? "agent" : "environment",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", sessions_info::MultipleSessionsInfo)
    first_session = first(sessions_info.sessions)
    print(
        io,
        "MultipleSessionsInfo\nSessions: ",
        n_sessions(sessions_info),
        "\nCognitive model: ",
    )
    print_model_name(io, first_session.cognitive_model)
    print(io, "\nEnvironment: ")
    print_model_name(io, first_session.environment)
    print_data_names(io, "Observations", keys(first_session.observations))
    print_data_names(io, "Actions", keys(first_session.actions))
    return nothing
end
