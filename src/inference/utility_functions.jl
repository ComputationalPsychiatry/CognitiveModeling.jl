## Preparing the data of a session ##

"""
    is_recorded(value)

Return whether a single action or observation in the data happened and was recorded, so that
it is scored: neither [`MissingValue`](@ref) nor a marker for nothing happening, such as
[`NoAction`](@ref) or [`NoObservation`](@ref). Used by [`recorded_values`](@ref) and
[`recorded_on_timestep`](@ref).
"""
is_recorded(value) = !(value isa MissingValue || value isa AbstractNoValue)

"""
    recorded_values(columns)

Return the recorded values of each action or observation in `columns`, a `NamedTuple` with
one vector per name, as a `NamedTuple` of vectors in the order of the timesteps. Fitting
scores these against the distributions the model gave on the same timesteps: the actions
against the cognitive model's, and the observations against the environment's. Each vector
has the element type of its column without [`MissingValue`](@ref) and the markers for
nothing happening, so that a session without recorded values for a name gets an empty vector
of the same type as other sessions, which scores nothing. A name whose column can only hold
those markers is left out, which is decided by its type, so that it is the same for every
session of a table. Used by [`SessionInfo`](@ref), which keeps the results in its fields
`recorded_actions` and `recorded_observations`.
"""
function recorded_values(columns::NamedTuple{names}) where {names}
    recorded = map(
        column ->
            Vector{Base.typesplit(eltype(column), Union{MissingValue,AbstractNoValue})}(
                filter(is_recorded, column)
            ),
        columns,
    )
    names_with_a_type = Tuple(
        name for name in names if eltype(getproperty(recorded, name)) !== Union{}
    )
    return NamedTuple{names_with_a_type}(recorded)
end

"""
    first_timestep_to_check_stopping(check_stopping_criterion, observations, actions,
                                     n_timesteps)

Return the first timestep before which a session checks the [`stopping_criterion`](@ref) of
its models, from the `check_stopping_criterion` of [`SessionInfo`](@ref): the timestep after
the last one with data when it is `nothing`, which is 1 when every value is
[`MissingValue`](@ref), 1 when it is `true`, and `n_timesteps + 1` when it is `false`, so
that no timestep is checked. Used by `SessionInfo`.
"""
function first_timestep_to_check_stopping(
    ::Nothing, observations::NamedTuple, actions::NamedTuple, n_timesteps
)
    return max(last_timestep_with_data(observations), last_timestep_with_data(actions)) + 1
end
function first_timestep_to_check_stopping(check::Bool, observations, actions, n_timesteps)
    return check ? 1 : n_timesteps + 1
end

"""
    last_timestep_with_data(columns)

Return the last timestep on which any column in `columns`, a `NamedTuple` of vectors, has a
value that isn't [`MissingValue`](@ref), and 0 when none has. A marker for nothing happening
counts as data, since it says that the timestep happened. Used by
[`first_timestep_to_check_stopping`](@ref).
"""
function last_timestep_with_data(columns::NamedTuple)
    last_timestep = 0
    for column in columns
        t = findlast(value -> !(value isa MissingValue), column)
        t === nothing || (last_timestep = max(last_timestep, t))
    end
    return last_timestep
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

#A call of its own, so that a value which can have two types gives one of two NamedTuples with a concrete field each: (; a = x) written out is inferred as a NamedTuple whose field type is the Union, which nothing is compiled for, while a call with a Union-typed argument is split by type
"""
    entry(::Val{name}, value)

Return `(; name = value)`. A `NamedTuple` which holds a value that can have two types is
built with `merge` of these, as [`row`](@ref) does. Used by [`row`](@ref), by
[`saved_session_values`](@ref) and by the timesteps of fitted sessions.
"""
entry(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of
vectors with one entry for each timestep, built with [`entry`](@ref) so that a column which
holds two types, such as an action that is [`MissingValue`](@ref) on some timesteps, gives
one of two `NamedTuple`s with concrete fields rather than one without a definite type. Used
by the timesteps of fitted sessions, and by the models of several sessions, which read the
parameters of one session with it.
"""
row(columns::NamedTuple{()}, t) = (;)
function row(columns::NamedTuple{names}, t) where {names}
    return merge(entry(Val(first(names)), first(columns)[t]), row(Base.tail(columns), t))
end

"""
    has_missing_value(data)

Return whether any entry of `data`, the `NamedTuple` of the actions or the observations of
one timestep, is [`MissingValue`](@ref). Used by the timesteps of fitted sessions, to skip
imputing when nothing is missing.
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
    cognitive_model::AbstractModel, observations::NamedTuple, actions::NamedTuple
)
    attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    if !has_every_name(attributes.inputs, keys(observations))
        throw(
            ArgumentError(
                "The observations in the data are named $(keys(observations)), but attribute_names lists the observations of $(describe_model(cognitive_model)), its inputs, as $(attributes.inputs). The data must have every observation the cognitive model lists.",
            ),
        )
    end
    if !same_names(keys(actions), attributes.outputs)
        throw(
            ArgumentError(
                "The actions in the data are named $(keys(actions)), but attribute_names lists the actions of $(describe_model(cognitive_model)), its outputs, as $(attributes.outputs).",
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
function check_data(cognitive_model::AbstractModel, observations, actions)
    return throw(
        ArgumentError(
            "The observations and the actions must each be a NamedTuple with one vector for each name, such as (; reward = [1.0, 0.0]), or a single vector when the model has only one of them in its attributes, but they have types $(typeof(observations)) and $(typeof(actions)).",
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
    cognitive_model::AbstractModel,
    ::NoEnvironment,
    observations::NamedTuple,
    observation_map,
    action_map,
)
    if observation_map !== nothing || action_map !== nothing
        throw(
            ArgumentError(
                "A session without an environment takes no observation_map or action_map, since nothing passes between two models.",
            ),
        )
    end
    if has_missing_entries(observations)
        throw(
            ArgumentError(
                "The observations have entries that are MissingValue(), but no environment was given to fill them in.",
            ),
        )
    end
    return nothing
end

function check_environment_fits(
    cognitive_model::AbstractModel,
    environment::AbstractModel,
    observations::NamedTuple,
    observation_map,
    action_map,
)
    check_names_fit(cognitive_model, environment, observation_map, action_map)
    return nothing
end

"""
    environment_view(values, environment, observation_map, filler)

Return `values`, a `NamedTuple` in the names under which the cognitive model receives the
observations, as `environment` gives them: under each output name that the environment
lists, the value that `observation_map` passes on from it, or `filler` for an output that
the map doesn't pass on. Without a map the names are the same. Without an environment
`values` is returned as it is. Used by [`SessionInfo`](@ref) for the environment's data,
with a column of [`MissingValue`](@ref) for each output that isn't passed on, so that it is
imputed, and for the observation the environment starts from, with
[`BeforeFirstTimestep`](@ref).
"""
environment_view(values, ::NoEnvironment, observation_map, filler) = values
function environment_view(values, environment::AbstractModel, observation_map, filler)
    outputs = check_type_attribute_names(attribute_names(environment), environment).outputs
    return NamedTuple{outputs}(
        map(name -> value_passed_on(values, name, observation_map, filler), outputs)
    )
end

"""
    value_passed_on(values, name, observation_map, filler)

Return the value in `values` that the environment's output `name` is received as: under the
same name without a map, under the name `observation_map` gives it, or `filler` when the map
doesn't pass it on. Used by [`environment_view`](@ref).
"""
value_passed_on(values, name, ::Nothing, filler) = values[name]
function value_passed_on(
    values, name, ::NameMap{Tgiven,Treceived}, filler
) where {Tgiven,Treceived}
    index = findfirst(==(name), Tgiven)
    return index === nothing ? filler : values[Treceived[index]]
end

## Splitting a table into sessions ##

"""
    data_names(cognitive_model, kind)

Return the names that [`attribute_names`](@ref) of `cognitive_model` lists for the data of
`kind`, `:observations` or `:actions`: its inputs for the observations and its outputs for
the actions. Used by [`column_map`](@ref) and [`named_data`](@ref).
"""
function data_names(cognitive_model, kind::Symbol)
    attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    return kind === :observations ? attributes.inputs : attributes.outputs
end

"""
    column_map(columns, cognitive_model, kind)

Return the columns of `kind`, `:observations` or `:actions`, as a `NamedTuple` from the name
the model uses to the specification of the column: a `NamedTuple` as it is, and a tuple or a
vector of names as the same names in both places. A single name is the column of the
attribute of that kind when the model has only one in its attributes, whatever the column is
called. It is an error when the model has several, and when the model lists the name as an
attribute of the other kind only, since `observation_columns` and `action_columns` may then
be swapped. When the model
lists none, it is a column of that name. A specification is the name of a column, or how to
build one from several, see [`columns_used`](@ref). Used by [`MultipleSessionsInfo`](@ref)
when it splits a table.
"""
column_map(columns::NamedTuple, cognitive_model, kind::Symbol) = columns
function column_map(column::Symbol, cognitive_model, kind::Symbol)
    names = data_names(cognitive_model, kind)
    other_kind = kind === :observations ? :actions : :observations
    keyword = kind === :observations ? "observation_columns" : "action_columns"

    #A name the model lists as an attribute of the other kind only may mean that the two keyword arguments are swapped
    if column in data_names(cognitive_model, other_kind) && !(column in names)
        throw(
            ArgumentError(
                "$keyword is $(repr(column)), which $(describe_model(cognitive_model)) lists among its $other_kind, not its $kind. Check that observation_columns and action_columns aren't swapped.",
            ),
        )
    end

    #A single name is the column of the attribute of that kind, when the model has only one in its attributes
    if length(names) == 1
        return NamedTuple{names}((column,))
    elseif length(names) > 1
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(cognitive_model))) lists the $kind $names, so $keyword must give the column of each by name, such as $(NamedTuple{names}(names)), and not the single name $(repr(column)).",
            ),
        )
    end

    #A model with no attributes of that kind gets the column under its own name
    return NamedTuple{(column,)}((column,))
end
function column_map(columns, cognitive_model, kind::Symbol)
    return NamedTuple{as_tuple_of_names(columns)}(as_tuple_of_names(columns))
end

"""
    named_data(data, cognitive_model, kind)

Return the data of `kind`, `:observations` or `:actions`, as a `NamedTuple` of vectors: a
single vector as the data of the attribute of that kind when the model has only one in its
attributes, which is an error when it has another number of them, and anything else as it
is, for [`check_data`](@ref) to check. A vector of `NamedTuple`s with the model's names is
the data written as one row for each timestep, which is returned as it is, for `check_data`
to report. Used by [`SessionInfo`](@ref).
"""
named_data(data, cognitive_model, kind::Symbol) = data
function named_data(data::AbstractVector, cognitive_model, kind::Symbol)
    names = data_names(cognitive_model, kind)

    #NamedTuples with the model's names are the data written as one row for each timestep, which check_data reports
    eltype(data) <: NamedTuple{names} && return data

    #Otherwise the vector is the data of the attribute of that kind, of which the model must have only one in its attributes
    if length(names) != 1
        throw(
            ArgumentError(
                "The $kind are a single vector, which can only be their data when the model has only one of them in its attributes, but a cognitive model of type $(typeof(cognitive_model)) has the $kind $names. Give them as a NamedTuple with one vector for each name.",
            ),
        )
    end
    return NamedTuple{names}((data,))
end

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
first, see [`narrow_column`](@ref), so that every session gets the same element type. Used
by [`select_rows`](@ref).
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
    column_table(table)

Return `table` as a `NamedTuple` with one vector for each column. A `NamedTuple` is returned
as it is, and any other table is read with Tables.jl, through the package's extension for
it, which loads with Tables.jl, as it does with DataFrames.jl, CSV.jl and the other packages
of tables, since they load Tables.jl. Without it, a table that isn't a `NamedTuple` is an
error. Used by [`MultipleSessionsInfo`](@ref) when it splits a table.
"""
column_table(table::NamedTuple) = table
function column_table(table)
    tables_extension = Base.get_extension(@__MODULE__, :CognitiveModelingTablesExt)
    if tables_extension === nothing
        throw(
            ArgumentError(
                "The table has type $(typeof(table)), which is read with Tables.jl. Load Tables.jl, or a package which loads it, such as DataFrames.jl, or give the table as a NamedTuple with one vector for each column.",
            ),
        )
    end
    return tables_extension.column_table(table)
end

"""
    check_table(table, columns)

Throw an informative error if `table`, a `NamedTuple`, doesn't have vectors of one length,
or doesn't have every column in `columns`. Used by [`MultipleSessionsInfo`](@ref) when it
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
[`MultipleSessionsInfo`](@ref) has, which is decided by the type. A session can have an
environment which doesn't run, see [`SessionInfo`](@ref). Used by
[`check_environment_parameters`](@ref).
"""
has_environment(::SessionInfo{<:Any,NoEnvironment}) = false
has_environment(::SessionInfo{<:Any,<:AbstractModel}) = true
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
`false`, and with `nothing` only when an observation or an action is missing, the
environment's history is saved, or its stopping criterion is checked on every timestep. A
session without an environment never includes one. With `false`, it is an error when an
observation is missing, the environment's history is saved, or its stopping criterion is
checked on every timestep, since those need the environment. Used by
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
        "the data has missing observations, which only the environment can fill in"
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

    #By default it is included when it is needed, or when an action is missing, which the observations it scores inform
    return needs_environment !== nothing || has_missing_entries(actions)
end

"""
    without_environment(session_info)

Return the session with `NoEnvironment()` as its environment and everything else the same,
so that it runs as a session without an environment. Used by fitted sessions whose
environment doesn't run, since every observation is in the data.
"""
function without_environment(session_info::SessionInfo)
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
        session_info.recorded_actions,
        session_info.recorded_observations,
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
    sessions_for_environment_parameters(environment_parameters_model, sessions_info)

Return the sessions as a full model runs them with `environment_parameters_model`: as they
are when it estimates nothing, and otherwise with every session that has an environment
including it, so that the observations in the data inform the parameters. Used by
[`prepare_full_model`](@ref).
"""
function sessions_for_environment_parameters(
    environment_parameters_model, sessions_info::MultipleSessionsInfo
)
    estimates_parameters(environment_parameters_model) || return sessions_info
    return MultipleSessionsInfo(
        map(including_environment, sessions_info.sessions),
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
values that nothing in the model uses.
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
    session_name(session_id)

Return the name of a session, as a `Symbol`, from its identifier: the value of a single
session column, as in `S1`, and the name and the value of each of several columns, joined
by `_` within a column and by `__` between columns, as in `id_S1__block_2`. The chains of a
fit name the entries of a session after it, such as `S1.learning_rate`, and
[`FixedParametersBySession`](@ref) can find the values of a session by it.

A name that isn't a valid Julia identifier, such as that of the identifier `1` or
`"sub-01"`, can't be typed into a `VarName`, but interpolation puts in any name: after
`using CognitiveModeling: session_name`, the learning rate of a session is
`chain[@varname(\$(session_name(session_id)).learning_rate)]`.

# Examples
```jldoctest
julia> CognitiveModeling.session_name("S1")
:S1

julia> CognitiveModeling.session_name((; id = "S1", block = 2))
:id_S1__block_2
```
"""
function session_name(session_id)

    #Asserted, since the compiler can't infer that Symbol gives a Symbol for an identifier of any type
    return Symbol(session_id)::Symbol
end
function session_name(session_id::NamedTuple)
    return Symbol(
        join((string(name, "_", value) for (name, value) in pairs(session_id)), "__")
    )
end

"""
    check_unique_session_names(session_ids)

Throw an informative error if two sessions have the same name, as [`session_name`](@ref)
gives it, since the chains of a fit and [`FixedParametersBySession`](@ref) find a session by
its name. Two sessions have the same name when they have the same identifier, or when their
identifiers give the same text, such as `1` and `"1"`. Used by
[`MultipleSessionsInfo`](@ref).
"""
function check_unique_session_names(session_ids::AbstractVector)
    names = map(session_name, session_ids)
    allunique(names) && return nothing
    for (i, name) in enumerate(names)
        first_i = findfirst(==(name), names)
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
                "The session identifiers $(repr(first_id)) and $(repr(session_id)) both give the session name $name, which the chains of a fit and FixedParametersBySession find a session by. Every session must have a name of its own.",
            ),
        )
    end
    return nothing
end

"""
    including_environment(session_info)

Return the session including its environment when it has one, and everything else the
same. A session that was prepared with `include_environment = false` is an error. Used by
[`sessions_for_environment_parameters`](@ref).
"""
function including_environment(session_info::SessionInfo)
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
        session_info.recorded_actions,
        session_info.recorded_observations,
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
    check_environment_parameters(session_info, environment_parameters)
    check_environment_parameters(sessions_info, environment_parameters)

Throw an informative error if environment parameters are given for a session, or for
sessions, without an environment model, which is decided by the types. Used by the models
of several sessions.
"""
function check_environment_parameters(
    ::SessionInfo{<:Any,<:AbstractModel}, environment_parameters::NamedTuple
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
    recorded_observation_names(session_info)

Return the names of the actions, or of the observations, that have recorded values in the
data of a session, which are scored against the distributions the cognitive model, or the
environment model, gave for them, in a `Val`. A session without an environment scores no
observations, so it has no such names. Used by fitted sessions, to select the distributions
they score.
"""
recorded_action_names(session_info::SessionInfo) = Val(keys(session_info.recorded_actions))
recorded_observation_names(::SessionInfo{<:Any,NoEnvironment}) = Val(())
function recorded_observation_names(session_info::SessionInfo{<:Any,<:AbstractModel})
    return Val(keys(session_info.recorded_observations))
end

"""
    session_index(sessions_info, session_id)

Return the index `i` of the session with the identifier `session_id` in a
[`MultipleSessionsInfo`](@ref), which is how the models of several sessions number it: its
parameters are element `i` of each vector a session parameter model gives, and its saved
values are element `i` of what the sessions return. The identifier is the value in the
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

## Scoring the recorded values ##

"""
    recorded_on_timestep(data_output, output_distribution, model)

Return whether each output of `model` on one timestep has a recorded value in the data, as a
`NamedTuple` of `Bool`s with the names of `data_output`: the actions of a cognitive model,
or the observations an environment model gives. A recorded value is scored against the
distribution the model gave for it in `output_distribution`, so it is an error when the
model gave nothing for it, [`NoAction`](@ref) or [`NoObservation`](@ref). Used by the
timesteps of fitted sessions.
"""
function recorded_on_timestep(
    data_output::NamedTuple{names}, output_distribution::NamedTuple, model
) where {names}
    return NamedTuple{names}(
        recorded_on_timestep(names, Tuple(data_output), output_distribution, model)
    )
end

function recorded_on_timestep(
    names::Tuple{}, data_values::Tuple{}, output_distribution, model
)
    return ()
end
function recorded_on_timestep(names::Tuple, data_values::Tuple, output_distribution, model)
    name = first(names)
    value = first(data_values)
    recorded = is_recorded(value)
    if recorded && getproperty(output_distribution, name) isa AbstractNoValue
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(model))) gave no output $name on a timestep where the data has $value for it.",
            ),
        )
    end
    return (
        recorded,
        recorded_on_timestep(
            Base.tail(names), Base.tail(data_values), output_distribution, model
        )...,
    )
end

"""
    initialize_distributions(recorded_distributions)

Return an empty vector for the distributions of each output that is recorded in a session,
typed by the distribution the model gave on the first timestep, so that the vectors keep one
type through the session when every timestep gives the same type. An output the model
doesn't give on the first timestep gets a `Vector{Union{}}()`, which the first distribution
kept sets the type of, as for [`push_value!`](@ref). Used by fitted sessions, after their
first timestep.
"""
function initialize_distributions(recorded_distributions::NamedTuple)
    return map(distribution -> if distribution isa AbstractNoValue
        Vector{Union{}}()
    else
        Vector{typeof(distribution)}()
    end, recorded_distributions)
end

"""
    store_distributions(distributions, recorded_distributions, recorded_on_timestep)

Add the distributions of the outputs that are recorded on one timestep, where
`recorded_on_timestep` says which, to the vectors in `distributions`, one for each output
that is recorded in the session, and return them. A distribution of another type than its
vector widens the vector, as for [`push_value!`](@ref). Used by fitted sessions, on every
timestep.
"""
function store_distributions(
    distributions::NamedTuple{names},
    recorded_distributions::NamedTuple,
    recorded_on_timestep::NamedTuple,
) where {names}
    return NamedTuple{names}(
        map(
            name -> if getproperty(recorded_on_timestep, name)
                push_value!(
                    getproperty(distributions, name),
                    getproperty(recorded_distributions, name),
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
    saved_session_values(session_info, timestep_values)

Return what a session saves of the values of one timestep, or of where it starts, as the
save histories of `session_info` select it, arranged as the session returns it: the
cognitive model's `states`, `observations`, `actions` and `action_distributions`, and with
an environment its `states`, `observations`, `actions` and `observation_distributions` under
`environment`, as [`map_saved`](@ref) arranges them. `timestep_values` has the cognitive
model's `state`, `observation`, `action` and `action_distribution`, and under `environment`
the environment's `state`, `observation`, `action` and `observation_distribution`, each in
its own names. Used by fitted sessions, on every timestep.
"""
function saved_session_values(
    session_info::SessionInfo{<:Any,NoEnvironment}, timestep_values
)
    return saved_cognitive_values(
        session_info.cognitive_model, timestep_values, session_info.cognitive_save_history
    )
end

function saved_session_values(
    session_info::SessionInfo{<:Any,<:AbstractModel}, timestep_values
)
    return merge(
        saved_cognitive_values(
            session_info.cognitive_model,
            timestep_values,
            session_info.cognitive_save_history,
        ),
        entry(
            Val(:environment),
            saved_environment_values(
                session_info.environment,
                timestep_values.environment,
                session_info.environment_save_history,
            ),
        ),
    )
end

"""
    saved_cognitive_values(cognitive_model, timestep_values, save_history)
    saved_environment_values(environment, environment_timestep_values, save_history)

Return what `save_history` selects of one model's part of the values of a timestep: the
`states`, `observations`, `actions` and `action_distributions` of the cognitive model, or
the `states`, `observations`, `actions` and `observation_distributions` of the environment,
from the values of the environment alone. The observation and the action can have one of
two types, so they go in through [`entry`](@ref). Used by [`saved_session_values`](@ref).
"""
function saved_cognitive_values(cognitive_model, timestep_values, save_history)
    return merge(
        (;
            states=save_state(
                cognitive_model, timestep_values.state, state_selection(save_history)
            ),
        ),
        entry(
            Val(:observations),
            select_names(timestep_values.observation, input_selection(save_history)),
        ),
        entry(
            Val(:actions),
            select_names(timestep_values.action, output_selection(save_history)),
        ),
        (;
            action_distributions=select_names(
                timestep_values.action_distribution,
                output_distribution_selection(save_history),
            ),
        ),
    )
end

function saved_environment_values(environment, environment_timestep_values, save_history)
    return merge(
        (;
            states=save_state(
                environment,
                environment_timestep_values.state,
                state_selection(save_history),
            ),
        ),
        entry(
            Val(:observations),
            select_names(
                environment_timestep_values.observation, output_selection(save_history)
            ),
        ),
        entry(
            Val(:actions),
            select_names(environment_timestep_values.action, input_selection(save_history)),
        ),
        (;
            observation_distributions=select_names(
                environment_timestep_values.observation_distribution,
                output_distribution_selection(save_history),
            ),
        ),
    )
end

"""
    initial_session_values(save_initial_values::Val, session_info, previous)

Return what a session saves from before the first timestep, as timestep 0, when
`save_initial_values` is `Val(true)`, and `nothing` otherwise. `previous` has the values
that the first timestep starts from, arranged as those of a timestep. The distributions of
timestep 0 are [`BeforeFirstTimestep`](@ref) for every output of each model, since none was
given yet. Used by fitted sessions, before their first timestep runs.
"""
function initial_session_values(
    ::Val{true}, session_info::SessionInfo{<:Any,NoEnvironment}, previous
)
    cognitive_model = session_info.cognitive_model

    #No action distribution was given before the first timestep
    return saved_session_values(
        session_info,
        merge(
            previous, (; action_distribution=output_before_first_timestep(cognitive_model))
        ),
    )
end

function initial_session_values(
    ::Val{true}, session_info::SessionInfo{<:Any,<:AbstractModel}, previous
)
    cognitive_model = session_info.cognitive_model

    #No observation distribution was given before the first timestep
    environment_timestep_values = merge(
        previous.environment,
        (;
            observation_distribution=output_before_first_timestep(session_info.environment),
        ),
    )

    #Nor an action distribution
    return saved_session_values(
        session_info,
        merge(
            previous,
            (;
                action_distribution=output_before_first_timestep(cognitive_model),
                environment=environment_timestep_values,
            ),
        ),
    )
end
initial_session_values(::Val{false}, session_info::SessionInfo, previous) = nothing

"""
    with_empty_environment_entry(saved_values)

Return `saved_values`, what a session without an environment saved, with an `environment`
entry whose vectors hold an empty `NamedTuple` for each saved timestep, for a session which
has an environment but didn't include it, so that every session with an environment returns
the same entries, in the same order, since the model's return otherwise has two types. Used
by fitted sessions.
"""
function with_empty_environment_entry(saved_values::NamedTuple)
    empty_vector = fill((;), length(saved_values.states))
    return (;
        saved_values.states,
        saved_values.observations,
        saved_values.actions,
        saved_values.action_distributions,
        environment=(;
            states=empty_vector,
            observations=empty_vector,
            actions=empty_vector,
            observation_distributions=empty_vector,
        ),
        saved_values.timestep_sizes,
    )
end

"""
    timestep_sizes_until(timestep_sizes, n_saved)

Return the vector of the sizes of the timesteps that a session returns, cut to its first
`n_saved` elements, for a session that ended before its data did when a
[`stopping_criterion`](@ref) was met. An empty vector of sizes, for a session that saves
nothing, is kept. Used by fitted sessions.
"""
function timestep_sizes_until(timestep_sizes::AbstractVector, n_saved)
    return isempty(timestep_sizes) ? timestep_sizes : timestep_sizes[1:n_saved]
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

#Shows a summary of a session in the REPL and notebooks, since the full type and the data would fill the screen
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

#Shows a summary of several sessions, with the models and the names of the first
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
