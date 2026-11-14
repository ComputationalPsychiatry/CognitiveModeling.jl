## Splitting a table into sessions ##

"""
    data_names(cognitive_model, kind)

Return the names in the attributes of `cognitive_model`, from [`attribute_names`](@ref), for
the data of `kind`, `:observations` or `:actions`: its inputs for the observations and its
outputs for the actions. Used by [`column_map`](@ref) and [`named_data`](@ref).
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
the model uses to the specification of the column, from what `observation_columns` or
`action_columns` of [`MultipleSessionsInfo`](@ref) was given:
- a tuple or a vector with an entry for each attribute, see [`column_entry`](@ref): a name,
  for a column that is named as the model names the attribute, or a pair from the
  specification of the column to the model's name, such as `:outcome => :reward`;
- one such pair on its own;
- a single name, which is the column of the attribute of that kind when the model has only
  one in its attributes, whatever the column is called. It is an error when the model has
  several, and when the model's attributes have the name as one of the other kind only,
  since `observation_columns` and `action_columns` may then be swapped. When its attributes
  have none of that kind, it is a column of that name;
- how to build a value from several columns, without the model's name: a `NamedTuple` of
  names, or a pair of a tuple of names and a function, for the attribute of that kind when
  the model has only one in its attributes.
A specification is the name of a column, or how to build one from several, see
[`columns_used`](@ref). Used by [`MultipleSessionsInfo`](@ref) when it splits a table.
"""
function column_map(columns::Union{Tuple,AbstractVector}, cognitive_model, kind::Symbol)

    #Each entry gives the name the model uses and the specification of its column
    entries = map(column_entry, Tuple(columns))
    names = map(first, entries)
    if !allunique(names)
        throw(
            ArgumentError(
                "The columns of the $kind give the names $names, but each name can only be given once.",
            ),
        )
    end
    return NamedTuple{names}(map(last, entries))
end

#A pair is one entry when it ends in the model's name. Otherwise it is a tuple of names and a function, which build the value of the model's only attribute of that kind
function column_map(column::Pair, cognitive_model, kind::Symbol)
    if names_its_attribute(column)
        return column_map((column,), cognitive_model, kind)
    end
    name = only_attribute_name(cognitive_model, kind, column)
    return NamedTuple{(name,)}((column,))
end

#A NamedTuple of names builds a NamedTuple in each row, as the value of the model's only attribute of that kind
function column_map(columns::NamedTuple, cognitive_model, kind::Symbol)
    name = only_attribute_name(cognitive_model, kind, columns)
    return NamedTuple{(name,)}((columns,))
end

function column_map(column::Symbol, cognitive_model, kind::Symbol)
    names = data_names(cognitive_model, kind)
    other_kind = kind === :observations ? :actions : :observations
    keyword = kind === :observations ? "observation_columns" : "action_columns"

    #A name of the other kind only, in the model's attributes, may mean that the two keyword arguments are swapped
    if column in data_names(cognitive_model, other_kind) && !(column in names)
        throw(
            ArgumentError(
                "$keyword is $(repr(column)), which is among the $other_kind in the attributes of $(describe_model(cognitive_model)), not its $kind. Check that observation_columns and action_columns aren't swapped.",
            ),
        )
    end

    #A single name is the column of the attribute of that kind, when the model has only one in its attributes
    if length(names) == 1
        return NamedTuple{names}((column,))
    elseif length(names) > 1
        throw(
            ArgumentError(
                "The attributes of $(describe_model(cognitive_model)) have the $kind $names, so $keyword must give the column of each, such as $names when the columns have those names, or with a pair for a column of another name, as in :outcome => $(repr(first(names))), and not the single name $(repr(column)).",
            ),
        )
    end

    #A model with no attributes of that kind gets the column under its own name
    return NamedTuple{(column,)}((column,))
end

"""
    column_entry(entry)

Return the name the model uses for an attribute and the specification of its column, as a
pair, from one entry of the columns of a kind. An entry is a name, for a column that is
named as the model names the attribute, or a pair from the specification of the column to
the model's name: `:outcome => :reward` for a column with another name,
`(:choicecol, :rtcol) => :ssm_action` or
`(; choice = :choicecol, rt = :rtcol) => :ssm_action` for a value built from several
columns, and `(:choicecol, :rtcol) => combine => :ssm_action` for what a function gives for
their values. Used by [`column_map`](@ref).
"""
column_entry(column::Symbol) = column => column
function column_entry(entry::Pair)
    specification, target = entry

    #The model's name comes after the specification of its column
    target isa Symbol && return target => specification

    #A function between the two builds the value from the columns before it
    if target isa Pair && last(target) isa Symbol
        return last(target) => (specification => first(target))
    end
    return throw(
        ArgumentError(
            "An entry of the columns must end in the name the model uses, such as :outcome => :reward or (:choicecol, :rtcol) => combine => :ssm_action, but it is $(repr(entry)).",
        ),
    )
end
function column_entry(entry)
    return throw(
        ArgumentError(
            "An entry of the columns is a name, or a pair from a column to the name the model uses, such as :outcome => :reward, but it is $(repr(entry)).",
        ),
    )
end

"""
    names_its_attribute(column)

Return whether `column`, a pair given as the columns of a kind, ends in the name the model
uses for the attribute, as in `:outcome => :reward` and
`(:choicecol, :rtcol) => combine => :ssm_action`, and not in a function, as in
`(:choicecol, :rtcol) => combine`. Used by [`column_map`](@ref).
"""
function names_its_attribute(column::Pair)
    target = last(column)
    return target isa Symbol || (target isa Pair && last(target) isa Symbol)
end

"""
    only_attribute_name(cognitive_model, kind, columns)

Return the name of the attribute of `kind`, `:observations` or `:actions`, when the model
has only one in its attributes, for `columns`, which build its value from several columns
without naming it. It is an error when the model has another number of them, since the
value then needs the model's name. Used by [`column_map`](@ref).
"""
function only_attribute_name(cognitive_model, kind::Symbol, columns)
    names = data_names(cognitive_model, kind)
    keyword = kind === :observations ? "observation_columns" : "action_columns"
    if length(names) != 1
        throw(
            ArgumentError(
                "$keyword is $(repr(columns)), which builds one value from several columns without the name the model uses for it. That is only possible when the model has only one kind of them in its attributes, but the attributes of $(describe_model(cognitive_model)) have the $kind $names. Give the name after the columns, such as (:choicecol, :rtcol) => :ssm_action.",
            ),
        )
    end
    return only(names)
end

"""
    named_data(data, cognitive_model, kind)

Return the data of `kind`, `:observations` or `:actions`, as a `NamedTuple` of vectors: a
single vector as the data of the attribute of that kind when the model has only one in its
attributes, which is an error when it has another number of them, and anything else as it
is, for [`check_data_fits`](@ref) to check. A vector of `NamedTuple`s with the model's names
is the data written as one row for each timestep, which is returned as it is, for
`check_data_fits` to report. Used by [`SessionInfo`](@ref).
"""
named_data(data, cognitive_model, kind::Symbol) = data
function named_data(data::AbstractVector, cognitive_model, kind::Symbol)
    names = data_names(cognitive_model, kind)

    #NamedTuples with the model's names are the data written as one row for each timestep, which check_data_fits reports
    eltype(data) <: NamedTuple{names} && return data

    #Otherwise the vector is the data of the attribute of that kind, of which the model must have only one in its attributes
    if length(names) != 1
        throw(
            ArgumentError(
                "The $kind are a single vector, which can only be their data when the model has only one kind of them in its attributes, but a cognitive model of type $(typeof(cognitive_model)) has the $kind $names. Give them as a NamedTuple with one vector for each name.",
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
Used by [`MultipleSessionsInfo`](@ref) when it checks a table, and by
[`build_column`](@ref).
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
    build_column(table, specification, missing_values, impute)

Return the column of an attribute for every row of `table`, as [`columns_used`](@ref)
describes the specification, with the values that are missing replaced by their marker, as
[`mark_missing_values`](@ref) does: the column itself, or one built row by row from several,
where a missing value in any of them makes the whole value of the row missing. The column is
made for the whole table, so that every session gets the same element type. Used by
[`build_columns`](@ref).
"""
function build_column(table::NamedTuple, specification::Symbol, missing_values, impute)
    return mark_missing_values(getproperty(table, specification), missing_values, impute)
end
function build_column(
    table::NamedTuple,
    specification::NamedTuple{names,<:Tuple{Vararg{Symbol}}},
    missing_values,
    impute,
) where {names}
    return build_column(
        table,
        values(specification) => (values...) -> NamedTuple{names}(values),
        missing_values,
        impute,
    )
end
function build_column(
    table::NamedTuple, specification::Tuple{Vararg{Symbol}}, missing_values, impute
)
    return build_column(table, specification => tuple, missing_values, impute)
end
function build_column(
    table::NamedTuple, specification::Pair{<:Tuple{Vararg{Symbol}}}, missing_values, impute
)
    columns = map(name -> getproperty(table, name), first(specification))
    combine = last(specification)
    marker = missing_value_marker(impute)
    built = map(eachindex(first(columns))) do i
        values = map(column -> column[i], columns)

        #A missing value in any of the columns makes the whole value missing
        is_missing = any(value -> is_missing_data_value(value, missing_values), values)
        return is_missing ? marker : combine(values...)
    end
    return narrow_element_type(built)
end

"""
    build_columns(table, columns, missing_values, impute)

Return the columns in `columns`, a `NamedTuple` from the name the model uses to the
specification of the column in `table`, for every row of the table, as a `NamedTuple` with
the model's names, each built with [`build_column`](@ref). Used by
[`MultipleSessionsInfo`](@ref) when it splits a table.
"""
function build_columns(table::NamedTuple, columns::NamedTuple, missing_values, impute)
    return map(
        specification -> build_column(table, specification, missing_values, impute), columns
    )
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
    select_rows(columns, rows)

Return the rows `rows` of each column in `columns`, a `NamedTuple` of vectors, which are the
data of one session. Used by [`MultipleSessionsInfo`](@ref) when it splits a table.
"""
select_rows(columns::NamedTuple, rows) = map(column -> column[rows], columns)

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
