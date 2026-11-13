## Preparing the data of a session ##

"""
    is_value_to_condition_on(value)

Return whether a single action or observation in the data is a value to condition on:
neither [`MissingValue`](@ref) nor an [`AbstractNoValue`](@ref), such as
[`UnimputedMissingValue`](@ref) or [`NoValue`](@ref). Used by
[`collect_values_to_condition_on`](@ref) and [`find_outputs_to_condition_on`](@ref).
"""
is_value_to_condition_on(value) = !(value isa MissingValue || value isa AbstractNoValue)

"""
    collect_values_to_condition_on(columns)

Return the values to condition on of each action or observation in `columns`, a `NamedTuple`
with one vector per name, as a `NamedTuple` of vectors in the order of the timesteps.
Fitting scores these against the distributions the model gave on the same timesteps: the
actions against the cognitive model's, and the observations against the environment's. Each
vector has the element type of its column without [`MissingValue`](@ref) and the other
markers, so that a session without values to condition on for a name gets an empty vector of
the same type as other sessions, which scores nothing. A name whose column can only hold
those markers is left out, which is decided by its type, so that it is the same for every
session of a table. Used by [`SessionInfo`](@ref), which keeps the results in its fields
`actions_to_condition_on` and `observations_to_condition_on`.
"""
function collect_values_to_condition_on(columns::NamedTuple{names}) where {names}
    values_to_condition_on = map(
        column ->
            Vector{Base.typesplit(eltype(column), Union{MissingValue,AbstractNoValue})}(
                filter(is_value_to_condition_on, column)
            ),
        columns,
    )
    names_with_a_type = Tuple(
        name for
        name in names if eltype(getproperty(values_to_condition_on, name)) !== Union{}
    )
    return NamedTuple{names_with_a_type}(values_to_condition_on)
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
value that isn't [`MissingValue`](@ref), and 0 when none has. Any other marker counts as
data, since it says that the timestep happened. Used by
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
    narrow_element_type(column)

Return `column` with the types it holds as its element type, when its element type is `Any`
or another abstract type that isn't a `Union`, and `column` itself otherwise. A vector
written as `[true, MissingValue(), true]` has the element type `Any`, since only `missing`
is promoted to a `Union`, and an `Any` in the data makes every timestep run with its types
unknown. Used by [`mark_missing_values`](@ref) and [`build_column`](@ref).
"""
function narrow_element_type(column::AbstractVector)
    T = eltype(column)
    (isconcretetype(T) || T isa Union) && return column
    narrowed_type = Union{}
    for value in column
        narrowed_type = Union{narrowed_type,typeof(value)}
    end
    return Vector{narrowed_type}(column)
end

"""
    column_can_hold(column, value)

Return whether `column` can hold `value`, by its element type, or holds a value which is
equal to it without having its type, such as `-1.0` for `-1`, as `isequal` compares them.
Used by [`mark_missing_values`](@ref).
"""
function column_can_hold(column::AbstractVector, value)
    return typeof(value) <: eltype(column) || any(isequal(value), column)
end

"""
    as_missing_values(missing_values)

Return `missing_values`, the `missing_observation_values` or the `missing_action_values` of
a [`SessionInfo`](@ref), as a tuple: a tuple as it is, and a vector as a tuple. Anything
else is an error. Used by [`SessionInfo`](@ref) and [`MultipleSessionsInfo`](@ref).
"""
as_missing_values(missing_values::Tuple) = missing_values
as_missing_values(missing_values::AbstractVector) = Tuple(missing_values)
function as_missing_values(missing_values)
    return throw(
        ArgumentError(
            "The values in the data which stand for a missing value must be given in a tuple or a vector, such as (missing,) or (missing, -1), but they are $(repr(missing_values)).",
        ),
    )
end

"""
    missing_value_marker(impute)

Return the marker which the missing values in the data of a session become:
[`MissingValue`](@ref) when they are imputed, and [`UnimputedMissingValue`](@ref) when
`impute` is `false`. Used by [`mark_missing_values`](@ref) and [`build_column`](@ref).
"""
missing_value_marker(impute::Bool) = impute ? MissingValue() : UnimputedMissingValue()

"""
    is_missing_data_value(value, missing_values)

Return whether `value` is missing in the data: [`MissingValue`](@ref) itself, or equal to
one of `missing_values`, as `isequal` compares them. Used by [`build_column`](@ref), for a
value which is built from several columns.
"""
function is_missing_data_value(value, missing_values::Tuple)
    return value isa MissingValue ||
           any(missing_value -> isequal(missing_value, value), missing_values)
end

"""
    mark_missing_values(columns, missing_values, impute)

Return `columns`, a `NamedTuple` with one vector per name, or a single vector, with every
value which is equal to one of `missing_values`, as `isequal` compares them, replaced by the
marker for a missing value: [`MissingValue`](@ref) when `impute` is `true`. When it is
`false`, the marker is [`UnimputedMissingValue`](@ref), which every `MissingValue()` that
is in the data already becomes too.

A column gets the type of the marker in its element type when its element type allows one
of the values, whether it holds one or not, so that sessions whose columns have the same
types keep them, or when it holds a value equal to one of them of another type, such as
`-1.0` for `-1`. Any other column is kept as it is, after [`narrow_element_type`](@ref).
Anything that isn't a `NamedTuple` of vectors is returned as it is, for
[`check_data_fits`](@ref) to report. Used by [`SessionInfo`](@ref) on the observations and
the actions, and by [`build_column`](@ref).
"""
function mark_missing_values(columns::NamedTuple, missing_values::Tuple, impute::Bool)
    return map(column -> mark_missing_values(column, missing_values, impute), columns)
end

function mark_missing_values(column::AbstractVector, missing_values::Tuple, impute::Bool)
    column = narrow_element_type(column)
    marker = missing_value_marker(impute)

    #Without imputation, a value which is marked as missing in the data already isn't imputed either
    values = impute ? missing_values : (missing_values..., MissingValue())

    #Only the values that the column can hold are replaced in it
    replaced_values = filter(value -> column_can_hold(column, value), values)
    isempty(replaced_values) && return column

    #The element type loses a replaced value which is the only one of its type, such as Missing, and gets the type of the marker
    T = eltype(column)
    for value in replaced_values
        Base.issingletontype(typeof(value)) && (T = Base.typesplit(T, typeof(value)))
    end
    marked = Vector{Union{T,typeof(marker)}}(undef, length(column))
    for i in eachindex(column)
        is_replaced = any(value -> isequal(value, column[i]), replaced_values)
        marked[i] = is_replaced ? marker : column[i]
    end
    return marked
end

mark_missing_values(x, missing_values::Tuple, impute::Bool) = x

#A call of its own, so that a value which can have two types gives one of two NamedTuples with a concrete field each: (; a = x) written out is inferred as a NamedTuple whose field type is the Union, which nothing is compiled for, while a call with a Union-typed argument is split by type
"""
    build_named_entry(::Val{name}, value)

Return `(; name = value)`. A `NamedTuple` which holds a value that can have two types is
built with `merge` of these, as [`select_row`](@ref) does. Used by [`select_row`](@ref), by
[`saved_session_values`](@ref) and by the timesteps of fitted sessions.
"""
build_named_entry(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    select_row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of
vectors with one entry for each timestep, built with [`build_named_entry`](@ref) so that a
column which holds two types, such as an action that is [`MissingValue`](@ref) on some
timesteps, gives one of two `NamedTuple`s with concrete fields rather than one without a
definite type. Used by the timesteps of fitted sessions, and by the models of several
sessions, which read the parameters of one session with it.
"""
select_row(columns::NamedTuple{()}, t) = (;)
function select_row(columns::NamedTuple{names}, t) where {names}
    return merge(
        build_named_entry(Val(first(names)), first(columns)[t]),
        select_row(Base.tail(columns), t),
    )
end

"""
    has_missing_value(data)

Return whether any entry of `data`, the `NamedTuple` of the actions or the observations of
one timestep, is [`MissingValue`](@ref). Used by the timesteps of fitted sessions, to skip
imputing when nothing is missing.
"""
has_missing_value(data::NamedTuple) = any(value -> value isa MissingValue, Tuple(data))

"""
    check_data_fits(cognitive_model, observations, actions)

Throw an informative error if the data of a session isn't a `NamedTuple` of vectors for the
observations and one for the actions, doesn't have the observations and the actions in the
cognitive model's attributes, from [`attribute_names`](@ref), or has columns of different
lengths, and return the number of timesteps otherwise. Used by [`SessionInfo`](@ref).
"""
function check_data_fits(
    cognitive_model::AbstractTimestepModel, observations::NamedTuple, actions::NamedTuple
)
    attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    if !has_every_name(attributes.inputs, keys(observations))
        throw(
            ArgumentError(
                "The observations in the data are named $(keys(observations)), but the observations in the attributes of $(describe_model(cognitive_model)), its inputs, are $(attributes.inputs). The data must have every observation in the cognitive model's attributes.",
            ),
        )
    end
    if !same_names(keys(actions), attributes.outputs)
        throw(
            ArgumentError(
                "The actions in the data are named $(keys(actions)), but the actions in the attributes of $(describe_model(cognitive_model)), its outputs, are $(attributes.outputs).",
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
function check_data_fits(cognitive_model::AbstractTimestepModel, observations, actions)
    return throw(
        ArgumentError(
            "The observations and the actions must each be a NamedTuple with one vector for each name, such as (; reward = [1.0, 0.0]), or a single vector when the model has only one kind of them in its attributes, but they have types $(typeof(observations)) and $(typeof(actions)).",
        ),
    )
end

"""
    check_environment_fits(cognitive_model, environment, observations)

Throw an informative error if a session can't run with `environment`: without one, the
observations in the data can't have a [`MissingValue`](@ref), since nothing imputes it, and
with one, its attributes must have the same observations and actions as those of
`cognitive_model`. Used by [`SessionInfo`](@ref).
"""
function check_environment_fits(
    cognitive_model::AbstractTimestepModel,
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
                "The observations have entries that are MissingValue(), but no environment was given to impute them. Give an environment model, or have the model receive them as UnimputedMissingValue(), with impute_missing_observations = false.",
            ),
        )
    end
    return nothing
end

function check_environment_fits(
    cognitive_model::AbstractTimestepModel,
    environment::AbstractTimestepModel,
    observations::NamedTuple,
    observation_map,
    action_map,
)
    check_names_fit(cognitive_model, environment, observation_map, action_map)
    return nothing
end

"""
    map_to_environment_outputs(values, environment, observation_map, value_if_not_passed_on)

Return `values`, a `NamedTuple` in the names under which the cognitive model receives the
observations, as `environment` gives them: under each output name in the environment's
attributes, the value that `observation_map` passes on from it, or `value_if_not_passed_on`
for an output that the map doesn't pass on. Without a map the names are the same. Without an
environment `values` is returned as it is. Used by [`SessionInfo`](@ref) for the
environment's data, with a column of [`MissingValue`](@ref) for each output that isn't
passed on, so that it is imputed, and for the observation the environment starts from, with
[`BeforeFirstTimestep`](@ref).
"""
function map_to_environment_outputs(
    values, ::NoEnvironment, observation_map, value_if_not_passed_on
)
    return values
end
function map_to_environment_outputs(
    values, environment::AbstractTimestepModel, observation_map, value_if_not_passed_on
)
    outputs = check_type_attribute_names(attribute_names(environment), environment).outputs
    return NamedTuple{outputs}(
        map(
            name -> value_passed_on(values, name, observation_map, value_if_not_passed_on),
            outputs,
        ),
    )
end

"""
    value_passed_on(values, name, observation_map, value_if_not_passed_on)

Return the value in `values` that the environment's output `name` is received as: under the
same name without a map, under the name `observation_map` gives it, or
`value_if_not_passed_on` when the map doesn't pass it on. Used by
[`map_to_environment_outputs`](@ref).
"""
value_passed_on(values, name, ::Nothing, value_if_not_passed_on) = values[name]
function value_passed_on(
    values, name, ::NameMap{Tgiven,Treceived}, value_if_not_passed_on
) where {Tgiven,Treceived}
    index = findfirst(==(name), Tgiven)
    return index === nothing ? value_if_not_passed_on : values[Treceived[index]]
end
