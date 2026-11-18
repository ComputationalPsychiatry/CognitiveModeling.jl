## Preparing the data of a session ##

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
    column_can_hold(column, missing_value)

Return whether `column` can hold `missing_value`, by its element type, or holds a value which
is equal to it without having its type, such as `-1.0` for `-1`, as `isequal` compares them.
For a function, return whether `column` holds a value it returns `true` for. Used by
[`mark_missing_values`](@ref).
"""
function column_can_hold(column::AbstractVector, missing_value)
    return typeof(missing_value) <: eltype(column) || any(isequal(missing_value), column)
end
function column_can_hold(column::AbstractVector, missing_value::Function)
    return any(value -> matches_missing_value(missing_value, value), column)
end

"""
    matches_missing_value(missing_value, value)

Return whether `value` in the data stands for a missing one by `missing_value`, one of the
values given for missing ones: whether `missing_value` returns `true` for `value` when it
is a function, and whether the two are equal, as `isequal` compares them, otherwise. Used by
[`is_missing_data_value`](@ref) and [`mark_missing_values`](@ref).
"""
matches_missing_value(missing_value, value) = isequal(missing_value, value)
matches_missing_value(missing_value::Function, value) = missing_value(value) === true

"""
    missing_value_marker(impute)

Return the marker which the missing values in the data of a session become:
[`MissingValue`](@ref) when they are imputed, and [`UnimputedMissingValue`](@ref) when
`impute` is `false`. Used by [`mark_missing_values`](@ref) and [`build_column`](@ref).
"""
missing_value_marker(impute::Bool) = impute ? MissingValue() : UnimputedMissingValue()

"""
    is_missing_data_value(value, missing_values)

Return whether `value` is missing in the data: [`MissingValue`](@ref) itself, or matched by
one of `missing_values`, as [`matches_missing_value`](@ref) says. Used by
[`build_column`](@ref), for a value which is built from several columns.
"""
function is_missing_data_value(value, missing_values::Tuple)
    return value isa MissingValue ||
           any(missing_value -> matches_missing_value(missing_value, value), missing_values)
end

"""
    narrow_parts(value)

Return `value` with each array in it narrowed to the types it holds, such as a
`Vector{Float64}` for a `Vector{Union{Missing,Float64}}` without a missing element, and
`value` itself when there is nothing to narrow. Used by
[`narrow_arrays_in_values`](@ref).
"""
narrow_parts(value) = value
narrow_parts(value::Union{Tuple,NamedTuple}) = map(narrow_parts, value)
function narrow_parts(value::AbstractArray)
    #An array whose element type is concrete and has no parts is narrow already
    T = eltype(value)
    isconcretetype(T) && !(T <: Union{AbstractArray,Tuple,NamedTuple}) && return value

    #Otherwise, a new array gets its element type from the values it holds, unless it is narrow already
    narrowed = map(narrow_parts, value)
    return typeof(narrowed) == typeof(value) ? value : narrowed
end

"""
    narrow_arrays_in_values(column)

Return `column` with the arrays in its values narrowed by [`narrow_parts`](@ref), and its
element type made to match them, keeping its other types, such as that of a marker.
`column` itself is returned when no value changes, so that a column marked for a whole
table keeps its element type in each session. Used by [`mark_missing_values`](@ref) and
[`build_column`](@ref).
"""
function narrow_arrays_in_values(column::AbstractVector)
    #Asserted to be a Bool, since for an AbstractVector of unknown type any could return something else, which JET reports
    any(value -> value isa Union{AbstractArray,Tuple,NamedTuple}, column)::Bool ||
        return column
    narrowed = map(narrow_parts, column)
    all(((new, old),) -> new === old, zip(narrowed, column)) && return column

    #The element type loses the types with parts, and gets those of the narrowed values
    T = Base.typesplit(eltype(column), Union{AbstractArray,Tuple,NamedTuple})
    for value in narrowed
        value isa Union{AbstractArray,Tuple,NamedTuple} && (T = Union{T,typeof(value)})
    end
    return convert(Vector{T}, narrowed)
end

"""
    mark_missing_values(columns, missing_values, impute)

Return `columns`, a `NamedTuple` with one vector per name, or a single vector, with every
value which is matched by one of `missing_values`, as [`matches_missing_value`](@ref) says,
replaced by the marker for a missing value: [`MissingValue`](@ref) when `impute` is `true`.
When it is `false`, the marker is [`UnimputedMissingValue`](@ref), which every
`MissingValue()` that is in the data already becomes too. The arrays in the values that are
kept are narrowed to the types they hold, by [`narrow_arrays_in_values`](@ref).

A column gets the type of the marker in its element type when its element type allows one
of the values, whether it holds one or not, so that sessions whose columns have the same
types keep them, or when it holds a value equal to one of them of another type, such as
`-1.0` for `-1`, or a value that one of them, a function, returns `true` for. Any other
column is kept as it is, after [`narrow_element_type`](@ref). Anything that isn't a
`NamedTuple` of vectors is returned as it is, for [`check_data_fits`](@ref) to report. Used
by [`SessionInfo`](@ref) on the observations and the actions, and by
[`build_column`](@ref).
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
    isempty(replaced_values) && return narrow_arrays_in_values(column)

    #The element type loses a replaced value which is the only one of its type, such as Missing, and gets the type of the marker
    T = eltype(column)
    for value in replaced_values
        value isa Function && continue
        Base.issingletontype(typeof(value)) && (T = Base.typesplit(T, typeof(value)))
    end
    marked = Vector{Union{T,typeof(marker)}}(undef, length(column))
    for i in eachindex(column)
        is_replaced = any(value -> matches_missing_value(value, column[i]), replaced_values)
        marked[i] = is_replaced ? marker : column[i]
    end

    #The arrays of the values that are kept, such as vectors which a function didn't find missing, hold only their own types
    return narrow_arrays_in_values(marked)
end

mark_missing_values(x, missing_values::Tuple, impute::Bool) = x

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

## Checking the data against the domains of the cognitive model ##
"""
    check_data_in_domains(cognitive_model, observations, actions, in_table)

Throw an `ArgumentError` if a value in `observations` or in `actions`, each a `NamedTuple`
of vectors under the names of `cognitive_model`, is outside its domain: an observation
outside the domain of its input in [`input_domains`](@ref), and an action outside the
domain of its output in [`output_domains`](@ref), as [`check_value_in_domain`](@ref) checks
them. Markers, such as [`MissingValue`](@ref) and [`NoValue`](@ref), aren't checked. The
error names the row of the table when `in_table` is `true`, and the timestep of the session
otherwise. Used by [`SessionInfo`](@ref), and by [`MultipleSessionsInfo`](@ref) for a whole
table.
"""
function check_data_in_domains(
    cognitive_model::AbstractTimestepModel,
    observations::NamedTuple,
    actions::NamedTuple,
    in_table::Bool,
)
    #The domains must be Domains under names of the model's inputs and outputs
    domains_of_inputs = input_domains(cognitive_model)
    check_data_domains_fit_model(cognitive_model, domains_of_inputs, :inputs)
    domains_of_outputs = output_domains(cognitive_model)
    check_data_domains_fit_model(cognitive_model, domains_of_outputs, :outputs)

    #The observations are checked against the domains of the inputs, and the actions against those of the outputs
    check_columns_in_domains(:observation, observations, domains_of_inputs, in_table)
    check_columns_in_domains(:action, actions, domains_of_outputs, in_table)
    return nothing
end

"""
    check_columns_in_domains(kind, columns, domains, in_table)

Check each value of each column in `columns` that has a domain in `domains` with
[`check_value_in_domain`](@ref), as a [`DataValue`](@ref) of `kind`, `:observation` or
`:action`, at its index in the column. A column without a domain, a domain without a
column, and a marker such as [`MissingValue`](@ref) or [`NoValue`](@ref) aren't checked.
Used by [`check_data_in_domains`](@ref).
"""
function check_columns_in_domains(
    kind::Symbol, columns::NamedTuple, domains::NamedTuple, in_table::Bool
)
    for name in keys(domains)

        #A domain of an input or an output that isn't in the data, such as one which only an environment gives, has nothing to check
        haskey(columns, name) || continue
        domain = getproperty(domains, name)
        for (index, value) in enumerate(getproperty(columns, name))

            #A marker of a missing value or of nothing happening isn't a value of the domain
            (value isa AbstractNoValue || value isa MissingValue) && continue
            check_value_in_domain(DataValue(kind, index, in_table), name, value, domain)
        end
    end
    return nothing
end

"""
    check_data_domains_fit_model(model, domains, attribute)

Throw an `ArgumentError` if `domains`, what [`input_domains`](@ref) or
[`output_domains`](@ref) returns for `model`, as `attribute`, `:inputs` or `:outputs`, says,
isn't a `NamedTuple` of [`Domain`](@ref)s, or names an input or an output that isn't in the
model's attributes, from [`attribute_names`](@ref). Used by
[`check_data_in_domains`](@ref).
"""
function check_data_domains_fit_model(
    ::AbstractTimestepModel, ::NamedTuple{()}, attribute::Symbol
)
    return nothing
end
function check_data_domains_fit_model(
    model::AbstractTimestepModel,
    ::NamedTuple{names,<:Tuple{Vararg{Domain}}},
    attribute::Symbol,
) where {names}
    attributes = check_type_attribute_names(attribute_names(model), model)
    names_in_attributes = getproperty(attributes, attribute)
    if !has_every_name(names, names_in_attributes)
        throw(
            ArgumentError(
                "$(domains_function_name(attribute)) of $(describe_model(model)) has domains for $names, but the $attribute in its attributes are $names_in_attributes.",
            ),
        )
    end
    return nothing
end
function check_data_domains_fit_model(
    model::AbstractTimestepModel, domains, attribute::Symbol
)
    return throw(
        ArgumentError(
            "$(domains_function_name(attribute)) of $(describe_model(model)) returns $(repr(domains)), but it must return a NamedTuple with a Domain under the name of each of its $attribute that has a domain, such as (; choice = Domain(; possible_values = (1, 2))).",
        ),
    )
end

"""
    domains_function_name(attribute)

Return the name of the function that gives the domains of `attribute`, `:inputs` or
`:outputs`, as errors write it. Used by [`check_data_domains_fit_model`](@ref).
"""
function domains_function_name(attribute::Symbol)
    return attribute === :inputs ? "input_domains" : "output_domains"
end
