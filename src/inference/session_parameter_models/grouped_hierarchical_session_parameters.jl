## Each parameter sampled from the distribution of a group of sessions ##

"""
    GroupedHierarchicalSessionParameters(parameters)

The session parameter model of a full model which samples each parameter of every session
from a group-level distribution, whose own parameters are sampled from priors, once for each
group of sessions. `parameters` is a `NamedTuple` with an entry for each parameter that is
fitted, a `NamedTuple` of its settings:
- `priors`: the priors of the parameters of the group-level distribution, in the order in
  which the distribution takes them, such as `(; mean = Normal(0, 1), sd = Exponential(1))`.
  Their names name the variables of the group-level parameters.
- `distribution=nothing`: the group-level distribution, as a function of its parameters,
  such as `Normal` or `(location, scale) -> Laplace(location, scale)`. `nothing` is a normal
  distribution, whose two priors are those of its mean and its standard deviation.
- `transformation=identity`: a function applied to the value of each session after it is
  sampled, such as `exp` for a parameter that must be positive or `logistic` for one between
  0 and 1, as an inverse link function does in a regression.
- `group_by=nothing`: a column of the table in sessions_info that the sessions were split
  from, or a tuple of columns, whose values group the sessions, so that each group has a
  group-level distribution of its own, such as `:diagnosis` for a clinical and a
  non-clinical group. Each session must have one value of each of the columns. Groups are
  numbered in the order of their first session. With `nothing`, every session is in one
  group.
- `centered=nothing`: for the default normal distribution, whether the value of each
  session is sampled from the distribution of its group, with `true`, or as a standardized
  value which the group's mean shifts and its standard deviation scales, with `false`,
  which is the default, since it usually samples better when the data of a session informs
  its parameters little. Any other distribution is always sampled from directly, and
  giving `centered` with one is an error.

The settings that aren't given take their defaults, and the parameters can be grouped by
different columns. The completed settings are kept in the field `parameters`. A full model
finds the group of every session once, before it is made, from the table that
[`MultipleSessionsInfo`](@ref) keeps, so sessions that were prepared one at a time can only
be in one group. The field `groups` is `nothing` until then, and afterwards has, for each
parameter, the group of every session, `group_of_each_session`, the number of groups,
`n_groups`, and the name of each group, `group_names`.

# Examples
```julia
using Turing
using LogExpFunctions: logistic

session_parameter_model = GroupedHierarchicalSessionParameters((;
    learning_rate = (;
        priors = (; mean = Normal(0, 1), sd = Exponential(1)),
        transformation = logistic,
        group_by = :diagnosis,
    ),
    inverse_temperature = (;
        priors = (; mean = Normal(0, 1), sd = Exponential(1)),
        transformation = exp,
    ),
))
```
"""
struct GroupedHierarchicalSessionParameters{
    Tparameters<:NamedTuple,Tgroups<:Union{Nothing,NamedTuple}
} <: AbstractSessionParametersModel
    parameters::Tparameters
    groups::Tgroups

    #Checks the settings of each parameter, and fills in those it leaves out. The groups of the sessions aren't found yet
    function GroupedHierarchicalSessionParameters(parameters::NamedTuple)
        completed = NamedTuple{keys(parameters)}(
            map(complete_hierarchical_settings, keys(parameters), values(parameters))
        )
        return new{typeof(completed),Nothing}(completed, nothing)
    end

    #The same model with the groups of the sessions it is prepared for
    function GroupedHierarchicalSessionParameters(
        parameters_model::GroupedHierarchicalSessionParameters, groups::NamedTuple
    )
        parameters = parameters_model.parameters
        return new{typeof(parameters),typeof(groups)}(parameters, groups)
    end
end

#It gives the parameters it has settings for
function parameter_names(parameters_model::GroupedHierarchicalSessionParameters)
    return keys(parameters_model.parameters)
end

#The group of every session is found once for each parameter, when a full model is prepared, and kept in the field for the groups
function prepare_parameters_model(
    parameters_model::GroupedHierarchicalSessionParameters,
    sessions_info::MultipleSessionsInfo,
)
    names = keys(parameters_model.parameters)
    groups = NamedTuple{names}(
        map(
            (name, settings) -> group_sessions(sessions_info, settings.group_by, name),
            names,
            values(parameters_model.parameters),
        ),
    )
    return GroupedHierarchicalSessionParameters(parameters_model, groups)
end

"""
    complete_hierarchical_settings(name, settings)

Return the settings of the parameter `name` of a
[`GroupedHierarchicalSessionParameters`](@ref), a `NamedTuple`, with every setting it leaves
out at its default, and `group_by` as a tuple of column names. It is an error when the
settings have no `priors`, when they have a setting there isn't, when the default normal
distribution doesn't have two priors, and when `centered` is given with another
distribution. Used by
the constructor of `GroupedHierarchicalSessionParameters`.
"""
function complete_hierarchical_settings(name::Symbol, settings::NamedTuple)

    #Every setting must be one there is
    known_settings = (:priors, :distribution, :transformation, :group_by, :centered)
    unknown_settings = Tuple(
        setting for setting in keys(settings) if !(setting in known_settings)
    )
    if !isempty(unknown_settings)
        throw(
            ArgumentError(
                "The settings of $name in GroupedHierarchicalSessionParameters include $unknown_settings, which aren't settings of it. Its settings are $known_settings.",
            ),
        )
    end

    #The priors of the group-level parameters must be given
    if !(
        haskey(settings, :priors) &&
        settings.priors isa NamedTuple &&
        !isempty(settings.priors)
    )
        throw(
            ArgumentError(
                "The settings of $name in GroupedHierarchicalSessionParameters must have priors, a NamedTuple with the prior of each parameter of the group-level distribution, such as (; mean = Normal(0, 1), sd = Exponential(1)).",
            ),
        )
    end

    #The default normal distribution takes the priors of its mean and its standard deviation
    distribution = get(settings, :distribution, nothing)
    if distribution === nothing && length(settings.priors) != 2
        throw(
            ArgumentError(
                "The group-level distribution of $name in GroupedHierarchicalSessionParameters is the default normal distribution, which takes two priors, of its mean and its standard deviation, but $name has the priors $(keys(settings.priors)).",
            ),
        )
    end

    #Only the default normal distribution has the choice of being sampled as standardized values, which is its default. Any other distribution is sampled from directly
    centered = get(settings, :centered, nothing)
    if centered !== nothing && distribution !== nothing
        throw(
            ArgumentError(
                "$name in GroupedHierarchicalSessionParameters has centered = $centered, but centered only applies to the default normal distribution, and its distribution is $distribution, which is always sampled from directly. Leave centered out.",
            ),
        )
    end
    centered = something(centered, distribution !== nothing)

    #The columns to group by are kept as a tuple of names, which is empty for one group
    group_by = get(settings, :group_by, nothing)
    group_by = group_by === nothing ? () : as_tuple_of_names(group_by)

    return (;
        priors=settings.priors,
        distribution,
        transformation=get(settings, :transformation, identity),
        group_by,
        centered,
    )
end

function complete_hierarchical_settings(name::Symbol, settings)
    return throw(
        ArgumentError(
            "The settings of $name in GroupedHierarchicalSessionParameters must be a NamedTuple, such as (; priors = (; mean = Normal(0, 1), sd = Exponential(1))), but they are $(repr(settings)).",
        ),
    )
end

"""
    group_sessions(sessions_info, group_by, parameter_name)

Return the groups of the sessions in `sessions_info` by the values of the columns in
`group_by`, a tuple of names, for the parameter `parameter_name` of a
[`GroupedHierarchicalSessionParameters`](@ref): a `NamedTuple` with the group of every
session, `group_of_each_session`, the number of groups, `n_groups`, and the name of each
group, `group_names`, as [`session_name`](@ref) names identifiers. Groups are numbered in
the order of their first session, and with no columns every session is in one group. A
session with more than one value, or a missing value, in one of the columns is an error.
Used by [`prepare_parameters_model`](@ref).
"""
function group_sessions(
    sessions_info::MultipleSessionsInfo, ::Tuple{}, parameter_name::Symbol
)
    return (;
        group_of_each_session=fill(1, n_sessions(sessions_info)),
        n_groups=1,
        group_names=[:all_sessions],
    )
end

function group_sessions(
    sessions_info::MultipleSessionsInfo,
    group_by::Tuple{Vararg{Symbol}},
    parameter_name::Symbol,
)

    #The value of each column in each session, which must be one value
    values_of_columns = map(group_by) do column
        values_of_sessions = session_column_values(sessions_info, column)
        return map(eachindex(values_of_sessions)) do i
            values_of_session = unique(values_of_sessions[i])
            if length(values_of_session) != 1 || ismissing(only(values_of_session))
                throw(
                    ArgumentError(
                        "GroupedHierarchicalSessionParameters groups $parameter_name by the column $column, which has the values $(values_of_session) in the session $(repr(sessions_info.session_ids[i])). Each session must have one value of every column its parameters are grouped by.",
                    ),
                )
            end
            return only(values_of_session)
        end
    end

    #The group of a session is its value of a single column, or the NamedTuple of its values of several
    group_of_sessions = map(eachindex(sessions_info.sessions)) do i
        values = map(values_of_column -> values_of_column[i], values_of_columns)
        return length(group_by) == 1 ? only(values) : NamedTuple{group_by}(values)
    end

    #Groups are numbered in the order of their first session
    groups = unique(group_of_sessions)
    return (;
        group_of_each_session=[
            findfirst(isequal(group), groups) for group in group_of_sessions
        ],
        n_groups=length(groups),
        group_names=map(session_name, groups),
    )
end
