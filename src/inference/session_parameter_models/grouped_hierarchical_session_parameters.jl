## Each parameter sampled from the distribution of a group of sessions ##

"""
    ConstructsType{T}()

A function which makes an object of the type `T` from its arguments, so that
`ConstructsType{Normal}()(0, 1)` is `Normal(0, 1)`. Used by [`HierarchicalParameter`](@ref)
for a group-level distribution that is given as a type.
"""
struct ConstructsType{T} end
(::ConstructsType{T})(arguments...) where {T} = T(arguments...)

"""
    HierarchicalParameter(priors; distribution=nothing, transformation=identity,
                          group_by=nothing, centered=nothing)

How one parameter of a [`GroupedHierarchicalSessionParameters`](@ref) is sampled for every
session: from a group-level distribution, whose own parameters are sampled from `priors`,
once for each group of sessions.

# Arguments
- `priors::NamedTuple`: the priors of the parameters of the group-level distribution, in the
  order in which the distribution takes them, such as
  `(; mean = Normal(0, 1), sd = Exponential(1))`. Their names name the variables of the
  group-level parameters.

# Keyword arguments
- `distribution=nothing`: the group-level distribution, as a function of its parameters,
  such as `Normal` or `(location, scale) -> Laplace(location, scale)`. `nothing` is a normal
  distribution, whose two priors are those of its mean and its standard deviation.
- `transformation=identity`: a function applied to the value of each session after it is
  sampled, such as `exp` for a parameter that must be positive or `logistic` for one between
  0 and 1, as an inverse link function does in a regression.
- `group_by=nothing`: a column of the table that the sessions were split from, or a tuple of
  columns, whose values group the sessions, so that each group has a group-level
  distribution of its own, such as `:diagnosis` for a clinical and a non-clinical group.
  Each session must have one value of each of the columns. Groups are numbered in the order
  of their first session. With `nothing`, every session is in one group.
- `centered=nothing`: for the default normal distribution, whether the value of each
  session is sampled from the distribution of its group, with `true`, or as a standardized
  value which the group's mean shifts and its standard deviation scales, with `false`,
  which is the default, since it usually samples better when the data of a session informs
  its parameters little. Any other distribution is always sampled from directly, and
  giving `centered` with one is an error.

Each argument is kept in a field of the same name, with `group_by` as a tuple of column
names, which is empty for one group, `centered` as `true` or `false`, and a `distribution`
that is given as a type inside a function which makes it.

# Examples
```jldoctest
julia> using Distributions: Normal, Exponential, Laplace

julia> learning_rate = HierarchicalParameter((; mean = Normal(0, 1), sd = Exponential(1)); transformation = exp, group_by = :diagnosis);

julia> learning_rate.group_by, learning_rate.centered
((:diagnosis,), false)

julia> inverse_temperature = HierarchicalParameter((; location = Normal(0, 1), scale = Exponential(1)); distribution = Laplace, transformation = exp);

julia> inverse_temperature.centered
true
```
"""
struct HierarchicalParameter{
    Tpriors<:NamedTuple,Tdistribution,Ttransformation,Tgroup_by<:Tuple{Vararg{Symbol}}
}
    priors::Tpriors
    distribution::Tdistribution
    transformation::Ttransformation
    group_by::Tgroup_by
    centered::Bool

    #Checks the settings, and keeps the columns to group by as a tuple of names, and whether the values are sampled from the distribution directly as the distribution decides it by default
    function HierarchicalParameter(
        priors;
        distribution=nothing,
        transformation=identity,
        group_by=nothing,
        centered::Union{Nothing,Bool}=nothing,
    )

        #The priors of the group-level parameters must be given
        if !(priors isa NamedTuple) || isempty(priors)
            throw(
                ArgumentError(
                    "A HierarchicalParameter takes the priors of the parameters of its group-level distribution in a NamedTuple, such as (; mean = Normal(0, 1), sd = Exponential(1)), but it was given $(repr(priors)).",
                ),
            )
        end

        #The default normal distribution takes the priors of its mean and its standard deviation
        if distribution === nothing && length(priors) != 2
            throw(
                ArgumentError(
                    "The group-level distribution of a HierarchicalParameter is a normal distribution by default, which takes two priors, of its mean and its standard deviation, but it was given the priors $(keys(priors)).",
                ),
            )
        end

        #Only the default normal distribution has the choice of being sampled as standardized values, which is its default. Any other distribution is sampled from directly
        if centered !== nothing && distribution !== nothing
            throw(
                ArgumentError(
                    "A HierarchicalParameter was given centered = $centered, but centered only applies to the default normal distribution, and its distribution is $distribution, which is always sampled from directly. Leave centered out.",
                ),
            )
        end
        sampled_directly = something(centered, distribution !== nothing)

        #The columns to group by are kept as a tuple of names, which is empty for one group
        columns_to_group_by =
            group_by === nothing ? () : as_tuple(group_by)::Tuple{Vararg{Symbol}}

        #A distribution given as a type, such as Normal, is kept inside a function which has the type in its own type. A type on its own is only known to be a type when a model is compiled, so that the method to call is looked up on every call, which is slower
        kept_distribution = if distribution isa Type
            ConstructsType{distribution}()
        else
            distribution
        end

        return new{
            typeof(priors),
            typeof(kept_distribution),
            typeof(transformation),
            typeof(columns_to_group_by),
        }(
            priors, kept_distribution, transformation, columns_to_group_by, sampled_directly
        )
    end
end

"""
    GroupedHierarchicalSessionParameters(parameters)

The session parameter model of a full model which samples each parameter of every session
from a group-level distribution, whose own parameters are sampled from priors, once for each
group of sessions. `parameters` is a `NamedTuple` with a [`HierarchicalParameter`](@ref) for
each parameter that is fitted, which has its priors, its group-level distribution, its
transformation and the columns that group the sessions, so the parameters can be grouped by
different columns. It is kept in the field `parameters`.

A full model finds the group of every session once, before it is made, from the table that
[`MultipleSessionsInfo`](@ref) keeps, so sessions that were prepared one at a time can only
be in one group. The field `groups` is `nothing` until then, and afterwards has, for each
parameter, the group of every session, `group_of_each_session`, the number of groups,
`n_groups`, and the symbol name of each group, `group_symbol_names`.

# Examples
```julia
using Turing
using LogExpFunctions: logistic

session_parameter_model = GroupedHierarchicalSessionParameters((;
    learning_rate = HierarchicalParameter(
        (; mean = Normal(0, 1), sd = Exponential(1));
        transformation = logistic,
        group_by = :diagnosis,
    ),
    inverse_temperature = HierarchicalParameter(
        (; mean = Normal(0, 1), sd = Exponential(1)); transformation = exp
    ),
))
```
"""
struct GroupedHierarchicalSessionParameters{
    Tparameters<:NamedTuple,Tgroups<:Union{Nothing,NamedTuple}
} <: AbstractSessionParametersModel
    parameters::Tparameters
    groups::Tgroups

    #Takes a HierarchicalParameter for each parameter. The groups of the sessions aren't found yet
    function GroupedHierarchicalSessionParameters(parameters::NamedTuple)
        foreach(check_is_hierarchical_parameter, keys(parameters), values(parameters))
        return new{typeof(parameters),Nothing}(parameters, nothing)
    end

    #The same model with the groups of the sessions it is prepared for
    function GroupedHierarchicalSessionParameters(
        parameters_model::GroupedHierarchicalSessionParameters, groups::NamedTuple
    )
        parameters = parameters_model.parameters
        return new{typeof(parameters),typeof(groups)}(parameters, groups)
    end
end

#It gives the parameters it has a HierarchicalParameter for
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
            (name, hierarchical_parameter) ->
                group_sessions(sessions_info, hierarchical_parameter.group_by, name),
            names,
            values(parameters_model.parameters),
        ),
    )
    return GroupedHierarchicalSessionParameters(parameters_model, groups)
end

"""
    check_is_hierarchical_parameter(name, hierarchical_parameter)

Throw an informative error if what a [`GroupedHierarchicalSessionParameters`](@ref) was
given for the parameter `name` isn't a [`HierarchicalParameter`](@ref). Used by the
constructor of `GroupedHierarchicalSessionParameters`.
"""
check_is_hierarchical_parameter(name::Symbol, ::HierarchicalParameter) = nothing
function check_is_hierarchical_parameter(name::Symbol, hierarchical_parameter)
    return throw(
        ArgumentError(
            "GroupedHierarchicalSessionParameters takes a HierarchicalParameter for each parameter, such as HierarchicalParameter((; mean = Normal(0, 1), sd = Exponential(1)); transformation = exp), but for $name it was given $(repr(hierarchical_parameter)).",
        ),
    )
end

"""
    group_sessions(sessions_info, group_by, parameter_name)

Return the groups of the sessions in `sessions_info` by the values of the columns in
`group_by`, a tuple of names, for the parameter `parameter_name` of a
[`GroupedHierarchicalSessionParameters`](@ref): a `NamedTuple` with the group of every
session, `group_of_each_session`, the number of groups, `n_groups`, and the symbol name of
each group, `group_symbol_names`, made from its values as [`session_id_to_symbol`](@ref)
makes the symbol name of a session from its identifier. Groups are numbered in the order of
their first session, and with no columns every session is in one group. A session with more
than one value, or a missing value, in one of the columns is an error.
Used by [`prepare_parameters_model`](@ref).
"""
function group_sessions(
    sessions_info::MultipleSessionsInfo, ::Tuple{}, parameter_name::Symbol
)
    return (;
        group_of_each_session=fill(1, n_sessions(sessions_info)),
        n_groups=1,
        group_symbol_names=[:all_sessions],
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
        group_symbol_names=map(session_id_to_symbol, groups),
    )
end
