### EACH PARAMETER SAMPLED FROM THE DISTRIBUTION OF A GROUP OF SESSIONS ###

#Each parameter of every session sampled from the distribution of its group, with the group of every session found when the full model was prepared, by the model that goes through the parameters
function CognitiveModeling.turing_parameters_model(
    parameters_model::HierarchicalGroupedParameters{<:NamedTuple,<:NamedTuple},
    ::MultipleSessionsInfo,
)
    return turing_hierarchical_grouped_parameters(
        parameters_model.parameters, parameters_model.groups
    )
end

#One that wasn't prepared is an error when its model is made, since it would find the group of every session on each evaluation
function CognitiveModeling.turing_parameters_model(
    ::HierarchicalGroupedParameters{<:NamedTuple,Nothing}, ::MultipleSessionsInfo
)
    return throw(
        ArgumentError(
            "A HierarchicalGroupedParameters must be prepared for the sessions with prepare_parameters_model(parameters_model, sessions_info) before turing_parameters_model makes its Turing.jl model, as turing_full_model does.",
        ),
    )
end

"""
    turing_hierarchical_grouped_parameters(parameters, groups)

Turing.jl model which samples one parameter at a time from `parameters`, the
[`HierarchicalParameter`](@ref) of each parameter of a
[`HierarchicalGroupedParameters`](@ref), with
[`turing_hierarchical_grouped_parameter`](@ref) under the parameter's name, and returns them
in a `NamedTuple`, each as a vector with a value for every session. `groups` has the groups
of the sessions for each parameter. Used by `HierarchicalGroupedParameters`.
"""
Turing.@model function turing_hierarchical_grouped_parameters(
    parameters::NamedTuple{names}, groups::NamedTuple
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_hierarchical_grouped_parameter(first(parameters), first(groups)),
            VarName{name}(),
        ),
        false,
    )
    other_values ~ to_submodel(
        turing_hierarchical_grouped_parameters(Base.tail(parameters), Base.tail(groups)),
        false,
    )
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_hierarchical_grouped_parameters(
    parameters::NamedTuple{()}, groups::NamedTuple
)
    return (;)
end

"""
    turing_hierarchical_grouped_parameter(hierarchical_parameter, groups)

Turing.jl model which samples one parameter for every session, as `hierarchical_parameter`,
a [`HierarchicalParameter`](@ref), says: the parameters of its group-level distribution for
each group, with [`turing_hierarchical_group_parameters`](@ref) from the priors in
`groups`, and then the value of every session. Centered, the value of every session is
sampled from the distribution of its group, which [`group_level_distribution`](@ref) makes,
as the variable `session`, named like `learning_rate.session` under the parameter's prefix.
Non-centered, a standard normal value is sampled for every session, as the variable
`session_z`, named like `learning_rate.session_z`, and the value of the session is the mean
of its group plus that value times the standard deviation of its group. It returns the
values with the transformation applied to each. Used by
[`turing_hierarchical_grouped_parameters`](@ref).
"""
Turing.@model function turing_hierarchical_grouped_parameter(hierarchical_parameter, groups)

    #Sample the parameters of the group-level distribution, each with a value for every group, or one value without groups
    group_parameters ~
    to_submodel(turing_hierarchical_group_parameters(groups.group_priors), false)

    #Sample the value of every session from the distribution of its group, or non-centered, as a standard normal value which the group's mean shifts and its standard deviation scales
    group_of_each_session = groups.group_of_each_session
    if hierarchical_parameter.centered
        session ~ arraydist(
            map(
                group -> group_level_distribution(
                    hierarchical_parameter.distribution, group_parameters, group
                ),
                group_of_each_session,
            ),
        )
        session_values = session
    else
        session_z ~ filldist(Normal(), length(group_of_each_session))
        means, standard_deviations = values(group_parameters)
        session_values =
            select_by_group(means, group_of_each_session) .+
            select_by_group(standard_deviations, group_of_each_session) .* session_z
    end

    #Transform every value to the scale of the parameter
    return map(hierarchical_parameter.transformation, session_values)
end

"""
    turing_hierarchical_group_parameters(group_priors)

Turing.jl model which samples one parameter of a group-level distribution at a time from
`group_priors`, the priors of the parameters under their names, each for every group with
[`turing_hierarchical_group_parameter`](@ref) under the name of its prior, and returns them
in a `NamedTuple`, each as a vector with a value for every group, or as one value without
groups. Used by [`turing_hierarchical_grouped_parameter`](@ref).
"""
Turing.@model function turing_hierarchical_group_parameters(
    group_priors::NamedTuple{names}
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(turing_hierarchical_group_parameter(first(group_priors)), VarName{name}()),
        false,
    )
    other_values ~
    to_submodel(turing_hierarchical_group_parameters(Base.tail(group_priors)), false)
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_hierarchical_group_parameters(group_priors::NamedTuple{()})
    return (;)
end

"""
    turing_hierarchical_group_parameter(prior)

Turing.jl model which samples the value of one parameter of a group-level distribution, as
the variable `group`, so that under the prefixes of the parameter and of the prior it is
named like `learning_rate.mean.group`. Without groups, `prior` is its prior, and the one
value is returned. With groups, `prior` is a `NamedTuple` with the prior of each group under
its symbol name, `group` has the value of each group under the same name, such as
`learning_rate.mean.group.clinical`, and the values are returned as a vector with a value
for every group. Used by [`turing_hierarchical_group_parameters`](@ref).
"""
Turing.@model function turing_hierarchical_group_parameter(prior)
    group ~ prior
    return group
end

Turing.@model function turing_hierarchical_group_parameter(priors_of_groups::NamedTuple)
    group ~ product_distribution(priors_of_groups)

    #The value of each group as a vector, to look up the group of every session
    return collect(values(group))
end

"""
    group_level_distribution(distribution, group_parameters, group)

Return the distribution of the group numbered `group`, made by `distribution` from the
values that the group has in `group_parameters`, a `NamedTuple` with a vector over the
groups, or one value without groups, for each parameter of the distribution, in the order
the distribution takes them. `nothing` makes a normal distribution. Used by
[`turing_hierarchical_grouped_parameter`](@ref).
"""
function group_level_distribution(distribution, group_parameters::NamedTuple, group)
    return distribution(
        map(values -> select_by_group(values, group), Tuple(group_parameters))...
    )
end
function group_level_distribution(::Nothing, group_parameters::NamedTuple, group)
    return group_level_distribution(Normal, group_parameters, group)
end

"""
    select_by_group(values, groups)

Return the values in `values`, a vector with a value for every group, of the groups
numbered in `groups`, which is one group number or a vector of them, as `values[groups]`
does. Without groups, `values` is one value that every group has, and is returned as it is.
Used by [`turing_hierarchical_grouped_parameter`](@ref) and
[`group_level_distribution`](@ref).
"""
select_by_group(values::AbstractVector, groups) = values[groups]
select_by_group(value::Real, _) = value
