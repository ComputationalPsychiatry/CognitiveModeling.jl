### EACH PARAMETER SAMPLED FROM THE DISTRIBUTION OF A GROUP OF SESSIONS ###

#Each parameter of every session sampled from the distribution of its group, with the group of every session found when the full model was prepared
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::GroupedHierarchicalSessionParameters{<:NamedTuple,<:NamedTuple},
    sessions_info::MultipleSessionsInfo,
)
    parameters ~ to_submodel(
        turing_hierarchical_parameters(
            parameters_model.parameters, parameters_model.groups
        ),
        false,
    )
    return parameters
end

#The same when it wasn't prepared, which finds the group of every session on each evaluation
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::GroupedHierarchicalSessionParameters{<:NamedTuple,Nothing},
    sessions_info::MultipleSessionsInfo,
)
    parameters ~ to_submodel(
        CognitiveModeling.turing_session_parameters_model(
            prepare_parameters_model(parameters_model, sessions_info), sessions_info
        ),
        false,
    )
    return parameters
end

"""
    turing_hierarchical_parameters(parameters, groups)

Turing.jl model which samples one parameter at a time from `parameters`, the
[`HierarchicalParameter`](@ref) of each parameter of a
[`GroupedHierarchicalSessionParameters`](@ref), with
[`turing_hierarchical_parameter`](@ref) under the parameter's name, and returns them
in a `NamedTuple`, each as a vector with a value for every session. `groups` has the groups
of the sessions for each parameter. Used by `GroupedHierarchicalSessionParameters`.
"""
Turing.@model function turing_hierarchical_parameters(
    parameters::NamedTuple{names}, groups::NamedTuple
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_hierarchical_parameter(first(parameters), first(groups)), VarName{name}()
        ),
        false,
    )
    other_values ~ to_submodel(
        turing_hierarchical_parameters(Base.tail(parameters), Base.tail(groups)), false
    )
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_hierarchical_parameters(
    parameters::NamedTuple{()}, groups::NamedTuple
)
    return (;)
end

"""
    turing_hierarchical_parameter(hierarchical_parameter, groups)

Turing.jl model which samples one parameter for every session, as `hierarchical_parameter`,
a [`HierarchicalParameter`](@ref), says: the parameters of its group-level distribution for
each group, with
[`turing_hierarchical_group_parameters`](@ref), then the value of every session from the
distribution of its group, with [`turing_hierarchical_centered_values`](@ref) or
[`turing_hierarchical_standardized_values`](@ref), and returns the values with the
transformation applied to each. Used by [`turing_hierarchical_parameters`](@ref).
"""
Turing.@model function turing_hierarchical_parameter(hierarchical_parameter, groups)

    #Sample the parameters of the group-level distribution, each as a vector with a value for every group
    group_parameters ~ to_submodel(
        turing_hierarchical_group_parameters(
            hierarchical_parameter.priors, groups.n_groups
        ),
        false,
    )

    #Sample the value of every session, from the distribution of its group or as a standardized value
    if hierarchical_parameter.centered
        session_values ~ to_submodel(
            turing_hierarchical_centered_values(
                hierarchical_parameter.distribution,
                group_parameters,
                groups.group_of_each_session,
            ),
            false,
        )
    else
        session_values ~ to_submodel(
            turing_hierarchical_standardized_values(
                group_parameters, groups.group_of_each_session
            ),
            false,
        )
    end

    #Transform every value to the scale of the parameter
    return map(hierarchical_parameter.transformation, session_values)
end

"""
    turing_hierarchical_group_parameters(priors, n_groups)

Turing.jl model which samples one parameter of a group-level distribution at a time from
`priors`, each for every group with [`turing_hierarchical_group_parameter`](@ref) under the
name of its prior, and returns them in a `NamedTuple`, each as a vector with a value for
every group. Used by [`turing_hierarchical_parameter`](@ref).
"""
Turing.@model function turing_hierarchical_group_parameters(
    priors::NamedTuple{names}, n_groups
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_hierarchical_group_parameter(filldist(first(priors), n_groups)),
            VarName{name}(),
        ),
        false,
    )
    other_values ~
    to_submodel(turing_hierarchical_group_parameters(Base.tail(priors), n_groups), false)
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_hierarchical_group_parameters(
    priors::NamedTuple{()}, n_groups
)
    return (;)
end

"""
    turing_hierarchical_group_parameter(distribution)

Turing.jl model which samples the values of one parameter of a group-level distribution for
every group from `distribution`, a distribution over a vector, as the variable `group`, so
that under the prefixes of the parameter and of the prior it is named like
`learning_rate.mean.group`. Used by [`turing_hierarchical_group_parameters`](@ref).
"""
Turing.@model function turing_hierarchical_group_parameter(distribution)
    group ~ distribution
    return group
end

"""
    turing_hierarchical_centered_values(distribution, group_parameters, group_of_each_session)

Turing.jl model which samples the value of one parameter for every session from the
distribution of the session's group, as the variable `session`, so that under the
parameter's prefix it is named like `learning_rate.session`. `distribution` makes the
distribution of a group from its parameters, and `nothing` is a normal distribution. Used by
[`turing_hierarchical_parameter`](@ref).
"""
Turing.@model function turing_hierarchical_centered_values(
    distribution, group_parameters::NamedTuple, group_of_each_session
)
    session ~ arraydist(
        map(
            group -> group_level_distribution(distribution, group_parameters, group),
            group_of_each_session,
        ),
    )
    return session
end

"""
    turing_hierarchical_standardized_values(group_parameters, group_of_each_session)

Turing.jl model which samples a standardized value of one parameter for every session, as
the variable `standardized`, so that under the parameter's prefix it is named like
`learning_rate.standardized`, and returns the value of every session: the mean of its
group plus the standardized value times the standard deviation of its group, which are the
two group-level parameters. This is the non-centered form of a normal distribution. Used by
[`turing_hierarchical_parameter`](@ref).
"""
Turing.@model function turing_hierarchical_standardized_values(
    group_parameters::NamedTuple, group_of_each_session
)
    standardized ~ filldist(Normal(), length(group_of_each_session))
    means, standard_deviations = values(group_parameters)
    return means[group_of_each_session] .+
           standard_deviations[group_of_each_session] .* standardized
end

"""
    group_level_distribution(distribution, group_parameters, group)

Return the distribution of the group numbered `group`, made by `distribution` from the
values that the group has in `group_parameters`, a `NamedTuple` with one vector over the
groups for each parameter of the distribution, in the order the distribution takes them.
`nothing` makes a normal distribution. Used by
[`turing_hierarchical_centered_values`](@ref).
"""
function group_level_distribution(distribution, group_parameters::NamedTuple, group)
    return distribution(map(values -> values[group], Tuple(group_parameters))...)
end
function group_level_distribution(::Nothing, group_parameters::NamedTuple, group)
    return group_level_distribution(Normal, group_parameters, group)
end
