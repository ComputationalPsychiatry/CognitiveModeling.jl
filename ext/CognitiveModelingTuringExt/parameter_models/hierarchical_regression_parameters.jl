### EACH PARAMETER GIVEN BY A REGRESSION ON COLUMNS OF THE SESSIONS' TABLE ###

#Each parameter of every session given by its regression, with the design of every regression built when the full model was prepared, by the model that goes through the parameters
function CognitiveModeling.turing_parameters_model(
    parameters_model::HierarchicalRegressionParameters{<:NamedTuple,<:NamedTuple},
    sessions_info::MultipleSessionsInfo,
)
    return turing_hierarchical_regression_parameters(
        parameters_model.parameters, parameters_model.designs, n_sessions(sessions_info)
    )
end

#One that wasn't prepared is an error when its model is made, since it would build the design of every regression on each evaluation
function CognitiveModeling.turing_parameters_model(
    ::HierarchicalRegressionParameters{<:NamedTuple,Nothing}, ::MultipleSessionsInfo
)
    return throw(
        ArgumentError(
            "A HierarchicalRegressionParameters must be prepared for the sessions with prepare_parameters_model(parameters_model, sessions_info) before turing_parameters_model makes its Turing.jl model, as full_model does.",
        ),
    )
end

"""
    turing_hierarchical_regression_parameters(parameters, designs, n_sessions)

Turing.jl model which gives one parameter at a time from `parameters`, the
[`RegressionParameter`](@ref) of each parameter of a
[`HierarchicalRegressionParameters`](@ref), with
[`turing_hierarchical_regression_parameter`](@ref) under the parameter's name, and returns
them in a `NamedTuple`, each as a vector with a value for each of `n_sessions` sessions.
`designs` has the design of each regression. Used by `HierarchicalRegressionParameters`.
"""
Turing.@model function turing_hierarchical_regression_parameters(
    parameters::NamedTuple{names}, designs::NamedTuple, n_sessions
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_hierarchical_regression_parameter(
                first(parameters), first(designs), n_sessions
            ),
            VarName{name}(),
        ),
        false,
    )
    other_values ~ to_submodel(
        turing_hierarchical_regression_parameters(
            Base.tail(parameters), Base.tail(designs), n_sessions
        ),
        false,
    )
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_hierarchical_regression_parameters(
    parameters::NamedTuple{()}, designs::NamedTuple, n_sessions
)
    return (;)
end

"""
    turing_hierarchical_regression_parameter(regression_parameter, design, n_sessions)

Turing.jl model which gives one parameter to every session, as `regression_parameter`, a
[`RegressionParameter`](@ref), says. It samples the non-varying coefficients of the
intercept, the predictors and the interactions as one variable `coef`, a `NamedTuple` with
each under its name, so that under the parameter's prefix one is named like
`learning_rate.coef.age`, and adds up their effects with [`add_up_effects`](@ref). It
samples the deviations of the varying coefficients and adds up their effects with
[`turing_regression_varying_effects`](@ref). It returns the sum of the effects on every
session, with the transformation applied to each. With a family, the other parameters of the
family's distribution are sampled once as one variable `family`, named like
`learning_rate.family.sd`, and the value of every session as the variable `sessions`, named
like `learning_rate.sessions`, from the distribution that
[`CognitiveModeling.build_family_distribution`](@ref) makes of the session's transformed sum
and the other parameters, and those values are returned instead. `design` has the design
columns and the levels of the regression, for each of `n_sessions` sessions. Used by
[`turing_hierarchical_regression_parameters`](@ref).
"""
Turing.@model function turing_hierarchical_regression_parameter(
    regression_parameter, design, n_sessions
)

    #Sample the non-varying coefficients, as one variable named coef, and add up their effects on every session, which are zero without coefficients
    coefficient_priors = design.coefficient_priors
    if isempty(coefficient_priors)
        non_varying_effects = zeros(n_sessions)
    else
        coef ~ product_distribution(coefficient_priors)
        non_varying_effects = add_up_effects(coef, design.design_columns, n_sessions)
    end

    #Sample the deviations of the varying coefficients of each grouping column, and add up their effects on every session
    varying_effects ~
    to_submodel(turing_regression_varying_effects(design.varying_effects), false)

    #Transform every sum to the scale of the parameter
    transformed_sums = map(
        regression_parameter.transformation, non_varying_effects .+ varying_effects
    )

    #Without a family, the transformed sums are the values of the sessions
    regression_family = regression_parameter.family
    if regression_family === nothing
        return transformed_sums
    end

    #With one, sample the other parameters of its distribution once, as one variable named family, and the value of every session from the distribution of its transformed sum
    family_priors = regression_family.priors
    if isempty(family_priors)
        sessions ~ arraydist(
            map(
                regression_outcome ->
                    build_family_distribution(regression_family, regression_outcome, ()),
                transformed_sums,
            ),
        )
    else
        family ~ product_distribution(family_priors)
        sessions ~ arraydist(
            map(
                regression_outcome -> build_family_distribution(
                    regression_family, regression_outcome, family
                ),
                transformed_sums,
            ),
        )
    end
    return sessions
end

"""
    add_up_effects(coefficients, design_columns, n_sessions)

Return the sum of the effects of the non-varying coefficients in `coefficients`, a
`NamedTuple`: each coefficient times the design column of the same name in
`design_columns`, as a vector with a value for each of `n_sessions` sessions. Used by
[`turing_hierarchical_regression_parameter`](@ref).
"""
function add_up_effects(
    coefficients::NamedTuple, design_columns::NamedTuple{names}, n_sessions
) where {names}

    #The coefficients in the order of the design columns
    ordered_coefficients = values(NamedTuple{names}(coefficients))
    effects = map(
        (coefficient, column) -> coefficient .* column,
        ordered_coefficients,
        values(design_columns),
    )
    return foldl(.+, effects; init=zeros(n_sessions))
end

"""
    turing_regression_varying_effects(designs)

Turing.jl model which goes through the varying effects of a regression one grouping column
at a time, from `designs`, the design of the varying effects of each grouping column under
their name. It samples the deviations of their varying coefficients with
[`turing_regression_varying_coefficients`](@ref) under the name of the varying effects, and
adds up their effects with [`add_up_varying_effects`](@ref). It returns the sum of the
effects of all varying effects on every session, as a vector with a value for every
session, or zero without varying effects. Used by
[`turing_hierarchical_regression_parameter`](@ref).
"""
Turing.@model function turing_regression_varying_effects(
    designs::NamedTuple{names}
) where {names}
    name = first(names)
    design = first(designs)
    deviations ~ to_submodel(
        prefix(
            turing_regression_varying_coefficients(
                design.coefficient_sd_priors, design.levels, design.centered
            ),
            VarName{name}(),
        ),
        false,
    )
    effects = add_up_varying_effects(
        deviations, design.design_columns, design.levels.level_of_each_session
    )
    other_effects ~
    to_submodel(turing_regression_varying_effects(Base.tail(designs)), false)
    return effects .+ other_effects
end

Turing.@model function turing_regression_varying_effects(designs::NamedTuple{()})
    return 0.0
end

"""
    add_up_varying_effects(deviations, design_columns, level_of_each_session)

Return the sum of the effects of the varying coefficients of one grouping column: for every
session, the deviation of the session's level, in `level_of_each_session`, times the
session's value in the coefficient's design column in `design_columns`, added up over the
coefficients, as a vector with a value for every session. `deviations` has the deviations
of each coefficient, a vector with a value for each level, in the order of
`design_columns`. Used by [`turing_regression_varying_effects`](@ref).
"""
function add_up_varying_effects(
    deviations::Tuple, design_columns::NamedTuple, level_of_each_session
)
    effects = map(
        (deviations_of_levels, column) ->
            deviations_of_levels[level_of_each_session] .* column,
        deviations,
        values(design_columns),
    )
    return foldl(.+, effects; init=0.0)
end

"""
    turing_regression_varying_coefficients(coefficient_sd_priors, levels, centered)

Turing.jl model which samples the deviations of the varying coefficients of one grouping
column, one coefficient at a time, with [`turing_regression_deviations`](@ref) under the
name of the coefficient and the prior of their standard deviation in
`coefficient_sd_priors`. It returns the deviations of each coefficient, a vector with a
value for each level, in a tuple in the order of `coefficient_sd_priors`, which is empty
without coefficients. `levels` has the number of levels and their groups, and `centered`
says how the deviations are sampled. Used by [`turing_regression_varying_effects`](@ref).
"""
Turing.@model function turing_regression_varying_coefficients(
    coefficient_sd_priors::NamedTuple{names}, levels, centered
) where {names}
    name = first(names)
    deviations ~ to_submodel(
        prefix(
            turing_regression_deviations(first(coefficient_sd_priors), levels, centered),
            VarName{name}(),
        ),
        false,
    )
    other_deviations ~ to_submodel(
        turing_regression_varying_coefficients(
            Base.tail(coefficient_sd_priors), levels, centered
        ),
        false,
    )
    return (deviations, other_deviations...)
end

Turing.@model function turing_regression_varying_coefficients(
    coefficient_sd_priors::NamedTuple{()}, levels, centered
)
    return ()
end

"""
    turing_regression_deviations(sd_prior, levels, centered)

Turing.jl model which samples the deviations of one varying coefficient, one for each
level, and returns them. Their standard deviation is the variable `sd`, so that under the
prefixes of the parameter, the varying effects and the coefficient it is named like
`learning_rate.participant.intercept.sd`: one value with `sd_prior` as its prior, or, when
the levels are grouped, a value for each group under its symbol name, such as
`learning_rate.participant.intercept.sd.clinical`, with `sd_prior` a `NamedTuple` with the
prior of each group. Centered, when `centered` is `true`, the deviations are the variable
`deviation`, named like `learning_rate.participant.intercept.deviation`, from normal
distributions around zero. Non-centered, a standard normal value is sampled for each level
as the variable `deviation_z`, named like `learning_rate.participant.intercept.deviation_z`,
and the deviation is that value times the standard deviation. `levels` has the number of
levels and their groups. Used by [`turing_regression_varying_coefficients`](@ref).
"""
Turing.@model function turing_regression_deviations(sd_prior, levels, centered)

    #Sample the standard deviation, one value, or one for each group when the levels are grouped, and give every level the one of its group
    sd ~ standard_deviation_distribution(sd_prior)
    sd_of_levels = standard_deviation_of_levels(sd, levels.groups)

    #Sample the deviation of every level from a normal distribution around zero, or non-centered, as a standard normal value which the standard deviation scales
    if centered
        deviation ~ deviation_distribution(sd_of_levels, levels.n_levels)
        return deviation
    else
        deviation_z ~ filldist(Normal(), levels.n_levels)
        return sd_of_levels .* deviation_z
    end
end

"""
    standard_deviation_distribution(sd_prior)
    standard_deviation_of_levels(sd, groups)

Return the distribution of the standard deviation of the deviations of one varying
coefficient: `sd_prior` itself, or, when it is a `NamedTuple` with the prior of each group,
a product of them. And return the standard deviation of every level: `sd` itself without
groups, where `groups` is `nothing`, or else the value of every level's group. Used by
[`turing_regression_deviations`](@ref).
"""
standard_deviation_distribution(sd_prior) = sd_prior
standard_deviation_distribution(sd_prior::NamedTuple) = product_distribution(sd_prior)
standard_deviation_of_levels(sd, ::Nothing) = sd
function standard_deviation_of_levels(sd::NamedTuple, groups::NamedTuple)

    #The value of each group as a vector, to look up the group of every level
    sd_of_groups = collect(values(sd))
    return sd_of_groups[groups.group_of_each_level]
end

"""
    deviation_distribution(sd_of_levels, n_levels)

Return the distribution of the deviations of `n_levels` levels of varying effects: normal
distributions around zero with the standard deviation `sd_of_levels`, which is one value for
every level, or a vector with the value of each. Used by
[`turing_regression_deviations`](@ref).
"""
deviation_distribution(sd::Real, n_levels) = filldist(Normal(0, sd), n_levels)
function deviation_distribution(sd_of_levels::AbstractVector, n_levels)
    return arraydist(map(sd -> Normal(0, sd), sd_of_levels))
end
