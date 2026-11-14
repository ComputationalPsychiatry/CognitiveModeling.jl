### EACH PARAMETER GIVEN BY A REGRESSION ON COLUMNS OF THE SESSIONS' TABLE ###

#Each parameter of every session given by its regression, with the design of every regression built when the full model was prepared
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::HierarchicalRegressionSessionParameters{<:NamedTuple,<:NamedTuple},
    sessions_info::MultipleSessionsInfo,
)
    parameters ~ to_submodel(
        turing_regression_parameters(
            parameters_model.parameters, parameters_model.designs, n_sessions(sessions_info)
        ),
        false,
    )
    return parameters
end

#The same when it wasn't prepared, which builds the design of every regression on each evaluation
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::HierarchicalRegressionSessionParameters{<:NamedTuple,Nothing},
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
    turing_regression_parameters(parameters, designs, n_sessions)

Turing.jl model which gives one parameter at a time from `parameters`, the
[`RegressionParameter`](@ref) of each parameter of a
[`HierarchicalRegressionSessionParameters`](@ref), with
[`turing_regression_parameter`](@ref) under the parameter's name, and returns them in a
`NamedTuple`, each as a vector with a value for each of `n_sessions` sessions. `designs` has
the design of each regression. Used by `HierarchicalRegressionSessionParameters`.
"""
Turing.@model function turing_regression_parameters(
    parameters::NamedTuple{names}, designs::NamedTuple, n_sessions
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_regression_parameter(first(parameters), first(designs), n_sessions),
            VarName{name}(),
        ),
        false,
    )
    other_values ~ to_submodel(
        turing_regression_parameters(Base.tail(parameters), Base.tail(designs), n_sessions),
        false,
    )
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_regression_parameters(
    parameters::NamedTuple{()}, designs::NamedTuple, n_sessions
)
    return (;)
end

"""
    turing_regression_parameter(regression_parameter, design, n_sessions)

Turing.jl model which gives one parameter to every session, as `regression_parameter`, a
[`RegressionParameter`](@ref), says. It samples the coefficients of the intercept, the
predictors and the interactions with [`turing_regression_coefficients`](@ref) and the
deviations of the varying effects with [`turing_regression_varying_effects`](@ref), and
returns the sum of their effects on every session, with the transformation applied to each.
`design` has the columns and the levels of the regression, for each of `n_sessions`
sessions. Used by [`turing_regression_parameters`](@ref).
"""
Turing.@model function turing_regression_parameter(regression_parameter, design, n_sessions)

    #Sample the coefficients of the intercept, the predictors and the interactions, and add up their effects on every session
    population_effects ~ to_submodel(
        turing_regression_coefficients(
            design.coefficient_priors, design.design_columns, n_sessions
        ),
        false,
    )

    #Sample the deviations of the levels of the varying effects of each column, and add up their effects on every session
    varying_effects ~
    to_submodel(turing_regression_varying_effects(design.varying_effects), false)

    #Transform every sum to the scale of the parameter
    return map(regression_parameter.transformation, population_effects .+ varying_effects)
end

"""
    turing_regression_coefficients(coefficient_priors, design_columns, n_sessions)

Turing.jl model which samples the coefficients of the intercept, the predictors and the
interactions of a regression from `coefficient_priors`, as one variable `coef` whose value
is a `NamedTuple` with each coefficient under its name, so that under the parameter's prefix
one is named like `learning_rate.coef.age`. It returns the sum of every coefficient times
its column in `design_columns`, from [`add_up_effects`](@ref), as a vector with a value for
each of `n_sessions` sessions, which is zero without coefficients. Used by
[`turing_regression_parameter`](@ref).
"""
Turing.@model function turing_regression_coefficients(
    coefficient_priors::NamedTuple, design_columns::NamedTuple, n_sessions
)
    coef ~ product_distribution(coefficient_priors)
    return add_up_effects(coef, design_columns, n_sessions)
end

Turing.@model function turing_regression_coefficients(
    coefficient_priors::NamedTuple{()}, design_columns::NamedTuple, n_sessions
)
    return zeros(n_sessions)
end

"""
    add_up_effects(coefficients, design_columns, n_sessions)

Return the sum of each coefficient in `coefficients`, a `NamedTuple`, times the column of
the same name in `design_columns`, as a vector with a value for each of `n_sessions`
sessions. Used by [`turing_regression_coefficients`](@ref).
"""
function add_up_effects(
    coefficients::NamedTuple, design_columns::NamedTuple{names}, n_sessions
) where {names}

    #The coefficients in the order of the columns
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

Turing.jl model which samples the deviations of the varying effects of one column at a
time, with [`turing_regression_varying_coefficients`](@ref) under the name of the varying
effects, from `designs`, the design of the varying effects of each column of a regression
under their name. It returns the sum of their effects on every session, as a vector with a
value for every session, or zero without varying effects. Used by
[`turing_regression_parameter`](@ref).
"""
Turing.@model function turing_regression_varying_effects(
    designs::NamedTuple{names}
) where {names}
    name = first(names)
    design = first(designs)
    effects ~ to_submodel(
        prefix(
            turing_regression_varying_coefficients(
                design.coefficient_sd_priors,
                design.design_columns,
                design.levels,
                design.centered,
            ),
            VarName{name}(),
        ),
        false,
    )
    other_effects ~
    to_submodel(turing_regression_varying_effects(Base.tail(designs)), false)
    return effects .+ other_effects
end

Turing.@model function turing_regression_varying_effects(designs::NamedTuple{()})
    return 0.0
end

"""
    turing_regression_varying_coefficients(coefficient_sd_priors, design_columns, levels,
                                           centered)

Turing.jl model which samples the deviations of the levels of the varying effects of one
column, for one of their coefficients at a time, with [`turing_regression_deviations`](@ref)
under the name of the coefficient and the prior of their standard deviation in
`coefficient_sd_priors`. It returns the sum over the coefficients of the deviation of every
session's level times the session's value in the coefficient's column in `design_columns`,
as a vector with a value for every session, or zero without coefficients. `levels` has the
level of every session and the groups of the levels, and `centered` says how the deviations
are sampled. Used by [`turing_regression_varying_effects`](@ref).
"""
Turing.@model function turing_regression_varying_coefficients(
    coefficient_sd_priors::NamedTuple{names}, design_columns::NamedTuple, levels, centered
) where {names}
    name = first(names)
    deviations ~ to_submodel(
        prefix(
            turing_regression_deviations(first(coefficient_sd_priors), levels, centered),
            VarName{name}(),
        ),
        false,
    )
    other_effects ~ to_submodel(
        turing_regression_varying_coefficients(
            Base.tail(coefficient_sd_priors), Base.tail(design_columns), levels, centered
        ),
        false,
    )
    return deviations[levels.level_of_each_session] .* first(design_columns) .+
           other_effects
end

Turing.@model function turing_regression_varying_coefficients(
    coefficient_sd_priors::NamedTuple{()}, design_columns::NamedTuple, levels, centered
)
    return 0.0
end

"""
    turing_regression_deviations(sd_prior, levels, centered)

Turing.jl model which samples the deviations of one varying coefficient, one for each
level: their standard deviation with [`turing_regression_standard_deviation`](@ref), and
then every deviation from a normal distribution around zero, with
[`turing_regression_centered_deviations`](@ref) when `centered` is `true`, or as a
standardized value which the standard deviation scales, with
[`turing_regression_standardized_deviations`](@ref). It returns the deviation of every
level. `sd_prior` is the prior of the standard deviation, a `NamedTuple` with the prior of
each group when the levels are grouped, and `levels` has the number of levels and their
groups. Used by [`turing_regression_varying_coefficients`](@ref).
"""
Turing.@model function turing_regression_deviations(sd_prior, levels, centered)

    #Sample the standard deviation, which is the one of every level's group when the levels are grouped
    sd_of_levels ~
    to_submodel(turing_regression_standard_deviation(sd_prior, levels.groups), false)

    #Sample the deviation of every level, from the normal distribution or as a standardized value
    if centered
        deviations ~ to_submodel(
            turing_regression_centered_deviations(sd_of_levels, levels.n_levels), false
        )
    else
        deviations ~ to_submodel(
            turing_regression_standardized_deviations(sd_of_levels, levels.n_levels), false
        )
    end
    return deviations
end

"""
    turing_regression_standard_deviation(sd_prior, groups)

Turing.jl model which samples the standard deviation of the deviations of one varying
coefficient, as the variable `sd`, so that under the prefixes of the parameter, the varying
effects and the coefficient it is named like `learning_rate.participant.intercept.sd`.
Without groups, where `groups` is `nothing`, it is one value with `sd_prior` as its prior,
which is returned. With groups, `sd_prior` is a `NamedTuple` with the prior of each group
under its symbol name, `sd` has the value of each group under the same name, such as
`learning_rate.participant.intercept.sd.clinical`, and the standard deviation of every level
is returned, which is the one of its group. Used by [`turing_regression_deviations`](@ref).
"""
Turing.@model function turing_regression_standard_deviation(sd_prior, ::Nothing)
    sd ~ sd_prior
    return sd
end

Turing.@model function turing_regression_standard_deviation(
    sd_prior::NamedTuple, groups::NamedTuple
)
    sd ~ product_distribution(sd_prior)

    #The value of each group as a vector, to look up the group of every level
    sd_of_groups = collect(values(sd))
    return sd_of_groups[groups.group_of_each_level]
end

"""
    turing_regression_centered_deviations(sd_of_levels, n_levels)

Turing.jl model which samples the deviation of each of `n_levels` levels of varying effects
from a normal distribution around zero, as the variable `deviation`, so that under the
prefixes of the parameter, the varying effects and the coefficient it is named like
`learning_rate.participant.intercept.deviation`. `sd_of_levels` is the standard deviation,
as one value or with a value for every level. Used by
[`turing_regression_deviations`](@ref).
"""
Turing.@model function turing_regression_centered_deviations(sd_of_levels, n_levels)
    deviation ~ deviation_distribution(sd_of_levels, n_levels)
    return deviation
end

"""
    turing_regression_standardized_deviations(sd_of_levels, n_levels)

Turing.jl model which samples a standardized value for each of `n_levels` levels of varying
effects, as the variable `standardized`, so that under the prefixes of the parameter, the
varying effects and the coefficient it is named like
`learning_rate.participant.intercept.standardized`, and returns the deviation of every
level: its standardized value times `sd_of_levels`, the standard deviation, as one value or
with a value for every level. This is the non-centered form of a normal distribution. Used
by [`turing_regression_deviations`](@ref).
"""
Turing.@model function turing_regression_standardized_deviations(sd_of_levels, n_levels)
    standardized ~ filldist(Normal(), n_levels)
    return sd_of_levels .* standardized
end

"""
    deviation_distribution(sd_of_levels, n_levels)

Return the distribution of the deviations of `n_levels` levels of varying effects: normal
distributions around zero with the standard deviation `sd_of_levels`, which is one value for
every level, or a vector with the value of each. Used by
[`turing_regression_centered_deviations`](@ref).
"""
deviation_distribution(sd::Real, n_levels) = filldist(Normal(0, sd), n_levels)
function deviation_distribution(sd_of_levels::AbstractVector, n_levels)
    return arraydist(map(sd -> Normal(0, sd), sd_of_levels))
end
