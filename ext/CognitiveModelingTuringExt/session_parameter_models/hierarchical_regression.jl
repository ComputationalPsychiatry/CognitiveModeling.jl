### EACH PARAMETER GIVEN BY A REGRESSION ON COLUMNS OF THE SESSIONS' TABLE ###

#Each parameter of every session given by its regression, with the design of every regression built when the full model was prepared
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::HierarchicalRegression{<:NamedTuple,<:NamedTuple},
    sessions_info::MultipleSessionsInfo,
)
    parameters ~ to_submodel(
        turing_sample_regression_parameters(
            parameters_model.parameters, parameters_model.designs, n_sessions(sessions_info)
        ),
        false,
    )
    return parameters
end

#The same when it wasn't prepared, which builds the design of every regression on each evaluation
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::HierarchicalRegression{<:NamedTuple,Nothing},
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
    turing_sample_regression_parameters(parameters, designs, n_sessions)

Turing.jl model which gives one parameter at a time from `parameters`, the
[`RegressionParameter`](@ref) of each parameter of a [`HierarchicalRegression`](@ref), with
[`turing_sample_regression_parameter`](@ref) under the parameter's name, and returns them in
a `NamedTuple`, each as a vector with a value for each of `n_sessions` sessions. `designs`
has the design of each regression. Used by `HierarchicalRegression`.
"""
Turing.@model function turing_sample_regression_parameters(
    parameters::NamedTuple{names}, designs::NamedTuple, n_sessions
) where {names}
    name = first(names)
    values ~ to_submodel(
        prefix(
            turing_sample_regression_parameter(
                first(parameters), first(designs), n_sessions
            ),
            VarName{name}(),
        ),
        false,
    )
    other_values ~ to_submodel(
        turing_sample_regression_parameters(
            Base.tail(parameters), Base.tail(designs), n_sessions
        ),
        false,
    )
    return merge(NamedTuple{(name,)}((values,)), other_values)
end

Turing.@model function turing_sample_regression_parameters(
    parameters::NamedTuple{()}, designs::NamedTuple, n_sessions
)
    return (;)
end

"""
    turing_sample_regression_parameter(regression_parameter, design, n_sessions)

Turing.jl model which gives one parameter to every session, as `regression_parameter`, a
[`RegressionParameter`](@ref), says. It samples the intercept with
[`turing_sample_intercept`](@ref), the coefficients of the predictors with
[`turing_sample_predictor_effects`](@ref), the random intercepts with
[`turing_sample_random_intercepts`](@ref) and the deviations of the sessions with
[`turing_sample_session_deviations`](@ref), and returns their sum for every session, with
the transformation applied to each. `design` has the columns and the levels of the
regression. Used by [`turing_sample_regression_parameters`](@ref).
"""
Turing.@model function turing_sample_regression_parameter(
    regression_parameter, design, n_sessions
)

    #Sample the intercept, which is zero for a regression without one
    intercept ~ to_submodel(turing_sample_intercept(regression_parameter.intercept), false)

    #Sample the coefficient of each predictor, and add up their effects on every session
    predictor_effects ~ to_submodel(
        turing_sample_predictor_effects(design.coefficient_priors, design.design_columns),
        false,
    )

    #Sample the random intercepts of each column, and add up those of the levels of every session
    random_intercepts ~ to_submodel(
        turing_sample_random_intercepts(
            regression_parameter.random_intercepts, design.random_intercept_levels
        ),
        false,
    )

    #Sample the deviation of every session
    session_deviations ~ to_submodel(
        turing_sample_session_deviations(
            regression_parameter.session_deviation, design.session_deviation_levels
        ),
        false,
    )

    #Add them up for every session, and transform every sum to the scale of the parameter
    sums =
        fill(intercept, n_sessions) .+ predictor_effects .+ random_intercepts .+
        session_deviations
    return map(regression_parameter.transformation, sums)
end

"""
    turing_sample_intercept(prior)

Turing.jl model which samples the intercept of a regression from `prior`, as the variable
`intercept`, so that under the parameter's prefix it is named like
`learning_rate.intercept`. With `nothing`, for a regression without an intercept, it samples
nothing and returns zero. Used by [`turing_sample_regression_parameter`](@ref).
"""
Turing.@model function turing_sample_intercept(prior)
    intercept ~ prior
    return intercept
end

Turing.@model function turing_sample_intercept(::Nothing)
    return 0.0
end

"""
    turing_sample_predictor_effects(coefficient_priors, design_columns)

Turing.jl model which samples the coefficient of one column at a time from
`coefficient_priors`, each with [`turing_sample_coefficient`](@ref) under the name of the
coefficient, and returns the sum of every coefficient times its column in `design_columns`,
as a vector with a value for every session, or zero without predictors. Used by
[`turing_sample_regression_parameter`](@ref).
"""
Turing.@model function turing_sample_predictor_effects(
    coefficient_priors::NamedTuple{names}, design_columns::NamedTuple
) where {names}
    name = first(names)
    coefficient ~ to_submodel(
        prefix(turing_sample_coefficient(first(coefficient_priors)), VarName{name}()), false
    )
    other_effects ~ to_submodel(
        turing_sample_predictor_effects(
            Base.tail(coefficient_priors), Base.tail(design_columns)
        ),
        false,
    )
    return coefficient .* first(design_columns) .+ other_effects
end

Turing.@model function turing_sample_predictor_effects(
    coefficient_priors::NamedTuple{()}, design_columns::NamedTuple
)
    return 0.0
end

"""
    turing_sample_coefficient(prior)

Turing.jl model which samples the coefficient of one column of a regression from `prior`, as
the variable `coefficient`, so that under the prefixes of the parameter and of the
coefficient's name it is named like `learning_rate.age.coefficient`. Used by
[`turing_sample_predictor_effects`](@ref).
"""
Turing.@model function turing_sample_coefficient(prior)
    coefficient ~ prior
    return coefficient
end

"""
    turing_sample_random_intercepts(random_intercepts, levels)

Turing.jl model which samples the random intercepts of one column at a time from
`random_intercepts`, the [`RandomIntercept`](@ref) of each column, with
[`turing_sample_deviations`](@ref) under the column's name, and returns the sum over the
columns of the random intercept of every session's level, as a vector with a value for every
session, or zero without random intercepts. `levels` has the levels of the sessions for each
column. Used by [`turing_sample_regression_parameter`](@ref).
"""
Turing.@model function turing_sample_random_intercepts(
    random_intercepts::NamedTuple{names}, levels::NamedTuple
) where {names}
    name = first(names)
    levels_of_column = first(levels)
    deviations ~ to_submodel(
        prefix(
            turing_sample_deviations(first(random_intercepts), levels_of_column),
            VarName{name}(),
        ),
        false,
    )
    other_random_intercepts ~ to_submodel(
        turing_sample_random_intercepts(Base.tail(random_intercepts), Base.tail(levels)),
        false,
    )
    return deviations[levels_of_column.level_of_each_session] .+ other_random_intercepts
end

Turing.@model function turing_sample_random_intercepts(
    random_intercepts::NamedTuple{()}, levels::NamedTuple
)
    return 0.0
end

"""
    turing_sample_session_deviations(session_deviation, levels)

Turing.jl model which samples the deviation of every session from `session_deviation`, a
[`RandomIntercept`](@ref) with every session as a level, with
[`turing_sample_deviations`](@ref) under `session_deviation`, and returns them as a vector
with a value for every session. With `nothing`, for a regression without them, it samples
nothing and returns zero. Used by [`turing_sample_regression_parameter`](@ref).
"""
Turing.@model function turing_sample_session_deviations(
    session_deviation::RandomIntercept, levels::NamedTuple
)
    deviations ~ to_submodel(
        prefix(
            turing_sample_deviations(session_deviation, levels), @varname(session_deviation)
        ),
        false,
    )
    return deviations
end

Turing.@model function turing_sample_session_deviations(::Nothing, ::Nothing)
    return 0.0
end

"""
    turing_sample_deviations(random_intercept, levels)

Turing.jl model which samples the deviations of one random intercept, one for each of its
levels: their standard deviation with [`turing_sample_standard_deviation`](@ref), and then
every deviation from a normal distribution around zero, with
[`turing_sample_centered_deviations`](@ref), or as a standardized value which the standard
deviation scales, with [`turing_sample_standardized_deviations`](@ref), as `centered` of
`random_intercept` says. It returns the deviation of every level. Used by
[`turing_sample_random_intercepts`](@ref) and [`turing_sample_session_deviations`](@ref).
"""
Turing.@model function turing_sample_deviations(random_intercept, levels)

    #Sample the standard deviation, which is the one of every level's group when the levels are grouped
    sd_of_levels ~ to_submodel(
        turing_sample_standard_deviation(random_intercept.sd_prior, levels.groups), false
    )

    #Sample the deviation of every level, from the normal distribution or as a standardized value
    if random_intercept.centered
        deviations ~
        to_submodel(turing_sample_centered_deviations(sd_of_levels, levels.n_levels), false)
    else
        deviations ~ to_submodel(
            turing_sample_standardized_deviations(sd_of_levels, levels.n_levels), false
        )
    end
    return deviations
end

"""
    turing_sample_standard_deviation(sd_prior, groups)

Turing.jl model which samples the standard deviation of a random intercept from `sd_prior`,
as the variable `sd`, so that under the prefixes of the parameter and of the column it is
named like `learning_rate.participant.sd`. Without groups, where `groups` is `nothing`, it
is one value, which is returned. With groups it is a vector with a value for each group, and
the standard deviation of every level is returned, which is the one of its group. Used by
[`turing_sample_deviations`](@ref).
"""
Turing.@model function turing_sample_standard_deviation(sd_prior, ::Nothing)
    sd ~ sd_prior
    return sd
end

Turing.@model function turing_sample_standard_deviation(sd_prior, groups::NamedTuple)
    sd ~ filldist(sd_prior, groups.n_groups)
    return sd[groups.group_of_each_level]
end

"""
    turing_sample_centered_deviations(sd_of_levels, n_levels)

Turing.jl model which samples the deviation of each of `n_levels` levels of a random
intercept from a normal distribution around zero, as the variable `deviation`, so that
under the prefixes of the parameter and of the column it is named like
`learning_rate.participant.deviation`. `sd_of_levels` is the standard deviation, as one
value or with a value for every level. Used by [`turing_sample_deviations`](@ref).
"""
Turing.@model function turing_sample_centered_deviations(sd_of_levels, n_levels)
    deviation ~ deviation_distribution(sd_of_levels, n_levels)
    return deviation
end

"""
    turing_sample_standardized_deviations(sd_of_levels, n_levels)

Turing.jl model which samples a standardized value for each of `n_levels` levels of a random
intercept, as the variable `standardized`, so that under the prefixes of the parameter and
of the column it is named like `learning_rate.participant.standardized`, and returns the
deviation of every level: its standardized value times `sd_of_levels`, the standard
deviation, as one value or with a value for every level. This is the non-centered form of
a normal distribution. Used by [`turing_sample_deviations`](@ref).
"""
Turing.@model function turing_sample_standardized_deviations(sd_of_levels, n_levels)
    standardized ~ filldist(Normal(), n_levels)
    return sd_of_levels .* standardized
end

"""
    deviation_distribution(sd_of_levels, n_levels)

Return the distribution of the deviations of `n_levels` levels of a random intercept: normal
distributions around zero with the standard deviation `sd_of_levels`, which is one value for
every level, or a vector with the value of each. Used by
[`turing_sample_centered_deviations`](@ref).
"""
deviation_distribution(sd::Real, n_levels) = filldist(Normal(0, sd), n_levels)
function deviation_distribution(sd_of_levels::AbstractVector, n_levels)
    return arraydist(map(sd -> Normal(0, sd), sd_of_levels))
end
