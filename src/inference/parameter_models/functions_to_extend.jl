## Functions that a parameter model can implement ##

"""
    parameter_names(parameters_model)

Return the names of the parameters that `parameters_model` gives every session, as a tuple
of `Symbol`s, or `nothing` when they aren't known before it runs.
[`prepare_full_model`](@ref) checks them against the parameters of the cognitive model and
of the environment model, and [`CompositeParameters`](@ref) checks that no two of its parts
give the same one.

# Implementation
A parameter model of one's own can add a method of this function. The default returns
`nothing`, so that nothing is checked before a fit.

# Examples
```jldoctest
julia> CognitiveModeling.parameter_names(FixedParameters((; scale = 2.0, offset = 0.0)))
(:scale, :offset)
```
"""
parameter_names(parameters_model::AbstractParametersModel) = nothing

"""
    estimates_parameters(parameters_model)

Return whether `parameters_model`, a parameter model, estimates the parameters it gives from
the data, instead of being given their values: `false` for [`NoParameters`](@ref),
[`FixedParameters`](@ref) and [`SessionwiseFixedParameters`](@ref), whether any part does
for a [`CompositeParameters`](@ref), and `true` for every other kind. When the environment's
parameters are estimated, a full model includes the environment in every session, so that
the observations in the data inform them, and it warns when the parameters of the two models
can have the same variables.

# Implementation
A parameter model of one's own can add a method of this function. The default returns
`true`, and a model which only gives values returns `false`.

# Examples
```jldoctest
julia> CognitiveModeling.estimates_parameters(FixedParameters((; scale = 2.0)))
false
```
"""
estimates_parameters(parameters_model) = true

"""
    prepare_parameters_model(parameters_model, sessions_info)

Return `parameters_model` prepared for the sessions in `sessions_info`, with what doesn't
change between the evaluations of an inference backend's model worked out once, before the
model is made. A [`SessionwiseFixedParameters`](@ref) has the values of every session looked
up, a [`HierarchicalGroupedParameters`](@ref) has the group of every session found, a
[`HierarchicalRegressionParameters`](@ref) has the design of every regression built, and a
[`CompositeParameters`](@ref) has each of its parts prepared. Used by
[`prepare_full_model`](@ref).

# Implementation
A parameter model of one's own can add a method of this function, which returns a parameter
model that the inference backends have methods for. The default returns `parameters_model`
as it is.
"""
function prepare_parameters_model(
    parameters_model::AbstractParametersModel, sessions_info::MultipleSessionsInfo
)
    return parameters_model
end

"""
    parameter_value_ranges(parameters_model)

Return the values that `parameters_model` can give each parameter, in a `NamedTuple` under
the names of the parameters: for each, a `NamedTuple` with the lowest and the highest value,
`lower` and `upper`, whether the values are `discrete`, and `source`, which says where they
come from in an error message. A parameter whose values aren't known is left out.
[`prepare_full_model`](@ref) checks them against the domains of the parameters, from
[`parameter_domains`](@ref), so that a prior or a transformation which reaches outside a
domain is an error before a fit.

[`IdenticalParameters`](@ref) and [`IndependentParameters`](@ref) give the values of each
prior that [`CognitiveModeling.distribution_support`](@ref) knows. A
[`HierarchicalGroupedParameters`](@ref) gives what the transformation of a parameter makes
of the real line, for a parameter with the default normal distribution. A
[`HierarchicalRegressionParameters`](@ref) that was prepared for the sessions gives what
the transformation of a parameter makes of the sums of its regression.

# Implementation
A parameter model of one's own can add a method of this function. The default returns an
empty `NamedTuple`, so that nothing is checked before a fit.

# Examples
```jldoctest
julia> parameters_model = HierarchicalGroupedParameters((; learning_rate = HierarchicalParameter((; mean = 0.0, sd = 1.0); transformation = exp)));

julia> CognitiveModeling.parameter_value_ranges(parameters_model).learning_rate.lower
0.0
```
"""
parameter_value_ranges(parameters_model::AbstractParametersModel) = (;)

## Functions that an inference backend can implement for its distributions ##

"""
    distribution_support(distribution)

Return the values that `distribution` can give, as a `NamedTuple` with the lowest and the
highest value, `lower` and `upper`, and whether the values are `discrete`, or `nothing` when
they aren't known. The parameter models read the values of their priors from it, for
[`CognitiveModeling.parameter_value_ranges`](@ref).

# Implementation
An inference backend can add a method of this function for the distributions it takes. The
default returns `nothing`.
"""
distribution_support(distribution) = nothing
