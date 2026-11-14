## Functions that a session parameter model can implement ##

"""
    parameter_names(parameters_model)

Return the names of the parameters that `parameters_model` gives every session, as a tuple
of `Symbol`s, or `nothing` when they aren't known before it runs.
[`prepare_full_model`](@ref) checks them against the parameters of the cognitive model and
of the environment model, and [`CompositeSessionParameters`](@ref) checks that no two of its
parts give the same one.

# Implementation
A session parameter model of one's own can add a method of this function. The default
returns `nothing`, so that nothing is checked before a fit.

# Examples
```jldoctest
julia> CognitiveModeling.parameter_names(FixedSessionParameters((; scale = 2.0, offset = 0.0)))
(:scale, :offset)
```
"""
parameter_names(parameters_model::AbstractSessionParametersModel) = nothing

"""
    estimates_parameters(parameters_model)

Return whether `parameters_model`, a session parameter model, estimates the parameters it
gives from the data, instead of being given their values: `false` for
[`NoParameters`](@ref), [`FixedSessionParameters`](@ref) and
[`FixedParametersBySession`](@ref), whether any part does for a
[`CompositeSessionParameters`](@ref), and `true` for every other kind. When the
environment's parameters are estimated, a full model includes the environment in every
session, so that the observations in the data inform them, and it warns when the parameters
of the two models can have the same variables.

# Implementation
A session parameter model of one's own can add a method of this function. The default
returns `true`, and a model which only gives values returns `false`.

# Examples
```jldoctest
julia> CognitiveModeling.estimates_parameters(FixedSessionParameters((; scale = 2.0)))
false
```
"""
estimates_parameters(parameters_model) = true

"""
    prepare_parameters_model(parameters_model, sessions_info)

Return `parameters_model` prepared for the sessions in `sessions_info`, with what doesn't
change between the evaluations of an inference backend's model worked out once, before the
model is made. A [`FixedParametersBySession`](@ref) has the values of every session looked
up, a [`GroupedHierarchicalSessionParameters`](@ref) has the group of every session found, a
[`HierarchicalRegression`](@ref) has the design of every regression built, and a
[`CompositeSessionParameters`](@ref) has each of its parts prepared. Used by
[`prepare_full_model`](@ref).

# Implementation
A session parameter model of one's own can add a method of this function, which returns a
session parameter model that the inference backends have methods for. The default returns
`parameters_model` as it is.
"""
function prepare_parameters_model(
    parameters_model::AbstractSessionParametersModel, sessions_info::MultipleSessionsInfo
)
    return parameters_model
end
