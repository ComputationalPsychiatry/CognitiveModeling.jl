## The supertype of session parameter models ##

"""
    AbstractSessionParametersModel

Supertype for session parameter models: the part of a full model, the model that an
inference backend fits to several sessions, which gives or estimates the parameters of every
session, for the cognitive model and for the environment model alike. The package comes with
[`NoParameters`](@ref), [`FixedSessionParameters`](@ref),
[`FixedParametersBySession`](@ref), [`IdenticalSessionParameters`](@ref),
[`IndependentSessionParameters`](@ref) and [`GroupedHierarchicalSessionParameters`](@ref),
and [`CompositeSessionParameters`](@ref) combines several, each giving some of the
parameters.

# Implementation
Create a subtype, and for each inference backend it is used with, add a method of that
backend's function for session parameter models, as the backend's documentation describes. A
method of [`CognitiveModeling.parameter_names`](@ref) lets the parameters it gives be
checked before a fit, and a method of [`CognitiveModeling.prepare_parameters_model`](@ref)
lets it work out once, before the backend's model is made, what doesn't change between
evaluations.
"""
abstract type AbstractSessionParametersModel end

## Whether a session parameter model estimates its parameters ##

"""
    estimates_parameters(parameters_model)

Return whether a session parameter model estimates the parameters it gives from the data,
instead of being given their values, which is decided by its type: `false` for
[`NoParameters`](@ref), [`FixedSessionParameters`](@ref) and
[`FixedParametersBySession`](@ref), whether any part does for a
[`CompositeSessionParameters`](@ref), and `true` for every other kind, also those of one's
own. Used by [`sessions_for_environment_parameters`](@ref), and by the full models of the
inference backends, to warn when the two models' parameters can have the same variables.
"""
estimates_parameters(parameters_model) = true
