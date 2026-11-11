## The same given values for every session ##

"""
    FixedSessionParameters(parameters)

The session parameter model of a full model which gives every session the same values, such
as `(; scale = 2.0)`, with nothing sampled. The values are kept in the field `parameters`.
"""
struct FixedSessionParameters{Tparameters<:NamedTuple} <: AbstractSessionParametersModel
    parameters::Tparameters
end

#It gives the parameters it holds values for, and estimates nothing
function parameter_names(parameters_model::FixedSessionParameters)
    return keys(parameters_model.parameters)
end
estimates_parameters(::FixedSessionParameters) = false
