## The same given values for every session ##

"""
    FixedParameters(parameters)

The parameter model of a full model which gives every session the same values, such as
`(; scale = 2.0)`, with nothing sampled. The values are kept in the field `parameters`.
"""
struct FixedParameters{Tparameters<:NamedTuple} <: AbstractParametersModel
    parameters::Tparameters
end

#It gives the parameters it holds values for, and estimates nothing
function parameter_names(parameters_model::FixedParameters)
    return keys(parameters_model.parameters)
end
estimates_parameters(::FixedParameters) = false
