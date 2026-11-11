## No parameters ##

"""
    NoParameters()

The session parameter model of a full model which gives no parameters, so that each
parameter takes the value the model gives it in [`default_parameters`](@ref). It is the
default for the parameters of the environment.
"""
struct NoParameters <: AbstractSessionParametersModel end

#It gives no parameters, and estimates nothing
parameter_names(::NoParameters) = ()
estimates_parameters(::NoParameters) = false
