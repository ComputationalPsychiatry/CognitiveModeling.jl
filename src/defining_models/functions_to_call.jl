## Functions that users call to read a model ##

"""
    discrete_parameter_names(model)

Return the names of the parameters of `model` that are discrete in
[`parameter_domains`](@ref), as a tuple. Every other parameter is continuous. Useful for
choosing how each parameter is sampled, since a discrete parameter needs another kind of
sampler than a continuous one.
"""
function discrete_parameter_names(model::AbstractTimestepModel)
    domains = parameter_domains(model)
    return Tuple(
        name for name in keys(domains) if get(getproperty(domains, name), :discrete, false)
    )
end
