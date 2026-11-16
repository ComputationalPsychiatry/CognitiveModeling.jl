## Functions that users call to read a model ##

"""
    discrete_parameter_names(model)

Return the names of the parameters of `model` that are discrete in
[`parameter_domains`](@ref), as a tuple. Every other parameter is continuous, see
[`continuous_parameter_names`](@ref). Useful for choosing how each parameter is sampled,
since a discrete parameter needs another kind of sampler than a continuous one.
"""
function discrete_parameter_names(model::AbstractTimestepModel)
    domains = parameter_domains(model)
    return Tuple(
        name for name in keys(domains) if get(getproperty(domains, name), :discrete, false)
    )
end

"""
    continuous_parameter_names(model)

Return the names of the parameters of `model` that are continuous, as a tuple, in the order
of its attributes: every parameter that isn't discrete in [`parameter_domains`](@ref), also
those without a domain. The others are in [`discrete_parameter_names`](@ref). Useful for
choosing how each parameter is sampled, since a sampler which uses gradients can only
sample the continuous ones.
"""
function continuous_parameter_names(model::AbstractTimestepModel)
    names_in_attributes =
        check_type_attribute_names(attribute_names(model), model).parameters
    discrete_names = discrete_parameter_names(model)
    return Tuple(name for name in names_in_attributes if !(name in discrete_names))
end
