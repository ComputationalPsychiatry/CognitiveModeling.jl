## Functions that users call to read a model ##

"""
    get_parameter_names(model; without_defaults::Bool = false)

Return the names of the parameters of `model` and of its submodels, as a tuple, in the
order of the attributes and under the flat names that
[`CognitiveModeling.get_attribute_names`](@ref) gives them. With `without_defaults = true`,
only the parameters without a default value in [`default_parameters`](@ref) of the model
or of its submodel: those that a fit needs priors or fixed values for. Not to be confused
with [`parameter_names`](@ref), which gives the names of the parameters that a parameter
model samples.

# Examples
```jldoctest
julia> struct Learner <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Learner) = ModelAttributes(;
           parameters = (:learning_rate, :inverse_temperature),
           states = (), inputs = (), outputs = ());

julia> CognitiveModeling.default_parameters(::Learner) = (; inverse_temperature = 1.0);

julia> CognitiveModeling.get_parameter_names(Learner())
(:learning_rate, :inverse_temperature)

julia> CognitiveModeling.get_parameter_names(Learner(); without_defaults = true)
(:learning_rate,)
```
"""
function get_parameter_names(model::AbstractSubmodel; without_defaults::Bool=false)
    names_in_attributes = get_attribute_names(model).parameters
    without_defaults || return names_in_attributes
    default_names = keys(collected_default_parameters(model))
    return Tuple(name for name in names_in_attributes if !(name in default_names))
end

"""
    discrete_parameter_names(model)

Return the names of the parameters of `model` that are discrete in
[`parameter_domains`](@ref), as a tuple, with those of its submodels under their flat names,
as [`CognitiveModeling.get_attribute_names`](@ref) gives them. Every other parameter is continuous, see
[`continuous_parameter_names`](@ref). Useful for choosing how each parameter is sampled,
since a discrete parameter needs another kind of sampler than a continuous one.
"""
function discrete_parameter_names(model::AbstractTimestepModel)
    domains = collected_parameter_domains(model)
    return Tuple(name for name in keys(domains) if is_discrete(getproperty(domains, name)))
end

"""
    continuous_parameter_names(model)

Return the names of the parameters of `model` that are continuous, as a tuple, in the order
of its attributes, with those of its submodels under their flat names: every parameter that isn't discrete in [`parameter_domains`](@ref), also
those without a domain. The others are in [`discrete_parameter_names`](@ref). Useful for
choosing how each parameter is sampled, since a sampler which uses gradients can only
sample the continuous ones.
"""
function continuous_parameter_names(model::AbstractTimestepModel)
    names_in_attributes = get_attribute_names(model).parameters
    discrete_names = discrete_parameter_names(model)
    return Tuple(name for name in names_in_attributes if !(name in discrete_names))
end
