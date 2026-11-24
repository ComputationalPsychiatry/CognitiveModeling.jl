## Functions that users call to read a model ##

"""
    get_parameter_names(model; with_default_parameters=true)

Return the names of the parameters of `model` and of its submodels, as a tuple, in the
order of the attributes and under the flat names that
[`CognitiveModeling.get_attribute_names`](@ref) gives them. With
`with_default_parameters = false`, the parameters with a default value in
[`default_parameters`](@ref) of the model or of its submodel are left out, which leaves
those that a fit needs priors or fixed values for. Not to be confused with
[`parameter_names`](@ref), which gives the names of the parameters that a parameter model
samples.

# Examples
```jldoctest
julia> struct Learner <: AbstractCognitiveModel end

julia> CognitiveModeling.attribute_names(::Learner) = ModelAttributes(;
           parameters = (:learning_rate, :inverse_temperature),
           states = (), inputs = (), outputs = ());

julia> CognitiveModeling.default_parameters(::Learner) = (; inverse_temperature = 1.0);

julia> CognitiveModeling.get_parameter_names(Learner())
(:learning_rate, :inverse_temperature)

julia> CognitiveModeling.get_parameter_names(Learner(); with_default_parameters = false)
(:learning_rate,)
```
"""
function get_parameter_names(model::AbstractSubmodel; with_default_parameters::Bool=true)
    return get_attribute_names(model; with_default_parameters).parameters
end

"""
    get_default_parameters(model)

Return the default values of the parameters of `model` and of its submodels, in a
`NamedTuple` under the flat names that [`CognitiveModeling.get_attribute_names`](@ref)
gives the parameters: the values that [`default_parameters`](@ref) gives for the model,
followed by those of each of its submodels. A parameter without a default value isn't in
it.

# Examples
```jldoctest
julia> struct Perception <: AbstractSubmodel end

julia> CognitiveModeling.attribute_names(::Perception) = ModelAttributes(;
           parameters = (:learning_rate, :initial_value),
           states = (), inputs = (), outputs = ());

julia> CognitiveModeling.default_parameters(::Perception) = (; initial_value = 0.5);

julia> struct PerceivesAndLapses <: AbstractCognitiveModel
           perception::Perception
       end

julia> CognitiveModeling.submodel_fields(::PerceivesAndLapses) = (:perception,);

julia> CognitiveModeling.attribute_names(::PerceivesAndLapses) = ModelAttributes(;
           parameters = :lapse_rate, states = (), inputs = (), outputs = ());

julia> CognitiveModeling.default_parameters(::PerceivesAndLapses) = (; lapse_rate = 0.0);

julia> CognitiveModeling.get_default_parameters(PerceivesAndLapses(Perception()))
(lapse_rate = 0.0, initial_value = 0.5)
```
"""
function get_default_parameters(model::AbstractSubmodel)
    check_submodels(model)
    return collected_default_parameters(model)
end

"""
    get_parameter_domains(model)

Return the [`Domain`](@ref)s of the parameters of `model` and of its submodels, in a
`NamedTuple` under the flat names that [`CognitiveModeling.get_attribute_names`](@ref)
gives the parameters: the domains that [`parameter_domains`](@ref) gives for the model,
followed by those of each of its submodels. A parameter without a domain isn't in it.

# Examples
```jldoctest
julia> struct Perception <: AbstractSubmodel end

julia> CognitiveModeling.attribute_names(::Perception) = ModelAttributes(;
           parameters = (:learning_rate, :initial_value),
           states = (), inputs = (), outputs = ());

julia> CognitiveModeling.parameter_domains(::Perception) = (;
           learning_rate = Domain(; lower = 0, upper = 1));

julia> struct PerceivesAndLapses <: AbstractCognitiveModel
           perception::Perception
       end

julia> CognitiveModeling.submodel_fields(::PerceivesAndLapses) = (:perception,);

julia> CognitiveModeling.attribute_names(::PerceivesAndLapses) = ModelAttributes(;
           parameters = :lapse_rate, states = (), inputs = (), outputs = ());

julia> CognitiveModeling.parameter_domains(::PerceivesAndLapses) = (;
           lapse_rate = Domain(; lower = 0, upper = 1));

julia> keys(CognitiveModeling.get_parameter_domains(PerceivesAndLapses(Perception())))
(:lapse_rate, :learning_rate)
```
"""
function get_parameter_domains(model::AbstractSubmodel)
    check_submodels(model)
    return collected_parameter_domains(model)
end

"""
    discrete_parameter_names(model; with_default_parameters=true)

Return the names of the parameters of `model` and of its submodels that are discrete in
[`parameter_domains`](@ref), as a tuple, in the order of the attributes and under the flat
names that [`CognitiveModeling.get_attribute_names`](@ref) gives them. Every other
parameter is continuous, see [`CognitiveModeling.continuous_parameter_names`](@ref).
With `with_default_parameters = false`, the parameters with a default value in
[`default_parameters`](@ref) are left out. Useful for choosing how each parameter is
sampled, since a discrete parameter needs another kind of sampler than a continuous one.
"""
function discrete_parameter_names(
    model::AbstractSubmodel; with_default_parameters::Bool=true
)
    names = get_parameter_names(model; with_default_parameters)
    domains = collected_parameter_domains(model)
    return Tuple(
        name for
        name in names if haskey(domains, name) && is_discrete(getproperty(domains, name))
    )
end

"""
    continuous_parameter_names(model; with_default_parameters=true)

Return the names of the parameters of `model` and of its submodels that are continuous, as a
tuple, in the order of the attributes and under the flat names that
[`CognitiveModeling.get_attribute_names`](@ref) gives them: every parameter that isn't
discrete in [`parameter_domains`](@ref), also those without a domain. The others are in
[`CognitiveModeling.discrete_parameter_names`](@ref). With
`with_default_parameters = false`, the parameters with a default value in
[`default_parameters`](@ref) are left out. Useful for choosing how each parameter is
sampled, since a sampler which uses gradients can only sample the continuous ones.
"""
function continuous_parameter_names(
    model::AbstractSubmodel; with_default_parameters::Bool=true
)
    names = get_parameter_names(model; with_default_parameters)
    discrete_names = discrete_parameter_names(model)
    return Tuple(name for name in names if !(name in discrete_names))
end
