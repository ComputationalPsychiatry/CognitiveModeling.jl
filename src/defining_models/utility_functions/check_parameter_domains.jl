## Checking parameters against their domains ##
"""
    check_parameter_domains(model, parameters)

Throw an [`InvalidParametersError`](@ref) if a parameter in `parameters` is outside its
domain in [`parameter_domains`](@ref) of `model`: below `lower`, above `upper`, or not a
whole number when it is `discrete`. Real numbers and arrays of them are checked, and
anything else is left alone, such as the traced numbers of Reactant.jl, whose values can't
be compared while a simulation is compiled. It is an `ArgumentError` when the domains name
a parameter that isn't in the model's attributes, or have a field other than `lower`,
`upper` and `discrete`. Used by [`simulate`](@ref), [`InstantiatedModel`](@ref) and fitted
sessions, before anything runs.
"""
function check_parameter_domains(model::AbstractTimestepModel, parameters::NamedTuple)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    check_each_parameter_domain(model, parameters, domains)
    return nothing
end

"""
    check_domains_fit_model(model, domains)

Throw an `ArgumentError` if `domains`, what [`parameter_domains`](@ref) returns for `model`,
names a parameter that isn't in its attributes, from [`attribute_names`](@ref), or has a
domain that isn't a `NamedTuple` with only `lower`, `upper` and `discrete`. Used by
[`check_parameter_domains`](@ref).
"""
check_domains_fit_model(::AbstractTimestepModel, ::NamedTuple{()}) = nothing
function check_domains_fit_model(model::AbstractTimestepModel, domains::NamedTuple)
    names_in_attributes =
        check_type_attribute_names(attribute_names(model), model).parameters
    if !has_every_name(keys(domains), names_in_attributes)
        throw(
            ArgumentError(
                "parameter_domains of $(describe_model(model)) has domains for $(keys(domains)), but the parameters in its attributes are $names_in_attributes.",
            ),
        )
    end
    for (name, domain) in pairs(domains)
        if !(
            domain isa NamedTuple &&
            has_every_name(keys(domain), (:lower, :upper, :discrete))
        )
            throw(
                ArgumentError(
                    "The domain of $name in parameter_domains of $(describe_model(model)) is $(repr(domain)), but it must be a NamedTuple with any of lower, upper and discrete, such as (; lower = 0, upper = 1).",
                ),
            )
        end
    end
    return nothing
end

"""
    check_each_parameter_domain(model, parameters, domains)

Check the value in `parameters` of each parameter with a domain in `domains`, one at a time,
with [`check_value_in_domain`](@ref). The names are read from the types, so that the values
are checked without any search for them when the model runs. Used by
[`check_parameter_domains`](@ref).
"""
function check_each_parameter_domain(
    ::AbstractTimestepModel, ::NamedTuple, ::NamedTuple{()}
)
    return nothing
end
function check_each_parameter_domain(
    model::AbstractTimestepModel, parameters::NamedTuple, domains::NamedTuple{names}
) where {names}
    name = first(names)::Symbol
    if haskey(parameters, name)
        check_value_in_domain(model, name, getfield(parameters, name), first(domains))
    end
    return check_each_parameter_domain(model, parameters, Base.tail(domains))
end

"""
    check_value_in_domain(model, name, value, domain)

Throw an [`InvalidParametersError`](@ref) if `value`, the value of the parameter `name` of
`model`, is outside `domain`, a `NamedTuple` with any of `lower`, `upper` and `discrete`.
Every element of an array of real numbers is checked, and a value that is neither is left
alone. Used by [`check_each_parameter_domain`](@ref).
"""
function check_value_in_domain(
    model::AbstractTimestepModel, name::Symbol, value::Real, domain::NamedTuple
)
    lower = get(domain, :lower, -Inf)
    upper = get(domain, :upper, Inf)
    if !(lower <= value <= upper)
        throw(
            InvalidParametersError(
                "The parameter $name of $(describe_model(model)) is $value, which is outside its domain, from $lower to $upper.",
            ),
        )
    end
    if get(domain, :discrete, false) && !isinteger(value)
        throw(
            InvalidParametersError(
                "The parameter $name of $(describe_model(model)) is $value, but it is discrete, so it must be a whole number.",
            ),
        )
    end
    return nothing
end

function check_value_in_domain(
    model::AbstractTimestepModel,
    name::Symbol,
    values::AbstractArray{<:Real},
    domain::NamedTuple,
)
    foreach(value -> check_value_in_domain(model, name, value, domain), values)
    return nothing
end

check_value_in_domain(::AbstractTimestepModel, ::Symbol, value, ::NamedTuple) = nothing

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
