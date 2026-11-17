## Checking what a parameter model can give against the domains of the parameters ##

"""
    check_given_values_in_domains(parameters_model, model)

Throw an informative error if `parameters_model`, a parameter model as
[`prepare_parameters_model`](@ref) prepared it, can give a parameter of `model`, a cognitive
model or an environment model, a value outside its domain in [`parameter_domains`](@ref).
The values in [`parameter_value_ranges`](@ref) are compared with the domains by
[`check_range_in_domain`](@ref), which throws an `ArgumentError`. The values of a
[`FixedParameters`](@ref) and of a [`SessionwiseFixedParameters`](@ref) are checked as
those of a session are when it runs, by [`check_value_in_domain`](@ref), which throws a
[`ParameterDomainError`](@ref), and every part of a [`CompositeParameters`](@ref) is
checked. It is an `ArgumentError` when the domains aren't [`ParameterDomain`](@ref)s under
the names of parameters in the model's attributes. Sessions without an environment have
`NoEnvironment()` as `model`, where nothing is checked. Used by
[`prepare_full_model`](@ref), for sessions that check the domains.
"""
function check_given_values_in_domains(
    parameters_model::AbstractParametersModel, model::AbstractTimestepModel
)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    for (name, range) in pairs(parameter_value_ranges(parameters_model))
        haskey(domains, name) || continue
        check_range_in_domain(model, name, range, getproperty(domains, name))
    end
    return nothing
end

function check_given_values_in_domains(
    parameters_model::FixedParameters, model::AbstractTimestepModel
)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    check_fixed_values_in_domains(model, parameters_model.parameters, domains)
    return nothing
end

function check_given_values_in_domains(
    parameters_model::SessionwiseFixedParameters, model::AbstractTimestepModel
)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    parameters_by_session = parameters_model.parameters_by_session

    #One vector for each parameter is checked element by element, and otherwise the parameters of one session at a time
    if has_vector_for_each_parameter(parameters_by_session)
        check_fixed_values_in_domains(model, parameters_by_session, domains)
    else
        foreach(
            parameters -> check_fixed_values_in_domains(model, parameters, domains),
            values(parameters_by_session),
        )
    end
    return nothing
end

function check_given_values_in_domains(
    parameters_model::CompositeParameters, model::AbstractTimestepModel
)
    foreach(part -> check_given_values_in_domains(part, model), parameters_model.parts)
    return nothing
end

function check_given_values_in_domains(::AbstractParametersModel, ::NoEnvironment)
    return nothing
end

"""
    check_fixed_values_in_domains(model, parameters, domains)

Throw a [`ParameterDomainError`](@ref) if a value in `parameters`, a `NamedTuple` of given
values, is outside the domain of its parameter in `domains`, the domains of `model`, as
[`check_value_in_domain`](@ref) decides. A vector with the values of every session is
checked element by element, and a parameter without a domain is left alone. Used by
[`check_given_values_in_domains`](@ref).
"""
function check_fixed_values_in_domains(
    model::AbstractTimestepModel, parameters::NamedTuple, domains::NamedTuple
)
    for (name, value) in pairs(parameters)
        haskey(domains, name) || continue
        check_value_in_domain(model, name, value, getproperty(domains, name))
    end
    return nothing
end

"""
    check_range_in_domain(model, name, range, domain)

Throw an `ArgumentError` if `range`, the values that a parameter model can give the
parameter `name` of `model`, as [`parameter_value_ranges`](@ref) returns them, reaches
outside `domain`, a [`ParameterDomain`](@ref): when the parameter's values aren't numbers,
when the lowest value is below the lower bound or the highest above the upper bound, as
[`bounds_of_domain`](@ref) gives them, or when the values are continuous and the parameter
is discrete. Used by [`check_given_values_in_domains`](@ref).
"""
function check_range_in_domain(
    model::AbstractTimestepModel,
    name::Symbol,
    range::NamedTuple,
    domain::ParameterDomain{Ttype},
) where {Ttype}

    #A range is of numbers, which a parameter whose type takes no real number can't be given
    if typeintersect(Ttype, Real) === Union{}
        throw(
            ArgumentError(
                "The parameter $name has values of type $Ttype in $(describe_model(model)), but it is given numbers by $(range.source).",
            ),
        )
    end
    lower, upper = bounds_of_domain(domain)
    if range.lower < lower || range.upper > upper
        throw(
            ArgumentError(
                "The parameter $name can be given values from $(range.lower) to $(range.upper), by $(range.source), but its domain in $(describe_model(model)) is from $lower to $upper. A value outside the domain is a ParameterDomainError when a session runs, which stops a fit. Give the parameter a prior or a transformation that stays within the domain, or leave the check out with check_parameter_domains = false in the sessions.",
            ),
        )
    end
    if is_discrete(domain) && !range.discrete
        throw(
            ArgumentError(
                "The parameter $name is discrete in $(describe_model(model)), with values of type $Ttype, but it is given continuous values by $(range.source). A value of another type is a ParameterDomainError when a session runs, which stops a fit.",
            ),
        )
    end
    return nothing
end

"""
    bounds_of_domain(domain)

Return the lowest and the highest value that `domain`, a [`ParameterDomain`](@ref), takes,
as a tuple: its lower and its upper bound, which are no lower than 0 and no higher than 1
for a domain of `Bool`, whose values are `true` and `false`. Used by
[`check_range_in_domain`](@ref).
"""
bounds_of_domain(domain::ParameterDomain) = (domain.lower, domain.upper)
function bounds_of_domain(domain::ParameterDomain{Bool})
    lower = domain.lower < 0 ? 0 : domain.lower
    upper = domain.upper > 1 ? 1 : domain.upper
    return (lower, upper)
end

"""
    value_ranges_of_priors(priors)

Return the values that each prior in `priors`, a `NamedTuple` with a prior under the name of
each parameter, can give, as [`parameter_value_ranges`](@ref) returns them, from
[`distribution_support`](@ref), with the prior as their source. A prior whose values aren't
known is left out. Used by the `parameter_value_ranges` of [`IdenticalParameters`](@ref)
and [`IndependentParameters`](@ref).
"""
function value_ranges_of_priors(priors::NamedTuple)
    ranges = Pair{Symbol,NamedTuple}[]
    for (name, prior) in pairs(priors)
        support = distribution_support(prior)
        support === nothing && continue
        push!(ranges, name => merge(support, (; source="its prior, $prior")))
    end
    return (; ranges...)
end

"""
    transformed_range(transformation, lower, upper)

Return the lowest and the highest value that `transformation` gives for the values from
`lower` to `upper`, as a `NamedTuple` with `lower` and `upper`: what it makes of the two
ends, in their order, which holds for a transformation that only rises or only falls, as
`exp` and `logistic` do. It returns `nothing` when the transformation throws for an end, or
gives something other than a real number. Used by the [`parameter_value_ranges`](@ref) of
[`HierarchicalGroupedParameters`](@ref) and [`HierarchicalRegressionParameters`](@ref).
"""
function transformed_range(transformation, lower::Real, upper::Real)
    ends = try
        (transformation(lower), transformation(upper))
    catch
        return nothing
    end
    all(value -> value isa Real, ends) || return nothing
    return (; lower=min(ends...), upper=max(ends...))
end
