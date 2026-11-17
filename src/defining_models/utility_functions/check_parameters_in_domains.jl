## Checking parameters against their domains ##
"""
    check_parameters_in_domains(model, parameters)

Throw a [`ParameterDomainError`](@ref) if a parameter in `parameters` is outside its domain
in [`parameter_domains`](@ref) of `model`: of another type than the domain's, or below its
lower bound or above its upper bound. Real numbers and values that aren't numbers are
checked, and so is every element of an array. A number that isn't a real number is left
alone. It is an `ArgumentError` when the domains aren't [`ParameterDomain`](@ref)s under
the names of parameters in the model's attributes. Used by [`simulate`](@ref),
[`InstantiatedModel`](@ref) and fitted sessions, before anything runs, unless they are given
`check_parameter_domains = false`.
"""
function check_parameters_in_domains(model::AbstractTimestepModel, parameters::NamedTuple)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    check_each_parameter_in_domain(model, parameters, domains)
    return nothing
end

#The names of the domains are compared with those in the model's attributes when the code is compiled, since both are in the types, so the check costs nothing when a model runs
"""
    check_domains_fit_model(model, domains)

Throw an `ArgumentError` if `domains`, what [`parameter_domains`](@ref) returns for `model`,
isn't a `NamedTuple` of [`ParameterDomain`](@ref)s, or names a parameter that isn't in the
model's attributes, from [`attribute_names`](@ref). Used by
[`check_parameters_in_domains`](@ref), [`discrete_parameter_names`](@ref) and the display of
a model.
"""
check_domains_fit_model(::AbstractTimestepModel, ::NamedTuple{()}) = nothing
function check_domains_fit_model(
    model::AbstractTimestepModel, ::NamedTuple{names,<:Tuple{Vararg{ParameterDomain}}}
) where {names}
    names_in_attributes =
        check_type_attribute_names(attribute_names(model), model).parameters
    if !has_every_name(names, names_in_attributes)
        throw(
            ArgumentError(
                "parameter_domains of $(describe_model(model)) has domains for $names, but the parameters in its attributes are $names_in_attributes.",
            ),
        )
    end
    return nothing
end
function check_domains_fit_model(model::AbstractTimestepModel, domains)
    return throw(
        ArgumentError(
            "parameter_domains of $(describe_model(model)) returns $(repr(domains)), but it must return a NamedTuple with a ParameterDomain under the name of each parameter that has a domain, such as (; learning_rate = ParameterDomain(; lower = 0, upper = 1)).",
        ),
    )
end

"""
    check_each_parameter_in_domain(model, parameters, domains)

Check the value in `parameters` of each parameter with a domain in `domains`, with
[`check_values_in_domains`](@ref). `parameters` must have every parameter with a domain,
which it has when its names are those in the model's attributes. Used by
[`check_parameters_in_domains`](@ref).
"""
function check_each_parameter_in_domain(
    model::AbstractTimestepModel, parameters::NamedTuple, domains::NamedTuple{names}
) where {names}

    #Read the values of the parameters with a domain as a tuple, in the order of the domains. The names are in the type of the domains, so nothing is searched for when the model runs
    values_with_domains = Tuple(NamedTuple{names}(parameters))

    #Check them one at a time, as tuples: a recursion over the NamedTuples themselves isn't inferred from three domains on, so that every check allocated
    return check_values_in_domains(model, names, values_with_domains, Tuple(domains))
end

"""
    check_values_in_domains(model, names, values, domains)

Check each of `values`, a tuple with the values of the parameters `names`, against the
domain at the same place in `domains`, a tuple of [`ParameterDomain`](@ref)s, with
[`check_value_in_domain`](@ref), one after the other. Used by
[`check_each_parameter_in_domain`](@ref).
"""
check_values_in_domains(::AbstractTimestepModel, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

#Each call checks the first value and calls itself with the remaining ones, so that the compiler knows the type of every value and of every domain
function check_values_in_domains(
    model::AbstractTimestepModel, names::Tuple, values::Tuple, domains::Tuple
)
    check_value_in_domain(model, first(names), first(values), first(domains))
    return check_values_in_domains(
        model, Base.tail(names), Base.tail(values), Base.tail(domains)
    )
end

"""
    check_value_in_domain(model, name, value, domain)

Throw a [`ParameterDomainError`](@ref) if `value`, the value of the parameter `name` of
`model`, is outside `domain`, a [`ParameterDomain`](@ref): a real number that the domain's
type doesn't take, as [`takes_value`](@ref) decides, or that is below the lower bound or
above the upper bound, which a `NaN` always is, or a value that isn't a number and isn't of
the domain's type. Every element of an array is checked as a value of its own. A number
that isn't a real number, and an array of them, is left alone. Used by
[`check_values_in_domains`](@ref).
"""
function check_value_in_domain(
    model::AbstractTimestepModel, name::Symbol, value::Real, domain::ParameterDomain{Ttype}
) where {Ttype}
    takes_value(Ttype, value) || throw_not_of_type(model, name, value, domain)
    if !(domain.lower <= value <= domain.upper)
        throw_outside_bounds(model, name, value, domain)
    end
    return nothing
end

#Numbers that aren't real numbers can't be compared with the bounds, such as the numbers that Reactant.jl traces while it compiles
function check_value_in_domain(
    ::AbstractTimestepModel, ::Symbol, ::Number, ::ParameterDomain
)
    return nothing
end

#An array has the domain of each of its elements. Which kind of elements it has is in its type, so an array that is left alone costs nothing
function check_value_in_domain(
    model::AbstractTimestepModel,
    name::Symbol,
    values::AbstractArray{Telement},
    domain::ParameterDomain,
) where {Telement}

    #An array of numbers that aren't real numbers is left alone as they are, since it can't be read one element at a time while Reactant.jl compiles
    Telement <: Number && !(Telement <: Real) && return nothing

    #Check each element as a value of its own
    foreach(value -> check_value_in_domain(model, name, value, domain), values)
    return nothing
end

#A value that isn't a number is of the domain's type or not
function check_value_in_domain(
    model::AbstractTimestepModel, name::Symbol, value, domain::ParameterDomain{Ttype}
) where {Ttype}
    value isa Ttype || throw_not_of_type(model, name, value, domain)
    return nothing
end

"""
    takes_value(type, value)

Return whether a [`ParameterDomain`](@ref) with `type` takes the real number `value`: when
it is of the type, when the type is a type of whole numbers and it is a whole number in
another number type, such as `3.0`, and when the type is `Bool` and it is 0 or 1. Used by
[`check_value_in_domain`](@ref).
"""
takes_value(::Type{T}, value::Real) where {T} = value isa T
takes_value(::Type{T}, value::Real) where {T<:Integer} = value isa T || isinteger(value)
takes_value(::Type{Bool}, value::Real) = value isa Bool || iszero(value) || isone(value)

#Functions of their own, which aren't inlined, so that the messages are only built when there is an error
"""
    throw_not_of_type(model, name, value, domain)
    throw_outside_bounds(model, name, value, domain)

Throw the [`ParameterDomainError`](@ref) for the parameter `name` of `model`, whose `value`
isn't of the type of `domain`, or is outside its bounds. Used by
[`check_value_in_domain`](@ref).
"""
@noinline function throw_not_of_type(model, name::Symbol, value, domain::ParameterDomain)

    #Say what the domain takes: a number that isn't whole is told so, and not that it is of another type than a type of whole numbers
    taken = if domain.type === Bool
        "true and false, or 0 and 1"
    elseif domain.type <: Integer && value isa Real
        "whole numbers, as its type is $(domain.type)"
    else
        "values of type $(domain.type)"
    end
    return throw(
        ParameterDomainError(
            "The parameter $name of $(describe_model(model)) is $(repr(value)), of type $(typeof(value)), but its domain takes $taken.",
        ),
    )
end
@noinline function throw_outside_bounds(model, name::Symbol, value, domain::ParameterDomain)
    return throw(
        ParameterDomainError(
            "The parameter $name of $(describe_model(model)) is $value, which is outside its domain, from $(domain.lower) to $(domain.upper).",
        ),
    )
end
