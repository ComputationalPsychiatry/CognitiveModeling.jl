## Checking parameters against their domains ##
"""
    check_parameters_in_domains(model, parameters)

Throw a [`ParameterDomainError`](@ref) if a parameter in `parameters` is outside its domain
in [`parameter_domains`](@ref) of `model`: of another type than the domain's, below its
lower bound or above its upper bound, not one of its possible values, or not satisfying its
constraint. Real numbers and values that aren't numbers are checked, and so is every element
of an array or a tuple. A number that isn't a real number is left alone. It is an
`ArgumentError` when the domains aren't [`Domain`](@ref)s under the names of parameters in
the model's attributes. The parameters of a model with submodels are given flat, and each
submodel's are checked against its own domains, with
[`check_parameters_of_submodels_in_domains`](@ref). Used by [`simulate_session`](@ref),
[`simulate_timestep`](@ref), [`InstantiatedModel`](@ref) and fitted sessions, before
anything runs, unless they are given `check_parameter_domains = false`.
"""
function check_parameters_in_domains(model::AbstractSubmodel, parameters::NamedTuple)
    check_parameters_of_submodels_in_domains(model, bundle_parameters(model, parameters))
    return nothing
end

#The names of the domains are compared with those in the model's attributes when the code is compiled, since both are in the types, so the check costs nothing when a model runs
"""
    check_domains_fit_model(model, domains)

Throw an `ArgumentError` if `domains`, what [`parameter_domains`](@ref) returns for `model`,
isn't a `NamedTuple` of [`Domain`](@ref)s, or names a parameter that isn't in the
model's attributes, from [`attribute_names`](@ref). Used by
[`check_parameters_in_domains`](@ref), [`get_parameter_domains`](@ref) and the display of a
model.
"""
check_domains_fit_model(::AbstractSubmodel, ::NamedTuple{()}) = nothing
function check_domains_fit_model(
    model::AbstractSubmodel, ::NamedTuple{names,<:Tuple{Vararg{Domain}}}
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
function check_domains_fit_model(model::AbstractSubmodel, domains)
    return throw(
        ArgumentError(
            "parameter_domains of $(describe_model(model)) returns $(repr(domains)), but it must return a NamedTuple with a Domain under the name of each parameter that has a domain, such as (; learning_rate = Domain(; lower = 0, upper = 1)).",
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
    model::AbstractSubmodel, parameters::NamedTuple, domains::NamedTuple{names}
) where {names}

    #Read the values of the parameters with a domain as a tuple, in the order of the domains. The names are in the type of the domains, so nothing is searched for when the model runs
    values_with_domains = Tuple(NamedTuple{names}(parameters))

    #Check them one at a time, as tuples: a recursion over the NamedTuples themselves isn't inferred from three domains on, so that every check allocated
    return check_values_in_domains(model, names, values_with_domains, Tuple(domains))
end

"""
    check_values_in_domains(model, names, values, domains)

Check each of `values`, a tuple with the values of the parameters `names`, against the
domain at the same place in `domains`, a tuple of [`Domain`](@ref)s, with
[`check_value_in_domain`](@ref), one after the other. Used by
[`check_each_parameter_in_domain`](@ref).
"""
check_values_in_domains(::AbstractSubmodel, ::Tuple{}, ::Tuple{}, ::Tuple{}) = nothing

#Each call checks the first value and calls itself with the remaining ones, so that the compiler knows the type of every value and of every domain
function check_values_in_domains(
    model::AbstractSubmodel, names::Tuple, values::Tuple, domains::Tuple
)
    check_value_in_domain(model, first(names), first(values), first(domains))
    return check_values_in_domains(
        model, Base.tail(names), Base.tail(values), Base.tail(domains)
    )
end

## What is checked against a domain ##
"""
    DataValue(kind, index, in_table)

A value in the data of a fitted session, which [`check_value_in_domain`](@ref) checks
against the domain of the input or the output of the cognitive model that it is: an
`:observation` or an `:action`, as `kind` says, at `index`, a row of the table when
`in_table` is `true`, and a timestep of the session otherwise. Used by
[`check_data_in_domains`](@ref).
"""
struct DataValue
    kind::Symbol
    index::Int
    in_table::Bool
end

"""
    describe_value(checked, name)

Return what is checked against a domain, as errors write it: `the parameter learning_rate of`
the model when `checked` is a model, and `the action choice in row 7 of the table` or
`the observation reward at timestep 3` when it is a [`DataValue`](@ref). Used by the
functions that throw the errors of [`check_value_in_domain`](@ref).
"""
function describe_value(model::AbstractSubmodel, name::Symbol)
    return "the parameter $name of $(describe_model(model))"
end
function describe_value(checked::DataValue, name::Symbol)
    place = if checked.in_table
        "in row $(checked.index) of the table"
    else
        "at timestep $(checked.index)"
    end
    return "the $(checked.kind) $name $place"
end

"""
    domain_error(checked, message)

Return the error for a value outside its domain, with `message`: a
[`ParameterDomainError`](@ref) for a parameter of a model, which stops a fit, and an
`ArgumentError` for a [`DataValue`](@ref). Used by the functions that throw the errors of
[`check_value_in_domain`](@ref).
"""
domain_error(::AbstractSubmodel, message::String) = ParameterDomainError(message)
domain_error(::DataValue, message::String) = ArgumentError(message)

"""
    check_value_in_domain(checked, name, value, domain)

Throw an error if `value`, the value of `name`, is outside `domain`, a [`Domain`](@ref): a
[`ParameterDomainError`](@ref) when `checked` is a model and `name` one of its parameters,
and an `ArgumentError` when `checked` is a [`DataValue`](@ref), a value in the data of a
fitted session. A domain of single values checks
`value` with [`check_single_value`](@ref). A domain of arrays takes an array with the number
of dimensions of its type, and a domain of tuples a tuple of the length of its type, if it
has one, and each element is then checked with `check_single_value` against the type of the
elements. The whole value is then checked against the domain's constraint, with
[`check_constraint`](@ref). A number that isn't a real number, and an array of them, is left
alone. Used by [`check_values_in_domains`](@ref) and [`check_data_in_domains`](@ref).
"""
function check_value_in_domain(
    checked::Union{AbstractSubmodel,DataValue}, name::Symbol, value, domain::Domain{Ttype}
) where {Ttype}
    check_value_in_domain(checked, name, value, domain, kind_of_values(Ttype))
    check_constraint(checked, name, value, domain.constraint)
    return nothing
end

#Which kind of values a domain takes is in its type, so it is chosen when the code is compiled and costs nothing when a model runs
"""
    SingleValues()

The kind of values of a [`Domain`](@ref) whose type isn't an array type or a tuple
type, as [`kind_of_values`](@ref) reads it. Used by [`check_value_in_domain`](@ref).
"""
struct SingleValues end

"""
    ArrayValues()

The kind of values of a [`Domain`](@ref) whose type is an array type, as
[`kind_of_values`](@ref) reads it. Used by [`check_value_in_domain`](@ref).
"""
struct ArrayValues end

"""
    TupleValues()

The kind of values of a [`Domain`](@ref) whose type is a tuple type, as
[`kind_of_values`](@ref) reads it. Used by [`check_value_in_domain`](@ref).
"""
struct TupleValues end

"""
    kind_of_values(type)

Return `ArrayValues()` for an array type, `TupleValues()` for a tuple type and
`SingleValues()` for any other type, the kind of values that a [`Domain`](@ref)
with `type` takes. Used by [`check_value_in_domain`](@ref).
"""
kind_of_values(::Type{<:AbstractArray}) = ArrayValues()
kind_of_values(::Type{<:Tuple}) = TupleValues()
kind_of_values(::Type) = SingleValues()

#A single value is checked as it is
function check_value_in_domain(
    checked::Union{AbstractSubmodel,DataValue},
    name::Symbol,
    value,
    domain::Domain{Ttype},
    ::SingleValues,
) where {Ttype}
    return check_single_value(checked, name, value, Ttype, domain, nothing)
end

#An array has the number of dimensions of the domain's type, and each of its elements is checked at its position. Which kind of elements it has is in its type, so an array that is left alone costs nothing
function check_value_in_domain(
    checked::Union{AbstractSubmodel,DataValue},
    name::Symbol,
    values,
    domain::Domain{Ttype,Telement},
    ::ArrayValues,
) where {Ttype,Telement}

    #A number that isn't a real number can't be compared with anything, such as a number that Reactant.jl traces while it compiles
    values isa Number && !(values isa Real) && return nothing

    #The value is an array with as many dimensions as the domain's type
    has_dimensions_of(values, Ttype) ||
        throw_not_of_type(checked, name, values, Ttype, nothing)

    #An array of numbers that aren't real numbers is left alone, since it can't be read one element at a time while Reactant.jl compiles
    eltype(values) <: Number && !(eltype(values) <: Real) && return nothing

    #Check each element as a single value, at its position in the array
    for (position, element) in pairs(values)
        check_single_value(checked, name, element, Telement, domain, position)
    end
    return nothing
end

#A tuple has the length of the domain's type, if it has one, and each of its elements is checked at its position
function check_value_in_domain(
    checked::Union{AbstractSubmodel,DataValue},
    name::Symbol,
    values,
    domain::Domain{Ttype,Telement},
    ::TupleValues,
) where {Ttype,Telement}

    #A number that isn't a real number can't be compared with anything, such as a number that Reactant.jl traces while it compiles
    values isa Number && !(values isa Real) && return nothing

    #The value is a tuple with the length of the domain's type
    has_length_of(values, Ttype) || throw_not_of_type(checked, name, values, Ttype, nothing)

    #Check each element as a single value, at its position in the tuple
    return check_tuple_elements(checked, name, values, Telement, domain, 1)
end

"""
    has_dimensions_of(value, type)

Return whether `value` is an array with as many dimensions as the array type `type`, of any
kind, which takes every number of them when it has none, as `AbstractArray{Real}`. Used by
[`check_value_in_domain`](@ref).
"""
function has_dimensions_of(values::AbstractArray, ::Type{T}) where {T}
    n_dimensions = dimensions_of(T)
    return n_dimensions === nothing || ndims(values) == n_dimensions
end
has_dimensions_of(value, ::Type) = false

"""
    has_length_of(value, type)

Return whether `value` is a tuple of the length of the tuple type `type`, which takes every
length when it has none, as `Tuple{Vararg{Real}}`. Used by [`check_value_in_domain`](@ref).
"""
function has_length_of(values::Tuple, ::Type{T}) where {T}
    tuple_length = length_of(T)
    return tuple_length === nothing || length(values) == tuple_length
end
has_length_of(value, ::Type) = false

"""
    dimensions_of(type)

Return the number of dimensions of the array type `type`, or `nothing` when the type leaves
it open, as `AbstractArray{Real}` does. Used by [`has_dimensions_of`](@ref).
"""
dimensions_of(::Type{<:AbstractArray{<:Any,N}}) where {N} = N
dimensions_of(::Type{<:AbstractArray}) = nothing

"""
    length_of(type)

Return the length of the tuple type `type`, or `nothing` when the type leaves it open, as
`Tuple{Vararg{Real}}` does. Used by [`has_length_of`](@ref) and the display of a model.
"""
length_of(::Type{<:NTuple{N,Any}}) where {N} = N
length_of(::Type{<:Tuple}) = nothing

"""
    check_tuple_elements(checked, name, values, type, domain, position)

Check each element of the tuple `values`, the value of `name`, as `checked` describes it, with
[`check_single_value`](@ref) against `type` and `domain`, the first at `position`. Used by
[`check_value_in_domain`](@ref).
"""
check_tuple_elements(checked, name::Symbol, ::Tuple{}, ::Type, domain, position) = nothing

#Each call checks the first element and calls itself with the remaining ones, so that the compiler knows the type of every element
function check_tuple_elements(
    checked, name::Symbol, values::Tuple, ::Type{T}, domain, position
) where {T}
    check_single_value(checked, name, first(values), T, domain, position)
    return check_tuple_elements(checked, name, Base.tail(values), T, domain, position + 1)
end

"""
    check_single_value(checked, name, value, type, domain, position)

Throw the error of [`check_value_in_domain`](@ref) if `value` isn't a value of `type` within
`domain`, a
[`Domain`](@ref): a real number that `type` doesn't take, as
[`takes_value`](@ref) decides, or that is below the lower bound or above the upper bound,
which a `NaN` always is, and which [`is_beyond_bounds`](@ref) decides for a number that
isn't one of Julia's own, or another value that isn't of `type`, or a value that isn't one
of the domain's possible values, as [`is_possible_value`](@ref) decides. A number that isn't
a real number is left alone. `value` is the value of `name`, as `checked` describes it, when
`position` is `nothing`, and its element at `position` otherwise. Used by
[`check_value_in_domain`](@ref) and [`check_tuple_elements`](@ref).
"""
function check_single_value(
    checked, name::Symbol, value::Real, ::Type{T}, domain::Domain, position
) where {T}
    takes_value(T, value) || throw_not_of_type(checked, name, value, T, position)
    if !(domain.lower <= value <= domain.upper) && is_beyond_bounds(value, domain)
        throw_outside_bounds(checked, name, value, domain, position)
    end
    is_possible_value(value, domain.possible_values) ||
        throw_not_possible_value(checked, name, value, domain, position)
    return nothing
end

#Any other value is of the type, except a number that isn't a real number, which can't be compared with anything, such as a number that Reactant.jl traces while it compiles
function check_single_value(
    checked, name::Symbol, value, ::Type{T}, domain::Domain, position
) where {T}
    value isa Number && return nothing
    value isa T || throw_not_of_type(checked, name, value, T, position)
    is_possible_value(value, domain.possible_values) ||
        throw_not_possible_value(checked, name, value, domain, position)
    return nothing
end

#Whether a domain has possible values is in its type, so a domain without them costs nothing. A number is compared by its value, as == compares it, so that 4.0 is among (2, 4, 8)
"""
    is_possible_value(value, possible_values)

Return whether `value` is one of `possible_values`, the possible values of a
[`Domain`](@ref), as `in` decides, and `true` when the domain has none. Used by
[`check_single_value`](@ref).
"""
is_possible_value(value, ::Nothing) = true
function is_possible_value(value, possible_values::Union{Tuple,AbstractRange})
    return value in possible_values
end

"""
    check_constraint(checked, name, value, constraint)

Throw the error of [`check_value_in_domain`](@ref) if `constraint`, the constraint of the
domain of `name`, as `checked` describes it, returns `false` for its `value`, and an
`ArgumentError` if it returns something other than `true` or `false`. Nothing is checked without a constraint, or
for a value with numbers that aren't real numbers, as [`holds_unreal_numbers`](@ref) decides.
Used by [`check_value_in_domain`](@ref).
"""
check_constraint(checked, name::Symbol, value, ::Nothing) = nothing
function check_constraint(checked, name::Symbol, value, constraint)

    #A number that isn't a real number can't be compared with anything, such as a number that Reactant.jl traces while it compiles
    holds_unreal_numbers(value) && return nothing

    #The constraint says whether the value satisfies it
    satisfies_constraint = constraint(value)
    satisfies_constraint isa Bool || throw_constraint_not_true_or_false(
        checked, name, value, constraint, satisfies_constraint
    )
    satisfies_constraint || throw_constraint_not_satisfied(checked, name, value, constraint)
    return nothing
end

"""
    holds_unreal_numbers(value)

Return whether `value` is a number that isn't a real number, or an array or a tuple with
such numbers in it, such as the numbers that Reactant.jl traces while it compiles. Used by
[`check_constraint`](@ref).
"""
holds_unreal_numbers(value::Number) = !(value isa Real)
function holds_unreal_numbers(values::AbstractArray)
    return eltype(values) <: Number && !(eltype(values) <: Real)
end
holds_unreal_numbers(values::Tuple) = any(holds_unreal_numbers, values)
holds_unreal_numbers(value) = false

#A number of automatic differentiation can compare as beyond a bound that it equals: ForwardDiff.jl orders numbers of the same value by their derivatives, so that a learning rate of exactly 1 whose derivative is above zero compares as above 1, as it does when a transformation such as logistic reaches 1 while a sampler explores. Such a number is compared with the nearest number beyond each bound instead, which its value alone decides. Julia's own numbers are compared with the bounds as they are
"""
    is_beyond_bounds(value, domain)

Return whether `value`, a real number which didn't compare as within the bounds of `domain`,
a [`Domain`](@ref), is beyond them. One of Julia's own number types always is. A
number of another type, such as one of automatic differentiation, is when it is `NaN`, or
when it is at or beyond the nearest floating-point number below the lower bound or above
the upper bound. Used by [`check_single_value`](@ref).
"""
function is_beyond_bounds(
    ::Union{AbstractFloat,Integer,Rational,AbstractIrrational}, ::Domain
)
    return true
end
function is_beyond_bounds(value::Real, domain::Domain)
    isnan(value) && return true
    is_below = isfinite(domain.lower) && value <= prevfloat(float(domain.lower))
    is_above = isfinite(domain.upper) && value >= nextfloat(float(domain.upper))
    return is_below || is_above
end

"""
    takes_value(type, value)

Return whether a [`Domain`](@ref) takes the real number `value` as a value of
`type`: when it is of the type, when the type is a type of whole numbers and it is a whole
number in another number type, such as `3.0`, and when the type is `Bool` and it is 0 or 1.
Used by [`check_single_value`](@ref).
"""
takes_value(::Type{T}, value::Real) where {T} = value isa T
takes_value(::Type{T}, value::Real) where {T<:Integer} = value isa T || isinteger(value)
takes_value(::Type{Bool}, value::Real) = value isa Bool || iszero(value) || isone(value)

#Functions of their own, which aren't inlined, so that the messages are only built when there is an error
"""
    throw_not_of_type(checked, name, value, type, position)

Throw the error of [`check_value_in_domain`](@ref) for `name`, as `checked` describes it,
whose `value` isn't of `type`. The value is the whole value when `position` is `nothing`,
and its element at `position` otherwise. Used by [`check_value_in_domain`](@ref) and
[`check_single_value`](@ref).
"""
@noinline function throw_not_of_type(checked, name::Symbol, value, type::Type, position)

    #Say what the domain takes: a number that isn't whole is told so, and not that it is of another type than a type of whole numbers
    whose_type = position === nothing ? "its type" : "the type of its elements"
    taken, reason = if type === Bool
        "true and false, or 0 and 1", ""
    elseif type <: Integer && value isa Real
        "whole numbers", ", as $whose_type is $type"
    else
        "values of type $type", ""
    end
    message = if position === nothing
        "$(uppercasefirst(describe_value(checked, name))) is $(repr(value)), of type $(typeof(value)), but its domain takes $taken$reason."
    else
        "$(uppercasefirst(describe_value(checked, name))) has $(describe_element(name, position)) = $(repr(value)), of type $(typeof(value)), but its domain takes $taken there$reason."
    end
    return throw(domain_error(checked, message))
end
"""
    throw_outside_bounds(checked, name, value, domain, position)

Throw the error of [`check_value_in_domain`](@ref) for `name`, as `checked` describes it,
whose `value` is outside the bounds of `domain`. The value is the whole value when
`position` is `nothing`, and its element at `position` otherwise. Used by
[`check_single_value`](@ref).
"""
@noinline function throw_outside_bounds(
    checked, name::Symbol, value, domain::Domain, position
)
    subject = if position === nothing
        "is $value"
    else
        "has $(describe_element(name, position)) = $value"
    end
    return throw(
        domain_error(
            checked,
            "$(uppercasefirst(describe_value(checked, name))) $subject, which is outside its domain, from $(domain.lower) to $(domain.upper).",
        ),
    )
end

"""
    throw_not_possible_value(checked, name, value, domain, position)

Throw the error of [`check_value_in_domain`](@ref) for `name`, as `checked` describes it,
whose `value` isn't one of the possible values of `domain`. The value is the whole value
when `position` is `nothing`, and its element at `position` otherwise. Used by
[`check_single_value`](@ref).
"""
@noinline function throw_not_possible_value(
    checked, name::Symbol, value, domain::Domain, position
)
    subject = if position === nothing
        "is $(repr(value))"
    else
        "has $(describe_element(name, position)) = $(repr(value))"
    end
    return throw(
        domain_error(
            checked,
            "$(uppercasefirst(describe_value(checked, name))) $subject, which isn't one of its possible values, $(repr(domain.possible_values)).",
        ),
    )
end

"""
    throw_constraint_not_satisfied(checked, name, value, constraint)

Throw the error of [`check_value_in_domain`](@ref) for `name`, as `checked` describes it,
whose `value` doesn't satisfy `constraint`, the constraint of its domain. Used by
[`check_constraint`](@ref).
"""
@noinline function throw_constraint_not_satisfied(checked, name::Symbol, value, constraint)
    return throw(
        domain_error(
            checked,
            "$(uppercasefirst(describe_value(checked, name))) is $(repr(value)), which doesn't satisfy $(describe_constraint(constraint)) of its domain.",
        ),
    )
end

"""
    throw_constraint_not_true_or_false(checked, name, value, constraint, returned)

Throw the `ArgumentError` for `constraint`, the constraint of the domain of `name`, as
`checked` describes it, which returned `returned` for its `value`, which is neither `true`
nor `false`. Used by [`check_constraint`](@ref).
"""
@noinline function throw_constraint_not_true_or_false(
    checked, name::Symbol, value, constraint, returned
)
    return throw(
        ArgumentError(
            "$(uppercasefirst(describe_constraint(constraint))) of the domain of $(describe_value(checked, name)) returned $(repr(returned)) for $(repr(value)), but a constraint returns true or false.",
        ),
    )
end

"""
    describe_constraint(constraint)

Return `constraint`, the constraint of a [`Domain`](@ref), as errors write it: with
its name, as in `the constraint sums_to_one`, and without one for an anonymous function, as
in `the constraint`. Used by [`throw_constraint_not_satisfied`](@ref) and
[`throw_constraint_not_true_or_false`](@ref).
"""
function describe_constraint(constraint)
    name = name_of_constraint(constraint)
    return name === nothing ? "the constraint" : "the constraint $name"
end

"""
    describe_element(name, position)

Return the element at `position` of the value of `name` as it is indexed, such as
`choice_probabilities[3]`, or `weights[2, 1]` for a position in a matrix. Used by
[`throw_not_of_type`](@ref), [`throw_outside_bounds`](@ref) and
[`throw_not_possible_value`](@ref).
"""
describe_element(name::Symbol, position::Integer) = "$name[$position]"
function describe_element(name::Symbol, position::CartesianIndex)
    return "$name[$(join(Tuple(position), ", "))]"
end
