## Displaying models, and describing them in messages ##
"""
    print_attribute_names(io, attributes; shorten=false, domains=(;), defaults=(;))

Print a line for each kind of attribute in `attributes`, a [`ModelAttributes`](@ref), with
`none` if it has no names of that kind. With `shorten`, a kind with more names than
[`shows_every_name`](@ref) allows shows only how many there are. When a parameter has a
[`ParameterDomain`](@ref) in `domains` or a default value in `defaults`, the parameters are
listed one on a line instead, as [`print_parameter_lines`](@ref) writes them.

The `show` methods of a `ModelAttributes`, of a cognitive model and of an environment model
each print a header of their own, and then these lines. Only those of the models shorten
them and give the domains and the default values, since a `ModelAttributes` is where the
full lists are read.
"""
function print_attribute_names(
    io::IO,
    attributes::ModelAttributes;
    shorten::Bool=false,
    domains::NamedTuple=(;),
    defaults::NamedTuple=(;),
)
    for (kind, names) in (
        ("Parameters", attributes.parameters),
        ("States", attributes.states),
        ("Inputs", attributes.inputs),
        ("Outputs", attributes.outputs),
    )
        print(io, "\n", kind, ":")
        if isempty(names)
            print(io, " none")
        elseif kind == "Parameters" &&
            any(name -> has_domain_or_default(name, domains, defaults), names)
            print_parameter_lines(io, names, domains, defaults; shorten)
        elseif shorten && !shows_every_name(io, names)
            print(io, " ", length(names), " (see attribute_names)")
        else
            print(io, " ")
            join(io, names, ", ")
        end
    end
    return nothing
end

"""
    has_domain_or_default(name, domains, defaults)

Return whether the parameter `name` has a [`ParameterDomain`](@ref) in `domains` or a
default value in `defaults`, so that the display of a model has more to show of it than its
name. Used by [`print_attribute_names`](@ref).
"""
function has_domain_or_default(name::Symbol, domains::NamedTuple, defaults::NamedTuple)
    return haskey(defaults, name) ||
           (haskey(domains, name) && getproperty(domains, name) isa ParameterDomain)
end

"""
    print_parameter_lines(io, names, domains, defaults; shorten=false)

Print the parameters `names` one on a line, each with its domain in `domains` after its
name, as [`describe_domain`](@ref) writes it, and then its default value in `defaults`, as
in `inverse_temperature ∈ [0, ∞) (default 2.0)`. A parameter without either has its name
alone. With `shorten`, only as many are listed as [`n_parameter_lines`](@ref) allows, and
the last line says how many more there are. Used by [`print_attribute_names`](@ref).
"""
function print_parameter_lines(
    io::IO,
    names::Tuple{Vararg{Symbol}},
    domains::NamedTuple,
    defaults::NamedTuple;
    shorten::Bool=false,
)
    n_listed = shorten ? n_parameter_lines(io, names) : length(names)
    compact_io = IOContext(io, :compact => true)
    for index in 1:n_listed
        name = names[index]
        print(io, "\n  ", name)
        if haskey(domains, name) && getproperty(domains, name) isa ParameterDomain
            print(io, " ∈ ", describe_domain(getproperty(domains, name)))
        end
        if haskey(defaults, name)
            print(io, " (default ")
            show(compact_io, getproperty(defaults, name))
            print(io, ")")
        end
    end
    if n_listed < length(names)
        print(io, "\n  … and ", length(names) - n_listed, " more (see attribute_names)")
    end
    return nothing
end

"""
    n_parameter_lines(io, names)

Return how many of the parameters `names` a display on `io` lists one on a line: all of them
when `io` isn't limited, as when printing to a file, and at most ten otherwise. Used by
[`print_parameter_lines`](@ref).
"""
function n_parameter_lines(io::IO, names::Tuple{Vararg{Symbol}})
    return get(io, :limit, false) ? min(length(names), 10) : length(names)
end

"""
    describe_domain(domain)

Return `domain`, a [`ParameterDomain`](@ref), as the display of a model writes it after the
name of its parameter: an interval for a continuous parameter, such as `[0, 1]` or `[0, ∞)`,
the whole numbers of a discrete one, such as `{1, 2, …}` or `{0, 1, …, 5}`, and the type of
a parameter that isn't a number, such as `AbstractString`. A parameter that is an array or
a tuple has the values of each element, such as `[0, 1] for each element of a vector`. Used
by [`print_parameter_lines`](@ref).
"""
function describe_domain(domain::ParameterDomain{Ttype,Telement}) where {Ttype,Telement}

    #The values of a single value, or of each element of an array or a tuple
    values = describe_values(Telement, domain.lower, domain.upper)
    Ttype <: AbstractArray || Ttype <: Tuple || return values
    return "$values for each element of $(describe_container(Ttype))"
end

"""
    describe_container(type)

Return the array type or tuple type `type` as the display of a model writes it: `a vector`,
`a matrix`, `an array`, `a tuple`, or `a tuple of 3` for a type with a length. Used by
[`describe_domain`](@ref).
"""
function describe_container(type::Type)
    type <: AbstractVector && return "a vector"
    type <: AbstractMatrix && return "a matrix"
    type <: AbstractArray && return "an array"
    tuple_length = length_of(type)
    return tuple_length === nothing ? "a tuple" : "a tuple of $tuple_length"
end

"""
    describe_values(type, lower, upper)

Return the values of `type` between `lower` and `upper`, as [`describe_domain`](@ref)
writes them for a single value or for each element of an array or a tuple.
"""
function describe_values(::Type{Ttype}, lower, upper) where {Ttype}

    #A parameter that isn't a number can be any value of its type
    Ttype <: Real || return string(Ttype)
    Ttype === Bool && return "{false, true}"

    #A continuous parameter has an interval, closed at a bound and open at infinity
    if !(Ttype <: Integer)
        return string(
            isfinite(lower) ? "[$lower" : "(-∞", ", ", isfinite(upper) ? "$upper]" : "∞)"
        )
    end

    #A discrete parameter has the whole numbers within its bounds
    first_value = isfinite(lower) ? ceil(Int, lower) : nothing
    last_value = isfinite(upper) ? floor(Int, upper) : nothing
    if first_value === nothing && last_value === nothing
        return "{…, -1, 0, 1, …}"
    elseif last_value === nothing
        return "{$first_value, $(first_value + 1), …}"
    elseif first_value === nothing
        return "{…, $(last_value - 1), $last_value}"
    elseif last_value - first_value <= 2
        return string("{", join(first_value:last_value, ", "), "}")
    else
        return "{$first_value, $(first_value + 1), …, $last_value}"
    end
end

"""
    shows_every_name(io, names)

Return whether a display on `io` lists every one of `names`: always when `io` isn't limited,
as when printing to a file, and otherwise when there are at most five. Used by the `show`
methods of a model and of an [`InstantiatedModel`](@ref), which give how many there are
instead.
"""
shows_every_name(io::IO, names) = !get(io, :limit, false) || length(names) <= 5

"""
    print_current_values(io, label, current_values, field_name)

Print a line with `label` and `current_values`, the fields of a model or the parameters or
the state of an [`InstantiatedModel`](@ref), in a `NamedTuple`: each name with its value,
`none` when it is empty, and only how many there are, with where to read them, when
[`shows_every_name`](@ref) doesn't allow listing them all. Values are printed compactly, and
large arrays are shortened in the REPL. Used by the `show` methods of a model and of an
`InstantiatedModel`.
"""
function print_current_values(io::IO, label, current_values::NamedTuple, field_name)
    print(io, "\n", label, ": ")
    compact_io = IOContext(io, :compact => true)
    if isempty(current_values)
        print(io, "none")
    elseif !shows_every_name(io, keys(current_values))
        print(io, length(current_values), " (see ", field_name, ")")
    else
        for (i, name) in enumerate(keys(current_values))
            i > 1 && print(io, ", ")
            print(io, name, " = ")
            show(compact_io, current_values[name])
        end
    end
    return nothing
end

#Shows the names on one line each in the REPL and notebooks
function Base.show(io::IO, ::MIME"text/plain", attributes::ModelAttributes)
    print(io, "ModelAttributes")
    print_attribute_names(io, attributes)
    return nothing
end

#Shows the kind and the type of a model in the REPL and notebooks, the values of its fields, and the names of its attributes, with the domain and the default value of each parameter that has one
function Base.show(io::IO, ::MIME"text/plain", model::AbstractTimestepModel)

    #Print the kind and the type of the model
    print(io, "-- ", uppercasefirst(model_kind(model)), " --")
    print(io, "\nType: ", nameof(typeof(model)))

    #Print the values of the model's fields, if it has any
    if fieldcount(typeof(model)) > 0
        field_names = fieldnames(typeof(model))
        fields = NamedTuple{field_names}(map(name -> getfield(model, name), field_names))
        print_current_values(io, "Fields", fields, "fieldnames")
    end

    #Print that the model has no method of its own for its names, or the names
    if has_attribute_names(model)
        attributes = check_type_attribute_names(attribute_names(model), model)
        print_attribute_names(
            io,
            attributes;
            shorten=true,
            domains=domains_to_display(model),
            defaults=defaults_to_display(model),
        )
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end

#Shows the type and the kind of a model on one line, as inside a vector
function Base.show(io::IO, model::AbstractTimestepModel)
    print(io, nameof(typeof(model)), " (", model_kind(model), ")")
    return nothing
end

"""
    domains_to_display(model)
    defaults_to_display(model)

Return the domains and the default values of the parameters of `model` for its display: what
[`parameter_domains`](@ref) and [`default_parameters`](@ref) return when it is a
`NamedTuple`, and none otherwise, so that a model whose domains or default values aren't
right is still shown. Used by the `show` method of a model.
"""
function domains_to_display(model::AbstractTimestepModel)
    domains = parameter_domains(model)
    return domains isa NamedTuple ? domains : (;)
end
function defaults_to_display(model::AbstractTimestepModel)
    defaults = default_parameters(model)
    return defaults isa NamedTuple ? defaults : (;)
end

"""
    model_kind(model)

Return which kind of model `model` is, as a display names it: `"cognitive model"`,
`"environment model"`, or `"model"` for one of neither kind. Used by the `show` methods of a
model and of an [`InstantiatedModel`](@ref).
"""
model_kind(::AbstractCognitiveModel) = "cognitive model"
model_kind(::AbstractEnvironmentModel) = "environment model"
model_kind(::AbstractTimestepModel) = "model"

"""
    describe_model(model)

Return how `model` is named in an error message: as a cognitive model or as an environment
model, with its type. Used in the error messages of the checks on models, in simulations and
in fitted sessions.
"""
function describe_model(cognitive_model::AbstractCognitiveModel)
    return "a cognitive model of type $(typeof(cognitive_model))"
end
function describe_model(environment::AbstractEnvironmentModel)
    return "an environment model of type $(typeof(environment))"
end
describe_model(model::AbstractTimestepModel) = "a model of type $(typeof(model))"
