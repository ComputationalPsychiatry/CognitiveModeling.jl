## Displaying models, and describing them in messages ##
"""
    print_attribute_names(io, attributes; shorten=false, domains=(;))

Print a line for each kind of attribute in `attributes`, a [`ModelAttributes`](@ref), with
`none` if it has no names of that kind. With `shorten`, a kind with more names than
[`shows_every_name`](@ref) allows shows only how many there are. A parameter with a
[`ParameterDomain`](@ref) in `domains` has it after its name, as
[`describe_domain`](@ref) writes it, such as `learning_rate ∈ [0, 1]`.

The `show` methods of a `ModelAttributes`, of a cognitive model and of an environment model
each print a header of their own, and then these lines. Only those of the models shorten
them and give the domains, since a `ModelAttributes` is where the full lists are read.
"""
function print_attribute_names(
    io::IO, attributes::ModelAttributes; shorten::Bool=false, domains::NamedTuple=(;)
)
    for (kind, names) in (
        ("Parameters", attributes.parameters),
        ("States", attributes.states),
        ("Inputs", attributes.inputs),
        ("Outputs", attributes.outputs),
    )
        print(io, "\n", kind, ": ")
        if isempty(names)
            print(io, "none")
        elseif shorten && !shows_every_name(io, names)
            print(io, length(names), " (see attribute_names)")
        elseif kind == "Parameters"
            join(io, (name_with_domain(name, domains) for name in names), ", ")
        else
            join(io, names, ", ")
        end
    end
    return nothing
end

"""
    name_with_domain(name, domains)

Return the name of a parameter as the display of a model writes it: followed by its domain
in `domains`, as in `learning_rate ∈ [0, 1]`, when it has a [`ParameterDomain`](@ref) there,
and alone otherwise. Used by [`print_attribute_names`](@ref).
"""
function name_with_domain(name::Symbol, domains::NamedTuple)
    haskey(domains, name) || return string(name)
    domain = getproperty(domains, name)
    domain isa ParameterDomain || return string(name)
    return string(name, " ∈ ", describe_domain(domain))
end

"""
    describe_domain(domain)

Return `domain`, a [`ParameterDomain`](@ref), as the display of a model writes it after the
name of its parameter: an interval for a continuous parameter, such as `[0, 1]` or
`[0, ∞)`, the whole numbers of a discrete one, such as `{1, 2, …}` or `{0, 1, …, 5}`, and the
type of a parameter that isn't a number, such as `AbstractString`. Used by
[`name_with_domain`](@ref).
"""
function describe_domain(domain::ParameterDomain{Ttype}) where {Ttype}
    lower, upper = domain.lower, domain.upper

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

#Shows the kind and the type of a model in the REPL and notebooks, the values of its fields, and the names of its attributes, with the domain of each parameter that has one
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
            io, attributes; shorten=true, domains=domains_to_display(model)
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

Return the domains of the parameters of `model` for its display: what
[`parameter_domains`](@ref) returns when it is a `NamedTuple`, and none otherwise, so that
a model whose domains aren't right is still shown. Used by the `show` method of a model.
"""
function domains_to_display(model::AbstractTimestepModel)
    domains = parameter_domains(model)
    return domains isa NamedTuple ? domains : (;)
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
