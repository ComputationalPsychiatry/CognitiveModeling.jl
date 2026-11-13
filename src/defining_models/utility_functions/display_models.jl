## Displaying models ##
"""
    print_attribute_names(io, attributes; shorten=false)

Print a line for each kind of attribute in `attributes`, a [`ModelAttributes`](@ref), with
`none` if it has no names of that kind. With `shorten`, a kind with more names than
[`shows_every_name`](@ref) allows shows only how many there are.

The `show` methods of a `ModelAttributes`, of a cognitive model and of an environment model
each print a header of their own, and then these lines. Only those of the models shorten
them, since a `ModelAttributes` is where the full lists are read.
"""
function print_attribute_names(io::IO, attributes::ModelAttributes; shorten::Bool=false)
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
        else
            join(io, names, ", ")
        end
    end
    return nothing
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

#Shows the kind and the type of a model in the REPL and notebooks, the values of its fields, and the names of its attributes
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
        print_attribute_names(io, attributes; shorten=true)
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
    model_kind(model)

Return which kind of model `model` is, as a display names it: `"cognitive model"`,
`"environment model"`, or `"model"` for one of neither kind. Used by the `show` methods of a
model and of an [`InstantiatedModel`](@ref).
"""
model_kind(::AbstractCognitiveModel) = "cognitive model"
model_kind(::AbstractEnvironmentModel) = "environment model"
model_kind(::AbstractTimestepModel) = "model"
