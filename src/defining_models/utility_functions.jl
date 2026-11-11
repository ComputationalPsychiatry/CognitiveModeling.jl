## Describing a model in messages ##
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
describe_model(model::AbstractModel) = "a model of type $(typeof(model))"

## Checking a model's own methods ##
"""
    check_type_attribute_names(attributes, model)

Return `attributes` if it is the [`ModelAttributes`](@ref) that a method of
[`attribute_names`](@ref) must return, and throw an informative error otherwise. `model` is
the cognitive model or the environment model whose method returned it. Used wherever the
names of a model are read: by the checks of simulations, instantiated models, save histories
and fitted sessions, and by the display of a model.
"""
function check_type_attribute_names(attributes::ModelAttributes, ::AbstractModel)
    return attributes
end

function check_type_attribute_names(attributes, model::AbstractModel)
    return error(
        "The attribute_names method of $(describe_model(model)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).",
    )
end

"""
    check_model_output(output, model)

Return `output` if it is the `ModelOutput` that an `apply_model` method of `model` must
return, and throw an error otherwise. Used on each timestep of [`simulate`](@ref),
[`simulate!`](@ref) and fitted sessions.
"""
function check_model_output(output::ModelOutput, ::AbstractModel)
    return output
end

#Untyped, so that there is always a method, whatever the model is inferred as
function check_model_output(output, model)
    return error(
        "The apply_model method of $(describe_model(model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).",
    )
end

"""
    has_attribute_names(model)

Return whether `model`, a cognitive model or an environment model, has its own method of
[`attribute_names`](@ref), instead of the fallback which throws. Used by the display of a
model, which shows one without a method of its own as `not specified`.
"""
function has_attribute_names(model::AbstractModel)
    method_signature = which(attribute_names, (typeof(model),)).sig
    return method_signature != Tuple{typeof(attribute_names),AbstractModel}
end

## Comparing names ##
"""
    has_name(name, names)
    has_every_name(names, other_names)

Return whether `name` is among the tuple `names`, and whether every name in the tuple
`names` is among `other_names`. Used by [`unique_names`](@ref), and by the checks that a
name map, a save history or the data has names that a model lists.
"""
has_name(name::Symbol, names::Tuple{}) = false
function has_name(name::Symbol, names::Tuple)
    return name === first(names) || has_name(name, Base.tail(names))
end
has_every_name(names::Tuple{}, other_names::Tuple) = true
function has_every_name(names::Tuple, other_names::Tuple)
    return has_name(first(names), other_names) &&
           has_every_name(Base.tail(names), other_names)
end

"""
    same_names(names, other_names)

Return whether two tuples of names have the same names, in any order. Used by the checks of
names in simulations and fitted sessions, such as [`check_names_fit`](@ref) and
[`check_match_attribute_names`](@ref).
"""
function same_names(names, other_names)
    return has_every_name(names, other_names) && has_every_name(other_names, names)
end

## Checking values against a model's names ##
"""
    check_match_attribute_names(model, parameters, state, previous_output)

Throw an informative error if the names of the parameters, of the previous output, or of a
state that is a `NamedTuple`, aren't those that [`attribute_names`](@ref) lists for `model`,
a cognitive model or an environment model. Used by [`InstantiatedModel`](@ref),
[`simulate`](@ref) and fitted sessions, before anything runs.
"""
function check_match_attribute_names(
    model::AbstractModel, parameters, state, previous_output
)
    attributes = check_type_attribute_names(attribute_names(model), model)
    if !same_names(keys(parameters), attributes.parameters)
        throw(
            ArgumentError(
                "The parameters are named $(keys(parameters)), but attribute_names lists the parameters of $(describe_model(model)) as $(attributes.parameters).",
            ),
        )
    end
    if !(
        previous_output isa NamedTuple &&
        same_names(keys(previous_output), attributes.outputs)
    )
        throw(
            ArgumentError(
                "The previous output is $previous_output, but it must be a NamedTuple with the outputs that attribute_names lists for $(describe_model(model)), $(attributes.outputs).",
            ),
        )
    end
    if state isa NamedTuple && !same_names(keys(state), attributes.states)
        throw(
            ArgumentError(
                "The state is named $(keys(state)), but attribute_names lists the states of $(describe_model(model)) as $(attributes.states).",
            ),
        )
    end
    return nothing
end

"""
    check_input_names(model, input)

Throw an informative error if `input`, given to a model without the other side of a
simulation, isn't a `NamedTuple` with every input that [`attribute_names`](@ref) lists for
`model`. It can have more than those. Used by [`simulate`](@ref) and [`simulate!`](@ref) on
a single input.
"""
function check_input_names(model::AbstractModel, input)
    attributes = check_type_attribute_names(attribute_names(model), model)
    if !(input isa NamedTuple && has_every_name(attributes.inputs, keys(input)))
        throw(
            ArgumentError(
                "The input is $input, but it must be a NamedTuple with the inputs that attribute_names lists for $(describe_model(model)), $(attributes.inputs).",
            ),
        )
    end
    return nothing
end

## Checking parameters against their domains ##
"""
    check_parameter_domains(model, parameters)

Throw an [`InvalidParametersError`](@ref) if a parameter in `parameters` is outside its
domain in [`parameter_domains`](@ref) of `model`: below `lower`, above `upper`, or not a
whole number when it is `discrete`. Real numbers and arrays of them are checked, and
anything else is left alone, such as the traced numbers of Reactant.jl, whose values can't
be compared while a simulation is compiled. It is an `ArgumentError` when the domains name
a parameter that the model doesn't list, or have a field other than `lower`, `upper` and
`discrete`. Used by [`simulate`](@ref), [`InstantiatedModel`](@ref) and fitted sessions,
before anything runs.
"""
function check_parameter_domains(model::AbstractModel, parameters::NamedTuple)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    check_each_parameter_domain(model, parameters, domains)
    return nothing
end

"""
    check_domains_fit_model(model, domains)

Throw an `ArgumentError` if `domains`, what [`parameter_domains`](@ref) returns for `model`,
names a parameter that [`attribute_names`](@ref) doesn't list for it, or has a domain that
isn't a `NamedTuple` with only `lower`, `upper` and `discrete`. Used by
[`check_parameter_domains`](@ref).
"""
check_domains_fit_model(::AbstractModel, ::NamedTuple{()}) = nothing
function check_domains_fit_model(model::AbstractModel, domains::NamedTuple)
    listed_names = check_type_attribute_names(attribute_names(model), model).parameters
    if !has_every_name(keys(domains), listed_names)
        throw(
            ArgumentError(
                "parameter_domains of $(describe_model(model)) has domains for $(keys(domains)), but attribute_names lists its parameters as $listed_names.",
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
check_each_parameter_domain(::AbstractModel, ::NamedTuple, ::NamedTuple{()}) = nothing
function check_each_parameter_domain(
    model::AbstractModel, parameters::NamedTuple, domains::NamedTuple{names}
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
    model::AbstractModel, name::Symbol, value::Real, domain::NamedTuple
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
    model::AbstractModel, name::Symbol, values::AbstractArray{<:Real}, domain::NamedTuple
)
    foreach(value -> check_value_in_domain(model, name, value, domain), values)
    return nothing
end

check_value_in_domain(::AbstractModel, ::Symbol, value, ::NamedTuple) = nothing

"""
    discrete_parameter_names(model)

Return the names of the parameters of `model` that are discrete in
[`parameter_domains`](@ref), as a tuple. Every other parameter is continuous. Useful for
choosing how each parameter is sampled, since a discrete parameter needs another kind of
sampler than a continuous one.
"""
function discrete_parameter_names(model::AbstractModel)
    domains = parameter_domains(model)
    return Tuple(
        name for name in keys(domains) if get(getproperty(domains, name), :discrete, false)
    )
end

## Outputs and inputs before the first timestep ##
"""
    output_before_first_timestep(model)
    input_before_first_timestep(model)
    output_before_first_timestep(attributes)
    input_before_first_timestep(attributes)

Return the output and the input from before the first timestep: a `NamedTuple` with the
output or the input names that [`attribute_names`](@ref) lists for `model`, a cognitive
model or an environment model, or that `attributes`, a [`ModelAttributes`](@ref), has, each
[`BeforeFirstTimestep`](@ref). A model's names are checked to be a `ModelAttributes` first.
Used wherever a model starts: by [`InstantiatedModel`](@ref), [`reset!`](@ref),
[`simulate`](@ref) and fitted sessions, for the previous output and for the output
distributions saved as timestep 0.
"""
function output_before_first_timestep(model::AbstractModel)
    return output_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

function input_before_first_timestep(model::AbstractModel)
    return input_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

function output_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Toutputs}(map(_ -> BeforeFirstTimestep(), Toutputs))
end

function input_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Tinputs}(map(_ -> BeforeFirstTimestep(), Tinputs))
end

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
the state of an [`InstantiatedModel`](@ref): each name with its value when they are a
`NamedTuple`, `none` when it is empty, and only how many there are, with where to read them,
when [`shows_every_name`](@ref) doesn't allow listing them all. A state that isn't a
`NamedTuple` is printed as it is. Values are printed compactly, and large arrays are
shortened in the REPL. Used by the `show` methods of a model and of an `InstantiatedModel`.
"""
function print_current_values(io::IO, label, current_values, field_name)
    print(io, "\n", label, ": ")
    compact_io = IOContext(io, :compact => true)
    if current_values isa NamedTuple && isempty(current_values)
        print(io, "none")
    elseif current_values isa NamedTuple && !shows_every_name(io, keys(current_values))
        print(io, length(current_values), " (see ", field_name, ")")
    elseif current_values isa NamedTuple
        for (i, name) in enumerate(keys(current_values))
            i > 1 && print(io, ", ")
            print(io, name, " = ")
            show(compact_io, current_values[name])
        end
    else
        show(compact_io, current_values)
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
function Base.show(io::IO, ::MIME"text/plain", model::AbstractModel)

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
function Base.show(io::IO, model::AbstractModel)
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
model_kind(::AbstractModel) = "model"
