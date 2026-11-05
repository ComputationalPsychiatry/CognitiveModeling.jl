#Return the names if they are a ModelAttributes, and throw an informative error otherwise
function check_attribute_names(attributes::ModelAttributes, ::AbstractCognitiveModel)
    return attributes
end

function check_attribute_names(attributes, model::AbstractCognitiveModel)
    error("The attribute_names method of a model of type $(typeof(model)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).")
end

#Whether a model has its own method of attribute_names, instead of the fallback which throws
function has_attribute_names(model::AbstractCognitiveModel)
    return which(attribute_names, (typeof(model),)).sig != Tuple{typeof(attribute_names), AbstractCognitiveModel}
end

#Throw an informative error if the names of the parameters, or of a NamedTuple state, aren't those in attribute_names
function check_names_match(model::AbstractCognitiveModel, parameters, state)
    attributes = check_attribute_names(attribute_names(model), model)
    if !issetequal(keys(parameters), attributes.parameters)
        throw(ArgumentError("The parameters are named $(keys(parameters)), but attribute_names lists the parameters of a model of type $(typeof(model)) as $(attributes.parameters)."))
    end
    if state isa NamedTuple && !issetequal(keys(state), attributes.states)
        throw(ArgumentError("The state is named $(keys(state)), but attribute_names lists the states of a model of type $(typeof(model)) as $(attributes.states)."))
    end
    return nothing
end

#Print a line for each kind of attribute, with none if there are no names of that kind
function print_attribute_names(io::IO, attributes::ModelAttributes)
    for (kind, names) in (("Parameters", attributes.parameters), ("States", attributes.states), ("Observations", attributes.observations), ("Actions", attributes.actions))
        print(io, "\n", kind, ": ")
        isempty(names) ? print(io, "none") : join(io, names, ", ")
    end
    return nothing
end

#Shows the names on one line each in the REPL and notebooks
function Base.show(io::IO, ::MIME"text/plain", attributes::ModelAttributes)
    print(io, "ModelAttributes")
    print_attribute_names(io, attributes)
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

Print a summary of `model` and the names of its attributes.

This is what the REPL and notebooks show. Everywhere else, such as in `print` or inside a
vector, a model is printed on one line by Julia's default `show(io, model)`.

The attributes show as `not specified` if `model` has no method of
[`attribute_names`](@ref), and a kind of attribute shows as `none` if the model has no names
of that kind.

# Examples
```jldoctest
julia> struct ExampleModel <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::ExampleModel)
           return ModelAttributes(;
               parameters = (:learning_rate, :noise),
               states = :expected_reward,
               observations = :reward,
               actions = (),
           )
       end;

julia> ExampleModel()
-- Cognitive model --
Type: ExampleModel
Parameters: learning_rate, noise
States: expected_reward
Observations: reward
Actions: none
```
"""
function Base.show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

    #Print the type of the model
    print(io, "-- Cognitive model --\nType: ", nameof(typeof(model)))

    #Print that the model has no method of its own for its names, or the names
    if has_attribute_names(model)
        print_attribute_names(io, check_attribute_names(attribute_names(model), model))
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end
