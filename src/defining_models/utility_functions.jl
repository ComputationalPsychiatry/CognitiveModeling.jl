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

#Whether a name is among other names, and whether every name in a tuple is. Both unrolled over the names, so that the compiler decides them from the names alone when it knows them, which it doesn't for issubset and issetequal with more than one name
has_name(name::Symbol, names::Tuple{}) = false
has_name(name::Symbol, names::Tuple) = name === first(names) || has_name(name, Base.tail(names))
has_every_name(names::Tuple{}, other_names::Tuple) = true
has_every_name(names::Tuple, other_names::Tuple) = has_name(first(names), other_names) && has_every_name(Base.tail(names), other_names)

#Whether two tuples of names have the same names, in any order
function same_names(names, other_names)
    return has_every_name(names, other_names) && has_every_name(other_names, names)
end

#Throw an informative error if the names of the parameters, the previous action or a NamedTuple state aren't those in attribute_names
function check_match_attribute_names(model::AbstractCognitiveModel, parameters, state, previous_action)
    attributes = check_attribute_names(attribute_names(model), model)
    if !same_names(keys(parameters), attributes.parameters)
        throw(ArgumentError("The parameters are named $(keys(parameters)), but attribute_names lists the parameters of a model of type $(typeof(model)) as $(attributes.parameters)."))
    end
    if !(previous_action isa NamedTuple && same_names(keys(previous_action), attributes.actions))
        throw(ArgumentError("The previous action is $previous_action, but it must be a NamedTuple with the actions that attribute_names lists for a model of type $(typeof(model)), $(attributes.actions)."))
    end
    if state isa NamedTuple && !same_names(keys(state), attributes.states)
        throw(ArgumentError("The state is named $(keys(state)), but attribute_names lists the states of a model of type $(typeof(model)) as $(attributes.states)."))
    end
    return nothing
end

#Throw an informative error if an observation given to a model isn't a NamedTuple with every observation that attribute_names lists
function check_observation_names(model::AbstractCognitiveModel, observation)
    attributes = check_attribute_names(attribute_names(model), model)
    if !(observation isa NamedTuple && has_every_name(attributes.observations, keys(observation)))
        throw(ArgumentError("The observation is $observation, but it must be a NamedTuple with the observations that attribute_names lists for a model of type $(typeof(model)), $(attributes.observations)."))
    end
    return nothing
end

#The action and the observation before the first timestep: a NamedTuple with the action or observation names that attribute_names lists for the model, each NoAction() or NoObservation()
function initial_action(model::AbstractCognitiveModel)
    return no_actions(check_attribute_names(attribute_names(model), model))
end

function initial_observation(model::AbstractCognitiveModel)
    return no_observations(check_attribute_names(attribute_names(model), model))
end

#Build the NamedTuples from the names in the type, so that they are known at compile time
function no_actions(::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}) where {Tparameters, Tstates, Tobservations, Tactions}
    return NamedTuple{Tactions}(map(_ -> NoAction(), Tactions))
end

function no_observations(::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}) where {Tparameters, Tstates, Tobservations, Tactions}
    return NamedTuple{Tobservations}(map(_ -> NoObservation(), Tobservations))
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
