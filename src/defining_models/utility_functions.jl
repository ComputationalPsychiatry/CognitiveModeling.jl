"""
    check_type_attribute_names(attributes, model)

Return `attributes` if it is the [`ModelAttributes`](@ref) that a method of
[`attribute_names`](@ref) must return, and throw an informative error otherwise. `model` is
the cognitive model or the environment model whose method returned it.
"""
function check_type_attribute_names(attributes::ModelAttributes, ::AbstractModel)
    return attributes
end

function check_type_attribute_names(attributes, model::AbstractModel)
    error("The attribute_names method of $(describe_model(model)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).")
end

"""
    has_attribute_names(model)

Return whether `model`, a cognitive model or an environment model, has its own method of
[`attribute_names`](@ref), instead of one of the two fallbacks which throw.
"""
function has_attribute_names(model::AbstractModel)
    method_signature = which(attribute_names, (typeof(model),)).sig
    return method_signature != Tuple{typeof(attribute_names), AbstractCognitiveModel} && method_signature != Tuple{typeof(attribute_names), AbstractEnvironmentModel}
end

"""
    has_name(name, names)
    has_every_name(names, other_names)

Return whether `name` is among the tuple `names`, and whether every name in the tuple
`names` is among `other_names`.

Both are unrolled over the names, so that the compiler decides them from the names alone
when it knows them, which it doesn't for `issubset` and `issetequal` with more than one
name.
"""
has_name(name::Symbol, names::Tuple{}) = false
has_name(name::Symbol, names::Tuple) = name === first(names) || has_name(name, Base.tail(names))
has_every_name(names::Tuple{}, other_names::Tuple) = true
has_every_name(names::Tuple, other_names::Tuple) = has_name(first(names), other_names) && has_every_name(Base.tail(names), other_names)

"""
    same_names(names, other_names)

Return whether two tuples of names have the same names, in any order.
"""
function same_names(names, other_names)
    return has_every_name(names, other_names) && has_every_name(other_names, names)
end

"""
    check_match_attribute_names(model, parameters, state, previous_action)
    check_match_attribute_names(environment, parameters, state, previous_observation)

Throw an informative error if the names of the parameters, of the previous action of a
cognitive model or the previous observation of an environment model, or of a state that is a
`NamedTuple`, aren't those that [`attribute_names`](@ref) lists.
"""
function check_match_attribute_names(model::AbstractCognitiveModel, parameters, state, previous_action)
    attributes = check_type_attribute_names(attribute_names(model), model)
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

"""
    check_observation_names(model, observation)

Throw an informative error if `observation`, given to a cognitive model without an
environment, isn't a `NamedTuple` with every observation that [`attribute_names`](@ref)
lists for `model`. It can have more than those.
"""
function check_observation_names(model::AbstractCognitiveModel, observation)
    attributes = check_type_attribute_names(attribute_names(model), model)
    if !(observation isa NamedTuple && has_every_name(attributes.observations, keys(observation)))
        throw(ArgumentError("The observation is $observation, but it must be a NamedTuple with the observations that attribute_names lists for a model of type $(typeof(model)), $(attributes.observations)."))
    end
    return nothing
end

"""
    initial_action(model)
    initial_observation(model)

Return the action and the observation from before the first timestep: a `NamedTuple` with
the action or the observation names that [`attribute_names`](@ref) lists for `model`, a
cognitive model or an environment model, each [`BeforeFirstTimestep`](@ref).
"""
function initial_action(model::AbstractModel)
    return actions_before_first_timestep(check_type_attribute_names(attribute_names(model), model))
end

function initial_observation(model::AbstractModel)
    return observations_before_first_timestep(check_type_attribute_names(attribute_names(model), model))
end

"""
    actions_before_first_timestep(attributes)
    observations_before_first_timestep(attributes)

Return a `NamedTuple` with the action or the observation names in `attributes`, a
[`ModelAttributes`](@ref), each [`BeforeFirstTimestep`](@ref). The names are read from the
type, so that the `NamedTuple` is known at compile time.
"""
function actions_before_first_timestep(::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}) where {Tparameters, Tstates, Tobservations, Tactions}
    return NamedTuple{Tactions}(map(_ -> BeforeFirstTimestep(), Tactions))
end

function observations_before_first_timestep(::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}) where {Tparameters, Tstates, Tobservations, Tactions}
    return NamedTuple{Tobservations}(map(_ -> BeforeFirstTimestep(), Tobservations))
end

"""
    print_attribute_names(io, attributes)

Print a line for each kind of attribute in `attributes`, a [`ModelAttributes`](@ref), with
`none` if it has no names of that kind.

The `show` methods of a `ModelAttributes`, of a cognitive model and of an environment model
each print a header of their own, and then these lines.
"""
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
        print_attribute_names(io, check_type_attribute_names(attribute_names(model), model))
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end
