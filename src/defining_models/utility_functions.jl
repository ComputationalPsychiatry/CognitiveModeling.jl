"""
    describe_model(model)

Return how `model` is named in an error message: as a cognitive model or as an environment
model, with its type.
"""
function describe_model(cognitive_model::AbstractCognitiveModel)
    return "a cognitive model of type $(typeof(cognitive_model))"
end
function describe_model(environment::AbstractEnvironmentModel)
    return "an environment model of type $(typeof(environment))"
end
describe_model(model::AbstractModel) = "a model of type $(typeof(model))"

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
    return error(
        "The attribute_names method of $(describe_model(model)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).",
    )
end

"""
    has_attribute_names(model)

Return whether `model`, a cognitive model or an environment model, has its own method of
[`attribute_names`](@ref), instead of the fallback which throws.
"""
function has_attribute_names(model::AbstractModel)
    method_signature = which(attribute_names, (typeof(model),)).sig
    return method_signature != Tuple{typeof(attribute_names),AbstractModel}
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

Return whether two tuples of names have the same names, in any order.
"""
function same_names(names, other_names)
    return has_every_name(names, other_names) && has_every_name(other_names, names)
end

"""
    check_match_attribute_names(model, parameters, state, previous_output)

Throw an informative error if the names of the parameters, of the previous output, or of a
state that is a `NamedTuple`, aren't those that [`attribute_names`](@ref) lists for `model`,
a cognitive model or an environment model.
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
`model`. It can have more than those.
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

"""
    initial_output(model)
    initial_input(model)

Return the output and the input from before the first timestep: a `NamedTuple` with the
output or the input names that [`attribute_names`](@ref) lists for `model`, a cognitive
model or an environment model, each [`BeforeFirstTimestep`](@ref).
"""
function initial_output(model::AbstractModel)
    return outputs_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

function initial_input(model::AbstractModel)
    return inputs_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

"""
    outputs_before_first_timestep(attributes)
    inputs_before_first_timestep(attributes)

Return a `NamedTuple` with the output or the input names in `attributes`, a
[`ModelAttributes`](@ref), each [`BeforeFirstTimestep`](@ref). The names are read from the
type, so that the `NamedTuple` is known at compile time.
"""
function outputs_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Toutputs}(map(_ -> BeforeFirstTimestep(), Toutputs))
end

function inputs_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Tinputs}(map(_ -> BeforeFirstTimestep(), Tinputs))
end

"""
    check_names_fit(cognitive_model, environment, observation_map=nothing,
                    action_map=nothing)

Throw an informative error if `cognitive_model` and `environment` can't be simulated
together. Without a map, the outputs of each must be the inputs of the other in
[`attribute_names`](@ref): the observations the environment gives are what the cognitive
model lists as its inputs, and the actions the cognitive model gives are what the
environment lists as its inputs. With a [`NameMap`](@ref) for the observations or the
actions, the map must take names the giver lists, and give exactly the names the receiver
lists.

The names are in the types, so the check is decided at compile time and costs nothing.
"""
function check_names_fit(
    cognitive_model::AbstractModel,
    environment::AbstractModel,
    observation_map=nothing,
    action_map=nothing,
)
    cognitive_attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    environment_attributes = check_type_attribute_names(
        attribute_names(environment), environment
    )
    check_observations_fit(
        cognitive_model,
        environment,
        cognitive_attributes.inputs,
        environment_attributes.outputs,
        observation_map,
    )
    check_actions_fit(
        cognitive_model,
        environment,
        cognitive_attributes.outputs,
        environment_attributes.inputs,
        action_map,
    )
    return nothing
end

"""
    check_observations_fit(cognitive_model, environment, cognitive_model_names,
                           environment_names, observation_map)
    check_actions_fit(cognitive_model, environment, cognitive_model_names,
                      environment_names, action_map)

Throw an informative error if the observations `environment` gives, its outputs, don't fit
those `cognitive_model` lists as its inputs, or the actions `cognitive_model` gives, its
outputs, don't fit those `environment` lists as its inputs, as [`check_names_fit`](@ref)
describes. The methods for a [`NameMap`](@ref) are with the simulation code, where that type
is defined.
"""
function check_observations_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(cognitive_model))) lists the observations $cognitive_model_names as its inputs, but $(describe_model(environment)) gives $environment_names. Without a map, the two must list the same observations. A NameMap as observation_map can select and rename them.",
            ),
        )
    end
    return nothing
end

#Without a map, the actions the cognitive model gives must be those the environment lists as its inputs. The method above is for the observations
function check_actions_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(cognitive_model))) gives the actions $cognitive_model_names, but $(describe_model(environment)) lists $environment_names as its inputs. Without a map, the two must list the same actions. A NameMap as action_map can select and rename them.",
            ),
        )
    end
    return nothing
end

"""
    print_attribute_names(io, attributes)

Print a line for each kind of attribute in `attributes`, a [`ModelAttributes`](@ref), with
`none` if it has no names of that kind.

The `show` methods of a `ModelAttributes`, of a cognitive model and of an environment model
each print a header of their own, and then these lines.
"""
function print_attribute_names(io::IO, attributes::ModelAttributes)
    for (kind, names) in (
        ("Parameters", attributes.parameters),
        ("States", attributes.states),
        ("Inputs", attributes.inputs),
        ("Outputs", attributes.outputs),
    )
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
    show(io::IO, ::MIME"text/plain", model::AbstractModel)

Print a summary of `model`, a cognitive model or an environment model, and the names of its
attributes.

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
               inputs = :reward,
               outputs = (),
           )
       end;

julia> ExampleModel()
-- Cognitive model --
Type: ExampleModel
Parameters: learning_rate, noise
States: expected_reward
Inputs: reward
Outputs: none

julia> struct ExampleEnvironment <: AbstractEnvironmentModel end

julia> function CognitiveModeling.attribute_names(environment::ExampleEnvironment)
           return ModelAttributes(;
               parameters = :reward_probability,
               states = (),
               inputs = :choice,
               outputs = :reward,
           )
       end;

julia> ExampleEnvironment()
-- Environment model --
Type: ExampleEnvironment
Parameters: reward_probability
States: none
Inputs: choice
Outputs: reward
```
"""
function Base.show(io::IO, ::MIME"text/plain", model::AbstractModel)

    #Print the kind and the type of the model
    print(io, model_header(model), "\nType: ", nameof(typeof(model)))

    #Print that the model has no method of its own for its names, or the names
    if has_attribute_names(model)
        print_attribute_names(io, check_type_attribute_names(attribute_names(model), model))
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end

"""
    model_header(model)

Return the first line that `show` prints for a model: which kind of model it is. Used by the
`show` method of an [`AbstractModel`](@ref).
"""
model_header(::AbstractCognitiveModel) = "-- Cognitive model --"
model_header(::AbstractEnvironmentModel) = "-- Environment model --"
model_header(::AbstractModel) = "-- Model --"
