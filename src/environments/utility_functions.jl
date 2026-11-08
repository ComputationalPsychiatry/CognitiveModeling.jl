#Throw an informative error if the names of the parameters, the previous observation or a NamedTuple state aren't those in attribute_names
function check_match_attribute_names(
    environment::AbstractEnvironmentModel, parameters, state, previous_observation
)
    attributes = check_type_attribute_names(attribute_names(environment), environment)
    if !same_names(keys(parameters), attributes.parameters)
        throw(
            ArgumentError(
                "The parameters are named $(keys(parameters)), but attribute_names lists the parameters of an environment model of type $(typeof(environment)) as $(attributes.parameters).",
            ),
        )
    end
    if !(
        previous_observation isa NamedTuple &&
        same_names(keys(previous_observation), attributes.observations)
    )
        throw(
            ArgumentError(
                "The previous observation is $previous_observation, but it must be a NamedTuple with the observations that attribute_names lists for an environment model of type $(typeof(environment)), $(attributes.observations).",
            ),
        )
    end
    if state isa NamedTuple && !same_names(keys(state), attributes.states)
        throw(
            ArgumentError(
                "The state is named $(keys(state)), but attribute_names lists the states of an environment model of type $(typeof(environment)) as $(attributes.states).",
            ),
        )
    end
    return nothing
end

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

"""
    check_names_fit(cognitive_model, environment, observation_map=nothing,
                    action_map=nothing)

Throw an informative error if `cognitive_model` and `environment` can't be simulated
together. Without a map, they must list the same observations and the same actions in
[`attribute_names`](@ref). With a [`NameMap`](@ref) for the observations or the actions, the
map must take names the giver lists, and give exactly the names the receiver lists.

The names are in the types, so the check is decided at compile time and costs nothing.
"""
function check_names_fit(
    cognitive_model::AbstractCognitiveModel,
    environment::AbstractEnvironmentModel,
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
        cognitive_attributes.observations,
        environment_attributes.observations,
        observation_map,
    )
    check_actions_fit(
        cognitive_model,
        environment,
        cognitive_attributes.actions,
        environment_attributes.actions,
        action_map,
    )
    return nothing
end

"""
    check_observations_fit(cognitive_model, environment, cognitive_model_names,
                           environment_names, observation_map)
    check_actions_fit(cognitive_model, environment, cognitive_model_names,
                      environment_names, action_map)

Throw an informative error if the observations `environment` gives don't fit those
`cognitive_model` lists, or the actions `cognitive_model` gives don't fit those
`environment` lists, as [`check_names_fit`](@ref) describes. The methods for a
[`NameMap`](@ref) are with the simulation code, where that type is defined.
"""
function check_observations_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "A cognitive model of type $(typeof(cognitive_model)) lists the observations $cognitive_model_names, but an environment model of type $(typeof(environment)) gives $environment_names. Without a map, the two must list the same observations. A NameMap as observation_map can select and rename them.",
            ),
        )
    end
    return nothing
end

#Without a map, the cognitive model and the environment must list the same actions. The method above is for the same observations
function check_actions_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "A cognitive model of type $(typeof(cognitive_model)) gives the actions $cognitive_model_names, but an environment model of type $(typeof(environment)) lists $environment_names. Without a map, the two must list the same actions. A NameMap as action_map can select and rename them.",
            ),
        )
    end
    return nothing
end

"""
    show(io::IO, ::MIME"text/plain", environment::AbstractEnvironmentModel)

Print a summary of `environment` and the names of its attributes, as for a cognitive model.

The attributes show as `not specified` if `environment` has no method of
[`attribute_names`](@ref), and a kind of attribute shows as `none` if the environment model
has no names of that kind.

# Examples
```jldoctest
julia> struct ExampleEnvironment <: AbstractEnvironmentModel end

julia> function CognitiveModeling.attribute_names(environment::ExampleEnvironment)
           return ModelAttributes(;
               parameters = :reward_probability,
               states = (),
               observations = :reward,
               actions = :choice,
           )
       end;

julia> ExampleEnvironment()
-- Environment model --
Type: ExampleEnvironment
Parameters: reward_probability
States: none
Observations: reward
Actions: choice
```
"""
function Base.show(io::IO, ::MIME"text/plain", environment::AbstractEnvironmentModel)

    #Print the type of the environment model
    print(io, "-- Environment model --\nType: ", nameof(typeof(environment)))

    #Print that the environment model has no method of its own for its names, or the names
    if has_attribute_names(environment)
        print_attribute_names(
            io, check_type_attribute_names(attribute_names(environment), environment)
        )
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end
