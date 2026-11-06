#Return the names if they are a ModelAttributes, and throw an informative error otherwise
function check_attribute_names(attributes::ModelAttributes, ::AbstractEnvironmentModel)
    return attributes
end

function check_attribute_names(attributes, environment::AbstractEnvironmentModel)
    error("The attribute_names method of an environment model of type $(typeof(environment)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).")
end

#Whether an environment model has its own method of attribute_names, instead of the fallback which throws
function has_attribute_names(environment::AbstractEnvironmentModel)
    return which(attribute_names, (typeof(environment),)).sig != Tuple{typeof(attribute_names), AbstractEnvironmentModel}
end

#Throw an informative error if the names of the parameters, the previous observation or a NamedTuple state aren't those in attribute_names
function check_match_attribute_names(environment::AbstractEnvironmentModel, parameters, state, previous_observation)
    attributes = check_attribute_names(attribute_names(environment), environment)
    if !same_names(keys(parameters), attributes.parameters)
        throw(ArgumentError("The parameters are named $(keys(parameters)), but attribute_names lists the parameters of an environment model of type $(typeof(environment)) as $(attributes.parameters)."))
    end
    if !(previous_observation isa NamedTuple && same_names(keys(previous_observation), attributes.observations))
        throw(ArgumentError("The previous observation is $previous_observation, but it must be a NamedTuple with the observations that attribute_names lists for an environment model of type $(typeof(environment)), $(attributes.observations)."))
    end
    if state isa NamedTuple && !same_names(keys(state), attributes.states)
        throw(ArgumentError("The state is named $(keys(state)), but attribute_names lists the states of an environment model of type $(typeof(environment)) as $(attributes.states)."))
    end
    return nothing
end

#The action and the observation before the first timestep, with the names that attribute_names lists for the environment model
function initial_action(environment::AbstractEnvironmentModel)
    return no_actions(check_attribute_names(attribute_names(environment), environment))
end

function initial_observation(environment::AbstractEnvironmentModel)
    return no_observations(check_attribute_names(attribute_names(environment), environment))
end

#How a model or an environment model is named in error messages
describe_model(model::AbstractCognitiveModel) = "a model of type $(typeof(model))"
describe_model(environment::AbstractEnvironmentModel) = "an environment model of type $(typeof(environment))"

#Throw an informative error if the model doesn't receive every observation it lists, or the environment every action it lists, directly or through a NameMap for each
function check_names_fit(model::AbstractCognitiveModel, environment::AbstractEnvironmentModel, observation_map = nothing, action_map = nothing)
    model_attributes = check_attribute_names(attribute_names(model), model)
    environment_attributes = check_attribute_names(attribute_names(environment), environment)
    check_observations_fit(model, environment, model_attributes.observations, environment_attributes.observations, observation_map)
    check_actions_fit(model, environment, model_attributes.actions, environment_attributes.actions, action_map)
    return nothing
end

#Without a map, the environment must give every observation the model lists
function check_observations_fit(model, environment, model_names, environment_names, ::Nothing)
    if !has_every_name(model_names, environment_names)
        throw(ArgumentError("A model of type $(typeof(model)) lists the observations $model_names, but an environment model of type $(typeof(environment)) gives $environment_names. The environment must give every observation the model lists. A NameMap as observation_map can rename them."))
    end
    return nothing
end

#Without a map, the model must give every action the environment lists
function check_actions_fit(model, environment, model_names, environment_names, ::Nothing)
    if !has_every_name(environment_names, model_names)
        throw(ArgumentError("An environment model of type $(typeof(environment)) lists the actions $environment_names, but a model of type $(typeof(model)) gives $model_names. The model must give every action the environment lists. A NameMap as action_map can rename them."))
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
        print_attribute_names(io, check_attribute_names(attribute_names(environment), environment))
    else
        print(io, "\nAttributes: not specified")
    end

    return nothing
end
