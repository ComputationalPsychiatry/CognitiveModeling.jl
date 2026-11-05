"""
    fill_missing_observations(observation, environment_observation)

Return `observation` with each entry that is [`MissingObservation`](@ref) replaced by the
entry of the same name in `environment_observation`, which an environment model gave for the
same timestep. The other entries are kept as they are.
"""
function fill_missing_observations(observation::NamedTuple{names}, environment_observation::NamedTuple) where {names}
    return NamedTuple{names}(map(name -> fill_missing_observation(getproperty(observation, name), environment_observation, name), names))
end

#Keeps a recorded observation, and takes a missing one from the environment
fill_missing_observation(value, environment_observation, name) = value
fill_missing_observation(::MissingObservation, environment_observation, name) = getproperty(environment_observation, name)

"""
    has_missing_observation(observation)

Return whether any entry of `observation` is [`MissingObservation`](@ref).
"""
function has_missing_observation(observation::NamedTuple)
    return any(value -> value isa MissingObservation, values(observation))
end

"""
    is_recorded(action)

Return whether a single action in the data was taken and recorded, so that it is scored:
neither [`MissingAction`](@ref) nor [`NoAction`](@ref).
"""
is_recorded(action) = !(action isa MissingAction || action isa NoAction)

"""
    recorded_actions(actions)

Return the recorded values of each action in `actions`, a `NamedTuple` with one vector per
action, as a `NamedTuple` of vectors in the order of the timesteps, leaving out every action
without recorded values. Fitting scores these against the model's action distributions on
the same timesteps.
"""
function recorded_actions(actions::NamedTuple{names}) where {names}
    recorded = map(column -> map(identity, filter(is_recorded, column)), actions)
    names_with_values = Tuple(name for name in names if !isempty(getproperty(recorded, name)))
    return NamedTuple{names_with_values}(recorded)
end

"""
    row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of vectors
with one entry for each timestep.
"""
row(columns::NamedTuple, t) = map(column -> column[t], columns)

#Throw an informative error if the data doesn't have the observations and actions that attribute_names lists, or its columns differ in length, and return the number of timesteps
function check_data(model::AbstractCognitiveModel, observations::NamedTuple, actions::NamedTuple)
    attributes = check_attribute_names(attribute_names(model), model)
    if !has_every_name(attributes.observations, keys(observations))
        throw(ArgumentError("The observations in the data are named $(keys(observations)), but attribute_names lists the observations of a model of type $(typeof(model)) as $(attributes.observations). The data must have every observation the model lists."))
    end
    if !same_names(keys(actions), attributes.actions)
        throw(ArgumentError("The actions in the data are named $(keys(actions)), but attribute_names lists the actions of a model of type $(typeof(model)) as $(attributes.actions)."))
    end
    lengths = (map(length, values(observations))..., map(length, values(actions))...)
    isempty(lengths) && throw(ArgumentError("The data has no observations and no actions, so the number of timesteps is unknown."))
    allequal(lengths) || throw(ArgumentError("Every observation and action in the data must have one value for each timestep, but their lengths are $(map(length, observations)) and $(map(length, actions))."))
    return first(lengths)
end

#The same for the data that isn't a NamedTuple of vectors
function check_data(model::AbstractCognitiveModel, observations, actions)
    throw(ArgumentError("The observations and the actions must each be a NamedTuple with one vector for each name, such as (; reward = [1.0, 0.0]), but they have types $(typeof(observations)) and $(typeof(actions))."))
end
