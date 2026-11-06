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
    without_missing_actions(actions)

Return `actions`, a `NamedTuple` with one vector per action, with every
[`MissingAction`](@ref) replaced by [`NoAction`](@ref), for a session where a missing action
is passed on as an action that wasn't taken instead of being imputed.
"""
function without_missing_actions(actions::NamedTuple)
    return map(column -> map(action -> action isa MissingAction ? NoAction() : action, column), actions)
end

"""
    row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of vectors
with one entry for each timestep.
"""
row(columns::NamedTuple, t) = map(column -> column[t], columns)

#Throw an informative error if the data doesn't have the observations and actions that attribute_names lists, or its columns differ in length, and return the number of timesteps
function check_data(cognitive_model::AbstractCognitiveModel, observations::NamedTuple, actions::NamedTuple)
    attributes = check_type_attribute_names(attribute_names(cognitive_model), cognitive_model)
    if !has_every_name(attributes.observations, keys(observations))
        throw(ArgumentError("The observations in the data are named $(keys(observations)), but attribute_names lists the observations of a cognitive model of type $(typeof(cognitive_model)) as $(attributes.observations). The data must have every observation the cognitive model lists."))
    end
    if !same_names(keys(actions), attributes.actions)
        throw(ArgumentError("The actions in the data are named $(keys(actions)), but attribute_names lists the actions of a cognitive model of type $(typeof(cognitive_model)) as $(attributes.actions)."))
    end
    lengths = (map(length, values(observations))..., map(length, values(actions))...)
    isempty(lengths) && throw(ArgumentError("The data has no observations and no actions, so the number of timesteps is unknown."))
    allequal(lengths) || throw(ArgumentError("Every observation and action in the data must have one value for each timestep, but their lengths are $(map(length, observations)) and $(map(length, actions))."))
    return first(lengths)
end

#The same for the data that isn't a NamedTuple of vectors
function check_data(cognitive_model::AbstractCognitiveModel, observations, actions)
    throw(ArgumentError("The observations and the actions must each be a NamedTuple with one vector for each name, such as (; reward = [1.0, 0.0]), but they have types $(typeof(observations)) and $(typeof(actions))."))
end
