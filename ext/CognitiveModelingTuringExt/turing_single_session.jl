#Checks the data and the names, fills in the defaults, and creates the Turing model of the session
function CognitiveModeling.turing_single_session(
        cognitive_model::AbstractCognitiveModel,
        cognitive_parameters,
        observations,
        actions;
        who_starts::Union{AgentStarts, EnvironmentStarts},
        environment::Union{Nothing, AbstractEnvironmentModel} = nothing,
        environment_parameters = (;),
        timestep_sizes = 1,
        impute_missing_actions::Bool = true,
        save_history::SaveHistory = SaveHistory(),
        catch_invalid_parameters::Bool = false,
        previous_cognitive_state = initialize_state(cognitive_model, merge_default_parameters(cognitive_model, cognitive_parameters)),
        previous_action = initial_action(cognitive_model),
        previous_observation = initial_observation(environment === nothing ? cognitive_model : environment),
        previous_environment_state = environment === nothing ? nothing : initialize_state(environment, merge_default_parameters(environment, environment_parameters)),
    )

    #Fill in the default parameters, and check the names of everything against attribute_names
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    check_match_attribute_names(cognitive_model, cognitive_parameters, previous_cognitive_state, previous_action)
    n_timesteps = check_data(cognitive_model, observations, actions)
    if environment === nothing
        if any(column -> any(value -> value isa MissingObservation, column), values(observations))
            throw(ArgumentError("The observations have entries that are MissingObservation(), but no environment was given to fill them in."))
        end
    else
        environment_parameters = merge_default_parameters(environment, environment_parameters)
        check_match_attribute_names(environment, environment_parameters, previous_environment_state, previous_observation)
        check_names_fit(cognitive_model, environment)
    end

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_timesteps
        throw(DimensionMismatch("timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and the data has $n_timesteps timesteps."))
    end

    #Missing actions that aren't imputed are passed on as actions that weren't taken
    if !impute_missing_actions
        actions = without_missing_actions(actions)
    end

    #The recorded actions are scored at the end
    recorded = recorded_actions(actions)

    #The sizes of the timesteps are returned with what is saved of the session
    timestep_sizes_to_return = saved_timestep_sizes(timestep_sizes, n_timesteps, save_history)

    session = single_session(cognitive_model, cognitive_parameters, observations, actions, recorded, n_timesteps, environment, environment_parameters, who_starts, timestep_sizes, timestep_sizes_to_return, save_history, previous_cognitive_state, previous_action, previous_observation, previous_environment_state)
    return catch_invalid_parameters ? catching_invalid_parameters(session) : session
end

#Runs the session, and gives the parameters a log density of -Inf if the cognitive model throws an InvalidParametersError
Turing.@model function catching_invalid_parameters(session)
    result = nothing
    try
        result ~ to_submodel(session, false)
    catch error
        error isa InvalidParametersError || rethrow()
        Turing.@addlogprob! -Inf
    end
    return result
end

#Runs the timesteps of the session, with the recorded actions scored together at the end
Turing.@model function single_session(cognitive_model, cognitive_parameters, observations, actions, recorded, n_timesteps, environment, environment_parameters, who_starts, timestep_sizes, timestep_sizes_to_return, save_history, previous_cognitive_state, previous_action, previous_observation, previous_environment_state)

    #Collect where the session starts
    previous = (;
        observation = previous_observation,
        action = previous_action,
        cognitive_state = previous_cognitive_state,
        environment_state = previous_environment_state,
    )

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs ~ to_submodel(prefix(session_timestep(cognitive_model, environment, cognitive_parameters, environment_parameters, who_starts, row(observations, 1), row(actions, 1), get_timestep_size(timestep_sizes, 1), keys(recorded), previous), @varname(timestep[1])), false)

    #Initialize storage for the saved outputs, and the distributions of the recorded actions
    storage = (;
        observations = initialize_storage(select_names(outputs.observation, observation_selection(save_history)), n_timesteps),
        actions = initialize_storage(select_names(outputs.action, action_selection(save_history)), n_timesteps),
        states = initialize_storage(save_state(cognitive_model, outputs.cognitive_state, state_selection(save_history)), n_timesteps),
    )
    distributions = store_distributions(map(_ -> Vector{Union{}}(), recorded), outputs.scored_distributions, outputs.scored)

    #Run the remaining timesteps, each starting from the outputs of the one before
    for t in 2:n_timesteps
        next_outputs ~ to_submodel(prefix(session_timestep(cognitive_model, environment, cognitive_parameters, environment_parameters, who_starts, row(observations, t), row(actions, t), get_timestep_size(timestep_sizes, t), keys(recorded), outputs), @varname(timestep[t])), false)
        outputs = next_outputs
        storage = (;
            observations = store_output!(storage.observations, select_names(outputs.observation, observation_selection(save_history)), t),
            actions = store_output!(storage.actions, select_names(outputs.action, action_selection(save_history)), t),
            states = store_output!(storage.states, save_state(cognitive_model, outputs.cognitive_state, state_selection(save_history)), t),
        )
        distributions = store_distributions(distributions, outputs.scored_distributions, outputs.scored)
    end

    #Score the recorded actions against the distributions of their timesteps, all at once
    if !isempty(recorded)
        recorded ~ product_distribution(map(product_distribution, distributions))
    end

    return (; observations = storage.observations, actions = storage.actions, states = storage.states, timestep_sizes = timestep_sizes_to_return)
end

#Runs one timestep, as a submodel with the timestep as its prefix, and returns what the next one starts from, together with what is saved and scored
Turing.@model function session_timestep(cognitive_model, environment, cognitive_parameters, environment_parameters, who_starts, data_observation, data_action, timestep_size, scored_names, previous)

    #When the environment starts, it gives this timestep's observation, in response to the previous action
    if who_starts isa EnvironmentStarts
        if environment === nothing
            observation = data_observation
            environment_state = previous.environment_state
        else
            environment_output ~ to_submodel(prefix(turing_apply_model(environment, EnvironmentInput(previous.action, previous.environment_state, previous.observation, environment_parameters, timestep_size)), @varname(environment)), false)
            environment_output = check_environment_output(environment_output, environment)
            filled_observation ~ to_submodel(prefix(impute_missing_observations(data_observation, environment_output.observation_distribution), @varname(environment)), false)
            observation = filled_observation
            environment_state = environment_output.new_state
        end
    else
        observation = previous.observation
        environment_state = previous.environment_state
    end

    #Run the cognitive model on the observation
    output ~ to_submodel(turing_apply_model(cognitive_model, ModelInput(observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size)), false)
    output = check_model_output(output, cognitive_model)

    #Impute the missing actions as latents named after the action, each with the cognitive model's distribution for it, and take the rest from the data
    missing_names = missing_action_names(data_action)
    imputed_values = ()
    for name in missing_names
        imputed ~ to_submodel(prefix(impute_action(getproperty(output.action_distribution, name)), VarName{name}()), false)
        imputed_values = (imputed_values..., imputed)
    end
    action = merge(data_action, NamedTuple{missing_names}(imputed_values))
    scored = scored_actions(data_action, output.action_distribution, cognitive_model)

    #When the agent starts, the environment responds to the action with the next observation
    if who_starts isa AgentStarts
        if environment === nothing
            observation = data_observation
        else
            environment_output ~ to_submodel(prefix(turing_apply_model(environment, EnvironmentInput(action, previous.environment_state, previous.observation, environment_parameters, timestep_size)), @varname(environment)), false)
            environment_output = check_environment_output(environment_output, environment)
            filled_observation ~ to_submodel(prefix(impute_missing_observations(data_observation, environment_output.observation_distribution), @varname(environment)), false)
            observation = filled_observation
            environment_state = environment_output.new_state
        end
    end

    return (;
        observation,
        action,
        cognitive_state = output.new_state,
        environment_state,
        scored,
        scored_distributions = NamedTuple{scored_names}(output.action_distribution),
    )
end

#A missing action as a latent, with the cognitive model's distribution for it
Turing.@model function impute_action(distribution)
    imputed ~ distribution
    return imputed
end

#Fills in the observations that are missing in the data, each from what the environment gives for it, under the name of the observation, and takes the rest from the data.
#It runs under the environment's prefix, so that an observation and an action with the same name get a latent each
Turing.@model function impute_missing_observations(data_observation, observation_distribution)
    missing_names = missing_observation_names(data_observation)
    imputed_values = ()
    for name in missing_names
        imputed ~ to_submodel(prefix(impute_observation(getproperty(observation_distribution, name)), VarName{name}()), false)
        imputed_values = (imputed_values..., imputed)
    end
    return merge(data_observation, NamedTuple{missing_names}(imputed_values))
end

#A missing observation as a latent, with the environment's distribution for it
Turing.@model function impute_observation(distribution)
    imputed ~ distribution
    return imputed
end

#An observation that isn't random is its value, with nothing to sample
Turing.@model function impute_observation(deterministic::Deterministic)
    return deterministic.value
end

#An observation that the environment doesn't give on this timestep stays NoObservation
Turing.@model function impute_observation(no_observation::NoObservation)
    return no_observation
end

#The names of the actions of a timestep that are MissingAction, unrolled over the names so that the types are known
missing_action_names(data_action::NamedTuple{names}) where {names} = missing_action_names(names, Tuple(data_action))
missing_action_names(names::Tuple{}, data_values::Tuple{}) = ()
function missing_action_names(names::Tuple, data_values::Tuple)
    rest = missing_action_names(Base.tail(names), Base.tail(data_values))
    return first(data_values) isa MissingAction ? (first(names), rest...) : rest
end

#The names of the observations of a timestep that are MissingObservation, unrolled over the names in the same way
missing_observation_names(data_observation::NamedTuple{names}) where {names} = missing_observation_names(names, Tuple(data_observation))
missing_observation_names(names::Tuple{}, data_values::Tuple{}) = ()
function missing_observation_names(names::Tuple, data_values::Tuple)
    rest = missing_observation_names(Base.tail(names), Base.tail(data_values))
    return first(data_values) isa MissingObservation ? (first(names), rest...) : rest
end

#Whether each action of a timestep is scored: those recorded in the data, which the cognitive model must have a distribution for
function scored_actions(data_action::NamedTuple{names}, action_distribution::NamedTuple, cognitive_model) where {names}
    return NamedTuple{names}(scored_actions(names, Tuple(data_action), action_distribution, cognitive_model))
end

scored_actions(names::Tuple{}, data_values::Tuple{}, action_distribution, cognitive_model) = ()
function scored_actions(names::Tuple, data_values::Tuple, action_distribution, cognitive_model)
    name = first(names)
    value = first(data_values)
    is_scored = is_recorded(value)
    if is_scored && getproperty(action_distribution, name) isa NoAction
        throw(ArgumentError("The cognitive model of type $(typeof(cognitive_model)) took no action $name on a timestep where the data has $value for it."))
    end
    return (is_scored, scored_actions(Base.tail(names), Base.tail(data_values), action_distribution, cognitive_model)...)
end

#Keeps the distributions of the actions that are scored on this timestep. The first one kept sets the type of each vector
function store_distributions(distributions::NamedTuple{names}, scored_distributions::NamedTuple, scored::NamedTuple) where {names}
    return NamedTuple{names}(map(name -> getproperty(scored, name) ? push_output!(getproperty(distributions, name), getproperty(scored_distributions, name)) : getproperty(distributions, name), names))
end
