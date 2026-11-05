#Resets the environment and uses the object itself as the state
function CognitiveModeling.initialize_state(environment::CommonRLEnvironment, parameters)
    reset!(environment.env)
    return environment.env
end

#Gives the initial observation before the first action and after an episode has ended, and acts otherwise
function CognitiveModeling.apply_model(environment::CommonRLEnvironment{Tenv, action_name}, input::EnvironmentInput) where {Tenv, action_name}
    env = input.previous_state
    action = getproperty(input.action, action_name)

    #Before the first action, the environment is at its initial observation
    if action isa NoAction
        return EnvironmentOutput(observation_at_start(env), env)
    end

    #After an episode has ended, start a new one
    if terminated(env)
        reset!(env)
        return EnvironmentOutput(observation_at_start(env), env)
    end

    #Take the model's action, and give what follows from it
    reward = act!(env, action)
    return EnvironmentOutput((; observation = observe(env), reward, terminated = terminated(env)), env)
end

#Copies the environment with its own clone, or with deepcopy if it has none
function CognitiveModeling.prepare_state_for_saving(environment::CommonRLEnvironment, env)
    return provided(clone, env) ? clone(env) : deepcopy(env)
end

#The observation at the start of an episode, where there is no reward yet
function observation_at_start(env)
    return (; observation = observe(env), reward = NoObservation(), terminated = false)
end
