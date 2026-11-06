#Resets the environment from CommonRLInterface.jl and uses the object itself as the state
function CognitiveModeling.initialize_state(environment::CommonRLEnvironment, parameters)
    reset!(environment.rl_env)
    return environment.rl_env
end

#Gives the initial observation before the first action, and acts otherwise
function CognitiveModeling.apply_model(environment::CommonRLEnvironment{Trl_env, action_name}, input::EnvironmentInput) where {Trl_env, action_name}
    rl_env = input.previous_state
    action = getproperty(input.action, action_name)

    #Before the first action, the environment is at its initial observation
    if action isa NoAction
        return EnvironmentOutput(observation_at_start(rl_env), rl_env)
    end

    #An episode that has ended can't take another action, and isn't started again here
    if terminated(rl_env)
        error("The episode of the environment of type $(typeof(rl_env)) from CommonRLInterface.jl has ended, so it can't take the action $action_name. Simulate no further than the timestep where terminated is true. A new call of simulate starts a new episode, and so does reset! of an Environment.")
    end

    #Take the model's action, and give what follows from it
    reward = act!(rl_env, action)
    return EnvironmentOutput((; observation = observe(rl_env), reward, terminated = terminated(rl_env)), rl_env)
end

#Copies the environment with its own clone, or with deepcopy if it has none
function CognitiveModeling.prepare_state_for_saving(environment::CommonRLEnvironment, rl_env)
    return provided(clone, rl_env) ? clone(rl_env) : deepcopy(rl_env)
end

#The observation at the start of an episode, where there is no reward yet
function observation_at_start(rl_env)
    return (; observation = observe(rl_env), reward = NoObservation(), terminated = false)
end
