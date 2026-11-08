#Resets the environment from CommonRLInterface.jl and uses the object itself as the state
function CognitiveModeling.initialize_state(environment::CommonRLEnvironment, parameters)
    reset!(environment.rl_env)
    return environment.rl_env
end

#Gives the observation the environment is at when there is no action to take, and acts otherwise
function CognitiveModeling.apply_model(
    environment::CommonRLEnvironment{Trl_env,action_name}, input::EnvironmentInput
) where {Trl_env,action_name}
    rl_env = input.previous_state
    action = getproperty(input.action, action_name)

    #Before the first action, and when the model takes none, the environment stays where it is
    if action isa AbstractNoValue
        return EnvironmentOutput(observation_without_action(rl_env), rl_env)
    end

    #An episode that has ended can't take another action, and isn't started again here
    if terminated(rl_env)
        error(
            "The episode of the environment of type $(typeof(rl_env)) from CommonRLInterface.jl has ended, so it can't take the action $action_name. A simulation stops by itself when the episode ends. A new call of simulate starts a new episode, and so does reset! of an Environment.",
        )
    end

    #Take the model's action, and give what follows from it, with nothing left to sample
    reward = act!(rl_env, action)
    return EnvironmentOutput(
        (;
            observation=Deterministic(observe(rl_env)),
            reward=Deterministic(reward),
            terminated=Deterministic(terminated(rl_env)),
        ),
        rl_env,
    )
end

#Copies the environment with its own clone, or with deepcopy if it has none
function CognitiveModeling.prepare_state_for_saving(
    environment::CommonRLEnvironment, rl_env
)
    return provided(clone, rl_env) ? clone(rl_env) : deepcopy(rl_env)
end

#A simulation stops when the episode of the environment from CommonRLInterface.jl has ended
function CognitiveModeling.stopping_criterion(
    environment::CommonRLEnvironment, rl_env, parameters
)
    return terminated(rl_env)
end

#The observation when no action is taken, where there is no reward
function observation_without_action(rl_env)
    return (;
        observation=Deterministic(observe(rl_env)),
        reward=NoObservation(),
        terminated=Deterministic(false),
    )
end
