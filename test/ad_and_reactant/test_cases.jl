using Random: AbstractRNG, Xoshiro

# Most models here give the probability of their action as the action itself, in a Deterministic, instead of
# something to sample from. It makes the simulation deterministic, and lets the gradient go all the way round
# the loop through the action and the environment

# An action that is sampled
struct SampledAction{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::SampledAction) = rand(rng) < action.probability

# A reward that is sampled: 1.0 with a probability, and 0.0 otherwise
struct SampledReward{T}
    probability::T
end
Base.rand(rng::AbstractRNG, reward::SampledReward) = float(rand(rng) < reward.probability)

# A Rescorla-Wagner model. Its state starts as a Float64, whatever number type the parameters have
struct LearnsRewards{Taction} <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsRewards, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsRewards{Taction}, context::ModelContext
) where {Taction}
    V₋₁ = context.previous_state.expected_reward
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + α * (context.input.reward - V₋₁)
    end
    return ModelOutput(
        (; choice=Taction(inv(1 + exp(-β * (Vₜ - 0.5))))), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::LearnsRewards)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# An environment whose reward depends on the action, and which has a state. The choice it receives is its input,
# and the reward it gives its output
struct RewardsFollowActions <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RewardsFollowActions, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::RewardsFollowActions, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; reward=Deterministic(context.parameters.reward_size * action)), (; n_timesteps)
    )
end
function CognitiveModeling.attribute_names(environment::RewardsFollowActions)
    return ModelAttributes(;
        parameters=:reward_size, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

# The same environment under other names than the model's, with an observation that isn't in the model's attributes, which
# can't share the name of the state it comes from, since a model uses a name for one attribute
struct PayoffsFollowLevers <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::PayoffsFollowLevers, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::PayoffsFollowLevers, context::ModelContext
)
    action = context.input.lever isa AbstractNoValue ? 0.5 : context.input.lever
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (;
            payoff=Deterministic(context.parameters.reward_size * action),
            timestep_number=Deterministic(n_timesteps),
        ),
        (; n_timesteps),
    )
end
function CognitiveModeling.attribute_names(environment::PayoffsFollowLevers)
    return ModelAttributes(;
        parameters=:reward_size,
        states=:n_timesteps,
        inputs=:lever,
        outputs=(:payoff, :timestep_number),
    )
end

# An environment which takes the choice, and whose reward is sampled whatever it is
struct SampledRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::SampledRewards, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::SampledRewards, context::ModelContext
)
    return ModelOutput((; reward=SampledReward(context.parameters.reward_probability)))
end
function CognitiveModeling.attribute_names(environment::SampledRewards)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

# A Rescorla-Wagner model which learns more the more time passes. It learns on the first timestep too, so that every size counts
struct LearnsOverTime <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsOverTime, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsOverTime, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    r = context.input.reward isa AbstractNoValue ? 0.5 : context.input.reward
    Vₜ = V₋₁ + (1 - exp(-α * context.timestep_size)) * (r - V₋₁)
    return ModelOutput(
        (; choice=Deterministic(inv(1 + exp(-β * (Vₜ - 0.5))))), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::LearnsOverTime)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# An environment whose reward grows with the time that passes
struct RewardsGrowOverTime <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsGrowOverTime, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::RewardsGrowOverTime, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    return ModelOutput((;
        reward=Deterministic(
            context.parameters.reward_size * action * context.timestep_size
        )
    ))
end
function CognitiveModeling.attribute_names(environment::RewardsGrowOverTime)
    return ModelAttributes(;
        parameters=:reward_size, states=(), inputs=:choice, outputs=:reward
    )
end

# A model and an environment which leave out their states
struct ChoosesWithoutState <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ChoosesWithoutState, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    model::ChoosesWithoutState, context::ModelContext
)
    return ModelOutput((;
        choice=Deterministic(inv(1 + exp(-context.parameters.inverse_temperature)))
    ))
end
function CognitiveModeling.attribute_names(model::ChoosesWithoutState)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=:reward, outputs=:choice
    )
end
struct RewardsWithoutState <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsWithoutState, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::RewardsWithoutState, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    return ModelOutput((; reward=Deterministic(context.parameters.reward_size * action)))
end
function CognitiveModeling.attribute_names(environment::RewardsWithoutState)
    return ModelAttributes(;
        parameters=:reward_size, states=(), inputs=:choice, outputs=:reward
    )
end

# A model whose state update has noise in it
struct LearnsWithNoise <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsWithNoise, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsWithNoise, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    r = context.input.reward isa AbstractNoValue ? 0.5 : context.input.reward
    Vₜ = V₋₁ + α * (r - V₋₁) + α * 0.1 * (rand(context.rng) - 0.5)
    return ModelOutput(
        (; choice=Deterministic(inv(1 + exp(-β * (Vₜ - 0.5))))), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::LearnsWithNoise)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# A Rescorla-Wagner model which changes a Vector in its state in place, and copies it when it's saved
struct LearnsInPlace <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsInPlace, parameters)
    return (;
        expected_rewards=[
            0.5 * one(parameters.learning_rate), 0.5 * one(parameters.learning_rate)
        ]
    )
end
function CognitiveModeling.apply_timestep_model(model::LearnsInPlace, context::ModelContext)
    V = context.previous_state.expected_rewards
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    r = context.input.reward isa AbstractNoValue ? 0.5 : context.input.reward
    V[1] += α * (r - V[1])
    return ModelOutput(
        (; choice=Deterministic(inv(1 + exp(-β * (V[1] - 0.5))))), context.previous_state
    )
end
function CognitiveModeling.prepare_state_for_saving(model::LearnsInPlace, state)
    return (; expected_rewards=copy(state.expected_rewards))
end
function CognitiveModeling.attribute_names(model::LearnsInPlace)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_rewards,
        inputs=:reward,
        outputs=:choice,
    )
end

# The same model, which copies its state with deepcopy when it's saved. ReverseDiff gave zero gradients through deepcopy
# of a container of tracked numbers before v1.18.8
struct LearnsInPlaceWithDeepcopy <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsInPlaceWithDeepcopy, parameters)
    return CognitiveModeling.initialize_state(LearnsInPlace(), parameters)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsInPlaceWithDeepcopy, context::ModelContext
)
    return CognitiveModeling.apply_timestep_model(LearnsInPlace(), context)
end
function CognitiveModeling.prepare_state_for_saving(model::LearnsInPlaceWithDeepcopy, state)
    return deepcopy(state)
end
function CognitiveModeling.attribute_names(model::LearnsInPlaceWithDeepcopy)
    return CognitiveModeling.attribute_names(LearnsInPlace())
end

# The same environment as RewardsFollowActions, which ends after the number of timesteps in its parameters
struct RewardsUntilEnd <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RewardsUntilEnd, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::RewardsUntilEnd, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; reward=Deterministic(context.parameters.reward_size * action)), (; n_timesteps)
    )
end
function CognitiveModeling.attribute_names(environment::RewardsUntilEnd)
    return ModelAttributes(;
        parameters=(:reward_size, :last_timestep),
        states=:n_timesteps,
        inputs=:choice,
        outputs=:reward,
    )
end
function CognitiveModeling.stopping_criterion(
    environment::RewardsUntilEnd, state, parameters
)
    return state.n_timesteps >= parameters.last_timestep
end

# The same environment as RewardsFollowActions, which gives no reward on every third timestep, so that the
# observation is sometimes NoValue and has two types inside the loop
struct RewardsNotAlwaysGiven <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RewardsNotAlwaysGiven, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::RewardsNotAlwaysGiven, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    n_timesteps = context.previous_state.n_timesteps + 1
    output_distribution = if iszero(mod(n_timesteps, 3))
        (; reward=NoValue())
    else
        (; reward=Deterministic(context.parameters.reward_size * action))
    end
    return ModelOutput(output_distribution, (; n_timesteps))
end
function CognitiveModeling.attribute_names(environment::RewardsNotAlwaysGiven)
    return ModelAttributes(;
        parameters=:reward_size, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

# Sums the expected rewards in the states of a simulation
function sum_expected_rewards(cognitive_states)
    total = zero(first(cognitive_states).expected_reward)
    for state in cognitive_states
        total += state.expected_reward
    end
    return total
end

# The same simulation with an environment under other names, which maps of names pass on
function simulate_with_name_maps(x, who_starts)
    simulation = simulate(
        LearnsRewards{Deterministic}(),
        PayoffsFollowLevers();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(SaveEverything()),
        observation_map=NameMap(:payoff => :reward),
        action_map=NameMap(:choice => :lever),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation which is deterministic, with the parameters from a vector
function simulate_deterministic(x, who_starts)
    simulation = simulate(
        LearnsRewards{Deterministic}(),
        RewardsFollowActions();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation which the environment ends before n_steps, with the parameters from a vector
function simulate_until_environment_ends(x, who_starts)
    simulation = simulate(
        LearnsRewards{Deterministic}(),
        RewardsUntilEnd();
        who_starts,
        n_steps=40,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8, last_timestep=20),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation where the reward isn't given on every timestep, saving everything or only the states, with the
# parameters from a vector. Enzyme reverse throws when the observations are saved, which is a bug in Enzyme: see
# KNOWN_FAILURES in ad/runtests.jl
function simulate_with_rewards_not_always_given(x, who_starts, cognitive_save_history)
    simulation = simulate(
        LearnsRewards{Deterministic}(),
        RewardsNotAlwaysGiven();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history,
    )
    return sum_expected_rewards(simulation.states)
end

# A model run on one observation at a time with simulate, passing its state and action on, with the parameters from a vector
function simulate_on_observations(x)
    parameters = (; learning_rate=x[1], inverse_temperature=x[2])
    action, state = simulate(LearnsRewards{Deterministic}(), (; reward=1.0); parameters)
    total = action.choice
    for reward in (0.0, 1.0, 1.0, 0.0)
        action, state = simulate(
            LearnsRewards{Deterministic}(),
            (; reward);
            parameters,
            previous_state=state,
            previous_output=action,
        )
        total += action.choice
    end
    return total
end

# A simulation with sampled actions and rewards from a seeded generator, with the parameters from a vector.
# Enzyme reverse gives wrong gradients here when the agent starts, with no error, which is a bug in Enzyme: see
# KNOWN_FAILURES in ad/runtests.jl. The number of timesteps is a constant because that's part of what triggers it
function simulate_sampled(x, who_starts)
    simulation = simulate(
        LearnsRewards{SampledAction}(),
        SampledRewards();
        who_starts,
        n_steps=30,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_probability=0.8),
        rng=Xoshiro(1),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation with a size for each timestep, with the parameters from a vector. The sizes are returned after
# BeforeFirstTimestep for timestep 0, which Enzyme reverse throws on, which is a bug in Enzyme: see
# KNOWN_FAILURES in ad/runtests.jl
const TIMESTEP_SIZES = collect(range(0.5, 1.5; length=20))
function simulate_with_timestep_sizes(x, who_starts)
    simulation = simulate(
        LearnsOverTime(),
        RewardsGrowOverTime();
        who_starts,
        n_steps=20,
        timestep_sizes=TIMESTEP_SIZES,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation with the timestep sizes themselves from a vector
function simulate_timestep_sizes_from_vector(x, who_starts)
    simulation = simulate(
        LearnsOverTime(),
        RewardsGrowOverTime();
        who_starts,
        n_steps=5,
        timestep_sizes=x,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation without states, with the parameters from a vector
function simulate_without_states(x, who_starts)
    simulation = simulate(
        ChoosesWithoutState(),
        RewardsWithoutState();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; inverse_temperature=x[1]),
        environment_parameters=(; reward_size=x[2]),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    #Timestep 0 has no reward, so the sum starts at timestep 1
    total = zero(x[2])
    for t in 2:length(simulation.observations)
        total += simulation.observations[t].reward
    end
    return total
end

# A simulation with noise in the model, with the parameters from a vector
function simulate_with_noise_in_model(x, who_starts)
    simulation = simulate(
        LearnsWithNoise(),
        RewardsWithoutState();
        who_starts,
        n_steps=30,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        rng=Xoshiro(1),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    return sum_expected_rewards(simulation.states)
end

# A simulation with a state changed in place, with the parameters from a vector
function simulate_in_place(x, who_starts, model=LearnsInPlace())
    simulation = simulate(
        model,
        RewardsWithoutState();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(SaveEverything()),
    )
    total = zero(x[1])
    for state in simulation.states
        total += state.expected_rewards[1]
    end
    return total
end

# A simulation which saves only the state of the model, with the parameters from a vector
function simulate_with_selection(x, who_starts)
    simulation = simulate(
        LearnsRewards{Deterministic}(),
        RewardsFollowActions();
        who_starts,
        n_steps=20,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_size=0.8),
        cognitive_save_history=SaveHistory(; states=:expected_reward),
        environment_save_history=SaveHistory(),
    )
    return sum_expected_rewards(simulation.states)
end

# An agent simulated with an environment, saving all or some of what happens, with the parameters from a vector.
# Both Enzyme forms with runtime activity fail here, and in read_history_after_environment, because the state starts as
# a constant: forward gives wrong gradients with no error, and reverse throws an error, which is a bug in Enzyme: see
# KNOWN_FAILURES in ad/runtests.jl
function simulate_agent_with_environment(x, who_starts, save_history)
    agent = InstantiatedModel(
        LearnsRewards{Deterministic}(),
        (; learning_rate=x[1], inverse_temperature=x[2]);
        save_history,
    )
    environment = InstantiatedModel(RewardsFollowActions(), (; reward_size=0.8))
    simulation = simulate!(agent, environment; who_starts, n_steps=20)
    return sum_expected_rewards(simulation.states)
end

# An agent simulated on one observation at a time, with or without its history, with the parameters from a vector
function simulate_agent_on_observations(x, save_history)
    agent = InstantiatedModel(
        LearnsRewards{Deterministic}(),
        (; learning_rate=x[1], inverse_temperature=x[2]);
        save_history,
    )
    total = zero(x[1])
    for reward in (1.0, 0.0, 1.0, 1.0, 0.0)
        total += simulate!(agent, (; reward)).choice
    end
    return total
end

# The states read back from the history of an agent, with the parameters from a vector
function read_history_after_environment(x, who_starts)
    agent = InstantiatedModel(
        LearnsRewards{Deterministic}(),
        (; learning_rate=x[1], inverse_temperature=x[2]);
        save_history=SaveHistory(SaveEverything()),
    )
    environment = InstantiatedModel(RewardsFollowActions(), (; reward_size=0.8))
    simulate!(agent, environment; who_starts, n_steps=20)
    return sum_expected_rewards(agent.history.states[2:end])
end

# The states read back from the history of an agent with a state changed in place, with the parameters from a vector
function read_history_in_place(x, model=LearnsInPlace())
    agent = InstantiatedModel(
        model,
        (; learning_rate=x[1], inverse_temperature=x[2]);
        save_history=SaveHistory(SaveEverything()),
    )
    for reward in (1.0, 0.0, 1.0, 1.0, 0.0)
        simulate!(agent, (; reward))
    end
    total = zero(x[1])
    for state in agent.history.states
        total += state.expected_rewards[1]
    end
    return total
end

# What the AD and Reactant tests check. In each entry,
# f takes a vector x and returns a number, such as a model's log likelihood for example data.
# Set reactant = false for what can't run on Reactant. Here that's Xoshiro, which Reactant doesn't support.
const TEST_CASES = [
    (;
        name="simulate, environment starts",
        f=x -> simulate_deterministic(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate, agent starts",
        f=x -> simulate_deterministic(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with sampling, environment starts",
        f=x -> simulate_sampled(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="simulate with sampling, agent starts",
        f=x -> simulate_sampled(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="simulate with timestep sizes, environment starts",
        f=x -> simulate_with_timestep_sizes(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with timestep sizes, agent starts",
        f=x -> simulate_with_timestep_sizes(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="gradient of the timestep sizes, environment starts",
        f=x -> simulate_timestep_sizes_from_vector(x, EnvironmentStarts()),
        x=[0.5, 2.0, 0.25, 1.5, 3.0],
        reactant=true,
    ),
    (;
        name="gradient of the timestep sizes, agent starts",
        f=x -> simulate_timestep_sizes_from_vector(x, AgentStarts()),
        x=[0.5, 2.0, 0.25, 1.5, 3.0],
        reactant=true,
    ),
    (;
        name="simulate without states, environment starts",
        f=x -> simulate_without_states(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate without states, agent starts",
        f=x -> simulate_without_states(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with noise in the model, environment starts",
        f=x -> simulate_with_noise_in_model(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="simulate with noise in the model, agent starts",
        f=x -> simulate_with_noise_in_model(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="simulate with a state changed in place, environment starts",
        f=x -> simulate_in_place(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a state changed in place, agent starts",
        f=x -> simulate_in_place(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a selection of what is saved, environment starts",
        f=x -> simulate_with_selection(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a selection of what is saved, agent starts",
        f=x -> simulate_with_selection(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! with a history, environment starts",
        f=x -> simulate_agent_with_environment(
            x, EnvironmentStarts(), SaveHistory(SaveEverything())
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! with a history, agent starts",
        f=x -> simulate_agent_with_environment(
            x, AgentStarts(), SaveHistory(SaveEverything())
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! saving only the states, environment starts",
        f=x -> simulate_agent_with_environment(
            x, EnvironmentStarts(), SaveHistory(; states=:expected_reward)
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! saving only the states, agent starts",
        f=x -> simulate_agent_with_environment(
            x, AgentStarts(), SaveHistory(; states=:expected_reward)
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! on single observations, with a history",
        f=x -> simulate_agent_on_observations(x, SaveHistory(SaveEverything())),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate! on single observations, without a history",
        f=x -> simulate_agent_on_observations(x, SaveHistory()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="reading the history of an agent, environment starts",
        f=x -> read_history_after_environment(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="reading the history of an agent, agent starts",
        f=x -> read_history_after_environment(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="reading the history of an agent with a state changed in place",
        f=read_history_in_place,
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a state saved with deepcopy, environment starts",
        f=x -> simulate_in_place(x, EnvironmentStarts(), LearnsInPlaceWithDeepcopy()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a state saved with deepcopy, agent starts",
        f=x -> simulate_in_place(x, AgentStarts(), LearnsInPlaceWithDeepcopy()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="reading the history of an agent with a state saved with deepcopy",
        f=x -> read_history_in_place(x, LearnsInPlaceWithDeepcopy()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with maps of names, environment starts",
        f=x -> simulate_with_name_maps(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with maps of names, agent starts",
        f=x -> simulate_with_name_maps(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate until the environment ends, environment starts",
        f=x -> simulate_until_environment_ends(x, EnvironmentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate until the environment ends, agent starts",
        f=x -> simulate_until_environment_ends(x, AgentStarts()),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate on single observations",
        f=simulate_on_observations,
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a reward that isn't always given, environment starts",
        f=x -> simulate_with_rewards_not_always_given(
            x, EnvironmentStarts(), SaveHistory(SaveEverything())
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a reward that isn't always given, agent starts",
        f=x -> simulate_with_rewards_not_always_given(
            x, AgentStarts(), SaveHistory(SaveEverything())
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a reward that isn't always given, saving only the states, environment starts",
        f=x -> simulate_with_rewards_not_always_given(
            x, EnvironmentStarts(), SaveHistory(; states=:expected_reward)
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
    (;
        name="simulate with a reward that isn't always given, saving only the states, agent starts",
        f=x -> simulate_with_rewards_not_always_given(
            x, AgentStarts(), SaveHistory(; states=:expected_reward)
        ),
        x=[0.3, 2.0],
        reactant=true,
    ),
]
