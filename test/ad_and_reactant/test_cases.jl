using CognitiveModeling: ModelInput, ModelOutput, EnvironmentInput, EnvironmentOutput
using Random: AbstractRNG, Xoshiro

# An action that is its own probability instead of a sample. It makes the simulation deterministic,
# and lets the gradient go all the way round the loop through the action and the environment
struct ProbabilityAsAction{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::ProbabilityAsAction) = action.probability

# An action that is sampled
struct SampledAction{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::SampledAction) = rand(rng) < action.probability

# A Rescorla-Wagner model. Its state starts as a Float64, whatever number type the parameters have
struct LearnsRewards{Taction} <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsRewards, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsRewards{Taction}, input::ModelInput) where {Taction}
    V₋₁ = input.previous_state.expected_reward
    α = input.parameters.learning_rate
    β = input.parameters.inverse_temperature
    Vₜ = input.observation isa NoObservation ? V₋₁ : V₋₁ + α * (input.observation.reward - V₋₁)
    return ModelOutput(Taction(inv(1 + exp(-β * (Vₜ - 0.5)))), (; expected_reward = Vₜ))
end

# An environment whose reward depends on the action, and which has a state
struct RewardsFollowActions <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsFollowActions, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::RewardsFollowActions, input::EnvironmentInput)
    action = input.action isa NoAction ? 0.5 : input.action.action
    n_timesteps = input.previous_state.n_timesteps + 1
    return EnvironmentOutput((; reward = input.parameters.reward_size * action), (; n_timesteps))
end

# An environment whose reward is sampled
struct SampledRewards <: AbstractEnvironmentModel end
function CognitiveModeling.apply_model(environment::SampledRewards, input::EnvironmentInput)
    return EnvironmentOutput((; reward = float(rand(input.rng) < input.parameters.reward_probability)))
end

# A Rescorla-Wagner model which learns more the more time passes. It learns on the first timestep too, so that every size counts
struct LearnsOverTime <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsOverTime, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsOverTime, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    α = input.parameters.learning_rate
    β = input.parameters.inverse_temperature
    r = input.observation isa NoObservation ? 0.5 : input.observation.reward
    Vₜ = V₋₁ + (1 - exp(-α * input.timestep_size)) * (r - V₋₁)
    return ModelOutput(ProbabilityAsAction(inv(1 + exp(-β * (Vₜ - 0.5)))), (; expected_reward = Vₜ))
end

# An environment whose reward grows with the time that passes
struct RewardsGrowOverTime <: AbstractEnvironmentModel end
function CognitiveModeling.apply_model(environment::RewardsGrowOverTime, input::EnvironmentInput)
    action = input.action isa NoAction ? 0.5 : input.action.action
    return EnvironmentOutput((; reward = input.parameters.reward_size * action * input.timestep_size))
end

# A model and an environment which leave out their states
struct ChoosesWithoutState <: AbstractCognitiveModel end
function CognitiveModeling.apply_model(model::ChoosesWithoutState, input::ModelInput)
    return ModelOutput(ProbabilityAsAction(inv(1 + exp(-input.parameters.inverse_temperature))))
end
struct RewardsWithoutState <: AbstractEnvironmentModel end
function CognitiveModeling.apply_model(environment::RewardsWithoutState, input::EnvironmentInput)
    action = input.action isa NoAction ? 0.5 : input.action.action
    return EnvironmentOutput((; reward = input.parameters.reward_size * action))
end

# A model whose state update has noise in it
struct LearnsWithNoise <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsWithNoise, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsWithNoise, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    α = input.parameters.learning_rate
    β = input.parameters.inverse_temperature
    r = input.observation isa NoObservation ? 0.5 : input.observation.reward
    Vₜ = V₋₁ + α * (r - V₋₁) + α * 0.1 * (rand(input.rng) - 0.5)
    return ModelOutput(ProbabilityAsAction(inv(1 + exp(-β * (Vₜ - 0.5)))), (; expected_reward = Vₜ))
end

# A Rescorla-Wagner model which changes a Vector in its state in place, and copies it when it's saved
struct LearnsInPlace <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsInPlace, parameters)
    return (; expected_rewards = [0.5 * one(parameters.learning_rate), 0.5 * one(parameters.learning_rate)])
end
function CognitiveModeling.apply_model(model::LearnsInPlace, input::ModelInput)
    V = input.previous_state.expected_rewards
    α = input.parameters.learning_rate
    β = input.parameters.inverse_temperature
    r = input.observation isa NoObservation ? 0.5 : input.observation.reward
    V[1] += α * (r - V[1])
    return ModelOutput(ProbabilityAsAction(inv(1 + exp(-β * (V[1] - 0.5)))), input.previous_state; save_state = state -> (; expected_rewards = copy(state.expected_rewards)))
end

# Sums the expected rewards in the states of a simulation
function sum_expected_rewards(cognitive_states)
    total = zero(first(cognitive_states).expected_reward)
    for state in cognitive_states
        total += state.expected_reward
    end
    return total
end

# A simulation which is deterministic, with the parameters from a vector
function simulate_deterministic(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsRewards{ProbabilityAsAction}(),
        RewardsFollowActions();
        who_starts,
        n_steps = 20,
        cognitive_parameters = (; learning_rate = x[1], inverse_temperature = x[2]),
        environment_parameters = (; reward_size = 0.8),
    )
    return sum_expected_rewards(cognitive_states)
end

# A simulation with sampled actions and rewards from a seeded generator, with the parameters from a vector.
# Enzyme reverse gives wrong gradients here when the agent starts, with no error. That is a bug to report to Enzyme,
# so those tests are left failing until it is fixed. The number of timesteps is a constant because that's part of what triggers it
function simulate_sampled(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsRewards{SampledAction}(),
        SampledRewards();
        who_starts,
        n_steps = 30,
        cognitive_parameters = (; learning_rate = x[1], inverse_temperature = x[2]),
        environment_parameters = (; reward_probability = 0.8),
        rng = Xoshiro(1),
    )
    return sum_expected_rewards(cognitive_states)
end

# A simulation with a size for each timestep, with the parameters from a vector
const TIMESTEP_SIZES = collect(range(0.5, 1.5; length = 20))
function simulate_with_timestep_sizes(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsOverTime(),
        RewardsGrowOverTime();
        who_starts,
        n_steps = 20,
        timestep_sizes = TIMESTEP_SIZES,
        cognitive_parameters = (; learning_rate = x[1], inverse_temperature = x[2]),
        environment_parameters = (; reward_size = 0.8),
    )
    return sum_expected_rewards(cognitive_states)
end

# A simulation with the timestep sizes themselves from a vector
function simulate_timestep_sizes_from_vector(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsOverTime(),
        RewardsGrowOverTime();
        who_starts,
        n_steps = 5,
        timestep_sizes = x,
        cognitive_parameters = (; learning_rate = 0.3, inverse_temperature = 2.0),
        environment_parameters = (; reward_size = 0.8),
    )
    return sum_expected_rewards(cognitive_states)
end

# A simulation without states, with the parameters from a vector
function simulate_without_states(x, who_starts)
    observations, _, _, _ = simulate(
        ChoosesWithoutState(),
        RewardsWithoutState();
        who_starts,
        n_steps = 20,
        cognitive_parameters = (; inverse_temperature = x[1]),
        environment_parameters = (; reward_size = x[2]),
    )
    total = zero(first(observations).reward)
    for observation in observations
        total += observation.reward
    end
    return total
end

# A simulation with noise in the model, with the parameters from a vector
function simulate_with_noise_in_model(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsWithNoise(),
        RewardsWithoutState();
        who_starts,
        n_steps = 30,
        cognitive_parameters = (; learning_rate = x[1], inverse_temperature = x[2]),
        environment_parameters = (; reward_size = 0.8),
        rng = Xoshiro(1),
    )
    return sum_expected_rewards(cognitive_states)
end

# A simulation with a state changed in place, with the parameters from a vector
function simulate_in_place(x, who_starts)
    _, _, cognitive_states, _ = simulate(
        LearnsInPlace(),
        RewardsWithoutState();
        who_starts,
        n_steps = 20,
        cognitive_parameters = (; learning_rate = x[1], inverse_temperature = x[2]),
        environment_parameters = (; reward_size = 0.8),
    )
    total = zero(x[1])
    for state in cognitive_states
        total += state.expected_rewards[1]
    end
    return total
end

# What the AD and Reactant tests check. In each entry,
# f takes a vector x and returns a number, such as a model's log likelihood for example data.
# Set reactant = false for what can't run on Reactant. Here that's Xoshiro, which Reactant doesn't support.
const TEST_CASES = [
    (; name = "simulate, environment starts", f = x -> simulate_deterministic(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate, agent starts", f = x -> simulate_deterministic(x, AgentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate with sampling, environment starts", f = x -> simulate_sampled(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = false),
    (; name = "simulate with sampling, agent starts", f = x -> simulate_sampled(x, AgentStarts()), x = [0.3, 2.0], reactant = false),
    (; name = "simulate with timestep sizes, environment starts", f = x -> simulate_with_timestep_sizes(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate with timestep sizes, agent starts", f = x -> simulate_with_timestep_sizes(x, AgentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "gradient of the timestep sizes, environment starts", f = x -> simulate_timestep_sizes_from_vector(x, EnvironmentStarts()), x = [0.5, 2.0, 0.25, 1.5, 3.0], reactant = true),
    (; name = "gradient of the timestep sizes, agent starts", f = x -> simulate_timestep_sizes_from_vector(x, AgentStarts()), x = [0.5, 2.0, 0.25, 1.5, 3.0], reactant = true),
    (; name = "simulate without states, environment starts", f = x -> simulate_without_states(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate without states, agent starts", f = x -> simulate_without_states(x, AgentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate with noise in the model, environment starts", f = x -> simulate_with_noise_in_model(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = false),
    (; name = "simulate with noise in the model, agent starts", f = x -> simulate_with_noise_in_model(x, AgentStarts()), x = [0.3, 2.0], reactant = false),
    (; name = "simulate with a state changed in place, environment starts", f = x -> simulate_in_place(x, EnvironmentStarts()), x = [0.3, 2.0], reactant = true),
    (; name = "simulate with a state changed in place, agent starts", f = x -> simulate_in_place(x, AgentStarts()), x = [0.3, 2.0], reactant = true),
]
