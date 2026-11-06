using Turing
using Turing.DynamicPPL: LogDensityFunction, LogDensityProblems

# Test cases for the log density of Turing.jl models of a session, which only this suite runs,
# since Turing.jl isn't loaded by the AD and Reactant suites of the package itself, in
# test/ad_and_reactant. Each case
# differentiates the log density the way a sampler does: through a LogDensityFunction made
# once, with the parameters and the latents in one vector in the order of the model. The
# parameters have flat priors, so the log density is the session's own.

# A Rescorla-Wagner model with a softmax choice, which doesn't learn when there is no observation
struct TuringRescorlaWagner <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TuringRescorlaWagner, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::TuringRescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    r = input.observation.reward
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + input.parameters.learning_rate * (r - V₋₁)
    return ModelOutput((; choice = BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::TuringRescorlaWagner) = ModelAttributes(; parameters = (:learning_rate, :inverse_temperature), states = :expected_reward, observations = :reward, actions = :choice)

# A continuous action, so that missing ones can be imputed
struct TuringContinuousAction <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TuringContinuousAction, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::TuringContinuousAction, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Normal(Vₜ, input.parameters.noise_scale)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::TuringContinuousAction) = ModelAttributes(; parameters = (:learning_rate, :noise_scale), states = :expected_reward, observations = :reward, actions = :choice)

# Samples noise within each timestep in Turing
struct TuringNoisyModel <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TuringNoisyModel, parameters) = (; expected_reward = 0.5)
Turing.@model function CognitiveModeling.turing_apply_model(model::TuringNoisyModel, input::ModelInput)
    noise ~ Normal(0, input.parameters.noise_scale)
    Vₜ = input.previous_state.expected_reward + input.parameters.learning_rate * (input.observation.reward - input.previous_state.expected_reward) + noise
    return ModelOutput((; choice = Normal(Vₜ, 1.0)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::TuringNoisyModel) = ModelAttributes(; parameters = (:learning_rate, :noise_scale), states = :expected_reward, observations = :reward, actions = :choice)

# Gives the previous choice as the reward, scaled by a parameter
struct TuringRewardsPreviousChoice <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::TuringRewardsPreviousChoice, parameters) = (;)
function CognitiveModeling.apply_model(environment::TuringRewardsPreviousChoice, input::EnvironmentInput)
    reward = input.action.choice isa AbstractNoValue ? 0.5 : input.action.choice
    return EnvironmentOutput((; reward = reward * input.parameters.scale))
end
CognitiveModeling.attribute_names(environment::TuringRewardsPreviousChoice) = ModelAttributes(; parameters = :scale, states = (), observations = :reward, actions = :choice)

const TURING_REWARDS = (; reward = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
const TURING_CHOICES = (; choice = [true, false, true, true, false, false, true, true, false, true])
const TURING_CONTINUOUS_CHOICES = (; choice = [0.3, 0.6, MissingAction(), 0.4, 0.7, MissingAction(), 0.5, 0.2, 0.8, 0.6])
const TURING_MISSING_REWARDS = (; reward = [MissingObservation(), 0.0, MissingObservation(), 1.0, MissingObservation(), 1.0, 0.0, MissingObservation(), 1.0, 1.0])

# The sessions, with flat priors on the parameters
Turing.@model function fit_rescorla_wagner(who_starts)
    learning_rate ~ Turing.Flat()
    inverse_temperature ~ Turing.Flat()
    session ~ to_submodel(prefix(turing_single_session(TuringRescorlaWagner(), (; learning_rate, inverse_temperature), TURING_REWARDS, TURING_CHOICES; who_starts), @varname(session)), false)
    return nothing
end

Turing.@model function fit_with_imputed_actions()
    learning_rate ~ Turing.Flat()
    noise_scale ~ Turing.Flat()
    session ~ to_submodel(prefix(turing_single_session(TuringContinuousAction(), (; learning_rate, noise_scale), TURING_REWARDS, TURING_CONTINUOUS_CHOICES; who_starts = EnvironmentStarts()), @varname(session)), false)
    return nothing
end

Turing.@model function fit_with_environment()
    learning_rate ~ Turing.Flat()
    inverse_temperature ~ Turing.Flat()
    scale ~ Turing.Flat()
    session ~ to_submodel(prefix(turing_single_session(TuringRescorlaWagner(), (; learning_rate, inverse_temperature), TURING_MISSING_REWARDS, TURING_CHOICES; who_starts = EnvironmentStarts(), environment = TuringRewardsPreviousChoice(), environment_parameters = (; scale)), @varname(session)), false)
    return nothing
end

Turing.@model function fit_with_noise_in_timesteps()
    learning_rate ~ Turing.Flat()
    noise_scale ~ Turing.Flat()
    session ~ to_submodel(prefix(turing_single_session(TuringNoisyModel(), (; learning_rate, noise_scale), TURING_REWARDS, (; choice = TURING_REWARDS.reward .- 0.5); who_starts = EnvironmentStarts()), @varname(session)), false)
    return nothing
end

# The log density as a function of the vector of parameters and latents, through a LogDensityFunction made once
function log_density_of(model)
    log_density_function = LogDensityFunction(model)
    return x -> LogDensityProblems.logdensity(log_density_function, x)
end

const TURING_CASES = [
    (; name = "turing session, environment starts", f = log_density_of(fit_rescorla_wagner(EnvironmentStarts())), x = [0.3, 2.0], reactant = false),
    (; name = "turing session, agent starts", f = log_density_of(fit_rescorla_wagner(AgentStarts())), x = [0.3, 2.0], reactant = false),
    (; name = "turing session with imputed actions", f = log_density_of(fit_with_imputed_actions()), x = [0.3, 0.8, 0.2, -0.1], reactant = false),
    (; name = "turing session with an environment", f = log_density_of(fit_with_environment()), x = [0.3, 2.0, 0.9], reactant = false),
    (; name = "turing session with noise in each timestep", f = log_density_of(fit_with_noise_in_timesteps()), x = [0.3, 0.8, 0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0, -0.3, 0.15, 0.1], reactant = false),
]
