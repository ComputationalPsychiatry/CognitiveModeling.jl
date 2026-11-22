#The example models in the Turing.jl suites: the kind of distributions which gives those of Distributions.jl, which
#Turing.jl scores, and the Turing.jl methods of the models which sample something of their own, with ~ instead of from a
#random number generator. A runner includes this file after ExampleModels.jl, with Turing.jl loaded

using Random: Random
using Turing: Distributions

#The distributions of Distributions.jl, and three of the same form, for values that aren't numbers
struct DistributionsJl end

ExampleModels.bernoulli(::DistributionsJl, probability) = Bernoulli(probability)
ExampleModels.bernoulli_logit(::DistributionsJl, logit) = BernoulliLogit(logit)
function ExampleModels.normal(::DistributionsJl, mean, standard_deviation)
    return Normal(mean, standard_deviation)
end
ExampleModels.categorical(::DistributionsJl, probabilities) = Categorical(probabilities)
function ExampleModels.independent_normals(::DistributionsJl, means, standard_deviation)
    n_means = length(means)
    identity_matrix = [row == column ? 1.0 : 0.0 for row in 1:n_means, column in 1:n_means]
    return MvNormal(means, standard_deviation^2 .* identity_matrix)
end

#A choice of :right with a probability, and :left otherwise
struct SymbolChoiceDistribution{T<:Real} <:
       Distributions.Distribution{Distributions.Univariate,Distributions.ValueSupport}
    probability_right::T
end
function Distributions.logpdf(distribution::SymbolChoiceDistribution, value::Symbol)
    return ExampleModels.log_density(
        ExampleModels.SymbolChoiceByHand(distribution.probability_right), value
    )
end
function Base.rand(rng::Random.AbstractRNG, distribution::SymbolChoiceDistribution)
    return rand(rng, ExampleModels.SymbolChoiceByHand(distribution.probability_right))
end
function ExampleModels.symbol_choice(::DistributionsJl, probability_right)
    return SymbolChoiceDistribution(probability_right)
end

#A NamedTuple of a choice, 1 with a probability and 2 otherwise, and a time from an exponential distribution with a rate
struct ChoiceAndTimeDistribution{T<:Real} <:
       Distributions.Distribution{Distributions.Multivariate,Distributions.ValueSupport}
    probability::T
    rate::T
end
function Distributions.logpdf(distribution::ChoiceAndTimeDistribution, value::NamedTuple)
    return ExampleModels.log_density(
        ExampleModels.ChoiceAndTimeByHand(distribution.probability, distribution.rate),
        value,
    )
end
function Base.rand(rng::Random.AbstractRNG, distribution::ChoiceAndTimeDistribution)
    return rand(
        rng, ExampleModels.ChoiceAndTimeByHand(distribution.probability, distribution.rate)
    )
end
function ExampleModels.choice_and_time(::DistributionsJl, probability, rate)
    return ChoiceAndTimeDistribution(promote(probability, rate)...)
end

#A NamedTuple of a reward, true with a probability, and a cue from a normal distribution
struct RewardAndCueDistribution{T<:Real} <:
       Distributions.Distribution{Distributions.Multivariate,Distributions.ValueSupport}
    probability::T
    cue_mean::T
    cue_noise::T
end
function written_by_hand(distribution::RewardAndCueDistribution)
    return ExampleModels.RewardAndCueByHand(
        distribution.probability, distribution.cue_mean, distribution.cue_noise
    )
end
function Distributions.logpdf(distribution::RewardAndCueDistribution, value::NamedTuple)
    return ExampleModels.log_density(written_by_hand(distribution), value)
end
function Base.rand(rng::Random.AbstractRNG, distribution::RewardAndCueDistribution)
    return rand(rng, written_by_hand(distribution))
end
function ExampleModels.reward_and_cue(::DistributionsJl, probability, cue_mean, cue_noise)
    return RewardAndCueDistribution(promote(probability, cue_mean, cue_noise)...)
end

## The models which sample something of their own ##
#The noise of each timestep, named timesteps[t].noise in a session
Turing.@model function CognitiveModeling.turing_timestep_model(
    model::Union{SamplingWithinTimestep,EveryFeature}, context::ModelContext
)
    noise ~ ExampleModels.normal(model.distributions, 0.0, context.parameters.update_noise)
    return ExampleModels.update_with_noise(model, context, noise)
end

#The offset of where the expected reward starts, named initial_cognitive_state.initial_offset in a session
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::Union{SampledInitialState,EveryFeature}, parameters
)
    initial_offset ~
    ExampleModels.normal(model.distributions, 0.0, parameters.initial_noise)
    state = CognitiveModeling.initialize_state(model, parameters)
    return merge(state, (; expected_reward=state.expected_reward + initial_offset))
end

#The drift of each timestep, named timesteps[t].environment.drift in a session
Turing.@model function CognitiveModeling.turing_timestep_model(
    environment::Union{SamplingWithinTimestepEnvironment,EveryFeatureEnvironment},
    context::ModelContext,
)
    drift ~
    ExampleModels.normal(environment.distributions, 0.0, context.parameters.drift_noise)
    return ExampleModels.update_with_drift(environment, context, drift)
end

## What the models sampled in a session ##
#What a model sampled of its own in a session, read with `variable`, which gives the value of a variable of the session by
#its name within the session, in the form that log_likelihood_by_hand and log_density_of_latents_by_hand take it. Most
#models sample nothing of their own
latents_from_variables(model, variable, n_timesteps) = (;)
function latents_from_variables(model::SamplingWithinTimestep, variable, n_timesteps)
    return (; noise=[variable("timesteps[$t].noise") for t in 1:n_timesteps])
end
function latents_from_variables(model::SampledInitialState, variable, n_timesteps)
    return (; initial_offset=variable("initial_cognitive_state.initial_offset"))
end
function latents_from_variables(model::EveryFeature, variable, n_timesteps)
    return (;
        initial_offset=variable("initial_cognitive_state.initial_offset"),
        noise=[variable("timesteps[$t].noise") for t in 1:n_timesteps],
    )
end
function latents_from_variables(model::EveryFeatureWithoutSampling, variable, n_timesteps)
    return (; initial_offset=0.0, noise=zeros(n_timesteps))
end
function latents_from_variables(
    environment::Union{SamplingWithinTimestepEnvironment,EveryFeatureEnvironment},
    variable,
    n_timesteps,
)
    return (; drift=[variable("timesteps[$t].environment.drift") for t in 1:n_timesteps])
end

#A function which gives the value of a variable of a model by the name of the variable, from values drawn from the model,
#with a prefix, such as "sessions[2].", for the variables of a session among several
function variables_of(values; prefix="")
    by_name = Dict(string(name) => values[name] for name in keys(values))
    return name -> by_name[prefix * name]
end
