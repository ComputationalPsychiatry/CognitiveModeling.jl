#The Turing.jl part of the code that the test suites share: the methods of the kind of distributions which gives those of
#Distributions.jl, which Turing.jl scores, with three of the same form for values that aren't numbers, the priors of the
#inference cases, and the Turing.jl methods of the models which sample something of their own, with ~ instead of from a
#random number generator. It loads with Turing.jl
module CommonTestFunctionalityTuringExt

using CognitiveModeling
using CommonTestFunctionality
using CommonTestFunctionality:
    SymbolChoiceByHand,
    ChoiceAndTimeByHand,
    RewardAndCueByHand,
    log_density,
    update_with_noise,
    update_with_drift
using Random: Random
using Turing
using Turing: Distributions

## The distributions of Distributions.jl, and three of the same form, for values that aren't numbers ##
CommonTestFunctionality.bernoulli(::DistributionsJl, probability) = Bernoulli(probability)
CommonTestFunctionality.bernoulli_logit(::DistributionsJl, logit) = BernoulliLogit(logit)
function CommonTestFunctionality.normal(::DistributionsJl, mean, standard_deviation)
    return Normal(mean, standard_deviation)
end
function CommonTestFunctionality.categorical(::DistributionsJl, probabilities)
    return Categorical(probabilities)
end
function CommonTestFunctionality.independent_normals(
    ::DistributionsJl, means, standard_deviation
)
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
    return CommonTestFunctionality.log_density(
        CommonTestFunctionality.SymbolChoiceByHand(distribution.probability_right), value
    )
end
function Base.rand(rng::Random.AbstractRNG, distribution::SymbolChoiceDistribution)
    return rand(
        rng, CommonTestFunctionality.SymbolChoiceByHand(distribution.probability_right)
    )
end
function CommonTestFunctionality.symbol_choice(::DistributionsJl, probability_right)
    return SymbolChoiceDistribution(probability_right)
end

#A NamedTuple of a choice, 1 with a probability and 2 otherwise, and a time from an exponential distribution with a rate
struct ChoiceAndTimeDistribution{T<:Real} <:
       Distributions.Distribution{Distributions.Multivariate,Distributions.ValueSupport}
    probability::T
    rate::T
end
function Distributions.logpdf(distribution::ChoiceAndTimeDistribution, value::NamedTuple)
    return CommonTestFunctionality.log_density(
        CommonTestFunctionality.ChoiceAndTimeByHand(
            distribution.probability, distribution.rate
        ),
        value,
    )
end
function Base.rand(rng::Random.AbstractRNG, distribution::ChoiceAndTimeDistribution)
    return rand(
        rng,
        CommonTestFunctionality.ChoiceAndTimeByHand(
            distribution.probability, distribution.rate
        ),
    )
end
function CommonTestFunctionality.choice_and_time(::DistributionsJl, probability, rate)
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
    return CommonTestFunctionality.RewardAndCueByHand(
        distribution.probability, distribution.cue_mean, distribution.cue_noise
    )
end
function Distributions.logpdf(distribution::RewardAndCueDistribution, value::NamedTuple)
    return CommonTestFunctionality.log_density(written_by_hand(distribution), value)
end
function Base.rand(rng::Random.AbstractRNG, distribution::RewardAndCueDistribution)
    return rand(rng, written_by_hand(distribution))
end
function CommonTestFunctionality.reward_and_cue(
    ::DistributionsJl, probability, cue_mean, cue_noise
)
    return RewardAndCueDistribution(promote(probability, cue_mean, cue_noise)...)
end

## The priors of the inference cases ##
function CommonTestFunctionality.beta(::DistributionsJl, first_shape, second_shape)
    return Beta(first_shape, second_shape)
end
function CommonTestFunctionality.log_normal(::DistributionsJl, mean, standard_deviation)
    return LogNormal(mean, standard_deviation)
end
CommonTestFunctionality.exponential(::DistributionsJl, scale) = Exponential(scale)
function CommonTestFunctionality.product_of(::DistributionsJl, distributions)
    return product_distribution(distributions)
end

## The models which sample something of their own ##
#The noise of each timestep, named timesteps[t].noise in a session
Turing.@model function CognitiveModeling.turing_timestep_model(
    model::Union{SamplingWithinTimestep,EveryFeature}, context::ModelContext
)
    noise ~ CommonTestFunctionality.normal(
        model.distributions, 0.0, context.parameters.update_noise
    )
    return CommonTestFunctionality.update_with_noise(model, context, noise)
end

#The offset of where the expected reward starts, named initial_cognitive_state.initial_offset in a session
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::Union{SampledInitialState,EveryFeature}, parameters
)
    initial_offset ~
    CommonTestFunctionality.normal(model.distributions, 0.0, parameters.initial_noise)
    state = CognitiveModeling.initialize_state(model, parameters)
    return merge(state, (; expected_reward=state.expected_reward + initial_offset))
end

#The drift of each timestep, named timesteps[t].environment.drift in a session
Turing.@model function CognitiveModeling.turing_timestep_model(
    environment::Union{SamplingWithinTimestepEnvironment,EveryFeatureEnvironment},
    context::ModelContext,
)
    drift ~ CommonTestFunctionality.normal(
        environment.distributions, 0.0, context.parameters.drift_noise
    )
    return CommonTestFunctionality.update_with_drift(environment, context, drift)
end

end
