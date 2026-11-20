## The distributions that the example models give ##
#Every example model that gives outputs, or samples anything of its own, takes a kind of distributions, and calls the
#functions below with it, such as bernoulli(model.distributions, probability). Each suite chooses the kind it needs, so
#that the same models run with the distributions written by hand here, with their means, which makes a simulation
#deterministic, and with those of Distributions.jl in the Turing.jl suites, whose kind turing_methods.jl adds

#The distributions written by hand below, which need no package of distributions, for the main suite and for Reactant
struct ByHand end

#The mean of each distribution instead of the distribution, as a Deterministic, so that a simulation is deterministic and
#its outputs are differentiable around the loop: the probability of a choice of two options, the mean of a normal
#distribution, and the probabilities of the options of a choice of several
struct Means end

#A choice of true with a probability, and false otherwise
function bernoulli end

#A number from a normal distribution with a mean and a standard deviation
function normal end

#The number of an option, from the probabilities of the options in a vector
function categorical end

#A choice of :right with a probability, and :left otherwise
function symbol_choice end

#A vector of numbers, each from a normal distribution with its own mean in a vector, and one standard deviation
function independent_normals end

#A NamedTuple of a choice, 1 with a probability and 2 otherwise, and a time from an exponential distribution with a rate,
#independent of the choice, as a stand-in for the distributions of SequentialSamplingModels.jl
function choice_and_time end

#A NamedTuple of a reward, true with a probability and false otherwise, and a cue from a normal distribution with a mean
#and a standard deviation, independent of the reward
function reward_and_cue end

## Written by hand ##
#Each has a method of rand, which a simulation samples with, and of log_density, which the likelihoods by hand use. The
#type parameters let the numbers be numbers of automatic differentiation
struct BernoulliByHand{T<:Real}
    probability::T
end
function Base.rand(rng::AbstractRNG, distribution::BernoulliByHand)
    return rand(rng) < distribution.probability
end
function log_density(distribution::BernoulliByHand, value)
    return value ? log(distribution.probability) : log1p(-distribution.probability)
end
bernoulli(::ByHand, probability) = BernoulliByHand(probability)

struct NormalByHand{T<:Real}
    mean::T
    standard_deviation::T
end
NormalByHand(mean, standard_deviation) = NormalByHand(promote(mean, standard_deviation)...)
function Base.rand(rng::AbstractRNG, distribution::NormalByHand)
    return distribution.mean + distribution.standard_deviation * randn(rng)
end
function log_density(distribution::NormalByHand, value)
    standardized = (value - distribution.mean) / distribution.standard_deviation
    return -standardized^2 / 2 - log(distribution.standard_deviation) - log(2π) / 2
end
normal(::ByHand, mean, standard_deviation) = NormalByHand(mean, standard_deviation)

struct CategoricalByHand{T<:Real}
    probabilities::Vector{T}
end
function Base.rand(rng::AbstractRNG, distribution::CategoricalByHand)
    #The first option whose cumulative probability passes a uniform number
    threshold = rand(rng)
    cumulative = zero(eltype(distribution.probabilities))
    for (option, probability) in enumerate(distribution.probabilities)
        cumulative += probability
        threshold < cumulative && return option
    end
    return length(distribution.probabilities)
end
log_density(distribution::CategoricalByHand, value) = log(distribution.probabilities[value])
categorical(::ByHand, probabilities) = CategoricalByHand(collect(probabilities))

struct SymbolChoiceByHand{T<:Real}
    probability_right::T
end
function Base.rand(rng::AbstractRNG, distribution::SymbolChoiceByHand)
    return rand(rng) < distribution.probability_right ? :right : :left
end
function log_density(distribution::SymbolChoiceByHand, value)
    if value === :right
        return log(distribution.probability_right)
    else
        return log1p(-distribution.probability_right)
    end
end
symbol_choice(::ByHand, probability_right) = SymbolChoiceByHand(probability_right)

struct IndependentNormalsByHand{T<:Real}
    means::Vector{T}
    standard_deviation::T
end
function Base.rand(rng::AbstractRNG, distribution::IndependentNormalsByHand)
    return [
        rand(rng, NormalByHand(mean, distribution.standard_deviation)) for
        mean in distribution.means
    ]
end
function log_density(distribution::IndependentNormalsByHand, values)
    return sum(
        log_density(NormalByHand(mean, distribution.standard_deviation), value) for
        (mean, value) in zip(distribution.means, values)
    )
end
function independent_normals(::ByHand, means, standard_deviation)
    element_type = promote_type(eltype(means), typeof(standard_deviation))
    return IndependentNormalsByHand{element_type}(collect(means), standard_deviation)
end

struct ChoiceAndTimeByHand{T<:Real}
    probability::T
    rate::T
end
function Base.rand(rng::AbstractRNG, distribution::ChoiceAndTimeByHand)
    choice = rand(rng) < distribution.probability ? 1 : 2
    return (; choice, time=(-log(rand(rng)) / distribution.rate))
end
function log_density(distribution::ChoiceAndTimeByHand, value)
    choice_density = if value.choice == 1
        log(distribution.probability)
    else
        log1p(-distribution.probability)
    end
    return choice_density + log(distribution.rate) - distribution.rate * value.time
end
function choice_and_time(::ByHand, probability, rate)
    return ChoiceAndTimeByHand(promote(probability, rate)...)
end

struct RewardAndCueByHand{T<:Real}
    probability::T
    cue_mean::T
    cue_noise::T
end
function Base.rand(rng::AbstractRNG, distribution::RewardAndCueByHand)
    reward = rand(rng, BernoulliByHand(distribution.probability))
    cue = rand(rng, NormalByHand(distribution.cue_mean, distribution.cue_noise))
    return (; reward, cue)
end
function log_density(distribution::RewardAndCueByHand, value)
    return log_density(BernoulliByHand(distribution.probability), value.reward) +
           log_density(
        NormalByHand(distribution.cue_mean, distribution.cue_noise), value.cue
    )
end
function reward_and_cue(::ByHand, probability, cue_mean, cue_noise)
    return RewardAndCueByHand(promote(probability, cue_mean, cue_noise)...)
end

## Their means ##
bernoulli(::Means, probability) = Deterministic(probability)
normal(::Means, mean, standard_deviation) = Deterministic(mean)
categorical(::Means, probabilities) = Deterministic(probabilities)
independent_normals(::Means, means, standard_deviation) = Deterministic(means)
function reward_and_cue(::Means, probability, cue_mean, cue_noise)
    return Deterministic((; reward=probability, cue=cue_mean))
end
