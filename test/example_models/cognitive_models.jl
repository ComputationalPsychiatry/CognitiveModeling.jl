## Cognitive models ##
#Every model is a learner of a task with rewards, which differs from the baseline in the feature it is named after. Each
#is followed by the log likelihood of its outputs by hand, written without the model, from a vector of timesteps, each a
#NamedTuple of the input that the model received, the output it gave, and the size of the timestep. An output that is a
#marker, such as an action that wasn't recorded, adds nothing. A model that samples within a timestep or when it starts
#takes what it sampled as latents, whose log density is added separately by log_density_of_latents_by_hand

## Helpers ##
#The expected reward after a reward, which stays as it was on a timestep without a reward, whose input is a marker
function learn(expected_reward, learning_rate, reward)
    return expected_reward + learning_rate * (reward - expected_reward)
end
learn(expected_reward, learning_rate, reward::AbstractNoValue) = expected_reward

#The log odds of choosing true, which grow with the expected reward. The models give their choices of true and false as
#log odds, which Distributions.jl's BernoulliLogit takes for any size of them
function choice_logit(expected_reward, inverse_temperature)
    return inverse_temperature * (expected_reward - 0.5)
end

#A value, or what stands in for it when it is a marker
value_or(value, default) = value
value_or(value::AbstractNoValue, default) = default

#The log density of an output in the data, which adds nothing when the output is a marker
log_density_if_given(distribution, value) = log_density(distribution, value)
log_density_if_given(distribution, value::AbstractNoValue) = 0.0

#The log likelihood of the outputs of a model by hand, which every model below has a method of
function log_likelihood_by_hand end

#What a model samples of its own in a simulation, read from the states it saved, which are vectors with timestep 0 first
latents_from_states(model, states) = (;)

#The log density of what a model samples of its own, which is nothing for most models
log_density_of_latents_by_hand(model, parameters, latents) = 0.0

#Runs timestep t of a model with what it sampled of its own on that timestep in a simulation, read with
#latents_from_states, so that it gives the distributions it gave in the simulation. Most models sample nothing of their own
timestep_model_with_latents(model, context, latents, t) = timestep_model(model, context)

## The baseline ##
#Learns the expected reward of choosing true, and chooses true with a probability that grows with it
struct Baseline{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(model::Baseline, parameters) = (; expected_reward=0.5)
function CognitiveModeling.timestep_model(model::Baseline, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learning_rate,
        context.input.reward,
    )
    logit = choice_logit(expected_reward, context.parameters.inverse_temperature)
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::Baseline)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(model::Baseline, parameters, timesteps, latents=(;))
    expected_reward = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5))
        )
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

## Without one kind of attribute ##
#Learns with a fixed learning rate of a half, and chooses true with the expected reward as the probability
struct WithoutParameters{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::WithoutParameters, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::WithoutParameters, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward, 0.5, context.input.reward
    )
    return ModelOutput(
        (; choice=bernoulli(model.distributions, expected_reward)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::WithoutParameters)
    return ModelAttributes(;
        parameters=(), states=:expected_reward, inputs=:reward, outputs=:choice
    )
end
function log_likelihood_by_hand(
    model::WithoutParameters, parameters, timesteps, latents=(;)
)
    expected_reward = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward = (expected_reward + timestep.input.reward) / 2
        end
        log_likelihood += log_density_if_given(
            BernoulliByHand(expected_reward), timestep.output.choice
        )
    end
    return log_likelihood
end

#Chooses true with a probability that grows with the reward of the timestep, and remembers nothing
struct WithoutStates{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(model::WithoutStates, parameters) = (;)
function CognitiveModeling.timestep_model(model::WithoutStates, context::ModelContext)
    reward = value_or(context.input.reward, 0.5)
    logit = choice_logit(reward, context.parameters.inverse_temperature)
    return ModelOutput((; choice=bernoulli_logit(model.distributions, logit)))
end
function CognitiveModeling.attribute_names(model::WithoutStates)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=:reward, outputs=:choice
    )
end
function log_likelihood_by_hand(model::WithoutStates, parameters, timesteps, latents=(;))
    log_likelihood = 0.0
    for timestep in timesteps
        reward = timestep.input.reward isa AbstractNoValue ? 0.5 : timestep.input.reward
        probability = inv(1 + exp(-parameters.inverse_temperature * (reward - 0.5)))
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

#Observes nothing: a habit of choosing true grows on every timestep, and the probability of choosing true with it
struct WithoutInputs{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(model::WithoutInputs, parameters) = (; habit=0.5)
function CognitiveModeling.timestep_model(model::WithoutInputs, context::ModelContext)
    habit = learn(context.previous_state.habit, context.parameters.learning_rate, 1.0)
    logit = choice_logit(habit, context.parameters.inverse_temperature)
    return ModelOutput((; choice=bernoulli_logit(model.distributions, logit)), (; habit))
end
function CognitiveModeling.attribute_names(model::WithoutInputs)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:habit,
        inputs=(),
        outputs=:choice,
    )
end
function log_likelihood_by_hand(model::WithoutInputs, parameters, timesteps, latents=(;))
    habit = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        habit += parameters.learning_rate * (1 - habit)
        probability = inv(1 + exp(-parameters.inverse_temperature * (habit - 0.5)))
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

#Learns the expected reward without acting, so it gives nothing to sample and has no distributions
struct WithoutOutputs <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::WithoutOutputs, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::WithoutOutputs, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learning_rate,
        context.input.reward,
    )
    return ModelOutput((;), (; expected_reward))
end
function CognitiveModeling.attribute_names(model::WithoutOutputs)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=:reward, outputs=()
    )
end
log_likelihood_by_hand(model::WithoutOutputs, parameters, timesteps, latents=(;)) = 0.0

## More and other kinds of values ##
#Three of each attribute: it learns from the reward, and remembers the last reward as a number, zero before the first,
#and how many timesteps it has seen.
#It chooses with the expected reward and a cue, rates its confidence around the expected reward and a bonus, and reports
#the last reward it saw, both with noise. A cue or a bonus that isn't given stands at a half and at zero
struct ManyAttributes{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::ManyAttributes, parameters)
    return (; expected_reward=0.5, last_reward=0.0, n_timesteps=0)
end
function CognitiveModeling.timestep_model(model::ManyAttributes, context::ModelContext)
    (; learning_rate, inverse_temperature, noise) = context.parameters
    (; reward, cue, bonus) = context.input
    expected_reward = learn(context.previous_state.expected_reward, learning_rate, reward)
    #As a number, also when the reward is a Bool, so that the state keeps its type
    last_reward = 1.0 * value_or(reward, context.previous_state.last_reward)
    n_timesteps = context.previous_state.n_timesteps + 1
    logit = choice_logit(expected_reward + value_or(cue, 0.5) - 0.5, inverse_temperature)
    return ModelOutput(
        (;
            choice=bernoulli_logit(model.distributions, logit),
            confidence=normal(
                model.distributions, expected_reward + value_or(bonus, 0.0), noise
            ),
            report=normal(model.distributions, last_reward, noise),
        ),
        (; expected_reward, last_reward, n_timesteps),
    )
end
function CognitiveModeling.attribute_names(model::ManyAttributes)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :noise),
        states=(:expected_reward, :last_reward, :n_timesteps),
        inputs=(:reward, :cue, :bonus),
        outputs=(:choice, :confidence, :report),
    )
end
function log_likelihood_by_hand(model::ManyAttributes, parameters, timesteps, latents=(;))
    expected_reward = 0.5
    last_reward = 0.0
    log_likelihood = 0.0
    for timestep in timesteps
        (; reward, cue, bonus) = timestep.input
        if !(reward isa AbstractNoValue)
            expected_reward += parameters.learning_rate * (reward - expected_reward)
            last_reward = reward
        end
        cue = cue isa AbstractNoValue ? 0.5 : cue
        bonus = bonus isa AbstractNoValue ? 0.0 : bonus
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward + cue - 1))
        )
        log_likelihood +=
            log_density_if_given(BernoulliByHand(probability), timestep.output.choice) +
            log_density_if_given(
                NormalByHand(expected_reward + bonus, parameters.noise),
                timestep.output.confidence,
            ) +
            log_density_if_given(
                NormalByHand(last_reward, parameters.noise), timestep.output.report
            )
    end
    return log_likelihood
end

#Vectors in every kind of attribute: a learning rate for each of three options, the expected reward of each, and the
#reward of each on every timestep. It chooses an option with a probability in proportion to its expected reward, and
#rates every option around its expected reward with noise
struct VectorAttributes{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::VectorAttributes, parameters)
    return (; expected_rewards=[0.5, 0.5, 0.5])
end
function CognitiveModeling.timestep_model(model::VectorAttributes, context::ModelContext)
    previous = context.previous_state.expected_rewards
    rewards = context.input.rewards
    expected_rewards = if rewards isa AbstractNoValue
        previous
    else
        previous .+ context.parameters.learning_rates .* (rewards .- previous)
    end
    return ModelOutput(
        (;
            choice=categorical(
                model.distributions, expected_rewards ./ sum(expected_rewards)
            ),
            ratings=independent_normals(
                model.distributions, expected_rewards, context.parameters.noise
            ),
        ),
        (; expected_rewards),
    )
end
function CognitiveModeling.attribute_names(model::VectorAttributes)
    return ModelAttributes(;
        parameters=(:learning_rates, :noise),
        states=:expected_rewards,
        inputs=:rewards,
        outputs=(:choice, :ratings),
    )
end
function log_likelihood_by_hand(model::VectorAttributes, parameters, timesteps, latents=(;))
    expected_rewards = [0.5, 0.5, 0.5]
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.rewards isa AbstractNoValue)
            expected_rewards = [
                expected_rewards[option] +
                parameters.learning_rates[option] *
                (timestep.input.rewards[option] - expected_rewards[option]) for
                option in 1:3
            ]
        end
        if !(timestep.output.choice isa AbstractNoValue)
            log_likelihood += log(
                expected_rewards[timestep.output.choice] / sum(expected_rewards)
            )
        end
        if !(timestep.output.ratings isa AbstractNoValue)
            for option in 1:3
                log_likelihood += log_density(
                    NormalByHand(expected_rewards[option], parameters.noise),
                    timestep.output.ratings[option],
                )
            end
        end
    end
    return log_likelihood
end

#NamedTuples as values: it observes a reward and a cue together, keeps the expected reward and the last cue together,
#and responds with a choice and a time together, choosing the first option with a probability that grows with both
struct NamedTupleAttributes{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::NamedTupleAttributes, parameters)
    return (; beliefs=(; expected_reward=0.5, last_cue=0.5))
end
function CognitiveModeling.timestep_model(
    model::NamedTupleAttributes, context::ModelContext
)
    previous = context.previous_state.beliefs
    feedback = context.input.feedback
    beliefs = if feedback isa AbstractNoValue
        previous
    else
        (;
            expected_reward=learn(
                previous.expected_reward,
                context.parameters.learning_rate,
                feedback.reward,
            ),
            last_cue=feedback.cue,
        )
    end
    probability = inv(
        1 + exp(
            -choice_logit(
                beliefs.expected_reward + beliefs.last_cue - 0.5,
                context.parameters.inverse_temperature,
            ),
        ),
    )
    return ModelOutput(
        (;
            response=choice_and_time(
                model.distributions, probability, context.parameters.rate
            )
        ),
        (; beliefs),
    )
end
function CognitiveModeling.attribute_names(model::NamedTupleAttributes)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :rate),
        states=:beliefs,
        inputs=:feedback,
        outputs=:response,
    )
end
function log_likelihood_by_hand(
    model::NamedTupleAttributes, parameters, timesteps, latents=(;)
)
    expected_reward = 0.5
    last_cue = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.feedback isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate *
                (timestep.input.feedback.reward - expected_reward)
            last_cue = timestep.input.feedback.cue
        end
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward + last_cue - 1))
        )
        log_likelihood += log_density_if_given(
            ChoiceAndTimeByHand(probability, parameters.rate), timestep.output.response
        )
    end
    return log_likelihood
end

#Rates how good the last reward was, around the expected reward with noise, instead of choosing
struct ContinuousOutput{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::ContinuousOutput, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::ContinuousOutput, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learning_rate,
        context.input.reward,
    )
    return ModelOutput(
        (; rating=normal(model.distributions, expected_reward, context.parameters.noise)),
        (; expected_reward),
    )
end
function CognitiveModeling.attribute_names(model::ContinuousOutput)
    return ModelAttributes(;
        parameters=(:learning_rate, :noise),
        states=:expected_reward,
        inputs=:reward,
        outputs=:rating,
    )
end
function log_likelihood_by_hand(model::ContinuousOutput, parameters, timesteps, latents=(;))
    expected_reward = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        log_likelihood += log_density_if_given(
            NormalByHand(expected_reward, parameters.noise), timestep.output.rating
        )
    end
    return log_likelihood
end

#Chooses :right with the probability with which the baseline chooses true, and :left otherwise
struct SymbolOutput{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::SymbolOutput, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::SymbolOutput, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learning_rate,
        context.input.reward,
    )
    probability = inv(
        1 + exp(-choice_logit(expected_reward, context.parameters.inverse_temperature))
    )
    return ModelOutput(
        (; choice=symbol_choice(model.distributions, probability)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::SymbolOutput)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(model::SymbolOutput, parameters, timesteps, latents=(;))
    expected_reward = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5))
        )
        log_likelihood += log_density_if_given(
            SymbolChoiceByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

## Kinds of parameters ##
#The baseline with a strategy, a Symbol, which is :learning for the baseline, and :guessing for a choice of true with a
#probability of a half on every timestep
struct WithSymbolParameter{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::WithSymbolParameter, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::WithSymbolParameter, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learning_rate,
        context.input.reward,
    )
    logit = if context.parameters.strategy === :learning
        choice_logit(expected_reward, context.parameters.inverse_temperature)
    else
        0.0
    end
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::WithSymbolParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :strategy),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function CognitiveModeling.parameter_domains(model::WithSymbolParameter)
    return (; strategy=Domain(; possible_values=(:learning, :guessing)))
end
function log_likelihood_by_hand(
    model::WithSymbolParameter, parameters, timesteps, latents=(;)
)
    parameters.strategy === :learning &&
        return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
    return sum(
        timestep -> log_density_if_given(BernoulliByHand(0.5), timestep.output.choice),
        timesteps;
        init=0.0,
    )
end

#The baseline with a parameter that is true or false, whether it learns at all: it learns with its learning rate times
#that, so that it is a number to the arithmetic of the model, without a branch on it
struct WithDiscreteParameter{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::WithDiscreteParameter, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::WithDiscreteParameter, context::ModelContext
)
    expected_reward = learn(
        context.previous_state.expected_reward,
        context.parameters.learns * context.parameters.learning_rate,
        context.input.reward,
    )
    logit = choice_logit(expected_reward, context.parameters.inverse_temperature)
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::WithDiscreteParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :learns),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function CognitiveModeling.parameter_domains(model::WithDiscreteParameter)
    return (; learns=Domain(; type=Bool))
end
function log_likelihood_by_hand(
    model::WithDiscreteParameter, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(
        Baseline(ByHand()),
        (;
            learning_rate=parameters.learns ? parameters.learning_rate : 0.0,
            parameters.inverse_temperature,
        ),
        timesteps,
    )
end

#The baseline with a default value for its inverse temperature
struct WithDefaultParameters{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::WithDefaultParameters, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::WithDefaultParameters, context::ModelContext
)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::WithDefaultParameters)
    return attribute_names(Baseline(model.distributions))
end
function CognitiveModeling.default_parameters(model::WithDefaultParameters)
    return (; inverse_temperature=2.0)
end
function log_likelihood_by_hand(
    model::WithDefaultParameters, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(
        Baseline(ByHand()), merge((; inverse_temperature=2.0), parameters), timesteps
    )
end

#The baseline with domains for its parameters, its reward and its choice
struct WithDomains{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(model::WithDomains, parameters) = (; expected_reward=0.5)
function CognitiveModeling.timestep_model(model::WithDomains, context::ModelContext)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::WithDomains)
    return attribute_names(Baseline(model.distributions))
end
function CognitiveModeling.parameter_domains(model::WithDomains)
    return (;
        learning_rate=Domain(; lower=0, upper=1), inverse_temperature=Domain(; lower=0)
    )
end
CognitiveModeling.input_domains(model::WithDomains) = (; reward=Domain(; lower=0, upper=1))
CognitiveModeling.output_domains(model::WithDomains) = (; choice=Domain(; type=Bool))
function log_likelihood_by_hand(model::WithDomains, parameters, timesteps, latents=(;))
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

## How it runs ##
#The baseline, which signals for a simulation to stop once its expected reward is above a threshold
struct WithStoppingCriterion{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::WithStoppingCriterion, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::WithStoppingCriterion, context::ModelContext
)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::WithStoppingCriterion)
    return attribute_names(Baseline(model.distributions))
end
function CognitiveModeling.stopping_criterion(
    model::WithStoppingCriterion, state, parameters
)
    return state.expected_reward > 0.8
end
#A fitted session checks the criterion before every timestep, and the cases have data on every timestep, so a criterion
#met before one of them, as when an imputed reward differs from the one that the data was simulated with, means that the
#parameters can't give the data
function log_likelihood_by_hand(
    model::WithStoppingCriterion, parameters, timesteps, latents=(;)
)
    expected_reward = 0.5
    for timestep in timesteps
        expected_reward > 0.8 && return -Inf
        expected_reward = learn(
            expected_reward, parameters.learning_rate, timestep.input.reward
        )
    end
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

#The baseline, which tends to repeat its previous choice by its stickiness. It counts its timesteps, so that on the first
#one it doesn't look at the choice before it, which a simulation or a session can be given. A previous choice that wasn't
#recorded and isn't imputed, a marker, counts as neither
struct DependsOnPreviousOutput{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::DependsOnPreviousOutput, parameters)
    return (; expected_reward=0.5, n_timesteps=0)
end
#The pull of the previous choice: towards true after true, towards false after false, and none without one
repetition(previous_choice) = 2 * previous_choice - 1
repetition(previous_choice::AbstractNoValue) = 0
function CognitiveModeling.timestep_model(
    model::DependsOnPreviousOutput, context::ModelContext
)
    (; learning_rate, inverse_temperature, stickiness) = context.parameters
    expected_reward = learn(
        context.previous_state.expected_reward, learning_rate, context.input.reward
    )
    n_timesteps = context.previous_state.n_timesteps + 1
    pull = n_timesteps == 1 ? 0 : repetition(context.previous_output.choice)
    logit = choice_logit(expected_reward, inverse_temperature) + stickiness * pull
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)),
        (; expected_reward, n_timesteps),
    )
end
function CognitiveModeling.attribute_names(model::DependsOnPreviousOutput)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :stickiness),
        states=(:expected_reward, :n_timesteps),
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(
    model::DependsOnPreviousOutput, parameters, timesteps, latents=(;)
)
    expected_reward = 0.5
    log_likelihood = 0.0
    for (t, timestep) in enumerate(timesteps)
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        pull = 0
        if t > 1 && !(timesteps[t - 1].output.choice isa AbstractNoValue)
            pull = timesteps[t - 1].output.choice ? 1 : -1
        end
        probability = inv(
            1 + exp(
                -(
                    parameters.inverse_temperature * (expected_reward - 0.5) +
                    parameters.stickiness * pull
                ),
            ),
        )
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

#The baseline, which learns more on a timestep the more time it covers, also on the first timestep
struct UsesTimestepSize{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::UsesTimestepSize, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::UsesTimestepSize, context::ModelContext)
    expected_reward = learn(
        context.previous_state.expected_reward,
        1 - exp(-context.parameters.learning_rate * context.timestep_size),
        context.input.reward,
    )
    logit = choice_logit(expected_reward, context.parameters.inverse_temperature)
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), (; expected_reward)
    )
end
function CognitiveModeling.attribute_names(model::UsesTimestepSize)
    return attribute_names(Baseline(model.distributions))
end
function log_likelihood_by_hand(model::UsesTimestepSize, parameters, timesteps, latents=(;))
    expected_reward = 0.5
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            step = 1 - exp(-parameters.learning_rate * timestep.timestep_size)
            expected_reward += step * (timestep.input.reward - expected_reward)
        end
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5))
        )
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

#The baseline, whose expected reward starts at a parameter
struct InitialStateFromParameter{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::InitialStateFromParameter, parameters)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.timestep_model(
    model::InitialStateFromParameter, context::ModelContext
)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::InitialStateFromParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :initial_value),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(
    model::InitialStateFromParameter, parameters, timesteps, latents=(;)
)
    expected_reward = parameters.initial_value
    log_likelihood = 0.0
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5))
        )
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end

#The baseline, whose expected reward starts at a parameter plus an offset that it samples, with a standard deviation that
#is a parameter. A simulation has no random number generator to sample a start from, so there it starts at the
#parameter, and the Turing.jl suites sample the offset, in turing_methods.jl
struct SampledInitialState{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::SampledInitialState, parameters)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.timestep_model(model::SampledInitialState, context::ModelContext)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::SampledInitialState)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :initial_value, :initial_noise),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(model::SampledInitialState, parameters, timesteps, latents)
    return log_likelihood_by_hand(
        InitialStateFromParameter(ByHand()),
        (;
            parameters.learning_rate,
            parameters.inverse_temperature,
            initial_value=parameters.initial_value + latents.initial_offset,
        ),
        timesteps,
    )
end
function latents_from_states(model::SampledInitialState, states)
    return (; initial_offset=0.0)
end
function log_density_of_latents_by_hand(model::SampledInitialState, parameters, latents)
    return log_density(NormalByHand(0.0, parameters.initial_noise), latents.initial_offset)
end

#The baseline, whose expected reward gets noise on every timestep, with a standard deviation that is a parameter. The
#state keeps the noise of the timestep, so that a simulation shows what it sampled
struct SamplingWithinTimestep{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::SamplingWithinTimestep, parameters)
    return (; expected_reward=0.5, noise=0.0)
end
#The state and the choice of the model after the noise of a timestep, which the Turing.jl method shares
function update_with_noise(model::SamplingWithinTimestep, context::ModelContext, noise)
    expected_reward =
        learn(
            context.previous_state.expected_reward,
            context.parameters.learning_rate,
            context.input.reward,
        ) + noise
    logit = choice_logit(expected_reward, context.parameters.inverse_temperature)
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), (; expected_reward, noise)
    )
end
function CognitiveModeling.timestep_model(
    model::SamplingWithinTimestep, context::ModelContext
)
    noise = rand(
        context.rng, normal(model.distributions, 0.0, context.parameters.update_noise)
    )
    return update_with_noise(model, context, noise)
end
function CognitiveModeling.attribute_names(model::SamplingWithinTimestep)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :update_noise),
        states=(:expected_reward, :noise),
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(
    model::SamplingWithinTimestep, parameters, timesteps, latents
)
    expected_reward = 0.5
    log_likelihood = 0.0
    for (t, timestep) in enumerate(timesteps)
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        expected_reward += latents.noise[t]
        probability = inv(
            1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5))
        )
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.choice
        )
    end
    return log_likelihood
end
function latents_from_states(model::SamplingWithinTimestep, states)
    return (; noise=[state.noise for state in states[2:end]])
end
function timestep_model_with_latents(model::SamplingWithinTimestep, context, latents, t)
    return update_with_noise(model, context, latents.noise[t])
end
function log_density_of_latents_by_hand(model::SamplingWithinTimestep, parameters, latents)
    return sum(
        noise -> log_density(NormalByHand(0.0, parameters.update_noise), noise),
        latents.noise;
        init=0.0,
    )
end

#The baseline, which gives a choice on every second timestep only, and nothing on the others
struct OutputNotEveryTimestep{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::OutputNotEveryTimestep, parameters)
    return (; expected_reward=0.5, n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    model::OutputNotEveryTimestep, context::ModelContext
)
    baseline_output = timestep_model(Baseline(model.distributions), context)
    n_timesteps = context.previous_state.n_timesteps + 1
    choice = isodd(n_timesteps) ? baseline_output.output_distribution.choice : NoValue()
    return ModelOutput(
        (; choice), (; baseline_output.new_state.expected_reward, n_timesteps)
    )
end
function CognitiveModeling.attribute_names(model::OutputNotEveryTimestep)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=(:expected_reward, :n_timesteps),
        inputs=:reward,
        outputs=:choice,
    )
end
function log_likelihood_by_hand(
    model::OutputNotEveryTimestep, parameters, timesteps, latents=(;)
)
    #The choices of the even timesteps are NoValue, which add nothing
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

#The baseline, which sets parameters_valid to false once its expected reward is above 0.99, as a model does whose
#parameters turn invalid with what it has received. With the learning rate of its pair that doesn't happen within ten
#timesteps, and with one near 1 it happens after the first rewards. Its distributions stay valid on that timestep
struct WithInvalidParameters{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
const HIGHEST_VALID_EXPECTED_REWARD = 0.99
function CognitiveModeling.initialize_state(model::WithInvalidParameters, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::WithInvalidParameters, context::ModelContext
)
    output = timestep_model(Baseline(model.distributions), context)
    return ModelOutput(
        output.output_distribution,
        output.new_state;
        parameters_valid=output.new_state.expected_reward <= HIGHEST_VALID_EXPECTED_REWARD,
    )
end
function CognitiveModeling.attribute_names(model::WithInvalidParameters)
    return attribute_names(Baseline(model.distributions))
end

#The log likelihood of the baseline up to the timestep on which the parameters turn invalid, where it is -Inf
function log_likelihood_by_hand(
    model::WithInvalidParameters, parameters, timesteps, latents=(;)
)
    expected_reward = 0.5
    for timestep in timesteps
        if !(timestep.input.reward isa AbstractNoValue)
            expected_reward +=
                parameters.learning_rate * (timestep.input.reward - expected_reward)
        end
        expected_reward <= HIGHEST_VALID_EXPECTED_REWARD || return -Inf
    end
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

## Kinds of states ##
#The baseline, which keeps its expected reward in a vector that it changes in place, and copies the vector when it is
#saved. The vector has the number type of the parameters, so that it can hold numbers of automatic differentiation
struct StateChangedInPlace{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::StateChangedInPlace, parameters)
    return (; expected_reward=[0.5 * one(parameters.learning_rate)])
end
function CognitiveModeling.timestep_model(model::StateChangedInPlace, context::ModelContext)
    expected_reward = context.previous_state.expected_reward
    expected_reward[1] = learn(
        expected_reward[1], context.parameters.learning_rate, context.input.reward
    )
    logit = choice_logit(expected_reward[1], context.parameters.inverse_temperature)
    return ModelOutput(
        (; choice=bernoulli_logit(model.distributions, logit)), context.previous_state
    )
end
function CognitiveModeling.prepare_state_for_saving(model::StateChangedInPlace, state)
    return (; expected_reward=copy(state.expected_reward))
end
function CognitiveModeling.attribute_names(model::StateChangedInPlace)
    return attribute_names(Baseline(model.distributions))
end
function log_likelihood_by_hand(
    model::StateChangedInPlace, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

#The same, which copies its state with deepcopy when it is saved
struct StateChangedInPlaceSavedWithDeepcopy{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    model::StateChangedInPlaceSavedWithDeepcopy, parameters
)
    return initialize_state(StateChangedInPlace(model.distributions), parameters)
end
function CognitiveModeling.timestep_model(
    model::StateChangedInPlaceSavedWithDeepcopy, context::ModelContext
)
    return timestep_model(StateChangedInPlace(model.distributions), context)
end
function CognitiveModeling.prepare_state_for_saving(
    model::StateChangedInPlaceSavedWithDeepcopy, state
)
    return deepcopy(state)
end
function CognitiveModeling.attribute_names(model::StateChangedInPlaceSavedWithDeepcopy)
    return attribute_names(Baseline(model.distributions))
end
function log_likelihood_by_hand(
    model::StateChangedInPlaceSavedWithDeepcopy, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

#The baseline, whose expected reward starts as an integer, 1, and is a number of another type after the first reward
struct StateChangesType{Tdistributions} <: AbstractCognitiveModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(model::StateChangesType, parameters)
    return (; expected_reward=1)
end
function CognitiveModeling.timestep_model(model::StateChangesType, context::ModelContext)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::StateChangesType)
    return attribute_names(Baseline(model.distributions))
end
function log_likelihood_by_hand(model::StateChangesType, parameters, timesteps, latents=(;))
    return log_likelihood_by_hand(
        InitialStateFromParameter(ByHand()),
        merge(parameters, (; initial_value=1)),
        timesteps,
    )
end

## Methods from a supertype ##
#A family of models, which defines the methods of its members once for all of them: each runs as the baseline
abstract type BaselineFamily <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::BaselineFamily, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::BaselineFamily, context::ModelContext)
    return timestep_model(Baseline(model.distributions), context)
end
function CognitiveModeling.attribute_names(model::BaselineFamily)
    return attribute_names(Baseline(model.distributions))
end

#A member of the family, which has no methods of its own
struct MethodsFromSupertype{Tdistributions} <: BaselineFamily
    distributions::Tdistributions
end
function log_likelihood_by_hand(
    model::MethodsFromSupertype, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(Baseline(ByHand()), parameters, timesteps)
end

## Every feature ##
#Two models with every feature at once, which share their methods as a family. Each learns an expected reward, with its
#learning rate times whether it learns at all and adjusted to the size of the timestep, from a value that starts at a
#parameter, and expected rewards of three options in a vector that it changes in place, from the rewards of all three,
#each with its own learning rate, and it remembers the last cue it saw in a NamedTuple. On two of every three timesteps it
#chooses, with a probability that grows with the expected reward and the last cue, and with the pull of its previous
#choice, unless its strategy is to guess. It rates the expected reward with noise, and chooses one of the three options
#in proportion to their expected rewards, on every timestep. It sets parameters_valid to false with a learning rate above
#0.9, and has domains, defaults, and a stopping criterion, an expected reward above 0.95
abstract type EveryFeatureFamily <: AbstractCognitiveModel end

#Every feature: it also adds noise to its expected reward on every timestep, samples an offset of where it starts in the
#Turing.jl suites, as SampledInitialState does, and responds with a choice and a time in a NamedTuple, and with a side,
#a Symbol, both with the probability of its choice
struct EveryFeature{Tdistributions} <: EveryFeatureFamily
    distributions::Tdistributions
end

#Every feature whose outputs have means, so that it runs with the means of its distributions too, and which samples
#nothing of its own: it has no noise, no offset, and no response or side
struct EveryFeatureWithoutSampling{Tdistributions} <: EveryFeatureFamily
    distributions::Tdistributions
end

function CognitiveModeling.initialize_state(model::EveryFeatureFamily, parameters)
    return (;
        expected_reward=parameters.initial_value,
        noise=0.0,
        n_timesteps=0,
        expected_rewards=[0.5 * one(eltype(parameters.learning_rates)) for _ in 1:3],
        beliefs=(; last_cue=0.5),
    )
end
function CognitiveModeling.prepare_state_for_saving(model::EveryFeatureFamily, state)
    return merge(state, (; expected_rewards=copy(state.expected_rewards)))
end

#The state and the outputs of a timestep with the noise of the timestep, which the Turing.jl method shares
function update_with_noise(model::EveryFeatureFamily, context::ModelContext, noise)
    parameters = context.parameters
    previous = context.previous_state
    (; reward, rewards, feedback) = context.input

    #The expected reward, with the learning rate adjusted to the size of the timestep, and the noise
    step = parameters.learns * (1 - exp(-parameters.learning_rate * context.timestep_size))
    expected_reward = learn(previous.expected_reward, step, reward) + noise

    #The expected rewards of the options, changed in place, one at a time
    expected_rewards = previous.expected_rewards
    if !(rewards isa AbstractNoValue)
        for option in 1:3
            expected_rewards[option] +=
                parameters.learning_rates[option] *
                (rewards[option] - expected_rewards[option])
        end
    end

    #The last cue it saw
    beliefs = feedback isa AbstractNoValue ? previous.beliefs : (; last_cue=feedback.cue)
    n_timesteps = previous.n_timesteps + 1

    #The log odds of choosing true, with the pull of the previous choice after the first timestep
    pull = n_timesteps == 1 ? 0 : repetition(context.previous_output.choice)
    logit = if parameters.strategy === :learning
        choice_logit(expected_reward + beliefs.last_cue - 0.5, parameters.inverse_temperature) + parameters.stickiness * pull
    else
        0.0
    end
    probability = inv(1 + exp(-logit))
    choice = if iszero(mod(n_timesteps, 3))
        NoValue()
    else
        bernoulli_logit(model.distributions, logit)
    end
    outputs = (;
        choice,
        rating=normal(model.distributions, expected_reward, parameters.rating_noise),
        option=categorical(model.distributions, expected_rewards ./ sum(expected_rewards)),
    )
    return ModelOutput(
        merge(outputs, outputs_with_sampling(model, probability, parameters)),
        (; expected_reward, noise, n_timesteps, expected_rewards, beliefs);
        parameters_valid=parameters.learning_rate <= 0.9,
    )
end
#The outputs that only EveryFeature gives
outputs_with_sampling(model::EveryFeatureWithoutSampling, probability, parameters) = (;)
function outputs_with_sampling(model::EveryFeature, probability, parameters)
    return (;
        response=choice_and_time(model.distributions, probability, parameters.rate),
        side=symbol_choice(model.distributions, probability),
    )
end
function CognitiveModeling.timestep_model(model::EveryFeature, context::ModelContext)
    noise = rand(
        context.rng, normal(model.distributions, 0.0, context.parameters.update_noise)
    )
    return update_with_noise(model, context, noise)
end
function CognitiveModeling.timestep_model(
    model::EveryFeatureWithoutSampling, context::ModelContext
)
    return update_with_noise(model, context, 0.0)
end

const EVERY_FEATURE_PARAMETERS = (
    :learning_rate,
    :inverse_temperature,
    :stickiness,
    :initial_value,
    :learning_rates,
    :rating_noise,
    :strategy,
    :learns,
)
const EVERY_FEATURE_STATES = (
    :expected_reward, :noise, :n_timesteps, :expected_rewards, :beliefs
)
const EVERY_FEATURE_INPUTS = (:reward, :rewards, :feedback)
function CognitiveModeling.attribute_names(model::EveryFeature)
    return ModelAttributes(;
        parameters=(EVERY_FEATURE_PARAMETERS..., :initial_noise, :update_noise, :rate),
        states=EVERY_FEATURE_STATES,
        inputs=EVERY_FEATURE_INPUTS,
        outputs=(:choice, :rating, :option, :response, :side),
    )
end
function CognitiveModeling.attribute_names(model::EveryFeatureWithoutSampling)
    return ModelAttributes(;
        parameters=EVERY_FEATURE_PARAMETERS,
        states=EVERY_FEATURE_STATES,
        inputs=EVERY_FEATURE_INPUTS,
        outputs=(:choice, :rating, :option),
    )
end
function CognitiveModeling.parameter_domains(model::EveryFeatureFamily)
    return (;
        learning_rate=Domain(; lower=0, upper=1),
        inverse_temperature=Domain(; lower=0),
        learning_rates=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        strategy=Domain(; possible_values=(:learning, :guessing)),
        learns=Domain(; type=Bool),
    )
end
function CognitiveModeling.input_domains(model::EveryFeatureFamily)
    return (; reward=Domain(; lower=0, upper=1))
end
CognitiveModeling.output_domains(model::EveryFeatureFamily) = (; choice=Domain(; type=Bool))
function CognitiveModeling.default_parameters(model::EveryFeatureFamily)
    return (; inverse_temperature=2.0, strategy=:learning, learns=true)
end
#Stops after a thousand timesteps, as its environment does by default, which no test reaches: a fit checks the criterion
#on every timestep, and one that the noise could meet while the data continues would end a session before the noise of
#its later timesteps, whose draws then have fewer variables, which keeps a sampler of Turing.jl from starting
function CognitiveModeling.stopping_criterion(model::EveryFeatureFamily, state, parameters)
    return state.n_timesteps >= 1000
end

function log_likelihood_by_hand(model::EveryFeatureFamily, parameters, timesteps, latents)
    parameters = merge(CognitiveModeling.default_parameters(model), parameters)
    expected_reward = parameters.initial_value + latents.initial_offset
    expected_rewards = [0.5, 0.5, 0.5]
    last_cue = 0.5
    log_likelihood = 0.0
    for (t, timestep) in enumerate(timesteps)
        (; reward, rewards, feedback) = timestep.input
        if !(reward isa AbstractNoValue) && parameters.learns
            step = 1 - exp(-parameters.learning_rate * timestep.timestep_size)
            expected_reward += step * (reward - expected_reward)
        end
        expected_reward += latents.noise[t]
        if !(rewards isa AbstractNoValue)
            expected_rewards = [
                expected_rewards[option] +
                parameters.learning_rates[option] *
                (rewards[option] - expected_rewards[option]) for option in 1:3
            ]
        end
        if !(feedback isa AbstractNoValue)
            last_cue = feedback.cue
        end

        #The choice, on two of every three timesteps, with the pull of the previous one
        pull = 0
        if t > 1 && !(timesteps[t - 1].output.choice isa AbstractNoValue)
            pull = timesteps[t - 1].output.choice ? 1 : -1
        end
        probability = if parameters.strategy === :learning
            inv(
                1 + exp(
                    -(
                        parameters.inverse_temperature * (expected_reward + last_cue - 1) + parameters.stickiness * pull
                    ),
                ),
            )
        else
            0.5
        end
        output = timestep.output
        log_likelihood +=
            log_density_if_given(BernoulliByHand(probability), output.choice) +
            log_density_if_given(
                NormalByHand(expected_reward, parameters.rating_noise), output.rating
            ) +
            log_density_if_given(
                CategoricalByHand(expected_rewards ./ sum(expected_rewards)), output.option
            )
        if model isa EveryFeature
            log_likelihood +=
                log_density_if_given(
                    ChoiceAndTimeByHand(probability, parameters.rate), output.response
                ) + log_density_if_given(SymbolChoiceByHand(probability), output.side)
        end
    end
    return log_likelihood
end
#A simulation starts at the initial value, without an offset
function latents_from_states(model::EveryFeature, states)
    return (; initial_offset=0.0, noise=[state.noise for state in states[2:end]])
end
function latents_from_states(model::EveryFeatureWithoutSampling, states)
    return (; initial_offset=0.0, noise=zeros(length(states) - 1))
end
function timestep_model_with_latents(model::EveryFeature, context, latents, t)
    return update_with_noise(model, context, latents.noise[t])
end
function log_density_of_latents_by_hand(model::EveryFeature, parameters, latents)
    return log_density(
        NormalByHand(0.0, parameters.initial_noise), latents.initial_offset
    ) + sum(
        noise -> log_density(NormalByHand(0.0, parameters.update_noise), noise),
        latents.noise;
        init=0.0,
    )
end
