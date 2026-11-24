## Environments ##
#Every environment gives rewards for choices, and differs from the baseline in the feature it is named after, or in the
#inputs and outputs that a cognitive model needs from it. Each is followed by the log likelihood of its outputs, the
#observations, by hand, as for the cognitive models, from a vector of timesteps with the input that the environment
#received, which is an action, and the output it gave

## Helpers ##
#The probability of a reward for a choice: the reward probability for true, half of it for false, and a half before the
#first choice. A choice which is a probability, from the means of the distributions, gives the expected one, so that the
#expected reward of a model which starts at a half moves
reward_probability(choice, probability) = probability * (1 + choice) / 2
reward_probability(choice::AbstractNoValue, probability) = 0.5

## The baseline ##
#Rewards a choice of true with a probability, and a choice of false with half of it, and counts its timesteps
struct BaselineEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(environment::BaselineEnvironment, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::BaselineEnvironment, context::ModelContext
)
    probability = reward_probability(
        context.input.choice, context.parameters.reward_probability
    )
    return ModelOutput(
        (; reward=bernoulli(environment.distributions, probability)),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::BaselineEnvironment)
    return ModelAttributes(;
        parameters=:reward_probability, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end
function log_likelihood_by_hand(
    environment::BaselineEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = 0.0
    for timestep in timesteps
        choice = timestep.input.choice
        probability = if choice isa AbstractNoValue
            0.5
        elseif choice
            parameters.reward_probability
        else
            parameters.reward_probability / 2
        end
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.reward
        )
    end
    return log_likelihood
end

## Submodels ##
#The schedule of the baseline's rewards as a submodel: it has the probability of a reward as its parameter, and counts
#the timesteps as its state
struct RewardSchedule <: AbstractSubmodel end
CognitiveModeling.initialize_state(schedule::RewardSchedule, parameters) = (; n_timesteps=0)
function CognitiveModeling.attribute_names(schedule::RewardSchedule)
    return ModelAttributes(;
        parameters=:reward_probability, states=:n_timesteps, inputs=(), outputs=()
    )
end

#The baseline with its schedule as a submodel, whose parameter it is given flat, and whose state it keeps under the name
#of the field
struct WithSubmodelsEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
    schedule::RewardSchedule
end
function WithSubmodelsEnvironment(distributions)
    return WithSubmodelsEnvironment(distributions, RewardSchedule())
end
CognitiveModeling.submodel_fields(environment::WithSubmodelsEnvironment) = (:schedule,)
function CognitiveModeling.attribute_names(environment::WithSubmodelsEnvironment)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end
function CognitiveModeling.initialize_state(
    environment::WithSubmodelsEnvironment, parameters
)
    return (; schedule=initialize_state(environment.schedule, parameters.schedule))
end
function CognitiveModeling.timestep_model(
    environment::WithSubmodelsEnvironment, context::ModelContext
)
    probability = reward_probability(
        context.input.choice, context.parameters.schedule.reward_probability
    )
    return ModelOutput(
        (; reward=bernoulli(environment.distributions, probability)),
        (; schedule=(; n_timesteps=context.previous_state.schedule.n_timesteps + 1)),
    )
end
function log_likelihood_by_hand(
    environment::WithSubmodelsEnvironment, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(BaselineEnvironment(ByHand()), parameters, timesteps)
end

## Without one kind of attribute ##
#The baseline without its count of timesteps
struct WithoutStatesEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(environment::WithoutStatesEnvironment, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::WithoutStatesEnvironment, context::ModelContext
)
    probability = reward_probability(
        context.input.choice, context.parameters.reward_probability
    )
    return ModelOutput((; reward=bernoulli(environment.distributions, probability)))
end
function CognitiveModeling.attribute_names(environment::WithoutStatesEnvironment)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end
function log_likelihood_by_hand(
    environment::WithoutStatesEnvironment, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(BaselineEnvironment(ByHand()), parameters, timesteps)
end

#The baseline with a reward probability of 0.75, which isn't a parameter
struct WithoutParametersEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::WithoutParametersEnvironment, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::WithoutParametersEnvironment, context::ModelContext
)
    probability = reward_probability(context.input.choice, 0.75)
    return ModelOutput(
        (; reward=bernoulli(environment.distributions, probability)),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::WithoutParametersEnvironment)
    return ModelAttributes(;
        parameters=(), states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end
function log_likelihood_by_hand(
    environment::WithoutParametersEnvironment, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(
        BaselineEnvironment(ByHand()), (; reward_probability=0.75), timesteps
    )
end

#Gives rewards on a schedule, with a probability, whatever the choices are, which it doesn't receive
struct WithoutInputsEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::WithoutInputsEnvironment, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::WithoutInputsEnvironment, context::ModelContext
)
    return ModelOutput(
        (;
            reward=bernoulli(
                environment.distributions, context.parameters.reward_probability
            )
        ),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::WithoutInputsEnvironment)
    return ModelAttributes(;
        parameters=:reward_probability, states=:n_timesteps, inputs=(), outputs=:reward
    )
end
function log_likelihood_by_hand(
    environment::WithoutInputsEnvironment, parameters, timesteps, latents=(;)
)
    return sum(
        timestep -> log_density_if_given(
            BernoulliByHand(parameters.reward_probability), timestep.output.reward
        ),
        timesteps;
        init=0.0,
    )
end

#Receives the choices and adds them up as numbers, so that its state keeps its type, but gives no observations, for a
#cognitive model without inputs
struct WithoutOutputsEnvironment <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::WithoutOutputsEnvironment, parameters
)
    return (; total_of_choices=0.0)
end
function CognitiveModeling.timestep_model(
    environment::WithoutOutputsEnvironment, context::ModelContext
)
    choice = 1.0 * value_or(context.input.choice, false)
    return ModelOutput(
        (;), (; total_of_choices=context.previous_state.total_of_choices + choice)
    )
end
function CognitiveModeling.attribute_names(environment::WithoutOutputsEnvironment)
    return ModelAttributes(;
        parameters=(), states=:total_of_choices, inputs=:choice, outputs=()
    )
end
function log_likelihood_by_hand(
    environment::WithoutOutputsEnvironment, parameters, timesteps, latents=(;)
)
    return 0.0
end

## How it runs ##
#The baseline, which gives a reward on every second timestep only, and nothing on the others
struct OutputNotEveryTimestepEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::OutputNotEveryTimestepEnvironment, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::OutputNotEveryTimestepEnvironment, context::ModelContext
)
    n_timesteps = context.previous_state.n_timesteps + 1
    reward = if isodd(n_timesteps)
        probability = reward_probability(
            context.input.choice, context.parameters.reward_probability
        )
        bernoulli(environment.distributions, probability)
    else
        NoValue()
    end
    return ModelOutput((; reward), (; n_timesteps))
end
function CognitiveModeling.attribute_names(environment::OutputNotEveryTimestepEnvironment)
    return attribute_names(BaselineEnvironment(environment.distributions))
end
function log_likelihood_by_hand(
    environment::OutputNotEveryTimestepEnvironment, parameters, timesteps, latents=(;)
)
    #The rewards of the even timesteps are NoValue, which add nothing
    return log_likelihood_by_hand(BaselineEnvironment(ByHand()), parameters, timesteps)
end

#The baseline, which signals for a simulation to stop after the timestep that is its last, a parameter
struct WithStoppingCriterionEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::WithStoppingCriterionEnvironment, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::WithStoppingCriterionEnvironment, context::ModelContext
)
    return timestep_model(BaselineEnvironment(environment.distributions), context)
end
function CognitiveModeling.attribute_names(environment::WithStoppingCriterionEnvironment)
    return ModelAttributes(;
        parameters=(:reward_probability, :last_timestep),
        states=:n_timesteps,
        inputs=:choice,
        outputs=:reward,
    )
end
function CognitiveModeling.parameter_domains(environment::WithStoppingCriterionEnvironment)
    return (; last_timestep=Domain(; type=Integer, lower=1))
end
function CognitiveModeling.stopping_criterion(
    environment::WithStoppingCriterionEnvironment, state, parameters
)
    return state.n_timesteps >= parameters.last_timestep
end
function log_likelihood_by_hand(
    environment::WithStoppingCriterionEnvironment, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(BaselineEnvironment(ByHand()), parameters, timesteps)
end

#The baseline, which gives no reward before the first choice, when it starts
struct NoOutputBeforeFirstInputEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::NoOutputBeforeFirstInputEnvironment, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::NoOutputBeforeFirstInputEnvironment, context::ModelContext
)
    baseline_output = timestep_model(
        BaselineEnvironment(environment.distributions), context
    )
    reward = if context.input.choice isa AbstractNoValue
        NoValue()
    else
        baseline_output.output_distribution.reward
    end
    return ModelOutput((; reward), baseline_output.new_state)
end
function CognitiveModeling.attribute_names(environment::NoOutputBeforeFirstInputEnvironment)
    return attribute_names(BaselineEnvironment(environment.distributions))
end
function log_likelihood_by_hand(
    environment::NoOutputBeforeFirstInputEnvironment, parameters, timesteps, latents=(;)
)
    #The reward before the first choice is NoValue, which adds nothing
    return log_likelihood_by_hand(BaselineEnvironment(ByHand()), parameters, timesteps)
end

#Gives a reward around a level times the choice, with noise, where the level drifts by noise that it samples on every
#timestep, with a standard deviation that is a parameter. The state keeps the drift of the timestep, so that a
#simulation shows what it sampled
struct SamplingWithinTimestepEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
CognitiveModeling.is_stochastic(environment::SamplingWithinTimestepEnvironment) = true
function CognitiveModeling.initialize_state(
    environment::SamplingWithinTimestepEnvironment, parameters
)
    return (; level=0.5, drift=0.0)
end
#The reward and the state of the environment after the drift of a timestep, which the Turing.jl method shares
function update_with_drift(
    environment::SamplingWithinTimestepEnvironment, context::ModelContext, drift
)
    level = context.previous_state.level + drift
    choice = value_or(context.input.choice, 0.5)
    return ModelOutput(
        (;
            reward=normal(
                environment.distributions, level * choice, context.parameters.reward_noise
            )
        ),
        (; level, drift),
    )
end
function CognitiveModeling.timestep_model(
    environment::SamplingWithinTimestepEnvironment, context::ModelContext
)
    drift = rand(
        context.rng, normal(environment.distributions, 0.0, context.parameters.drift_noise)
    )
    return update_with_drift(environment, context, drift)
end
function CognitiveModeling.attribute_names(environment::SamplingWithinTimestepEnvironment)
    return ModelAttributes(;
        parameters=(:drift_noise, :reward_noise),
        states=(:level, :drift),
        inputs=:choice,
        outputs=:reward,
    )
end
function log_likelihood_by_hand(
    environment::SamplingWithinTimestepEnvironment, parameters, timesteps, latents
)
    level = 0.5
    log_likelihood = 0.0
    for (t, timestep) in enumerate(timesteps)
        level += latents.drift[t]
        choice = timestep.input.choice isa AbstractNoValue ? 0.5 : timestep.input.choice
        log_likelihood += log_density_if_given(
            NormalByHand(level * choice, parameters.reward_noise), timestep.output.reward
        )
    end
    return log_likelihood
end
function latents_from_states(environment::SamplingWithinTimestepEnvironment, states)
    return (; drift=[state.drift for state in states[2:end]])
end
function timestep_model_with_latents(
    environment::SamplingWithinTimestepEnvironment, context, latents, t
)
    return update_with_drift(environment, context, latents.drift[t])
end
function log_density_of_latents_by_hand(
    environment::SamplingWithinTimestepEnvironment, parameters, latents
)
    return sum(
        drift -> log_density(NormalByHand(0.0, parameters.drift_noise), drift),
        latents.drift;
        init=0.0,
    )
end

#The baseline under other names than the cognitive models': it receives a lever, gives a payoff, and gives the number of
#the timestep too, which the cognitive models don't observe. Maps of names pass them on
struct OtherNamesEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(environment::OtherNamesEnvironment, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::OtherNamesEnvironment, context::ModelContext
)
    n_timesteps = context.previous_state.n_timesteps + 1
    probability = reward_probability(
        context.input.lever, context.parameters.reward_probability
    )
    return ModelOutput(
        (;
            payoff=bernoulli(environment.distributions, probability),
            timestep_number=Deterministic(n_timesteps),
        ),
        (; n_timesteps),
    )
end
function CognitiveModeling.attribute_names(environment::OtherNamesEnvironment)
    return ModelAttributes(;
        parameters=:reward_probability,
        states=:n_timesteps,
        inputs=:lever,
        outputs=(:payoff, :timestep_number),
    )
end
function log_likelihood_by_hand(
    environment::OtherNamesEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = 0.0
    for timestep in timesteps
        lever = timestep.input.lever
        probability = if lever isa AbstractNoValue
            0.5
        elseif lever
            parameters.reward_probability
        else
            parameters.reward_probability / 2
        end
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.payoff
        )
    end
    return log_likelihood
end

## For the cognitive models with other inputs and outputs ##
#The partner of ManyAttributes, with three of each attribute: it rewards the choice as the baseline does, gives a cue
#around the confidence, and a bonus around the report times its size, both with noise. It remembers the last choice and
#the last confidence, and counts its timesteps. The last choice is kept as a number, so that the state keeps its type
struct ManyAttributesEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::ManyAttributesEnvironment, parameters
)
    return (; n_timesteps=0, last_choice=0.0, last_confidence=0.5)
end
function CognitiveModeling.timestep_model(
    environment::ManyAttributesEnvironment, context::ModelContext
)
    (; reward_probability, noise, bonus_size) = context.parameters
    (; choice, confidence, report) = context.input
    return ModelOutput(
        (;
            reward=bernoulli(
                environment.distributions,
                CommonTestFunctionality.reward_probability(choice, reward_probability),
            ),
            cue=normal(environment.distributions, value_or(confidence, 0.5), noise),
            bonus=normal(
                environment.distributions, bonus_size * value_or(report, 0.5), noise
            ),
        ),
        (;
            n_timesteps=context.previous_state.n_timesteps + 1,
            last_choice=1.0 * value_or(choice, context.previous_state.last_choice),
            last_confidence=value_or(confidence, context.previous_state.last_confidence),
        ),
    )
end
function CognitiveModeling.attribute_names(environment::ManyAttributesEnvironment)
    return ModelAttributes(;
        parameters=(:reward_probability, :noise, :bonus_size),
        states=(:n_timesteps, :last_choice, :last_confidence),
        inputs=(:choice, :confidence, :report),
        outputs=(:reward, :cue, :bonus),
    )
end
function log_likelihood_by_hand(
    environment::ManyAttributesEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = log_likelihood_by_hand(
        BaselineEnvironment(ByHand()),
        parameters,
        [(; timestep.input, output=(; timestep.output.reward)) for timestep in timesteps],
    )
    for timestep in timesteps
        (; confidence, report) = timestep.input
        confidence = confidence isa AbstractNoValue ? 0.5 : confidence
        report = report isa AbstractNoValue ? 0.5 : report
        log_likelihood +=
            log_density_if_given(
                NormalByHand(confidence, parameters.noise), timestep.output.cue
            ) + log_density_if_given(
                NormalByHand(parameters.bonus_size * report, parameters.noise),
                timestep.output.bonus,
            )
    end
    return log_likelihood
end

#The partner of VectorAttributes: the reward of each of three options around its payoff, a parameter in a vector, with
#noise, where the chosen option gets a bonus of 0.1. A choice which is the probabilities of the options, from the means
#of the distributions, gives each option that share of the bonus. It receives the ratings too, and ignores them
struct VectorAttributesEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::VectorAttributesEnvironment, parameters
)
    return (; n_timesteps=0)
end
#The share of the bonus that each option gets
chosen_shares(choice::Integer) = [option == choice ? 1.0 : 0.0 for option in 1:3]
chosen_shares(probabilities::AbstractVector) = probabilities
chosen_shares(choice::AbstractNoValue) = [0.0, 0.0, 0.0]
function CognitiveModeling.timestep_model(
    environment::VectorAttributesEnvironment, context::ModelContext
)
    means = context.parameters.payoffs .+ 0.1 .* chosen_shares(context.input.choice)
    return ModelOutput(
        (;
            rewards=independent_normals(
                environment.distributions, means, context.parameters.noise
            )
        ),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::VectorAttributesEnvironment)
    return ModelAttributes(;
        parameters=(:payoffs, :noise),
        states=:n_timesteps,
        inputs=(:choice, :ratings),
        outputs=:rewards,
    )
end
function log_likelihood_by_hand(
    environment::VectorAttributesEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = 0.0
    for timestep in timesteps
        timestep.output.rewards isa AbstractNoValue && continue
        for option in 1:3
            bonus = timestep.input.choice == option ? 0.1 : 0.0
            log_likelihood += log_density(
                NormalByHand(parameters.payoffs[option] + bonus, parameters.noise),
                timestep.output.rewards[option],
            )
        end
    end
    return log_likelihood
end

#The partner of NamedTupleAttributes: it gives a reward and a cue together, rewarding the first option with a probability
#and the second with half of it, as the baseline does, and the cue around a mean with noise
struct NamedTupleAttributesEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::NamedTupleAttributesEnvironment, parameters
)
    return (; n_timesteps=0)
end
#Whether the response chose the first option, or nothing before the first response
chose_first(response) = response.choice == 1
chose_first(response::AbstractNoValue) = response
function CognitiveModeling.timestep_model(
    environment::NamedTupleAttributesEnvironment, context::ModelContext
)
    (; reward_probability, cue_mean, cue_noise) = context.parameters
    probability = CommonTestFunctionality.reward_probability(
        chose_first(context.input.response), reward_probability
    )
    return ModelOutput(
        (;
            feedback=reward_and_cue(
                environment.distributions, probability, cue_mean, cue_noise
            )
        ),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::NamedTupleAttributesEnvironment)
    return ModelAttributes(;
        parameters=(:reward_probability, :cue_mean, :cue_noise),
        states=:n_timesteps,
        inputs=:response,
        outputs=:feedback,
    )
end
function log_likelihood_by_hand(
    environment::NamedTupleAttributesEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = 0.0
    for timestep in timesteps
        response = timestep.input.response
        probability = if response isa AbstractNoValue
            0.5
        elseif response.choice == 1
            parameters.reward_probability
        else
            parameters.reward_probability / 2
        end
        log_likelihood += log_density_if_given(
            RewardAndCueByHand(probability, parameters.cue_mean, parameters.cue_noise),
            timestep.output.feedback,
        )
    end
    return log_likelihood
end

#The partner of ContinuousOutput: a reward around the rating times a scale, plus the rest of the scale up to one, with
#noise, so that a rating of one gets a reward around one, and the expected reward of a model which rates it moves
struct ContinuousInputEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
function CognitiveModeling.initialize_state(
    environment::ContinuousInputEnvironment, parameters
)
    return (;)
end
function CognitiveModeling.timestep_model(
    environment::ContinuousInputEnvironment, context::ModelContext
)
    scale = context.parameters.reward_scale
    return ModelOutput((;
        reward=normal(
            environment.distributions,
            scale * value_or(context.input.rating, 0.5) + 1 - scale,
            context.parameters.reward_noise,
        )
    ))
end
function CognitiveModeling.attribute_names(environment::ContinuousInputEnvironment)
    return ModelAttributes(;
        parameters=(:reward_scale, :reward_noise),
        states=(),
        inputs=:rating,
        outputs=:reward,
    )
end
function log_likelihood_by_hand(
    environment::ContinuousInputEnvironment, parameters, timesteps, latents=(;)
)
    log_likelihood = 0.0
    for timestep in timesteps
        rating = timestep.input.rating isa AbstractNoValue ? 0.5 : timestep.input.rating
        mean = parameters.reward_scale * rating + 1 - parameters.reward_scale
        log_likelihood += log_density_if_given(
            NormalByHand(mean, parameters.reward_noise), timestep.output.reward
        )
    end
    return log_likelihood
end

#The partner of SymbolOutput: it rewards :right with a probability, and :left with half of it
struct SymbolInputEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
CognitiveModeling.initialize_state(environment::SymbolInputEnvironment, parameters) = (;)
#Whether the choice is :right, or nothing before the first choice
chose_right(choice::Symbol) = choice === :right
chose_right(choice::AbstractNoValue) = choice
function CognitiveModeling.timestep_model(
    environment::SymbolInputEnvironment, context::ModelContext
)
    probability = reward_probability(
        chose_right(context.input.choice), context.parameters.reward_probability
    )
    return ModelOutput((; reward=bernoulli(environment.distributions, probability)))
end
function CognitiveModeling.attribute_names(environment::SymbolInputEnvironment)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end
function log_likelihood_by_hand(
    environment::SymbolInputEnvironment, parameters, timesteps, latents=(;)
)
    return log_likelihood_by_hand(
        BaselineEnvironment(ByHand()),
        parameters,
        [
            (; input=(; choice=chose_right(timestep.input.choice)), timestep.output) for
            timestep in timesteps
        ],
    )
end

## Every feature ##
#The partner of the models with every feature, with features of its own: it rewards the choice as the baseline does,
#except on every fourth timestep, gives the reward of each of three options around its payoff with a bonus for the chosen
#option, as VectorAttributesEnvironment does, and a reward and a cue together, as NamedTupleAttributesEnvironment does,
#with the cue around its mean plus a level that drifts by noise that it samples on every timestep. It stops after the
#timestep that is its last, a parameter with a default. It receives the choice, the rating and the option, and a map of
#names leaves out the response and the side of EveryFeature
struct EveryFeatureEnvironment{Tdistributions} <: AbstractEnvironmentModel
    distributions::Tdistributions
end
CognitiveModeling.is_stochastic(environment::EveryFeatureEnvironment) = true
function CognitiveModeling.initialize_state(
    environment::EveryFeatureEnvironment, parameters
)
    return (; n_timesteps=0, level=0.0, drift=0.0)
end
#The observations and the state of a timestep with the drift of the timestep, which the Turing.jl method shares
function update_with_drift(
    environment::EveryFeatureEnvironment, context::ModelContext, drift
)
    parameters = context.parameters
    (; choice, option) = context.input
    n_timesteps = context.previous_state.n_timesteps + 1
    level = context.previous_state.level + drift
    reward = if iszero(mod(n_timesteps, 4))
        NoValue()
    else
        bernoulli(
            environment.distributions,
            reward_probability(choice, parameters.reward_probability),
        )
    end
    means = parameters.payoffs .+ 0.1 .* chosen_shares(option)
    return ModelOutput(
        (;
            reward,
            rewards=independent_normals(environment.distributions, means, parameters.noise),
            feedback=reward_and_cue(
                environment.distributions,
                reward_probability(choice, parameters.reward_probability),
                parameters.cue_mean + level,
                parameters.noise,
            ),
        ),
        (; n_timesteps, level, drift),
    )
end
function CognitiveModeling.timestep_model(
    environment::EveryFeatureEnvironment, context::ModelContext
)
    drift = rand(
        context.rng, normal(environment.distributions, 0.0, context.parameters.drift_noise)
    )
    return update_with_drift(environment, context, drift)
end
function CognitiveModeling.attribute_names(environment::EveryFeatureEnvironment)
    return ModelAttributes(;
        parameters=(
            :reward_probability, :payoffs, :noise, :cue_mean, :drift_noise, :last_timestep
        ),
        states=(:n_timesteps, :level, :drift),
        inputs=(:choice, :rating, :option),
        outputs=(:reward, :rewards, :feedback),
    )
end
function CognitiveModeling.parameter_domains(environment::EveryFeatureEnvironment)
    return (;
        reward_probability=Domain(; lower=0, upper=1),
        last_timestep=Domain(; type=Integer, lower=1),
    )
end
function CognitiveModeling.default_parameters(environment::EveryFeatureEnvironment)
    return (; last_timestep=1000)
end
function CognitiveModeling.stopping_criterion(
    environment::EveryFeatureEnvironment, state, parameters
)
    return state.n_timesteps >= parameters.last_timestep
end
function log_likelihood_by_hand(
    environment::EveryFeatureEnvironment, parameters, timesteps, latents
)
    level = 0.0
    log_likelihood = 0.0
    for (t, timestep) in enumerate(timesteps)
        level += latents.drift[t]
        (; choice, option) = timestep.input
        probability = if choice isa AbstractNoValue
            0.5
        elseif choice
            parameters.reward_probability
        else
            parameters.reward_probability / 2
        end
        log_likelihood += log_density_if_given(
            BernoulliByHand(probability), timestep.output.reward
        )
        if !(timestep.output.rewards isa AbstractNoValue)
            for each_option in 1:3
                bonus = option == each_option ? 0.1 : 0.0
                log_likelihood += log_density(
                    NormalByHand(parameters.payoffs[each_option] + bonus, parameters.noise),
                    timestep.output.rewards[each_option],
                )
            end
        end
        log_likelihood += log_density_if_given(
            RewardAndCueByHand(probability, parameters.cue_mean + level, parameters.noise),
            timestep.output.feedback,
        )
    end
    return log_likelihood
end
function latents_from_states(environment::EveryFeatureEnvironment, states)
    return (; drift=[state.drift for state in states[2:end]])
end
function timestep_model_with_latents(
    environment::EveryFeatureEnvironment, context, latents, t
)
    return update_with_drift(environment, context, latents.drift[t])
end
function log_density_of_latents_by_hand(
    environment::EveryFeatureEnvironment, parameters, latents
)
    return sum(
        drift -> log_density(NormalByHand(0.0, parameters.drift_noise), drift),
        latents.drift;
        init=0.0,
    )
end
