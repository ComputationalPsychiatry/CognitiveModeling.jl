using JET: JET
using Random: AbstractRNG, Xoshiro

## Models and environments for the tests ##
#An action made by hand, so that the tests don't need a distribution library
struct ChooseWithProbability{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::ChooseWithProbability) = rand(rng) < action.probability

#An observation made by hand in the same way: a reward of 1.0 with a probability, and of 0.0 otherwise
struct ZeroOrOne{T}
    probability::T
end
Base.rand(rng::AbstractRNG, reward::ZeroOrOne) = float(rand(rng) < reward.probability)

#Learns an expected reward, and doesn't learn on timesteps with no observation, or before the first one
struct LearnsRewards <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsRewards, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_timestep_model(model::LearnsRewards, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    end
    return ModelOutput((; choice=ChooseWithProbability(Vₜ)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::LearnsRewards)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=:reward, outputs=:choice
    )
end

#Has no state
struct OnlyChooses <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyChooses, parameters) = (;)
function CognitiveModeling.apply_timestep_model(model::OnlyChooses, context::ModelContext)
    return ModelOutput((; choice=ChooseWithProbability(0.7)))
end
function CognitiveModeling.attribute_names(model::OnlyChooses)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=:choice)
end

#Has no actions
struct OnlyLearns <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyLearns, parameters) = (; expected_reward=0.5)
function CognitiveModeling.apply_timestep_model(model::OnlyLearns, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    return ModelOutput(
        (;),
        (;
            expected_reward=V₋₁ +
                            context.parameters.learning_rate * (context.input.reward - V₋₁)
        ),
    )
end
function CognitiveModeling.attribute_names(model::OnlyLearns)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=:reward, outputs=()
    )
end

#Has two named actions, and receives a reward which it does nothing with
struct TwoChoices <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TwoChoices, parameters) = (;)
function CognitiveModeling.apply_timestep_model(model::TwoChoices, context::ModelContext)
    return ModelOutput((;
        first_choice=ChooseWithProbability(0.0), second_choice=ChooseWithProbability(1.0)
    ))
end
function CognitiveModeling.attribute_names(model::TwoChoices)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:reward, outputs=(:first_choice, :second_choice)
    )
end

#Has noise in its state update
struct LearnsWithNoise <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsWithNoise, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsWithNoise, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    return ModelOutput(
        (; choice=ChooseWithProbability(0.5)),
        (; expected_reward=V₋₁ + rand(context.rng) - 0.5),
    )
end
function CognitiveModeling.attribute_names(model::LearnsWithNoise)
    return ModelAttributes(;
        parameters=(), states=:expected_reward, inputs=:reward, outputs=:choice
    )
end

#Changes a Vector in its state in place, and prepares its state for saving with the function it is given
struct LearnsInPlace{Tprepare_state} <: AbstractCognitiveModel
    prepare_state::Tprepare_state
end
function CognitiveModeling.prepare_state_for_saving(model::LearnsInPlace, state)
    return model.prepare_state(state)
end
CognitiveModeling.initialize_state(model::LearnsInPlace, parameters) = (; values=[0.5, 0.5])
function CognitiveModeling.apply_timestep_model(model::LearnsInPlace, context::ModelContext)
    values = context.previous_state.values
    values[1] += context.parameters.learning_rate * (context.input.reward - values[1])
    return ModelOutput((; choice=ChooseWithProbability(0.5)), context.previous_state)
end
function CognitiveModeling.attribute_names(model::LearnsInPlace)
    return ModelAttributes(;
        parameters=:learning_rate, states=:values, inputs=:reward, outputs=:choice
    )
end

#Starts with an integer as its state, which becomes a Float64 from the second timestep
struct StartsWithInteger <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::StartsWithInteger, parameters)
    return (; expected_reward=0)
end
function CognitiveModeling.apply_timestep_model(
    model::StartsWithInteger, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + 0.5 * (context.input.reward - V₋₁)
    end
    return ModelOutput((; choice=ChooseWithProbability(0.5)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::StartsWithInteger)
    return ModelAttributes(;
        parameters=(), states=:expected_reward, inputs=:reward, outputs=:choice
    )
end

#An environment which keeps who started as its state, to check what the environment is told
struct EnvironmentRecordsWhoStarts <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::EnvironmentRecordsWhoStarts, parameters
)
    return (; who_starts=UnknownWhoStarts())
end
function CognitiveModeling.apply_timestep_model(
    environment::EnvironmentRecordsWhoStarts, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)), (; who_starts=context.who_starts))
end
function CognitiveModeling.attribute_names(environment::EnvironmentRecordsWhoStarts)
    return ModelAttributes(;
        parameters=(), states=:who_starts, inputs=:choice, outputs=:reward
    )
end

#Keeps who started as its state, to check what the model is told
struct RecordsWhoStarts <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::RecordsWhoStarts, parameters)
    return (; who_starts=UnknownWhoStarts())
end
function CognitiveModeling.apply_timestep_model(
    model::RecordsWhoStarts, context::ModelContext
)
    return ModelOutput((; choice=Deterministic(true)), (; who_starts=context.who_starts))
end
function CognitiveModeling.attribute_names(model::RecordsWhoStarts)
    return ModelAttributes(;
        parameters=(), states=:who_starts, inputs=:reward, outputs=:choice
    )
end

#Has two states, so that one of them can be selected for saving
struct LearnsAndCounts <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsAndCounts, parameters)
    return (; expected_reward=0.5, n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    model::LearnsAndCounts, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    end
    return ModelOutput(
        (; choice=ChooseWithProbability(Vₜ)),
        (; expected_reward=Vₜ, n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(model::LearnsAndCounts)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=(:expected_reward, :n_timesteps),
        inputs=:reward,
        outputs=:choice,
    )
end

#Has an action that is never taken, and a NamedTuple of actions
struct ChoosesInGroups <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ChoosesInGroups, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    model::ChoosesInGroups, context::ModelContext
)
    return ModelOutput((;
        choice=ChooseWithProbability(1.0),
        skipped=NoValue(),
        movement=(; left=ChooseWithProbability(0.0), right=ChooseWithProbability(1.0)),
    ))
end
function CognitiveModeling.attribute_names(model::ChoosesInGroups)
    return ModelAttributes(;
        parameters=(), states=(), inputs=(), outputs=(:choice, :skipped, :movement)
    )
end

#Has default parameters and names, and an initial state which depends on a parameter
struct HasDefaults <: AbstractCognitiveModel end
function CognitiveModeling.default_parameters(model::HasDefaults)
    return (; learning_rate=0.1, initial_value=0.5)
end
function CognitiveModeling.initialize_state(model::HasDefaults, parameters)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.apply_timestep_model(model::HasDefaults, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    return ModelOutput(
        (; choice=ChooseWithProbability(V₋₁)),
        (;
            expected_reward=V₋₁ +
                            context.parameters.learning_rate * (context.input.reward - V₋₁)
        ),
    )
end
function CognitiveModeling.attribute_names(model::HasDefaults)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

#Throws the error for invalid parameters when the learning rate is above 1
struct RejectsHighLearningRates <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RejectsHighLearningRates, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    model::RejectsHighLearningRates, context::ModelContext
)
    context.parameters.learning_rate > 1 &&
        throw(InvalidParametersError("The learning rate must be at most 1."))
    return ModelOutput((; choice=ChooseWithProbability(0.5)))
end
function CognitiveModeling.attribute_names(model::RejectsHighLearningRates)
    return ModelAttributes(;
        parameters=:learning_rate, states=(), inputs=:reward, outputs=:choice
    )
end

#Has no names, so it can't be simulated
struct WithoutNames <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::WithoutNames, parameters) = (;)
function CognitiveModeling.apply_timestep_model(model::WithoutNames, context::ModelContext)
    return ModelOutput((; choice=ChooseWithProbability(0.5)))
end

#Takes the agent's choice, and gives rewards with a fixed probability, whatever it is
struct RewardsWithProbability <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsWithProbability, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::RewardsWithProbability, context::ModelContext
)
    return ModelOutput((; reward=ZeroOrOne(context.parameters.reward_probability)))
end
function CognitiveModeling.attribute_names(environment::RewardsWithProbability)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

#Has default parameters, and reads the agent's choice
struct RewardsWithDefaults <: AbstractEnvironmentModel end
function CognitiveModeling.default_parameters(environment::RewardsWithDefaults)
    return (; reward_size=1.0, bias=0.0)
end
function CognitiveModeling.initialize_state(environment::RewardsWithDefaults, parameters)
    return (; last_choice=NoValue())
end
function CognitiveModeling.apply_timestep_model(
    environment::RewardsWithDefaults, context::ModelContext
)
    reward = if context.input.choice isa AbstractNoValue
        context.parameters.bias
    else
        context.parameters.bias + context.parameters.reward_size * context.input.choice
    end
    return ModelOutput(
        (; reward=Deterministic(reward)), (; last_choice=context.input.choice)
    )
end
function CognitiveModeling.attribute_names(environment::RewardsWithDefaults)
    return ModelAttributes(;
        parameters=(:reward_size, :bias),
        states=:last_choice,
        inputs=:choice,
        outputs=:reward,
    )
end

#Gives a reward and a cue, and takes no actions, so that a model needs maps of names to be used with it
struct RewardsAndCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsAndCues, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::RewardsAndCues, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0), cue=Deterministic(2)))
end
function CognitiveModeling.attribute_names(environment::RewardsAndCues)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=(:reward, :cue))
end

#Gives only a cue, which a model which learns from rewards can't use
struct OnlyCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::OnlyCues, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::OnlyCues, context::ModelContext
)
    return ModelOutput((; cue=Deterministic(2)))
end
function CognitiveModeling.attribute_names(environment::OnlyCues)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=:cue)
end

#Has no names, so it can't be simulated
struct EnvironmentWithoutNames <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::EnvironmentWithoutNames, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::EnvironmentWithoutNames, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)))
end

#Takes the agent's choice, and gives fixed rewards whatever it is, counting the timesteps in its state
struct FixedRewards <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::FixedRewards, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::FixedRewards, context::ModelContext
)
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; reward=Deterministic(context.parameters.rewards[n_timesteps])), (; n_timesteps)
    )
end
function CognitiveModeling.attribute_names(environment::FixedRewards)
    return ModelAttributes(;
        parameters=:rewards, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

#Changes a Vector in its state in place, and copies its state when it's saved
struct CountsInPlace <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::CountsInPlace, parameters) = (; counts=[0])
function CognitiveModeling.apply_timestep_model(
    environment::CountsInPlace, context::ModelContext
)
    context.previous_state.counts[1] += 1
    return ModelOutput((; reward=Deterministic(1.0)), context.previous_state)
end
function CognitiveModeling.prepare_state_for_saving(environment::CountsInPlace, state)
    return deepcopy(state)
end
function CognitiveModeling.attribute_names(environment::CountsInPlace)
    return ModelAttributes(; parameters=(), states=:counts, inputs=:choice, outputs=:reward)
end

#Rewards an action of true, and has nothing to say before the first action
struct RewardsTrue <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsTrue, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::RewardsTrue, context::ModelContext
)
    context.input.choice isa AbstractNoValue && return ModelOutput((; reward=NoValue()))
    return ModelOutput((; reward=Deterministic(float(context.input.choice))))
end
function CognitiveModeling.attribute_names(environment::RewardsTrue)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end

#Gives no observation on every 5th timestep
struct SometimesSilent <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::SometimesSilent, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::SometimesSilent, context::ModelContext
)
    n_timesteps = context.previous_state.n_timesteps + 1
    observation_distribution = if iszero(mod(n_timesteps, 5))
        (; reward=NoValue())
    else
        (; reward=Deterministic(1.0))
    end
    return ModelOutput(observation_distribution, (; n_timesteps))
end
function CognitiveModeling.attribute_names(environment::SometimesSilent)
    return ModelAttributes(;
        parameters=(), states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

#Saves the timestep size it gets in its state, and does nothing with the one it observes. The state can't share the
#name of the observation, since a model uses each name for one attribute
struct SavesTimestepSize <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::SavesTimestepSize, parameters)
    return (; received_timestep_size=0)
end
function CognitiveModeling.apply_timestep_model(
    model::SavesTimestepSize, context::ModelContext
)
    return ModelOutput(
        (; choice=ChooseWithProbability(0.5)),
        (; received_timestep_size=context.timestep_size),
    )
end
function CognitiveModeling.attribute_names(model::SavesTimestepSize)
    return ModelAttributes(;
        parameters=(),
        states=:received_timestep_size,
        inputs=:timestep_size,
        outputs=:choice,
    )
end

#Gives the timestep size it gets as its observation
struct ObservesTimestepSize <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesTimestepSize, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::ObservesTimestepSize, context::ModelContext
)
    return ModelOutput((; timestep_size=Deterministic(context.timestep_size)))
end
function CognitiveModeling.attribute_names(environment::ObservesTimestepSize)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:choice, outputs=:timestep_size
    )
end

#Returns a tuple instead of a ModelOutput
struct ReturnsTuple <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ReturnsTuple, parameters) = (;)
function CognitiveModeling.apply_timestep_model(model::ReturnsTuple, context::ModelContext)
    return (ChooseWithProbability(0.5), (;))
end
function CognitiveModeling.attribute_names(model::ReturnsTuple)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end

#Returns the observation instead of a ModelOutput
struct ReturnsObservation <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ReturnsObservation, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::ReturnsObservation, context::ModelContext
)
    return (; reward=1.0)
end
function CognitiveModeling.attribute_names(environment::ReturnsObservation)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end

#Have a method for the initial state, but none for a timestep
struct OnlyInitializesState <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyInitializesState, parameters) = (;)
function CognitiveModeling.attribute_names(model::OnlyInitializesState)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
struct EnvironmentOnlyInitializesState <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::EnvironmentOnlyInitializesState, parameters
)
    return (;)
end
function CognitiveModeling.attribute_names(environment::EnvironmentOnlyInitializesState)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end

#Have a method for a timestep, but none for the initial state, so they must be given a state
struct NeedsGivenState <: AbstractCognitiveModel end
function CognitiveModeling.apply_timestep_model(
    model::NeedsGivenState, context::ModelContext
)
    return ModelOutput((; choice=ChooseWithProbability(0.5)), context.previous_state)
end
function CognitiveModeling.attribute_names(model::NeedsGivenState)
    return ModelAttributes(; parameters=(), states=:value, inputs=:reward, outputs=:choice)
end
struct EnvironmentNeedsGivenState <: AbstractEnvironmentModel end
function CognitiveModeling.apply_timestep_model(
    environment::EnvironmentNeedsGivenState, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)), context.previous_state)
end
function CognitiveModeling.attribute_names(environment::EnvironmentNeedsGivenState)
    return ModelAttributes(; parameters=(), states=:value, inputs=:choice, outputs=:reward)
end

#Rewards a lever of true with a payoff, and gives a cue: the same as RewardsTrue, under other names than the models use
struct PaysForLever <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::PaysForLever, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::PaysForLever, context::ModelContext
)
    context.input.lever isa AbstractNoValue &&
        return ModelOutput((; payoff=NoValue(), cue=Deterministic(2)))
    return ModelOutput((;
        payoff=Deterministic(float(context.input.lever)), cue=Deterministic(2)
    ))
end
function CognitiveModeling.attribute_names(environment::PaysForLever)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:lever, outputs=(:payoff, :cue)
    )
end

#Gives payoffs with a fixed probability, whatever the lever is, so that its outputs always have the same types
struct PaysWithProbability <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::PaysWithProbability, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::PaysWithProbability, context::ModelContext
)
    return ModelOutput((;
        payoff=ZeroOrOne(context.parameters.reward_probability), cue=Deterministic(2)
    ))
end
function CognitiveModeling.attribute_names(environment::PaysWithProbability)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:lever, outputs=(:payoff, :cue)
    )
end

#Ends after the number of timesteps in its parameters, which it counts in its state
struct EndsAfterTimesteps <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::EndsAfterTimesteps, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    environment::EndsAfterTimesteps, context::ModelContext
)
    return ModelOutput(
        (; reward=Deterministic(1.0)),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::EndsAfterTimesteps)
    return ModelAttributes(;
        parameters=:last_timestep, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end
function CognitiveModeling.stopping_criterion(
    environment::EndsAfterTimesteps, state, parameters
)
    return state.n_timesteps >= parameters.last_timestep
end

#A cognitive model which ends after the number of timesteps in its parameters, like EndsAfterTimesteps does
struct StopsAfterTimesteps <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::StopsAfterTimesteps, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_timestep_model(
    model::StopsAfterTimesteps, context::ModelContext
)
    return ModelOutput(
        (; choice=ChooseWithProbability(0.5)),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(model::StopsAfterTimesteps)
    return ModelAttributes(;
        parameters=:last_timestep, states=:n_timesteps, inputs=:reward, outputs=:choice
    )
end
function CognitiveModeling.stopping_criterion(model::StopsAfterTimesteps, state, parameters)
    return state.n_timesteps >= parameters.last_timestep
end

#Has an observation that is never made, and a NamedTuple of observations, and takes no actions
struct ObservesInGroups <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesInGroups, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::ObservesInGroups, context::ModelContext
)
    return ModelOutput((;
        reward=ZeroOrOne(1.0),
        skipped=NoValue(),
        cues=(; left=ZeroOrOne(0.0), right=Deterministic(2)),
    ))
end
function CognitiveModeling.attribute_names(environment::ObservesInGroups)
    return ModelAttributes(;
        parameters=(), states=(), inputs=(), outputs=(:reward, :skipped, :cues)
    )
end

#Keeps the reward it gave on the timestep before in its state, as it receives it
struct RemembersReward <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RemembersReward, parameters)
    return (; previous_reward=BeforeFirstTimestep())
end
function CognitiveModeling.apply_timestep_model(
    environment::RemembersReward, context::ModelContext
)
    return ModelOutput(
        (; reward=ZeroOrOne(0.5)), (; previous_reward=context.previous_output.reward)
    )
end
function CognitiveModeling.attribute_names(environment::RemembersReward)
    return ModelAttributes(;
        parameters=(), states=:previous_reward, inputs=:choice, outputs=:reward
    )
end

#Give a value itself where something to sample from is needed
struct GivesUnwrapped{T} <: AbstractEnvironmentModel
    reward::T
end
CognitiveModeling.initialize_state(environment::GivesUnwrapped, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    environment::GivesUnwrapped, context::ModelContext
)
    return ModelOutput((; reward=environment.reward))
end
function CognitiveModeling.attribute_names(environment::GivesUnwrapped)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end
struct ChoosesUnwrapped{T} <: AbstractCognitiveModel
    choice::T
end
CognitiveModeling.initialize_state(model::ChoosesUnwrapped, parameters) = (;)
function CognitiveModeling.apply_timestep_model(
    model::ChoosesUnwrapped, context::ModelContext
)
    return ModelOutput((; choice=model.choice))
end
function CognitiveModeling.attribute_names(model::ChoosesUnwrapped)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end

const LEARNING = (; learning_rate=0.5)
const REWARDS = (; reward_probability=0.8)

#The maps between the names of LearnsRewards and those of PaysForLever and PaysWithProbability
const PAYOFF_AS_REWARD = NameMap(:payoff => :reward)
const CHOICE_AS_LEVER = NameMap(:choice => :lever)

#As many names as a large model has, for checking that comparing them costs nothing
const MANY_NAMES = Tuple(Symbol(:state_, i) for i in 1:65)
const MANY_NAMES_REVERSED = reverse(MANY_NAMES)

#Whether a call compiles to code which only returns, with no calls or branches left. With code coverage on, as in CI, each line
#of code leaves a statement for the coverage, and a statement that was removed leaves nothing in its place, which both are fine
function folds(f, argument_types...)
    statements = only(code_typed(f, argument_types))[1].code
    return all(
        statement ->
            statement isa Core.ReturnNode ||
            statement === nothing ||
            Meta.isexpr(statement, :code_coverage_effect),
        statements,
    )
end

#Saves everything but timestep 0, so that element t of what a simulation returns is timestep t
const EVERY_TIMESTEP = SaveHistory(SaveEverything(); save_initial_timestep_values=false)

#Shortens the calls below
function simulate_for_test(
    model,
    environment;
    who_starts=EnvironmentStarts(),
    n_steps=50,
    cognitive_parameters=LEARNING,
    environment_parameters=REWARDS,
    rng=Xoshiro(1),
    cognitive_save_history=EVERY_TIMESTEP,
    environment_save_history=EVERY_TIMESTEP,
    keywords...,
)
    return simulate(
        model,
        environment;
        who_starts,
        n_steps,
        cognitive_parameters,
        environment_parameters,
        rng,
        cognitive_save_history,
        environment_save_history,
        keywords...,
    )
end

#The eight vectors a simulation returns: the cognitive model's four, and the environment's four
function saved_vectors(simulation)
    return (
        simulation.observations,
        simulation.actions,
        simulation.states,
        simulation.action_distributions,
        simulation.environment.states,
        simulation.environment.observations,
        simulation.environment.actions,
        simulation.environment.observation_distributions,
    )
end

#What a simulation returns of the cognitive model, without the environment's part, for comparing simulations with different environments
function cognitive_values(simulation)
    return (;
        simulation.observations,
        simulation.actions,
        simulation.states,
        simulation.action_distributions,
        simulation.timestep_sizes,
    )
end

#A model whose learning rate has a domain, for the check of the parameters when a simulation starts
struct LearnsWithinDomain <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(::LearnsWithinDomain, parameters) = (; value=0.0)
function CognitiveModeling.apply_timestep_model(::LearnsWithinDomain, context::ModelContext)
    return ModelOutput(
        (; choice=Deterministic(true)), (; value=context.parameters.learning_rate)
    )
end
function CognitiveModeling.attribute_names(::LearnsWithinDomain)
    return ModelAttributes(;
        parameters=:learning_rate, states=:value, inputs=:reward, outputs=:choice
    )
end
function CognitiveModeling.parameter_domains(::LearnsWithinDomain)
    return (; learning_rate=(; lower=0, upper=1),)
end

@testset "A parameter outside its domain is an error before anything runs" begin
    @test simulate(
        LearnsWithinDomain(), (; reward=1.0); parameters=(; learning_rate=0.5)
    )[2] == (; value=0.5)
    @test_throws InvalidParametersError(
        "The parameter learning_rate of a cognitive model of type LearnsWithinDomain is 1.5, which is outside its domain, from 0 to 1.",
    ) simulate(LearnsWithinDomain(), (; reward=1.0); parameters=(; learning_rate=1.5))
    @test_throws InvalidParametersError InstantiatedModel(
        LearnsWithinDomain(), (; learning_rate=-0.1)
    )
end

@testset "simulate" begin
    expected_rewards(cognitive_states) =
        [state.expected_reward for state in cognitive_states]

    @testset "Who starts decides what the model learns from" begin
        fixed_rewards = (; rewards=[1.0, 0.0, 1.0])
        #Learns from the reward of the same timestep
        simulation = simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            who_starts=EnvironmentStarts(),
            n_steps=3,
            environment_parameters=fixed_rewards,
        )
        @test expected_rewards(simulation.states) == [0.75, 0.375, 0.6875]
        @test simulation.environment.states[end] == (; n_timesteps=3)
        #Learns from the reward of the timestep before, and from nothing on the first
        simulation = simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            who_starts=AgentStarts(),
            n_steps=3,
            environment_parameters=fixed_rewards,
        )
        @test expected_rewards(simulation.states) == [0.5, 0.75, 0.375]
        @test simulation.environment.states[end] == (; n_timesteps=3)
    end

    @testset "The environment responds to the agent's actions" begin
        #The environment sees the action of the same timestep
        (; observations, actions) = simulate_for_test(
            LearnsRewards(),
            RewardsTrue();
            who_starts=AgentStarts(),
            environment_parameters=(;),
        )
        @test [observation.reward for observation in observations] == [float(action.choice) for action in actions]
        #The environment sees the action of the timestep before, and nothing on the first
        (; observations, actions) = simulate_for_test(
            LearnsRewards(),
            RewardsTrue();
            who_starts=EnvironmentStarts(),
            environment_parameters=(;),
        )
        @test observations[1] === (; reward=NoValue())
        @test [observation.reward for observation in observations[2:end]] == [float(action.choice) for action in actions[1:(end - 1)]]
    end

    @testset "The same seed gives the same simulation, for the model and the environment" begin
        for who_starts in (EnvironmentStarts(), AgentStarts()),
            (model, cognitive_parameters) in
            ((LearnsRewards(), LEARNING), (LearnsWithNoise(), (;)))

            first_simulation = simulate_for_test(
                model,
                RewardsWithProbability();
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(3),
            )
            second_simulation = simulate_for_test(
                model,
                RewardsWithProbability();
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(3),
            )
            other_seed = simulate_for_test(
                model,
                RewardsWithProbability();
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(4),
            )
            @test first_simulation == second_simulation
            @test first_simulation != other_seed
        end
    end

    @testset "It returns one NamedTuple, with the environment's values under environment" begin
        simulation = simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps=3)
        @test simulation isa NamedTuple
        @test keys(simulation) == (
            :states,
            :observations,
            :actions,
            :action_distributions,
            :environment,
            :timestep_sizes,
        )
        @test keys(simulation.environment) ==
            (:states, :observations, :actions, :observation_distributions)
        #The names can be destructured
        (; observations, actions, states, environment, timestep_sizes) = simulation
        @test observations === simulation.observations &&
            actions === simulation.actions &&
            states === simulation.states &&
            environment === simulation.environment &&
            timestep_sizes === simulation.timestep_sizes
        @test all(vector -> length(vector) == 3, saved_vectors(simulation))
    end

    @testset "The return type is known and the loop doesn't allocate" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate(
            LearnsRewards(),
            RewardsWithProbability();
            who_starts,
            n_steps,
            cognitive_parameters=LEARNING,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa NamedTuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                who_starts, 10
            )
            run_simulation(who_starts, 10)
            #Doubling the timesteps only adds the storage: for the cognitive model, 8 bytes each for an observation, an action
            #distribution and a state, and 1 for an action, and a byte each for the observation, the action and the action
            #distribution to say which of two types it has, since timestep 0 has BeforeFirstTimestep. The environment's observations,
            #actions and observation distributions take the same again, without a state
            extra_bytes =
                @allocated(run_simulation(who_starts, 20_000)) -
                @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 50 * 10_000
        end
    end

    @testset "Outputs whose types change are all kept" begin
        #An observation that is sometimes missing
        (; observations, states) = simulate_for_test(
            LearnsRewards(), SometimesSilent(); environment_parameters=(;)
        )
        @test count(observation -> observation.reward isa NoValue, observations) == 10
        @test expected_rewards(states)[4] == expected_rewards(states)[5]
        #A first observation that is missing
        (; observations) = simulate_for_test(
            LearnsRewards(), RewardsTrue(); environment_parameters=(;)
        )
        @test observations[1] === (; reward=NoValue())
        @test observations[2].reward isa Float64
        #A state that changes from an integer to a Float64
        (; states) = simulate_for_test(
            StartsWithInteger(),
            FixedRewards();
            who_starts=AgentStarts(),
            n_steps=3,
            cognitive_parameters=(;),
            environment_parameters=(; rewards=[1.0, 1.0, 1.0]),
        )
        @test expected_rewards(states) == [0, 0.5, 0.75]
    end

    @testset "Models and environments can leave things out" begin
        #A model without observations or a state, with an environment without a state: a map passes none of the observations on
        simulation = simulate_for_test(
            OnlyChooses(),
            RewardsWithProbability();
            cognitive_parameters=(;),
            observation_map=NameMap(),
        )
        @test eltype(simulation.observations) === typeof((;))
        @test eltype(simulation.states) === typeof((;))
        @test eltype(simulation.environment.states) === typeof((;))
        #The environment still gave the observations the model didn't receive
        @test eltype(simulation.environment.observations) === @NamedTuple{reward::Float64}
        #A model without actions, with an environment which takes none
        simulation = simulate_for_test(
            OnlyLearns(),
            RewardsAndCues();
            environment_parameters=(;),
            observation_map=NameMap(:reward),
        )
        @test eltype(simulation.actions) === typeof((;))
        @test eltype(simulation.states) <: NamedTuple
        @test eltype(simulation.environment.actions) === typeof((;))
    end

    @testset "A single action keeps the name the model gave it" begin
        (; actions) = simulate_for_test(LearnsRewards(), RewardsWithProbability())
        @test actions isa Vector{@NamedTuple{choice::Bool}}
    end

    @testset "Each of several named actions is sampled" begin
        (; actions) = simulate_for_test(
            TwoChoices(),
            RewardsAndCues();
            cognitive_parameters=(;),
            environment_parameters=(;),
            observation_map=NameMap(:reward),
            action_map=NameMap(),
        )
        @test all(action -> action === (; first_choice=false, second_choice=true), actions)
    end

    @testset "NoValue and a NamedTuple inside the actions are sampled on their own" begin
        (; actions) = simulate_for_test(
            ChoosesInGroups(),
            RewardsAndCues();
            cognitive_parameters=(;),
            environment_parameters=(;),
            observation_map=NameMap(),
            action_map=NameMap(),
        )
        @test all(
            action ->
                action ===
                (; choice=true, skipped=NoValue(), movement=(; left=false, right=true)),
            actions,
        )
        @test simulate!(InstantiatedModel(ChoosesInGroups(), (;)), (;)) ===
            (; choice=true, skipped=NoValue(), movement=(; left=false, right=true))
    end

    @testset "Each observation is sampled from what the environment gives, and a Deterministic is its value" begin
        #RewardsWithProbability gives something to sample the reward from
        (; observations) = simulate_for_test(LearnsRewards(), RewardsWithProbability())
        @test observations isa Vector{@NamedTuple{reward::Float64}}
        @test Set(observation.reward for observation in observations) == Set([0.0, 1.0])
        (; observations) = simulate_for_test(
            LearnsRewards(),
            RewardsWithProbability();
            environment_parameters=(; reward_probability=1.0),
        )
        @test all(observation -> observation === (; reward=1.0), observations)
        #FixedRewards gives each reward in a Deterministic
        (; observations) = simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            n_steps=3,
            environment_parameters=(; rewards=[1.0, 0.0, 0.25]),
        )
        @test observations == [(; reward=1.0), (; reward=0.0), (; reward=0.25)]
        #One that is sampled and one that isn't, in the same observation, which the environment's history has both of as its outputs
        environment = InstantiatedModel(
            PaysWithProbability(),
            REWARDS;
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=50,
            rng=Xoshiro(1),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        @test all(observation -> observation.cue === 2, environment.history.outputs[2:end])
        @test Set(
            observation.payoff for observation in environment.history.outputs[2:end]
        ) == Set([0.0, 1.0])
    end

    @testset "NoValue and a NamedTuple inside the observations are sampled on their own" begin
        environment = InstantiatedModel(
            ObservesInGroups(), (;); save_history=SaveHistory(SaveEverything())
        )
        simulate!(
            InstantiatedModel(OnlyLearns(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            observation_map=NameMap(:reward),
        )
        sampled = (; reward=1.0, skipped=NoValue(), cues=(; left=0.0, right=2))
        @test all(
            observation -> observation === sampled, environment.history.outputs[2:end]
        )
        @test environment.previous_output === sampled
    end

    @testset "A number, a string or a symbol that isn't wrapped gives an informative error" begin
        for value in (1.0, 1, true, "left", :left)
            @test_throws "A model gave $(repr(value)), of type $(typeof(value)), as what to sample an output from, but it is a value. An output that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))." simulate_for_test(
                LearnsRewards(), GivesUnwrapped(value); environment_parameters=(;)
            )
            @test_throws "A model gave $(repr(value)), of type $(typeof(value)), as what to sample an output from, but it is a value. An output that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))." simulate_for_test(
                ChoosesUnwrapped(value), RewardsWithProbability(); cognitive_parameters=(;)
            )
            @test_throws ArgumentError simulate!(
                InstantiatedModel(ChoosesUnwrapped(value), (;)), (; reward=1.0)
            )
        end
        #Wrapped, the same values are what is observed and done
        (; observations, actions) = simulate_for_test(
            ChoosesUnwrapped(Deterministic(:left)),
            GivesUnwrapped(Deterministic(1));
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
        )
        @test observations == fill((; reward=1), 2) && actions == fill((; choice=:left), 2)
        #A collection is still something to sample from, so one that isn't wrapped gives one of its elements
        (; observations) = simulate_for_test(
            ChoosesUnwrapped(Deterministic(:left)),
            GivesUnwrapped([2.0, 2.0]);
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
        )
        @test observations == fill((; reward=2.0), 2)
    end

    @testset "The environment receives the observation that was sampled, on the next timestep" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            simulation = simulate_for_test(
                LearnsRewards(), RemembersReward(); who_starts, environment_parameters=(;)
            )
            environment_states = simulation.environment.states
            @test environment_states[1] === (; previous_reward=BeforeFirstTimestep())
            @test [state.previous_reward for state in environment_states[2:end]] == [
                observation.reward for observation in simulation.observations[1:(end - 1)]
            ]
        end
    end

    @testset "An InvalidParametersError from a model stops the simulation, with its message" begin
        @test_throws InvalidParametersError simulate_for_test(
            RejectsHighLearningRates(),
            RewardsWithProbability();
            cognitive_parameters=(; learning_rate=1.5),
        )
        @test_throws "The learning rate must be at most 1." simulate_for_test(
            RejectsHighLearningRates(),
            RewardsWithProbability();
            cognitive_parameters=(; learning_rate=1.5),
        )
        @test simulate_for_test(
            RejectsHighLearningRates(),
            RewardsWithProbability();
            cognitive_parameters=(; learning_rate=0.5),
        ) isa NamedTuple
    end

    @testset "cognitive_save_history and environment_save_history select what is saved of each kind" begin
        fixed_rewards = (; rewards=[1.0, 0.0, 1.0])
        run_with(; keywords...) = simulate_for_test(
            LearnsAndCounts(),
            FixedRewards();
            n_steps=3,
            environment_parameters=fixed_rewards,
            keywords...,
        )
        #The helper saves everything, which selecting everything also does
        everything = run_with()
        @test run_with(;
            cognitive_save_history=SaveHistory(
                SaveEverything(); save_initial_timestep_values=false
            ),
            environment_save_history=SaveHistory(
                SaveEverything(); save_initial_timestep_values=false
            ),
        ) == everything
        @test everything.states == [
            (; expected_reward=0.75, n_timesteps=1),
            (; expected_reward=0.375, n_timesteps=2),
            (; expected_reward=0.6875, n_timesteps=3),
        ]
        #Names keep those entries, and a kind that isn't given keeps nothing
        simulation = run_with(;
            cognitive_save_history=SaveHistory(;
                states=:n_timesteps, save_initial_timestep_values=false
            ),
        )
        @test simulation.observations == fill((;), 3) &&
            eltype(simulation.observations) === typeof((;))
        @test simulation.actions == fill((;), 3)
        @test simulation.states == [(; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        @test simulation.environment == everything.environment
        #Several names keep the entries in the order given
        (; states) = run_with(;
            cognitive_save_history=SaveHistory(;
                states=(:n_timesteps, :expected_reward), save_initial_timestep_values=false
            ),
        )
        @test states[1] === (; n_timesteps=1, expected_reward=0.75)
        #The environment's selection only affects what is returned under environment, in its own names: its outputs are the
        #observations and its inputs the actions
        simulation = run_with(;
            environment_save_history=SaveHistory(;
                outputs=:reward, inputs=:choice, save_initial_timestep_values=false
            ),
        )
        @test cognitive_values(simulation) == cognitive_values(everything)
        @test simulation.environment.states == fill((;), 3)
        @test simulation.environment.observations == everything.environment.observations
        @test simulation.environment.actions == everything.environment.actions
        @test simulation.environment.observation_distributions == fill((;), 3)
        #Nothing at all gives empty NamedTuples of everything, followed by the size of the timesteps
        nothing_saved = run_with(;
            cognitive_save_history=SaveHistory(; save_initial_timestep_values=false),
            environment_save_history=SaveHistory(; save_initial_timestep_values=false),
        )
        @test all(==(fill((;), 3)), saved_vectors(nothing_saved))
        @test nothing_saved.timestep_sizes === 1
    end

    @testset "Nothing is saved by default, for timestep 0 and each timestep after it" begin
        @test all(
            ==(fill((;), 4)),
            saved_vectors(
                simulate(
                    LearnsRewards(),
                    RewardsWithProbability();
                    who_starts=EnvironmentStarts(),
                    n_steps=3,
                    cognitive_parameters=LEARNING,
                    environment_parameters=REWARDS,
                    rng=Xoshiro(1),
                ),
            ),
        )
    end

    @testset "Timestep 0 is saved by default, with the previous observation and action and the states before the first timestep" begin
        rewards = (; rewards=[1.0, 0.0, 1.0])
        everything = SaveHistory(SaveEverything())
        for who_starts in (EnvironmentStarts(), AgentStarts())
            with_timestep_zero = simulate_for_test(
                LearnsAndCounts(),
                FixedRewards();
                who_starts,
                n_steps=3,
                environment_parameters=rewards,
                cognitive_save_history=everything,
                environment_save_history=everything,
            )
            @test with_timestep_zero.observations[1] === (; reward=BeforeFirstTimestep())
            @test with_timestep_zero.actions[1] === (; choice=BeforeFirstTimestep())
            @test with_timestep_zero.states[1] === (; expected_reward=0.5, n_timesteps=0)
            @test with_timestep_zero.environment.states[1] === (; n_timesteps=0)
            #The environment's observation and action before the first timestep, and its observation distributions, which are
            #BeforeFirstTimestep since none was given yet
            @test with_timestep_zero.environment.observations[1] ===
                (; reward=BeforeFirstTimestep())
            @test with_timestep_zero.environment.actions[1] ===
                (; choice=BeforeFirstTimestep())
            @test with_timestep_zero.environment.observation_distributions[1] ===
                (; reward=BeforeFirstTimestep())
            #After it come the timesteps, as without it
            without_timestep_zero = simulate_for_test(
                LearnsAndCounts(),
                FixedRewards();
                who_starts,
                n_steps=3,
                environment_parameters=rewards,
            )
            for (with_vector, without_vector) in
                zip(saved_vectors(with_timestep_zero), saved_vectors(without_timestep_zero))
                @test isequal(with_vector[2:end], without_vector)
            end
        end
        #What is given as previous is what timestep 0 has, and when it has the types of the later timesteps, the vectors have them only
        simulation = simulate_for_test(
            LearnsAndCounts(),
            FixedRewards();
            n_steps=2,
            environment_parameters=rewards,
            previous_observation=(; reward=0.25),
            previous_action=(; choice=true),
            previous_cognitive_state=(; expected_reward=0.1, n_timesteps=5),
            cognitive_save_history=everything,
            environment_save_history=everything,
        )
        @test (
            simulation.observations[1],
            simulation.actions[1],
            simulation.states[1],
            simulation.environment.states[1],
        ) === (
            (; reward=0.25),
            (; choice=true),
            (; expected_reward=0.1, n_timesteps=5),
            (; n_timesteps=0),
        )
        @test simulation.observations isa Vector{@NamedTuple{reward::Float64}} &&
            simulation.actions isa Vector{@NamedTuple{choice::Bool}}
        @test simulation.environment.observations[1] === (; reward=0.25) &&
            simulation.environment.actions[1] === (; choice=true)
        #With maps, the observation and the action have the cognitive model's names, and under environment the environment's
        simulation = simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            n_steps=2,
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            cognitive_save_history=everything,
            environment_save_history=everything,
        )
        @test simulation.observations[1] === (; reward=BeforeFirstTimestep()) &&
            simulation.actions[1] === (; choice=BeforeFirstTimestep())
        @test simulation.environment.observations[1] ===
              (; payoff=BeforeFirstTimestep(), cue=BeforeFirstTimestep()) &&
            simulation.environment.actions[1] === (; lever=BeforeFirstTimestep())
        #A selection applies to it too, and cognitive_save_history decides for the environment's vectors as well, so that all of
        #them have the same length
        simulation = simulate_for_test(
            LearnsAndCounts(),
            FixedRewards();
            n_steps=2,
            environment_parameters=rewards,
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
        )
        @test simulation.states == [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        @test simulation.environment.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        @test all(vector -> length(vector) == 3, saved_vectors(simulation))
        #The model is told who starts, and that nobody is known to when it runs on a single input
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test simulate_for_test(
                RecordsWhoStarts(),
                FixedRewards();
                who_starts,
                n_steps=2,
                cognitive_parameters=(;),
                environment_parameters=rewards,
                cognitive_save_history=SaveHistory(; states=:who_starts),
            ).states[end] === (; who_starts)
            @test simulate_for_test(
                RecordsWhoStarts(),
                EnvironmentRecordsWhoStarts();
                who_starts,
                n_steps=2,
                cognitive_parameters=(;),
                environment_parameters=(;),
                cognitive_save_history=SaveHistory(; states=:who_starts),
            ).environment.states[end] === (; who_starts)
        end
        @test simulate(RecordsWhoStarts(), (; reward=1.0); parameters=(;))[2] ===
            (; who_starts=UnknownWhoStarts())
        #The action distributions are saved when selected, with BeforeFirstTimestep for every action at timestep 0
        (; actions, action_distributions) = simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            n_steps=3,
            environment_parameters=rewards,
            cognitive_save_history=SaveHistory(;
                outputs=:choice, output_distributions=:choice
            ),
        )
        @test action_distributions[1] === (; choice=BeforeFirstTimestep())
        @test [
            distribution.choice.probability for distribution in action_distributions[2:end]
        ] ≈ [0.75, 0.375, 0.6875]
        @test length(action_distributions) == length(actions) == 4
        @test simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            n_steps=3,
            environment_parameters=rewards,
            cognitive_save_history=SaveHistory(; outputs=:choice),
        ).action_distributions == fill((;), 4)
        @test_throws "The SaveHistory selects the output distributions (:bet,), but the outputs in the attributes of a cognitive model of type LearnsRewards are (:choice,). A history can only select names in the model's attributes." InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(; output_distributions=:bet)
        )
        #A state changed in place is saved before the first timestep changes it
        (; states) = simulate_for_test(
            LearnsInPlace(state -> (; values=copy(state.values))),
            FixedRewards();
            n_steps=2,
            environment_parameters=(; rewards=[1.0, 1.0]),
            cognitive_save_history=everything,
        )
        @test [state.values[1] for state in states] == [0.5, 0.75, 0.875]
    end

    @testset "The environment's observations, actions and observation distributions are returned as it gave and received them" begin
        everything = SaveHistory(SaveEverything())
        for who_starts in (EnvironmentStarts(), AgentStarts())
            simulation = simulate_for_test(
                LearnsRewards(),
                PaysForLever();
                who_starts,
                n_steps=3,
                environment_parameters=(;),
                observation_map=PAYOFF_AS_REWARD,
                action_map=CHOICE_AS_LEVER,
                cognitive_save_history=everything,
                environment_save_history=everything,
            )
            environment = simulation.environment
            #The observations are as the environment gave them, before the observation_map, with the cue the model didn't receive
            @test all(
                observation -> keys(observation) == (:payoff, :cue),
                environment.observations,
            )
            @test isequal(
                [observation.payoff for observation in environment.observations],
                [observation.reward for observation in simulation.observations],
            )
            @test all(observation -> observation.cue === 2, environment.observations[2:end])
            #The actions are as the environment received them, after the action_map
            @test all(action -> keys(action) == (:lever,), environment.actions)
            @test isequal(
                [action.lever for action in environment.actions],
                [action.choice for action in simulation.actions],
            )
            #The observation distributions are what the environment gave its observations to be sampled from
            @test all(
                distribution -> keys(distribution) == (:payoff, :cue),
                environment.observation_distributions,
            )
            @test all(
                distribution -> distribution.cue === Deterministic(2),
                environment.observation_distributions[2:end],
            )
            for (observation, distribution) in zip(
                environment.observations[2:end],
                environment.observation_distributions[2:end],
            )
                expected_distribution = if observation.payoff === NoValue()
                    NoValue()
                else
                    Deterministic(observation.payoff)
                end
                @test distribution.payoff === expected_distribution
            end
            #At timestep 0 every observation distribution is BeforeFirstTimestep
            @test environment.observation_distributions[1] ===
                (; payoff=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
            #The cognitive model's action distributions are what it gave its actions to be sampled from
            @test [
                distribution.choice.probability for
                distribution in simulation.action_distributions[2:end]
            ] == expected_rewards(simulation.states[2:end])
            #When the environment starts, it receives the previous action, so its first payoff isn't there
            if who_starts === EnvironmentStarts()
                @test environment.observation_distributions[2] ===
                    (; payoff=NoValue(), cue=Deterministic(2))
            else
                @test environment.observation_distributions[2] === (;
                    payoff=Deterministic(float(simulation.actions[2].choice)),
                    cue=Deterministic(2),
                )
            end
        end
        #The environment's selection applies to them, in its own names
        simulation = simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            n_steps=3,
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            environment_save_history=SaveHistory(;
                outputs=:cue, save_initial_timestep_values=false
            ),
        )
        @test simulation.environment.observations == fill((; cue=2), 3)
        @test simulation.environment.actions == fill((;), 3)
        @test simulation.environment.observation_distributions == fill((;), 3)
        @test simulation.environment.states == fill((;), 3)
        simulation = simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            n_steps=3,
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            environment_save_history=SaveHistory(;
                inputs=:lever,
                output_distributions=:payoff,
                save_initial_timestep_values=false,
            ),
        )
        @test simulation.environment.observations == fill((;), 3)
        @test [action.lever for action in simulation.environment.actions] == [action.choice for action in simulation.actions]
        @test all(
            distribution -> keys(distribution) == (:payoff,),
            simulation.environment.observation_distributions,
        )
    end

    @testset "An entry that is NoValue is saved like any other, and a model without actions saves empty NamedTuples" begin
        (; observations, actions) = simulate_for_test(
            LearnsRewards(),
            RewardsTrue();
            n_steps=3,
            environment_parameters=(;),
            cognitive_save_history=SaveHistory(;
                inputs=:reward, outputs=:choice, save_initial_timestep_values=false
            ),
        )
        @test observations[1] === (; reward=NoValue())
        @test observations[2] == (; reward=float(actions[1].choice))
        for cognitive_save_history in
            (SaveHistory(; outputs=(), save_initial_timestep_values=false), EVERY_TIMESTEP)
            (; actions) = simulate_for_test(
                OnlyLearns(),
                RewardsAndCues();
                n_steps=3,
                environment_parameters=(;),
                observation_map=NameMap(:reward),
                cognitive_save_history,
            )
            @test actions == fill((;), 3)
        end
    end

    @testset "The output before the first timestep has every output as BeforeFirstTimestep" begin
        #For a cognitive model, the outputs are its actions
        @test CognitiveModeling.output_before_first_timestep(LearnsRewards()) ===
            (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.output_before_first_timestep(OnlyLearns()) === (;)
        #The names are in the type, so it is inferred
        @test @inferred(CognitiveModeling.output_before_first_timestep(TwoChoices())) ===
            (; first_choice=BeforeFirstTimestep(), second_choice=BeforeFirstTimestep())
        @test InstantiatedModel(TwoChoices(), (;)).previous_output ===
            (; first_choice=BeforeFirstTimestep(), second_choice=BeforeFirstTimestep())
        #The environment receives it on the first timestep when it starts, and RewardsTrue gives no reward for it
        (; observations) = simulate_for_test(
            LearnsRewards(), RewardsTrue(); n_steps=2, environment_parameters=(;)
        )
        @test observations[1] === (; reward=NoValue())
        #For an environment model, the outputs are the observations it gives
        @test CognitiveModeling.output_before_first_timestep(RewardsWithProbability()) ===
            (; reward=BeforeFirstTimestep())
        @test CognitiveModeling.output_before_first_timestep(ObservesTimestepSize()) ===
            (; timestep_size=BeforeFirstTimestep())
        @test @inferred(
            CognitiveModeling.output_before_first_timestep(RewardsAndCues())
        ) === (; reward=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
        @test InstantiatedModel(RewardsAndCues(), (;)).previous_output ===
            (; reward=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
    end

    @testset "The input before the first timestep has every input as BeforeFirstTimestep" begin
        #For a cognitive model, the inputs are its observations
        @test CognitiveModeling.input_before_first_timestep(LearnsRewards()) ===
            (; reward=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(OnlyChooses()) === (;)
        #The names are in the type, so it is inferred
        @test @inferred(
            CognitiveModeling.input_before_first_timestep(SavesTimestepSize())
        ) === (; timestep_size=BeforeFirstTimestep())
        #The model receives it on the first timestep when it starts, so it learns nothing there
        (; states) = simulate_for_test(
            LearnsRewards(),
            FixedRewards();
            who_starts=AgentStarts(),
            n_steps=2,
            environment_parameters=(; rewards=[1.0, 1.0]),
        )
        @test expected_rewards(states) == [0.5, 0.75]
        #For an environment model, the inputs are the actions it receives, which it can have none of
        @test CognitiveModeling.input_before_first_timestep(RewardsTrue()) ===
            (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(PaysForLever()) ===
            (; lever=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(RewardsAndCues()) === (;)
    end

    @testset "The markers for nothing there share a supertype, and the one for before the first timestep is its own" begin
        @test NoValue() isa AbstractNoValue && BeforeFirstTimestep() isa AbstractNoValue
        @test !(BeforeFirstTimestep() isa NoValue)
        @test !(MissingValue() isa AbstractNoValue)
    end

    @testset "A state isn't prepared for saving when none of it is saved, also at timestep 0" begin
        (; states) = simulate_for_test(
            LearnsInPlace(state -> error("prepared anyway")),
            RewardsWithProbability();
            cognitive_save_history=SaveHistory(; inputs=SaveEverything()),
        )
        @test states == fill((;), 51)
    end

    @testset "Selecting a name that isn't there, or from something that isn't a NamedTuple, gives an error" begin
        @test_throws "The SaveHistory selects the states (:surprise,), but the states in the attributes of a cognitive model of type LearnsAndCounts are (:expected_reward, :n_timesteps)." simulate_for_test(
            LearnsAndCounts(),
            RewardsWithProbability();
            cognitive_save_history=SaveHistory(; states=:surprise),
        )
        @test_throws "The SaveHistory selects the inputs (:cue,), but the inputs in the attributes of a cognitive model of type LearnsAndCounts are (:reward,)." simulate_for_test(
            LearnsAndCounts(),
            RewardsWithProbability();
            cognitive_save_history=SaveHistory(; inputs=:cue),
        )
        @test_throws "can only select names from a NamedTuple" simulate_for_test(
            LearnsInPlace(state -> state.values),
            RewardsWithProbability();
            cognitive_save_history=SaveHistory(; states=:values),
        )
    end

    @testset "A selection keeps the return type known and the loop free of allocations" begin
        rng = Xoshiro(1)
        run_simulation(cognitive_save_history, n_steps) = simulate(
            LearnsAndCounts(),
            RewardsWithProbability();
            who_starts=EnvironmentStarts(),
            n_steps,
            cognitive_parameters=LEARNING,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history,
            environment_save_history=SaveHistory(),
        )
        for cognitive_save_history in (
            SaveHistory(; states=:n_timesteps),
            SaveHistory(; inputs=:reward, states=(:expected_reward, :n_timesteps)),
            SaveHistory(),
        )
            @test @inferred(run_simulation(cognitive_save_history, 10)) isa NamedTuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                cognitive_save_history, 10
            )
        end
        #Saving only the integer state adds 8 bytes for each timestep
        run_simulation(SaveHistory(; states=:n_timesteps), 10)
        extra_bytes =
            @allocated(run_simulation(SaveHistory(; states=:n_timesteps), 20_000)) -
            @allocated(run_simulation(SaveHistory(; states=:n_timesteps), 10_000))
        @test extra_bytes <= 9 * 10_000
    end

    @testset "A method which copies the state keeps the saved values of a state changed in place" begin
        for prepare_state in (deepcopy, state -> (; values=copy(state.values)))
            (; states) = simulate_for_test(
                LearnsInPlace(prepare_state), RewardsWithProbability()
            )
            @test length(unique(state.values[1] for state in states)) > 1
            @test allunique(objectid(state.values) for state in states)
        end
    end

    @testset "A method which does nothing copies nothing" begin
        (; states) = simulate_for_test(LearnsInPlace(identity), RewardsWithProbability())
        @test all(state -> state.values === first(states).values, states)
    end

    @testset "An environment can also have a method which copies its state" begin
        simulation = simulate_for_test(
            LearnsRewards(), CountsInPlace(); environment_parameters=(;)
        )
        environment_states = simulation.environment.states
        @test [state.counts[1] for state in environment_states] == 1:length(environment_states)
    end

    @testset "What prepare_state_for_saving returns is what is saved" begin
        (; states) = simulate_for_test(
            LearnsInPlace(state -> (; values=copy(state.values))), RewardsWithProbability()
        )
        @test eltype(states) === @NamedTuple{values::Vector{Float64}}
        @test length(unique(states)) > 1
    end

    @testset "There must be at least one timestep" begin
        @test length(
            simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps=1).actions
        ) == 1
        @test_throws ArgumentError simulate_for_test(
            LearnsRewards(), RewardsWithProbability(); n_steps=0
        )
    end

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "There is no apply_timestep_model method for a cognitive model of type OnlyInitializesState with a context of type " simulate_for_test(
            OnlyInitializesState(), RewardsWithProbability(); cognitive_parameters=(;)
        )
        @test_throws "Each model must define CognitiveModeling.apply_timestep_model(model::OnlyInitializesState, context::CognitiveModeling.ModelContext)." simulate_for_test(
            OnlyInitializesState(), RewardsWithProbability(); cognitive_parameters=(;)
        )
        @test_throws "There is no apply_timestep_model method for an environment model of type EnvironmentOnlyInitializesState" simulate_for_test(
            LearnsRewards(), EnvironmentOnlyInitializesState(); environment_parameters=(;)
        )
    end

    @testset "A missing method for the initial state gives an informative error, unless a state is given" begin
        @test_throws "There is no initialize_state method for a cognitive model of type NeedsGivenState. Each model must define CognitiveModeling.initialize_state(model::NeedsGivenState, parameters), which returns (;) if the model has no state." simulate_for_test(
            NeedsGivenState(), RewardsWithProbability(); cognitive_parameters=(;)
        )
        @test_throws "There is no initialize_state method for an environment model of type EnvironmentNeedsGivenState" simulate_for_test(
            LearnsRewards(), EnvironmentNeedsGivenState(); environment_parameters=(;)
        )
        simulation = simulate(
            NeedsGivenState(),
            EnvironmentNeedsGivenState();
            who_starts=EnvironmentStarts(),
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
            previous_cognitive_state=(; value=1.0),
            previous_environment_state=(; value=2.0),
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        #The given states are those of timestep 0
        @test simulation.states == fill((; value=1.0), 3)
        @test simulation.environment.states == fill((; value=2.0), 3)
    end

    @testset "A timestep method which returns the wrong type gives an informative error" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test_throws "The apply_timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput, but it returned a value of type Tuple" simulate_for_test(
                ReturnsTuple(),
                RewardsWithProbability();
                who_starts,
                cognitive_parameters=(;),
            )
            @test_throws "The apply_timestep_model method of an environment model of type ReturnsObservation must return a CognitiveModeling.ModelOutput, but it returned a value of type @NamedTuple" simulate_for_test(
                LearnsRewards(),
                ReturnsObservation();
                who_starts,
                environment_parameters=(;),
            )
        end
    end

    @testset "The model and the environment both get the size of each timestep" begin
        observed_sizes(observations) =
            [observation.timestep_size for observation in observations]
        received_sizes(states) = [state.received_timestep_size for state in states]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_with_sizes(; keywords...) = simulate(
                SavesTimestepSize(),
                ObservesTimestepSize();
                who_starts,
                n_steps=3,
                cognitive_parameters=(;),
                environment_parameters=(;),
                rng=Xoshiro(1),
                cognitive_save_history=EVERY_TIMESTEP,
                environment_save_history=EVERY_TIMESTEP,
                keywords...,
            )
            #The default is the integer 1
            (; observations, states) = run_with_sizes()
            @test observed_sizes(observations) == [1, 1, 1] &&
                eltype(observed_sizes(observations)) === Int
            @test received_sizes(states) == [1, 1, 1]
            #A single number is the size of every timestep
            (; observations, states) = run_with_sizes(timestep_sizes=0.5)
            @test observed_sizes(observations) == [0.5, 0.5, 0.5]
            @test received_sizes(states) == [0.5, 0.5, 0.5]
            #A vector has one size for each timestep
            (; observations, states) = run_with_sizes(timestep_sizes=[0.5, 2.0, 0.25])
            @test observed_sizes(observations) == [0.5, 2.0, 0.25]
            @test received_sizes(states) == [0.5, 2.0, 0.25]
            #Any vector can be used, like a range
            (; observations) = run_with_sizes(timestep_sizes=1:3)
            @test observed_sizes(observations) == [1, 2, 3]
            #The type of a timestep size isn't restricted
            (; observations) = run_with_sizes(timestep_sizes=(; fast=0.1, slow=10.0))
            @test observed_sizes(observations) == fill((; fast=0.1, slow=10.0), 3)
            #A vector with the wrong length isn't accepted
            @test_throws "it has 2 sizes and n_steps is 3" run_with_sizes(
                timestep_sizes=[0.5, 2.0]
            )
        end
    end

    @testset "Timestep sizes keep the return type known" begin
        rng = Xoshiro(1)
        run_simulation(timestep_sizes) = simulate(
            LearnsRewards(),
            RewardsWithProbability();
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes,
            cognitive_parameters=LEARNING,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        for timestep_sizes in (1, 0.5, [0.5, 2.0, 0.25], 1:3)
            @test @inferred(run_simulation(timestep_sizes)) isa NamedTuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                timestep_sizes
            )
        end
    end

    @testset "The sizes of the timesteps are returned as they were given" begin
        run_with_sizes(timestep_sizes; keywords...) = simulate_for_test(
            LearnsRewards(),
            RewardsWithProbability();
            n_steps=3,
            timestep_sizes,
            keywords...,
        ).timestep_sizes
        #One size for every timestep is returned as it is, whatever it is
        @test run_with_sizes(1) === 1
        @test run_with_sizes(0.5) === 0.5
        @test run_with_sizes((; fast=0.1, slow=10.0)) === (; fast=0.1, slow=10.0)
        #A vector has the size of each timestep
        @test run_with_sizes([0.5, 2.0, 0.25]) == [0.5, 2.0, 0.25]
        @test run_with_sizes(1:3) == 1:3
        #They are returned whatever is saved
        @test run_with_sizes(
            0.5;
            cognitive_save_history=SaveHistory(),
            environment_save_history=SaveHistory(),
        ) === 0.5
        #With timestep 0, a vector starts with BeforeFirstTimestep for it, so that it has an element for each element of the others, and one size stays as it is
        @test isequal(
            run_with_sizes(
                [0.5, 2.0, 0.25]; cognitive_save_history=SaveHistory(SaveEverything())
            ),
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25],
        )
        @test run_with_sizes(
            [0.5, 2.0, 0.25]; cognitive_save_history=SaveHistory(SaveEverything())
        ) isa Vector{Union{BeforeFirstTimestep,Float64}}
        @test run_with_sizes(0.5; cognitive_save_history=SaveHistory(SaveEverything())) ===
            0.5
    end

    @testset "An environment's stopping criterion ends the simulation, and n_steps is the most timesteps there can be" begin
        @test CognitiveModeling.stopping_criterion(
            RewardsWithProbability(), (;), REWARDS
        ) === false
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_until_end(n_steps; timestep_sizes=1, keywords...) = simulate_for_test(
                LearnsRewards(),
                EndsAfterTimesteps();
                who_starts,
                n_steps,
                timestep_sizes,
                environment_parameters=(; last_timestep=7),
                keywords...,
            )
            #Everything returned has the timesteps that ran
            simulation = run_until_end(20)
            @test all(vector -> length(vector) == 7, saved_vectors(simulation))
            @test simulation.environment.states[end] == (; n_timesteps=7)
            @test simulation.timestep_sizes === 1
            #The same as a simulation of that many timesteps
            @test run_until_end(20) == run_until_end(7)
            #Fewer timesteps than it takes to end run in full
            @test length(run_until_end(5).observations) == 5
            #A vector of sizes has one for each timestep there can be, and those of the timesteps that ran are returned
            @test run_until_end(20; timestep_sizes=collect(0.5:0.5:10.0)).timestep_sizes ==
                collect(0.5:0.5:3.5)
            #With timestep 0, everything has it too
            with_timestep_zero = run_until_end(
                20;
                timestep_sizes=collect(0.5:0.5:10.0),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )
            @test all(vector -> length(vector) == 8, saved_vectors(with_timestep_zero))
            @test length(with_timestep_zero.timestep_sizes) == 8
            @test with_timestep_zero.environment.states[1] == (; n_timesteps=0) &&
                with_timestep_zero.environment.states[end] == (; n_timesteps=7)
            #The return type is still known
            rng = Xoshiro(1)
            typed_run(n_steps) = simulate(
                LearnsRewards(),
                EndsAfterTimesteps();
                who_starts,
                n_steps,
                cognitive_parameters=LEARNING,
                environment_parameters=(; last_timestep=7),
                rng,
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )
            @test @inferred(typed_run(20)) isa NamedTuple
            JET.@test_opt target_modules = (CognitiveModeling,) typed_run(20)
        end
        #When the environment starts, the model has observed and acted on the last timestep
        simulation = simulate_for_test(
            LearnsRewards(),
            EndsAfterTimesteps();
            n_steps=20,
            environment_parameters=(; last_timestep=2),
        )
        @test expected_rewards(simulation.states) == [0.75, 0.875] &&
            length(simulation.actions) == 2
        #An environment that has ended already can't be simulated
        @test_throws "The stopping criterion of an environment model of type EndsAfterTimesteps is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate_for_test(
            LearnsRewards(),
            EndsAfterTimesteps();
            environment_parameters=(; last_timestep=0),
        )
    end

    @testset "A cognitive model's stopping criterion ends the simulation too, and check_stopping_criterion = false runs every timestep" begin
        @test CognitiveModeling.stopping_criterion(LearnsRewards(), (;), LEARNING) === false
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_until_end(; keywords...) = simulate_for_test(
                StopsAfterTimesteps(),
                RewardsWithProbability();
                who_starts,
                n_steps=20,
                cognitive_parameters=(; last_timestep=3),
                keywords...,
            )
            #The model ends the simulation after its third timestep
            simulation = run_until_end()
            @test all(vector -> length(vector) == 3, saved_vectors(simulation))
            @test simulation.states[end] == (; n_timesteps=3)
            #Without the check, every timestep runs
            every_timestep = run_until_end(; check_stopping_criterion=false)
            @test all(vector -> length(vector) == 20, saved_vectors(every_timestep))
            @test every_timestep.states[end] == (; n_timesteps=20)
            @test every_timestep.timestep_sizes === 1
            #The same goes for an environment which would end the simulation
            environment_ends = simulate_for_test(
                LearnsRewards(),
                EndsAfterTimesteps();
                who_starts,
                n_steps=20,
                environment_parameters=(; last_timestep=7),
                check_stopping_criterion=false,
            )
            @test all(vector -> length(vector) == 20, saved_vectors(environment_ends))
            @test environment_ends.environment.states[end] == (; n_timesteps=20)
            #A model whose criterion is met already can't run a timestep, unless the check is off
            @test_throws "The stopping criterion of a cognitive model of type StopsAfterTimesteps is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." run_until_end(
                cognitive_parameters=(; last_timestep=0)
            )
            @test length(
                run_until_end(;
                    cognitive_parameters=(; last_timestep=0), check_stopping_criterion=false
                ).actions,
            ) == 20
            @test length(
                simulate_for_test(
                    LearnsRewards(),
                    EndsAfterTimesteps();
                    who_starts,
                    n_steps=20,
                    environment_parameters=(; last_timestep=0),
                    check_stopping_criterion=false,
                ).actions,
            ) == 20
        end
        #Only a Bool is taken
        @test_throws TypeError simulate_for_test(
            LearnsRewards(), RewardsWithProbability(); check_stopping_criterion=1
        )
    end
end

@testset "simulate!" begin
    @testset "Two simulations in a row give the same as one long one" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate(
                LearnsRewards(),
                RewardsTrue();
                who_starts,
                n_steps=60,
                cognitive_parameters=LEARNING,
                environment_parameters=(;),
                rng=Xoshiro(9),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )

            agent = InstantiatedModel(
                LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
            )
            environment = InstantiatedModel(
                RewardsTrue(), (;); save_history=SaveHistory(SaveEverything())
            )
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps=25, rng)
            second_part = simulate!(agent, environment; who_starts, n_steps=35, rng)

            #simulate! returns no timestep 0, which the one simulation starts with, and which the histories have from where they started
            for (first_vector, second_vector, whole_vector) in zip(
                saved_vectors(first_part),
                saved_vectors(second_part),
                saved_vectors(one_simulation),
            )
                @test isequal(vcat(first_vector, second_vector), whole_vector[2:end])
            end
            @test isequal(agent.history.inputs, one_simulation.observations)
            @test isequal(agent.history.outputs, one_simulation.actions)
            @test isequal(agent.history.states, one_simulation.states)
            @test isequal(
                agent.history.output_distributions, one_simulation.action_distributions
            )
            @test isequal(environment.history.states, one_simulation.environment.states)
            @test isequal(
                environment.history.outputs, one_simulation.environment.observations
            )
            @test isequal(environment.history.inputs, one_simulation.environment.actions)
            @test isequal(
                environment.history.output_distributions,
                one_simulation.environment.observation_distributions,
            )
            #The size of the timesteps is returned as it was given
            @test first_part.timestep_sizes === 1 &&
                second_part.timestep_sizes === 1 &&
                one_simulation.timestep_sizes === 1
            @test agent.state == one_simulation.states[end]
            @test agent.previous_output == one_simulation.actions[end]
            @test environment.state == one_simulation.environment.states[end]
            @test environment.previous_output == one_simulation.observations[end]
        end
    end

    @testset "Timestep sizes in two simulations in a row give the same as in one long one" begin
        sizes = [0.5, 2.0, 0.25, 1.5, 3.0]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate(
                SavesTimestepSize(),
                ObservesTimestepSize();
                who_starts,
                n_steps=5,
                timestep_sizes=sizes,
                cognitive_parameters=(;),
                environment_parameters=(;),
                rng=Xoshiro(9),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )

            agent = InstantiatedModel(
                SavesTimestepSize(), (;); save_history=SaveHistory(SaveEverything())
            )
            environment = InstantiatedModel(
                ObservesTimestepSize(), (;); save_history=SaveHistory(SaveEverything())
            )
            rng = Xoshiro(9)
            first_part = simulate!(
                agent, environment; who_starts, n_steps=2, timestep_sizes=sizes[1:2], rng
            )
            second_part = simulate!(
                agent, environment; who_starts, n_steps=3, timestep_sizes=sizes[3:5], rng
            )

            #The vectors, and the sizes of the timesteps that follow them, after timestep 0, which the histories have too
            for (first_vector, second_vector, whole_vector) in zip(
                saved_vectors(first_part),
                saved_vectors(second_part),
                saved_vectors(one_simulation),
            )
                @test isequal(vcat(first_vector, second_vector), whole_vector[2:end])
            end
            @test isequal(
                vcat(first_part.timestep_sizes, second_part.timestep_sizes),
                one_simulation.timestep_sizes[2:end],
            )
            @test isequal(agent.history.timestep_sizes, one_simulation.timestep_sizes)
            @test isequal(environment.history.timestep_sizes, one_simulation.timestep_sizes)
        end
    end

    @testset "Later simulations don't change the outputs of earlier ones" begin
        agent = InstantiatedModel(
            LearnsInPlace(deepcopy), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        (; states) = simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        last_value = states[end].values[1]
        simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        @test states[end].values[1] == last_value
    end

    @testset "simulate! continues from the state itself, whatever is saved" begin
        agent = InstantiatedModel(
            LearnsInPlace(state -> (; values=copy(state.values))),
            LEARNING;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        first_states = simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
        ).states
        @test agent.state.values[1] == first_states[end].values[1]
        second_states = simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
        ).states
        @test agent.state.values[1] == second_states[end].values[1]
    end

    @testset "An agent fills in the default parameters, and starts from them" begin
        agent = InstantiatedModel(HasDefaults(), (;))
        @test agent.parameters == (; learning_rate=0.1, initial_value=0.5)
        @test agent.state == (; expected_reward=0.5)
        agent = InstantiatedModel(HasDefaults(), (; initial_value=0.9))
        @test agent.parameters == (; learning_rate=0.1, initial_value=0.9)
        @test agent.state == (; expected_reward=0.9)
        @test InstantiatedModel(
            HasDefaults(),
            (; learning_rate=0.3),
            (; expected_reward=0.2),
            (; choice=NoValue()),
        ).parameters == (; learning_rate=0.3, initial_value=0.5)
        #A model without defaults keeps the parameters as they are
        @test InstantiatedModel(LearnsRewards(), LEARNING).parameters === LEARNING
        #The model is kept as it is
        @test InstantiatedModel(LearnsRewards(), LEARNING).model === LearnsRewards()
    end

    @testset "An agent checks the names of the parameters and the state against attribute_names" begin
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but the parameters in the attributes of a cognitive model of type HasDefaults are (:learning_rate, :initial_value)" InstantiatedModel(
            HasDefaults(), (; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type HasDefaults are (:expected_reward,)" InstantiatedModel(
            HasDefaults(), (;), (; value=0.5), (; choice=NoValue())
        )
        @test_throws "The previous output is NoValue(), but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type HasDefaults, (:choice,)." InstantiatedModel(
            HasDefaults(), (;), (; expected_reward=0.5), NoValue()
        )
        @test_throws "must be a NamedTuple with the outputs in the attributes of" InstantiatedModel(
            HasDefaults(), (;), (; expected_reward=0.5), (; bet=true)
        )
        #The order of the names doesn't matter
        @test InstantiatedModel(HasDefaults(), (; initial_value=0.5, learning_rate=0.1)).parameters.learning_rate ==
            0.1
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of a cognitive model of type HasDefaults, (:expected_reward,), also when it is a single object." InstantiatedModel(
            HasDefaults(), (;), [0.5], (; choice=NoValue())
        )
        #A model without names can't be simulated or be instantiated
        @test_throws "There is no attribute_names method for a cognitive model of type WithoutNames. Each model must define CognitiveModeling.attribute_names(model::WithoutNames), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs." InstantiatedModel(
            WithoutNames(), (;)
        )
        @test_throws "There is no attribute_names method for a cognitive model of type WithoutNames" simulate_for_test(
            WithoutNames(), RewardsWithProbability(); cognitive_parameters=(;)
        )
    end

    @testset "An agent and an environment need a method for the initial state, unless a state is given" begin
        @test_throws "There is no initialize_state method for a cognitive model of type NeedsGivenState" InstantiatedModel(
            NeedsGivenState(), (;)
        )
        @test_throws "There is no initialize_state method for an environment model of type EnvironmentNeedsGivenState" InstantiatedModel(
            EnvironmentNeedsGivenState(), (;)
        )
        @test InstantiatedModel(
            NeedsGivenState(), (;), (; value=1.0), (; choice=NoValue())
        ).state == (; value=1.0)
        @test InstantiatedModel(
            EnvironmentNeedsGivenState(), (;), (; value=2.0), (; reward=NoValue())
        ).state == (; value=2.0)
    end

    @testset "An environment fills in the default parameters, and checks the names like an agent" begin
        environment = InstantiatedModel(RewardsWithDefaults(), (;))
        @test environment.parameters == (; reward_size=1.0, bias=0.0)
        @test InstantiatedModel(RewardsWithDefaults(), (; bias=0.5)).parameters ==
            (; reward_size=1.0, bias=0.5)
        @test InstantiatedModel(
            RewardsWithDefaults(), (; bias=0.5), (; last_choice=true), (; reward=1.0)
        ).parameters == (; reward_size=1.0, bias=0.5)
        @test InstantiatedModel(RewardsWithProbability(), REWARDS).parameters === REWARDS
        @test_throws "The parameters are named (:reward_size, :bias, :noise), but the parameters in the attributes of an environment model of type RewardsWithDefaults are (:reward_size, :bias)" InstantiatedModel(
            RewardsWithDefaults(), (; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of an environment model of type RewardsWithDefaults are (:last_choice,)" InstantiatedModel(
            RewardsWithDefaults(), (;), (; value=0.5), (; reward=NoValue())
        )
        @test_throws "The previous output is NoValue(), but it must be a NamedTuple with the outputs in the attributes of an environment model of type RewardsWithDefaults, (:reward,)." InstantiatedModel(
            RewardsWithDefaults(), (;), (; last_choice=NoValue()), NoValue()
        )
        @test_throws "must be a NamedTuple with the outputs in the attributes of" InstantiatedModel(
            RewardsWithDefaults(), (;), (; last_choice=NoValue()), (; cue=1)
        )
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of an environment model of type RewardsWithDefaults, (:last_choice,), also when it is a single object." InstantiatedModel(
            RewardsWithDefaults(), (;), [0.5], (; reward=NoValue())
        )
        #An environment model without names can't be simulated or be instantiated
        @test_throws "There is no attribute_names method for an environment model of type EnvironmentWithoutNames. Each model must define CognitiveModeling.attribute_names(model::EnvironmentWithoutNames)" InstantiatedModel(
            EnvironmentWithoutNames(), (;)
        )
        @test_throws "There is no attribute_names method for an environment model of type EnvironmentWithoutNames" simulate_for_test(
            LearnsRewards(), EnvironmentWithoutNames(); environment_parameters=(;)
        )
    end

    @testset "simulate fills in the default parameters, and checks the names like an agent" begin
        run_with(model, environment; keywords...) =
            simulate_for_test(model, environment; n_steps=2, keywords...)
        #The defaults are used, also for the initial state, and the given values replace them
        (; states) = run_with(
            HasDefaults(), RewardsWithProbability(); cognitive_parameters=(;)
        )
        @test states[1].expected_reward != 0.5
        (; states) = run_with(
            HasDefaults(),
            RewardsWithProbability();
            cognitive_parameters=(; learning_rate=0.0, initial_value=0.9),
        )
        @test [state.expected_reward for state in states] == [0.9, 0.9]
        simulation = run_with(
            LearnsRewards(), RewardsWithDefaults(); environment_parameters=(; bias=0.5)
        )
        @test all(observation -> observation.reward >= 0.5, simulation.observations)
        @test simulation.environment.states[end].last_choice isa Bool
        #The names of the parameters, the states, the previous action and the previous observation are checked
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but the parameters in the attributes of a cognitive model of type HasDefaults" run_with(
            HasDefaults(), RewardsWithProbability(); cognitive_parameters=(; noise=0.1)
        )
        @test_throws "The parameters are named (:reward_probability, :noise), but the parameters in the attributes of an environment model of type RewardsWithProbability" run_with(
            LearnsRewards(),
            RewardsWithProbability();
            environment_parameters=(; reward_probability=0.8, noise=0.1),
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type LearnsRewards" run_with(
            LearnsRewards(),
            RewardsWithProbability();
            previous_cognitive_state=(; value=0.5),
        )
        @test_throws "The state is named (:value,), but the states in the attributes of an environment model of type FixedRewards" run_with(
            LearnsRewards(),
            FixedRewards();
            environment_parameters=(; rewards=[1.0, 0.0]),
            previous_environment_state=(; value=0),
        )
        @test_throws "The previous output is (bet = true,), but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type LearnsRewards, (:choice,)." run_with(
            LearnsRewards(), RewardsWithProbability(); previous_action=(; bet=true)
        )
        @test_throws "The previous output is (cue = 1,), but it must be a NamedTuple with the outputs in the attributes of an environment model of type RewardsWithProbability, (:reward,)." run_with(
            LearnsRewards(), RewardsWithProbability(); previous_observation=(; cue=1)
        )
        @test_throws "must be a NamedTuple with the outputs" run_with(
            LearnsRewards(), RewardsWithProbability(); previous_observation=NoValue()
        )
        #The order of the names doesn't matter
        @test run_with(
            HasDefaults(),
            RewardsWithProbability();
            cognitive_parameters=(; initial_value=0.5, learning_rate=0.1),
        ) isa NamedTuple
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of a cognitive model of type NeedsGivenState, (:value,), also when it is a single object." run_with(
            NeedsGivenState(),
            RewardsWithProbability();
            cognitive_parameters=(;),
            previous_cognitive_state=[1.0],
        )
    end

    @testset "The checks of the names are decided at compile time, so they cost nothing" begin
        #Each compiles to code which only returns, however many names there are
        @test folds(
            CognitiveModeling.check_names_fit, LearnsRewards, RewardsWithProbability
        )
        @test folds(
            CognitiveModeling.check_match_attribute_names,
            LearnsAndCounts,
            typeof(LEARNING),
            typeof((; expected_reward=0.5, n_timesteps=0)),
            typeof((; choice=NoValue())),
        )
        @test folds(
            CognitiveModeling.check_match_attribute_names,
            RewardsAndCues,
            typeof((;)),
            typeof((;)),
            typeof((; reward=NoValue(), cue=NoValue())),
        )
        @test folds(
            CognitiveModeling.check_input_names,
            LearnsRewards,
            typeof((; reward=1.0, cue=2)),
        )
        @test folds(() -> CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED))
        @test CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED)
        @test !CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES[1:64])
        @test CognitiveModeling.has_every_name((:reward,), (:cue, :reward)) &&
            !CognitiveModeling.has_every_name((:reward, :cue), (:reward,))
    end

    @testset "Without maps, the model and the environment must have the same observations and actions in their attributes" begin
        #An environment which gives something else, or more, than the inputs in the model's attributes
        @test_throws "The attributes of a cognitive model of type LearnsRewards have the observations (:reward,) as inputs, but an environment model of type OnlyCues gives (:cue,). Without a map, the two must have the same observations in their attributes. A NameMap as observation_map can select and rename them." simulate_for_test(
            LearnsRewards(), OnlyCues(); environment_parameters=(;)
        )
        @test_throws "The attributes of a cognitive model of type LearnsRewards have the observations (:reward,) as inputs, but an environment model of type RewardsAndCues gives (:reward, :cue). Without a map, the two must have the same observations in their attributes." simulate_for_test(
            LearnsRewards(), RewardsAndCues(); environment_parameters=(;)
        )
        #A model with fewer observations in its attributes than the environment gives
        @test_throws "The attributes of a cognitive model of type OnlyChooses have the observations () as inputs, but an environment model of type RewardsTrue gives (:reward,)." simulate_for_test(
            OnlyChooses(),
            RewardsTrue();
            cognitive_parameters=(;),
            environment_parameters=(;),
        )
        #A model which gives fewer actions than the inputs in the environment's attributes, or more
        @test_throws "A cognitive model of type OnlyLearns gives the actions (), but the attributes of an environment model of type RewardsTrue have (:choice,) as inputs. Without a map, the two must have the same actions in their attributes. A NameMap as action_map can select and rename them." simulate_for_test(
            OnlyLearns(), RewardsTrue(); environment_parameters=(;)
        )
        @test_throws "A cognitive model of type LearnsRewards gives the actions (:choice,), but the attributes of an environment model of type RewardsAndCues have () as inputs. Without a map, the two must have the same actions in their attributes." simulate_for_test(
            LearnsRewards(),
            RewardsAndCues();
            environment_parameters=(;),
            observation_map=NameMap(:reward),
        )
        #The same names in another order fit
        @test CognitiveModeling.check_names_fit(
            TwoChoices(),
            RewardsWithProbability(),
            nothing,
            NameMap(:second_choice => :choice),
        ) === nothing
        #simulate! checks it too
        @test_throws "Without a map, the two must have the same observations in their attributes." simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            InstantiatedModel(OnlyCues(), (;));
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test_throws "Without a map, the two must have the same actions in their attributes." simulate!(
            InstantiatedModel(OnlyLearns(), LEARNING),
            InstantiatedModel(RewardsTrue(), (;));
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            InstantiatedModel(RewardsAndCues(), (;));
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=NameMap(:reward),
            action_map=NameMap(),
        ) isa NamedTuple
    end

    @testset "Maps of names let a model and an environment with other names be used together" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #The same simulation as with an environment which uses the model's names, for the cognitive model's part
            same_names = simulate_for_test(
                LearnsRewards(),
                RewardsTrue();
                who_starts,
                environment_parameters=(;),
                rng=Xoshiro(5),
            )
            mapped = simulate_for_test(
                LearnsRewards(),
                PaysForLever();
                who_starts,
                environment_parameters=(;),
                rng=Xoshiro(5),
                observation_map=PAYOFF_AS_REWARD,
                action_map=CHOICE_AS_LEVER,
            )
            @test isequal(cognitive_values(mapped), cognitive_values(same_names))
            #The observations are returned as the model received them, without the cue, and the actions as it gave them
            @test keys(mapped.observations[end]) == (:reward,)
            @test keys(mapped.actions[end]) == (:choice,)
            #Under environment, they are as the environment gave and received them
            @test keys(mapped.environment.observations[end]) == (:payoff, :cue)
            @test keys(mapped.environment.actions[end]) == (:lever,)
            @test isequal(
                [observation.payoff for observation in mapped.environment.observations],
                [observation.reward for observation in same_names.environment.observations],
            )
            @test [action.lever for action in mapped.environment.actions] == [action.choice for action in same_names.environment.actions]
        end
        #Without the maps, the two don't fit
        @test_throws "Without a map, the two must have the same observations in their attributes. A NameMap as observation_map can select and rename them." simulate_for_test(
            LearnsRewards(), PaysForLever(); environment_parameters=(;)
        )
        @test_throws "Without a map, the two must have the same actions in their attributes. A NameMap as action_map can select and rename them." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
        )
    end

    @testset "A map selects what the receiver gets" begin
        #The model receives only the observations the map takes, and a map without names passes no actions on
        simulation = simulate_for_test(
            LearnsRewards(),
            RewardsAndCues();
            n_steps=2,
            environment_parameters=(;),
            observation_map=NameMap(:reward),
            action_map=NameMap(),
        )
        @test simulation.observations == fill((; reward=1.0), 2)
        @test simulation.environment.observations == fill((; reward=1.0, cue=2), 2)
        @test simulation.environment.actions == fill((;), 2)
        #An environment receives only the actions the map takes, here the second of two
        (; observations, actions) = simulate_for_test(
            TwoChoices(),
            RewardsTrue();
            n_steps=3,
            cognitive_parameters=(;),
            environment_parameters=(;),
            action_map=NameMap(:second_choice => :choice),
        )
        @test [observation.reward for observation in observations] == [NoValue(), 1.0, 1.0]
        @test actions == fill((; first_choice=false, second_choice=true), 3)
        simulation = simulate_for_test(
            TwoChoices(),
            RewardsTrue();
            n_steps=3,
            cognitive_parameters=(;),
            environment_parameters=(;),
            action_map=NameMap(:first_choice => :choice),
        )
        @test [observation.reward for observation in simulation.observations] == [NoValue(), 0.0, 0.0]
        @test simulation.environment.actions == fill((; choice=false), 3)
    end

    @testset "A map must take names that are given, and give exactly the names in the receiver's attributes" begin
        @test_throws "The observation_map takes the observations (:prize,), but an environment model of type PaysForLever gives (:payoff, :cue). A map can only take observations the environment gives." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=NameMap(:prize => :reward),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The observation_map gives the observations (:reward, :cue), but the attributes of a cognitive model of type LearnsRewards have (:reward,) as inputs. A map must give exactly the observations in the cognitive model's attributes." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=NameMap(:payoff => :reward, :cue),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The observation_map gives the observations (:cue,), but the attributes of a cognitive model of type LearnsRewards have (:reward,) as inputs." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=NameMap(:cue),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The action_map takes the actions (:press,), but a cognitive model of type LearnsRewards gives (:choice,). A map can only take actions the cognitive model gives." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
            action_map=NameMap(:press => :lever),
        )
        @test_throws "The action_map gives the actions (:choice,), but the attributes of an environment model of type PaysForLever have (:lever,) as inputs. A map must give exactly the actions in the environment's attributes." simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=PAYOFF_AS_REWARD,
            action_map=NameMap(:choice),
        )
        #Only a NameMap is taken
        @test_throws TypeError simulate_for_test(
            LearnsRewards(),
            PaysForLever();
            environment_parameters=(;),
            observation_map=(:payoff => :reward,),
        )
    end

    @testset "Maps keep the return type known and the loop free of allocations, and their checks cost nothing" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate(
            LearnsRewards(),
            PaysWithProbability();
            who_starts,
            n_steps,
            cognitive_parameters=LEARNING,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa NamedTuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                who_starts, 10
            )
            run_simulation(who_starts, 10)
            #As without maps for the cognitive model: 8 bytes each for an observation, an action distribution and a state, 1 for an
            #action, and a byte each for which type they have. The environment's observations and observation distributions have the
            #cue too, 16 bytes each, and its actions 1
            extra_bytes =
                @allocated(run_simulation(who_starts, 20_000)) -
                @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 70 * 10_000
        end
        @test folds(
            CognitiveModeling.check_names_fit,
            LearnsRewards,
            PaysForLever,
            typeof(PAYOFF_AS_REWARD),
            typeof(CHOICE_AS_LEVER),
        )
    end

    @testset "With maps, each history has the names of its own side" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(
            PaysForLever(), (;); save_history=SaveHistory(SaveEverything())
        )
        simulation = simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        #The agent's are the ones returned as observations and actions, under the model's names
        @test agent.history.inputs[2:end] == simulation.observations
        @test agent.history.outputs[2:end] == simulation.actions
        @test length(agent.history.output_distributions) == length(agent.history.outputs) &&
            agent.history.output_distributions[1] === (; choice=BeforeFirstTimestep())
        @test all(observation -> keys(observation) == (:reward,), agent.history.inputs)
        @test all(action -> keys(action) == (:choice,), agent.history.outputs)
        #The environment's are the same values, under its own names, with the cue the model didn't receive, and are the ones
        #returned under environment
        @test all(
            observation -> keys(observation) == (:payoff, :cue), environment.history.outputs
        )
        @test all(action -> keys(action) == (:lever,), environment.history.inputs)
        @test isequal(
            [observation.payoff for observation in environment.history.outputs],
            [observation.reward for observation in agent.history.inputs],
        )
        @test [action.lever for action in environment.history.inputs[2:end]] == [action.choice for action in simulation.actions]
        @test environment.history.outputs[2:end] == simulation.environment.observations
        @test environment.history.inputs[2:end] == simulation.environment.actions
        @test environment.history.output_distributions[2:end] ==
            simulation.environment.observation_distributions
        @test environment.history.output_distributions[1] ===
            (; payoff=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
        #Where they are left for the next simulation, each side has its own names too
        @test keys(environment.previous_output) == (:payoff, :cue)
        @test keys(agent.previous_output) == (:choice,)
        @test simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        ) isa NamedTuple
        #simulate! checks the maps too
        @test_throws "A map must give exactly the observations in the cognitive model's attributes." simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=NameMap(:cue),
            action_map=CHOICE_AS_LEVER,
        )
    end

    @testset "It stops when the environment's stopping criterion is met, and the environment must then be reset" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(
            EndsAfterTimesteps(),
            (; last_timestep=4);
            save_history=SaveHistory(SaveEverything()),
        )
        #Two timesteps run in full, and the next call stops after the two that are left
        @test length(
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2).observations,
        ) == 2
        simulation = simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=10,
            timestep_sizes=collect(1.0:10.0),
        )
        @test all(vector -> length(vector) == 2, saved_vectors(simulation))
        @test simulation.timestep_sizes == [1.0, 2.0]
        @test environment.state == (; n_timesteps=4)
        #Both histories have the timesteps that ran, after where they started
        @test length(agent.history.states) == length(agent.history.timestep_sizes) == 5
        @test agent.history.timestep_sizes == [BeforeFirstTimestep(), 1, 1, 1.0, 2.0]
        @test environment.history.timestep_sizes == agent.history.timestep_sizes
        @test length(environment.history.states) == 5
        #It has ended, so it can't be simulated further until it is reset
        @test_throws "The stopping criterion of an environment model of type EndsAfterTimesteps is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=10
        )
        @test length(agent.history.states) == 5
        #Unless the check is off, in which case it runs on
        @test length(
            simulate!(
                agent,
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=3,
                check_stopping_criterion=false,
            ).observations,
        ) == 3
        @test environment.state == (; n_timesteps=7)
        reset!(environment)
        @test length(
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10).observations,
        ) == 4
    end

    @testset "It stops when the agent's stopping criterion is met, and the agent must then be reset" begin
        agent = InstantiatedModel(
            StopsAfterTimesteps(),
            (; last_timestep=4);
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(
            RewardsWithProbability(), REWARDS; save_history=SaveHistory(SaveEverything())
        )
        #Two timesteps run in full, and the next call stops after the two that are left
        @test length(
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2).actions
        ) == 2
        simulation = simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=10
        )
        @test all(vector -> length(vector) == 2, saved_vectors(simulation))
        @test agent.state == (; n_timesteps=4)
        @test length(agent.history.states) == length(environment.history.states) == 5
        #It has ended, so it can't be simulated further until it is reset
        @test_throws "The stopping criterion of a cognitive model of type StopsAfterTimesteps is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=10
        )
        reset!(agent)
        @test length(
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10).actions,
        ) == 4
    end

    @testset "An error in a timestep method gives the same errors as in simulate" begin
        agent = InstantiatedModel(ReturnsTuple(), (;))
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        @test_throws "The apply_timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=1
        )
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        environment = InstantiatedModel(ReturnsObservation(), (;))
        @test_throws "The apply_timestep_model method of an environment model of type ReturnsObservation must return a CognitiveModeling.ModelOutput" simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=1
        )
    end
end

@testset "simulate! on a single input" begin
    @testset "It updates the agent and returns the sampled action" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        action = simulate!(agent, (; reward=1.0); rng=Xoshiro(1))
        @test action isa @NamedTuple{choice::Bool}
        @test agent.previous_output === action
        @test agent.state == (; expected_reward=0.75)
    end

    @testset "Several in a row give the same as a simulation with the same observations" begin
        rewards = [1.0, 0.0, 1.0, 1.0]
        one_simulation = simulate(
            LearnsRewards(),
            FixedRewards();
            who_starts=EnvironmentStarts(),
            n_steps=4,
            cognitive_parameters=LEARNING,
            environment_parameters=(; rewards),
            rng=Xoshiro(9),
            cognitive_save_history=SaveHistory(SaveEverything()),
        )

        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        rng = Xoshiro(9)
        actions = [simulate!(agent, (; reward); rng) for reward in rewards]

        #The simulation starts with timestep 0
        @test actions == one_simulation.actions[2:end]
        @test agent.state == one_simulation.states[end]
    end

    @testset "The same seed gives the same action" begin
        action_with_seed(seed) = [
            simulate!(
                InstantiatedModel(LearnsRewards(), LEARNING),
                (; reward=1.0);
                rng=Xoshiro(seed),
            ) for _ in 1:20
        ]
        @test action_with_seed(3) == action_with_seed(3)
        @test action_with_seed(3) != action_with_seed(4)
    end

    @testset "An observation can be NoValue" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        simulate!(agent, (; reward=NoValue()))
        @test agent.state == (; expected_reward=0.5)
    end

    @testset "The input must be a NamedTuple with the inputs in the model's attributes, and can have more" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        @test_throws "The input is 1.0, but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type LearnsRewards, (:reward,)." simulate!(
            agent, 1.0
        )
        @test_throws "must be a NamedTuple with the inputs" simulate!(agent, NoValue())
        @test_throws "must be a NamedTuple with the inputs" simulate!(agent, (; cue=1.0))
        @test simulate!(agent, (; reward=1.0, cue=1.0)) isa NamedTuple
        @test agent.state == (; expected_reward=0.75)
        #A model without inputs takes an empty NamedTuple
        @test simulate!(InstantiatedModel(OnlyChooses(), (;)), (;)) isa NamedTuple
        #The check costs nothing when the types are known
        JET.@test_opt target_modules = (CognitiveModeling,) CognitiveModeling.simulate_timestep(
            LearnsRewards(),
            (; reward=1.0, cue=1.0),
            (; expected_reward=0.5),
            (; choice=NoValue()),
            LEARNING,
            1,
            Xoshiro(1),
            SaveHistory(),
        )
    end

    @testset "An environment model can be run on a single input too, which is an action" begin
        environment = InstantiatedModel(
            RewardsWithDefaults(), (; bias=0.5); save_history=SaveHistory(SaveEverything())
        )
        observation = simulate!(environment, (; choice=true); rng=Xoshiro(1))
        @test observation === (; reward=1.5)
        @test environment.previous_output === observation
        @test environment.state === (; last_choice=true)
        @test environment.history.inputs ==
            [(; choice=BeforeFirstTimestep()), (; choice=true)]
        @test environment.history.outputs ==
            [(; reward=BeforeFirstTimestep()), (; reward=1.5)]
        @test environment.history.output_distributions ==
            [(; reward=BeforeFirstTimestep()), (; reward=Deterministic(1.5))]
        @test_throws "The input is (cue = 1,), but it must be a NamedTuple with the inputs in the attributes of an environment model of type RewardsWithDefaults, (:choice,)." simulate!(
            environment, (; cue=1)
        )
    end

    @testset "The model gets the size of the timestep" begin
        agent = InstantiatedModel(SavesTimestepSize(), (;))
        #The default is the integer 1
        simulate!(agent, (; timestep_size=NoValue()))
        @test agent.state.received_timestep_size === 1
        simulate!(agent, (; timestep_size=NoValue()); timestep_size=0.5)
        @test agent.state.received_timestep_size === 0.5
    end

    @testset "It works without actions, and with several actions" begin
        @test simulate!(InstantiatedModel(OnlyLearns(), LEARNING), (; reward=1.0)) === (;)
        @test simulate!(InstantiatedModel(TwoChoices(), (;)), (; reward=1.0)) ===
            (; first_choice=false, second_choice=true)
    end

    @testset "An error in the timestep method gives the same errors as in simulate" begin
        @test_throws "The apply_timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate!(
            InstantiatedModel(ReturnsTuple(), (;)), (; reward=1.0)
        )
        @test_throws "There is no apply_timestep_model method for a cognitive model of type OnlyInitializesState" simulate!(
            InstantiatedModel(OnlyInitializesState(), (;)), (; reward=1.0)
        )
    end
end

@testset "simulate on a single input" begin
    @testset "It returns the sampled output and the new state, starting from the initial state by default" begin
        action, cognitive_state = simulate(
            LearnsRewards(), (; reward=1.0); parameters=LEARNING, rng=Xoshiro(1)
        )
        @test action isa @NamedTuple{choice::Bool}
        @test cognitive_state == (; expected_reward=0.75)
        #The same as simulate! of an agent
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        @test simulate!(agent, (; reward=1.0); rng=Xoshiro(1)) === action
        @test agent.state === cognitive_state
        #An environment model runs on an action in the same way
        @test simulate(RewardsWithDefaults(), (; choice=true); parameters=(; bias=0.5)) ===
            ((; reward=1.5), (; last_choice=true))
    end

    @testset "Passing the state and the output on gives the same as a simulation with the same observations" begin
        rewards = [1.0, 0.0, 1.0, 1.0]
        one_simulation = simulate(
            LearnsRewards(),
            FixedRewards();
            who_starts=EnvironmentStarts(),
            n_steps=4,
            cognitive_parameters=LEARNING,
            environment_parameters=(; rewards),
            rng=Xoshiro(9),
            cognitive_save_history=SaveHistory(SaveEverything()),
        )

        rng = Xoshiro(9)
        action, cognitive_state = simulate(
            LearnsRewards(), (; reward=rewards[1]); parameters=LEARNING, rng
        )
        actions = [action]
        for reward in rewards[2:end]
            action, cognitive_state = simulate(
                LearnsRewards(),
                (; reward);
                parameters=LEARNING,
                previous_state=cognitive_state,
                previous_output=action,
                rng,
            )
            push!(actions, action)
        end

        #The simulation starts with timestep 0
        @test actions == one_simulation.actions[2:end]
        @test cognitive_state == one_simulation.states[end]
    end

    @testset "It fills in the default parameters, and checks the names and the input" begin
        _, cognitive_state = simulate(
            HasDefaults(),
            (; reward=1.0);
            parameters=(; initial_value=0.9, learning_rate=0.0),
        )
        @test cognitive_state == (; expected_reward=0.9)
        @test simulate(HasDefaults(), (; reward=1.0); parameters=(;))[2].expected_reward ≈
            0.55
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but the parameters in the attributes of a cognitive model of type HasDefaults" simulate(
            HasDefaults(), (; reward=1.0); parameters=(; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type LearnsRewards" simulate(
            LearnsRewards(),
            (; reward=1.0);
            parameters=LEARNING,
            previous_state=(; value=0.5),
        )
        @test_throws "The previous output is (bet = true,), but it must be a NamedTuple with the outputs" simulate(
            LearnsRewards(),
            (; reward=1.0);
            parameters=LEARNING,
            previous_output=(; bet=true),
        )
        @test_throws "The input is 1.0, but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type LearnsRewards, (:reward,)." simulate(
            LearnsRewards(), 1.0; parameters=LEARNING
        )
        #An input can have more than the inputs in the model's attributes, and an observation can be NoValue
        @test simulate(LearnsRewards(), (; reward=1.0, cue=2); parameters=LEARNING)[2] ==
            (; expected_reward=0.75)
        @test simulate(LearnsRewards(), (; reward=NoValue()); parameters=LEARNING)[2] ==
            (; expected_reward=0.5)
    end

    @testset "The model gets the size of the timestep, and the same seed gives the same action" begin
        @test simulate(
            SavesTimestepSize(), (; timestep_size=NoValue()); parameters=(;)
        )[2] === (; received_timestep_size=1)
        @test simulate(
            SavesTimestepSize(),
            (; timestep_size=NoValue());
            parameters=(;),
            timestep_size=0.5,
        )[2] === (; received_timestep_size=0.5)
        actions_with_seed(seed) = [
            simulate(
                LearnsRewards(),
                (; reward=1.0);
                parameters=LEARNING,
                rng=Xoshiro(seed + i),
            )[1] for i in 1:20
        ]
        @test actions_with_seed(3) == actions_with_seed(3)
        @test actions_with_seed(3) != actions_with_seed(40)
    end

    @testset "The return type is known, and nothing is allocated" begin
        rng = Xoshiro(1)
        run_timestep(cognitive_state, action) = simulate(
            LearnsRewards(),
            (; reward=1.0);
            parameters=LEARNING,
            previous_state=cognitive_state,
            previous_output=action,
            rng,
        )
        @test @inferred(run_timestep((; expected_reward=0.5), (; choice=true))) isa Tuple
        JET.@test_opt target_modules = (CognitiveModeling,) run_timestep(
            (; expected_reward=0.5), (; choice=true)
        )
        @test @allocated(run_timestep((; expected_reward=0.5), (; choice=true))) == 0
        #The first timestep, from the defaults, is known too
        @test @inferred(
            simulate(LearnsRewards(), (; reward=1.0); parameters=LEARNING, rng)
        ) isa Tuple
    end
end

@testset "The history of an agent" begin
    @testset "It starts where the agent starts" begin
        history = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        ).history
        @test history isa CognitiveModeling.ModelHistory
        @test history.inputs == [(; reward=BeforeFirstTimestep())]
        @test history.outputs == [(; choice=BeforeFirstTimestep())]
        @test history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test history.states == [(; expected_reward=0.5)]
        @test history.timestep_sizes == [BeforeFirstTimestep()]
        #The vectors are untyped, since what is saved can change type between timesteps
        @test history.states isa Vector{Any} && history.timestep_sizes isa Vector{Any}
        #Also if the state and the previous output are given
        history = InstantiatedModel(
            LearnsRewards(),
            LEARNING,
            (; expected_reward=0.25),
            (; choice=true);
            save_history=SaveHistory(SaveEverything()),
        ).history
        @test history.inputs == [(; reward=BeforeFirstTimestep())]
        @test history.outputs == [(; choice=true)]
        @test history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test history.states == [(; expected_reward=0.25)]
    end

    @testset "It starts empty when the initial timestep values aren't saved, and then has one element for each timestep" begin
        save_history = SaveHistory(SaveEverything(); save_initial_timestep_values=false)
        agent = InstantiatedModel(LearnsRewards(), LEARNING; save_history)
        environment = InstantiatedModel(
            FixedRewards(), (; rewards=[1.0, 0.0]); save_history
        )
        @test isempty(agent.history.states) &&
            isempty(agent.history.timestep_sizes) &&
            isempty(environment.history.states)
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2)
        simulate!(agent, (; reward=1.0); timestep_size=0.5)
        @test agent.history.states == [
            (; expected_reward=0.75), (; expected_reward=0.375), (; expected_reward=0.6875)
        ]
        @test agent.history.inputs == [(; reward=1.0), (; reward=0.0), (; reward=1.0)]
        @test agent.history.timestep_sizes == [1, 1, 0.5]
        @test environment.history.states == [(; n_timesteps=1), (; n_timesteps=2)]
        @test environment.history.outputs == [(; reward=1.0), (; reward=0.0)]
        @test environment.history.output_distributions ==
            [(; reward=Deterministic(1.0)), (; reward=Deterministic(0.0))]
        #reset! starts it empty again
        reset!(agent)
        @test isempty(agent.history.states) && isempty(agent.history.timestep_sizes)
    end

    @testset "Nothing is saved by default, and a SaveHistory selects what is" begin
        @test InstantiatedModel(LearnsRewards(), LEARNING).save_history === SaveHistory()
        @test InstantiatedModel(
            LearnsRewards(), LEARNING, (; expected_reward=0.25), (; choice=NoValue())
        ).save_history === SaveHistory()
        @test InstantiatedModel(RewardsWithProbability(), REWARDS).save_history ===
            SaveHistory()
        @test InstantiatedModel(
            RewardsWithProbability(), REWARDS, (;), (; reward=NoValue())
        ).save_history === SaveHistory()
        save_history = SaveHistory(; states=:n_timesteps)
        agent = InstantiatedModel(LearnsAndCounts(), LEARNING; save_history)
        @test agent.save_history === save_history
        @test InstantiatedModel(
            LearnsAndCounts(),
            LEARNING,
            (; expected_reward=0.25, n_timesteps=0),
            (; choice=NoValue());
            save_history,
        ).save_history === save_history
        #The history starts with what is selected
        @test agent.history.inputs == [(;)]
        @test agent.history.outputs == [(;)]
        @test agent.history.output_distributions == [(;)]
        @test agent.history.states == [(; n_timesteps=0)]
        #It gets what is selected on a single input and with an environment, which selects its own
        simulate!(agent, (; reward=1.0))
        environment = InstantiatedModel(
            FixedRewards(), (; rewards=[1.0, 0.0]); save_history=SaveHistory()
        )
        @test environment.save_history === SaveHistory()
        simulation = simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=2
        )
        @test agent.history.inputs == fill((;), 4)
        @test agent.history.outputs == fill((;), 4)
        @test agent.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        @test simulation.states == [(; n_timesteps=2), (; n_timesteps=3)]
        @test all(==([(;), (;)]), saved_vectors(simulation)[5:end])
        #The agent and the environment are still updated in full
        @test agent.state.n_timesteps == 3 && agent.state.expected_reward != 0.5
        @test environment.state == (; n_timesteps=2)
    end

    @testset "A history can only select names in the model's attributes" begin
        @test_throws "The SaveHistory selects the inputs (:reward,), but the inputs in the attributes of a cognitive model of type OnlyChooses are (). A history can only select names in the model's attributes." InstantiatedModel(
            OnlyChooses(), (;); save_history=SaveHistory(; inputs=:reward)
        )
        @test_throws "The SaveHistory selects the outputs (:bet,), but the outputs in the attributes of a cognitive model of type LearnsRewards are (:choice,)." InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(; outputs=:bet)
        )
        #For an environment, the inputs are the actions it receives, and the outputs the observations it gives
        @test_throws "The SaveHistory selects the inputs (:choice,), but the inputs in the attributes of an environment model of type RewardsAndCues are ()." InstantiatedModel(
            RewardsAndCues(), (;); save_history=SaveHistory(; inputs=:choice)
        )
        @test_throws "The SaveHistory selects the outputs (:cue,), but the outputs in the attributes of an environment model of type FixedRewards are (:reward,)" InstantiatedModel(
            FixedRewards(), (; rewards=[1.0]); save_history=SaveHistory(; outputs=:cue)
        )
        @test_throws "The SaveHistory selects the output distributions (:cue,), but the outputs in the attributes of an environment model of type FixedRewards are (:reward,)" InstantiatedModel(
            FixedRewards(),
            (; rewards=[1.0]);
            save_history=SaveHistory(; output_distributions=:cue),
        )
        #The states are checked too, since prepare_state_for_saving returns a state with the names in the model's attributes
        @test_throws "The SaveHistory selects the states (:first_value,), but the states in the attributes of a cognitive model of type LearnsInPlace" InstantiatedModel(
            LearnsInPlace(deepcopy),
            LEARNING;
            save_history=SaveHistory(; states=:first_value),
        )
        #Everything can always be selected, and an input with more than the inputs in the model's attributes is saved in full
        agent = InstantiatedModel(
            OnlyChooses(), (;); save_history=SaveHistory(SaveEverything())
        )
        simulate!(agent, (; cue=1.0))
        @test agent.history.inputs == [(;), (; cue=1.0)]
        #A selected input is kept from an input with more
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(; inputs=:reward)
        )
        simulate!(agent, (; reward=1.0, cue=1.0))
        @test agent.history.inputs == [(; reward=BeforeFirstTimestep()), (; reward=1.0)]
    end

    @testset "It adds what simulate! returns" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            agent = InstantiatedModel(
                LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
            )
            environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps=3, rng)
            second_part = simulate!(agent, environment; who_starts, n_steps=2, rng)

            @test agent.history.inputs == vcat(
                [(; reward=BeforeFirstTimestep())],
                first_part.observations,
                second_part.observations,
            )
            @test agent.history.outputs == vcat(
                (; choice=BeforeFirstTimestep()), first_part.actions, second_part.actions
            )
            @test agent.history.output_distributions == vcat(
                (; choice=BeforeFirstTimestep()),
                first_part.action_distributions,
                second_part.action_distributions,
            )
            @test agent.history.states ==
                vcat((; expected_reward=0.5), first_part.states, second_part.states)
        end
    end

    @testset "It adds what happens on a single input" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        first_action = simulate!(agent, (; reward=1.0))
        second_action = simulate!(agent, (; reward=NoValue()))

        @test agent.history.inputs ==
            [(; reward=BeforeFirstTimestep()), (; reward=1.0), (; reward=NoValue())]
        @test agent.history.outputs ==
            [(; choice=BeforeFirstTimestep()), first_action, second_action]
        @test agent.history.output_distributions == [
            (; choice=BeforeFirstTimestep()),
            (; choice=ChooseWithProbability(0.75)),
            (; choice=ChooseWithProbability(0.75)),
        ]
        @test agent.history.states ==
            [(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.75)]
    end

    @testset "A single input and an environment can be used on the same agent" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        simulate!(agent, (; reward=1.0))
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        simulate!(agent, (; reward=0.0))
        @test length(agent.history.inputs) ==
            length(agent.history.outputs) ==
            length(agent.history.output_distributions) ==
            length(agent.history.states) ==
            6
        @test agent.history.states[end] == agent.state
    end

    @testset "It stays empty by default" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        simulate!(agent, (; reward=1.0))
        simulation = simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3
        )
        @test isempty(agent.history.inputs)
        @test isempty(agent.history.outputs)
        @test isempty(agent.history.output_distributions)
        @test isempty(agent.history.states)
        #The agent is still updated, and the simulation still returns empty NamedTuples for each timestep
        @test agent.state != (; expected_reward=0.5)
        @test simulation.states == fill((;), 3)
        @test eltype(simulation.environment.states) === typeof((;))
    end

    @testset "It keeps the values of a state changed in place, with a method which copies it" begin
        for prepare_state in (deepcopy, state -> (; values=copy(state.values)))
            agent = InstantiatedModel(
                LearnsInPlace(prepare_state),
                LEARNING;
                save_history=SaveHistory(SaveEverything()),
            )
            environment = InstantiatedModel(FixedRewards(), (; rewards=[1.0, 0.0]))
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2)
            simulate!(agent, (; reward=1.0))
            #The initial state is copied by the same method
            @test [state.values[1] for state in agent.history.states] == [0.5, 0.75, 0.375, 0.6875]
            @test allunique(objectid(state.values) for state in agent.history.states)
        end
    end

    @testset "It saves what the method gives, also of the initial state" begin
        agent = InstantiatedModel(
            LearnsInPlace(state -> (; first_value=state.values[1])),
            LEARNING;
            save_history=SaveHistory(SaveEverything()),
        )
        simulate!(agent, (; reward=1.0))
        @test agent.history.states == [(; first_value=0.5), (; first_value=0.75)]
    end

    @testset "The initial state is the same object as the agent's, with a method which does nothing" begin
        agent = InstantiatedModel(
            LearnsInPlace(identity), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        @test agent.history.states[1].values === agent.state.values
    end

    @testset "Later simulations don't change it" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(RewardsWithProbability(), REWARDS)
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        states_so_far = copy(agent.history.states)
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        @test agent.history.states[1:4] == states_so_far
    end

    @testset "It has the size of each timestep, when anything is saved" begin
        agent = InstantiatedModel(
            SavesTimestepSize(), (;); save_history=SaveHistory(; outputs=:choice)
        )
        environment = InstantiatedModel(
            ObservesTimestepSize(),
            (;);
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        #A new history starts with the marker for before the first timestep, as the rest of it
        @test agent.history.timestep_sizes == [BeforeFirstTimestep()] &&
            environment.history.timestep_sizes == [BeforeFirstTimestep()]
        #A vector of sizes, and one size for every timestep
        simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes=[0.5, 2.0, 0.25],
        )
        simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=2, timestep_sizes=4
        )
        @test agent.history.timestep_sizes == [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4]
        @test environment.history.timestep_sizes ==
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4]
        #One entry for each entry of the outputs, which start with the previous output
        @test length(agent.history.outputs) == length(agent.history.timestep_sizes)
        #They are the sizes the environment gave the model as observations, its outputs
        @test [observation.timestep_size for observation in environment.history.outputs] == agent.history.timestep_sizes
        #A single input adds the size of its timestep, which is 1 by default
        simulate!(agent, (; timestep_size=NoValue()); timestep_size=1.5)
        simulate!(agent, (; timestep_size=NoValue()))
        @test agent.history.timestep_sizes ==
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4, 1.5, 1]
        #reset! starts from the marker again
        reset!(agent)
        @test agent.history.timestep_sizes == [BeforeFirstTimestep()]
        #Nothing is kept when nothing is saved, on either side
        agent = InstantiatedModel(SavesTimestepSize(), (;))
        environment = InstantiatedModel(
            ObservesTimestepSize(),
            (;);
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        simulate!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes=[0.5, 2.0, 0.25],
        )
        simulate!(agent, (; timestep_size=1.0); timestep_size=1.5)
        @test isempty(agent.history.timestep_sizes)
        @test environment.history.timestep_sizes == [BeforeFirstTimestep(), 0.5, 2.0, 0.25]
    end

    @testset "The environment has a history too, with the same observations and actions as its outputs and inputs" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(
            FixedRewards(),
            (; rewards=[1.0, 0.0, 1.0]);
            save_history=SaveHistory(SaveEverything()),
        )
        @test environment.history isa CognitiveModeling.ModelHistory
        @test environment.history.outputs == [(; reward=BeforeFirstTimestep())]
        #It starts with the actions in its attributes as inputs, which an environment can have none of
        @test environment.history.inputs == [(; choice=BeforeFirstTimestep())]
        @test InstantiatedModel(
            RewardsAndCues(), (;); save_history=SaveHistory(SaveEverything())
        ).history.inputs == [(;)]
        @test environment.history.states == [(; n_timesteps=0)]
        #Before the first timestep, it has given nothing to sample the observations from
        @test environment.history.output_distributions == [(; reward=BeforeFirstTimestep())]
        simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        @test environment.history.outputs == agent.history.inputs
        @test environment.history.inputs[2:end] == agent.history.outputs[2:end]
        @test environment.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        #The observation distributions are what it gave its observations to be sampled from
        @test environment.history.output_distributions == [
            (; reward=BeforeFirstTimestep()),
            (; reward=Deterministic(1.0)),
            (; reward=Deterministic(0.0)),
            (; reward=Deterministic(1.0)),
        ]
        #Its selection is its own
        environment = InstantiatedModel(
            FixedRewards(),
            (; rewards=[1.0, 0.0, 1.0]);
            save_history=SaveHistory(; states=:n_timesteps),
        )
        simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test environment.history.outputs == fill((;), 3)
        @test environment.history.inputs == fill((;), 3)
        @test environment.history.output_distributions == fill((;), 3)
        @test environment.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        #It stays empty by default
        environment = InstantiatedModel(FixedRewards(), (; rewards=[1.0, 0.0, 1.0]))
        simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test isempty(environment.history.outputs) &&
            isempty(environment.history.inputs) &&
            isempty(environment.history.output_distributions) &&
            isempty(environment.history.states)
    end
end

@testset "reset!" begin
    @testset "An agent goes back to where a new one starts, with a new history" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        environment = InstantiatedModel(RewardsTrue(), (;))
        first_run = simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=5, rng=Xoshiro(2)
        )
        history = agent.history
        @test reset!(agent) === agent
        @test agent.state === (; expected_reward=0.5)
        @test agent.previous_output === (; choice=BeforeFirstTimestep())
        @test agent.parameters === LEARNING
        #The history is a new one, and the one from before is as it was
        @test agent.history !== history
        @test length(history.states) == 6
        @test agent.history.inputs == [(; reward=BeforeFirstTimestep())]
        @test agent.history.outputs == [(; choice=BeforeFirstTimestep())]
        @test agent.history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test agent.history.states == [(; expected_reward=0.5)]
        #With the environment reset too, the same simulation runs again
        reset!(environment)
        @test isequal(
            simulate!(
                agent,
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=5,
                rng=Xoshiro(2),
            ),
            first_run,
        )
    end

    @testset "An agent keeps its parameters and its selection, and gets a new state object" begin
        agent = InstantiatedModel(
            HasDefaults(),
            (; learning_rate=0.3);
            save_history=SaveHistory(; states=SaveEverything()),
        )
        parameters = agent.parameters
        simulate!(agent, (; reward=1.0))
        reset!(agent)
        @test agent.parameters === parameters
        @test agent.save_history === SaveHistory(; states=SaveEverything())
        @test agent.history.inputs == [(;)] && length(agent.history.states) == 1
        #A state that is changed in place is made again
        agent = InstantiatedModel(LearnsInPlace(deepcopy), LEARNING)
        values = agent.state.values
        simulate!(agent, (; reward=1.0))
        reset!(agent)
        @test agent.state.values == [0.5, 0.5]
        @test agent.state.values !== values
        #A history that saves nothing stays empty
        @test isempty(agent.history.inputs) &&
            isempty(agent.history.outputs) &&
            isempty(agent.history.states)
    end

    @testset "An agent that was given a state of its own goes back to the initial state" begin
        agent = InstantiatedModel(
            LearnsRewards(), LEARNING, (; expected_reward=0.25), (; choice=true)
        )
        reset!(agent)
        @test agent.state === (; expected_reward=0.5)
        @test agent.previous_output === (; choice=BeforeFirstTimestep())
    end

    @testset "An environment goes back to where a new one starts, with a new history" begin
        environment = InstantiatedModel(
            FixedRewards(),
            (; rewards=[1.0, 0.0, 1.0]);
            save_history=SaveHistory(SaveEverything()),
        )
        simulate!(
            InstantiatedModel(LearnsRewards(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        history = environment.history
        @test reset!(environment) === environment
        @test environment.state === (; n_timesteps=0)
        @test environment.previous_output === (; reward=BeforeFirstTimestep())
        @test environment.history !== history
        @test length(history.states) == 3
        @test environment.history.outputs == [(; reward=BeforeFirstTimestep())]
        @test environment.history.inputs == [(; choice=BeforeFirstTimestep())]
        @test environment.history.output_distributions == [(; reward=BeforeFirstTimestep())]
        @test environment.history.states == [(; n_timesteps=0)]
        #It can then be simulated from the start again
        (; observations) = simulate!(
            InstantiatedModel(
                LearnsRewards(), LEARNING; save_history=SaveHistory(SaveEverything())
            ),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
        )
        @test observations == [(; reward=1.0), (; reward=0.0), (; reward=1.0)]
    end
end

#Has more parameters than a display lists, and a large array in its state
struct HasManyParameters <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::HasManyParameters, parameters)
    return (; values=collect(1.0:100.0))
end
function CognitiveModeling.attribute_names(model::HasManyParameters)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f), states=:values, inputs=(), outputs=()
    )
end

@testset "An instantiated model counts its timesteps and shows them" begin
    @testset "Both forms of simulate! add to n_timesteps, whatever is saved, and reset! sets it to 0" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        environment = InstantiatedModel(
            RewardsTrue(), (;); save_history=SaveHistory(SaveEverything())
        )
        @test agent.n_timesteps == environment.n_timesteps == 0
        simulate!(agent, (; reward=1.0))
        @test agent.n_timesteps == 1
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        @test agent.n_timesteps == 4
        @test environment.n_timesteps == 3
        @test reset!(agent).n_timesteps == 0
        @test environment.n_timesteps == 3
    end

    @testset "Only the timesteps that ran are counted when a stopping criterion ends them" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        environment = InstantiatedModel(EndsAfterTimesteps(), (; last_timestep=4))
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10)
        @test agent.n_timesteps == environment.n_timesteps == 4
    end

    @testset "The REPL shows its model's kind and type, the timesteps, and the values of the parameters and the state" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        simulate!(agent, (; reward=1.0))
        @test repr("text/plain", agent) == """
        -- Instantiated cognitive model --
        Type: LearnsRewards
        Timesteps simulated: 1
        Parameters: learning_rate = 0.5
        State: expected_reward = 0.75"""
        #An environment without parameters or a state
        @test repr("text/plain", InstantiatedModel(RewardsTrue(), (;))) == """
        -- Instantiated environment model --
        Type: RewardsTrue
        Timesteps simulated: 0
        Parameters: none
        State: none"""
    end

    @testset "The REPL shows only how many values there are when there are more than five, and shortens large arrays" begin
        agent = InstantiatedModel(HasManyParameters(), (; a=1, b=2, c=3, d=4, e=5, f=6))
        shown = repr("text/plain", agent; context=:limit => true)
        @test occursin("\nParameters: 6 (see .parameters)\n", shown)
        @test occursin("State: values = [1.0, 2.0, 3.0", shown) && occursin("…", shown)
        #Without a limit, as when printing to a file, everything is listed
        unlimited = repr("text/plain", agent)
        @test occursin("Parameters: a = 1, b = 2, c = 3, d = 4, e = 5, f = 6", unlimited)
        @test occursin("100.0]", unlimited) && !occursin("…", unlimited)
    end

    @testset "The REPL says so when the stopping criterion is met" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        environment = InstantiatedModel(EndsAfterTimesteps(), (; last_timestep=2))
        @test !occursin("Stopping criterion", repr("text/plain", environment))
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10)
        @test endswith(
            repr("text/plain", environment),
            "\nStopping criterion met: reset! to simulate it again",
        )
        @test !occursin("Stopping criterion", repr("text/plain", reset!(environment)))
    end

    @testset "On one line it shows its model's type and kind, and how many timesteps it has run" begin
        agent = InstantiatedModel(LearnsRewards(), LEARNING)
        @test repr(agent) ==
            "LearnsRewards (instantiated cognitive model, 0 timesteps simulated)"
        simulate!(agent, (; reward=1.0))
        @test repr(agent) ==
            "LearnsRewards (instantiated cognitive model, 1 timestep simulated)"
        @test occursin(
            "RewardsTrue (instantiated environment model, 0 timesteps simulated)",
            repr("text/plain", [InstantiatedModel(RewardsTrue(), (;))]),
        )
    end
end
