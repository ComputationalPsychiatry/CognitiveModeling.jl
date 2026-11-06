using JET: JET
using Random: AbstractRNG, Xoshiro

## Models and environments for the tests ##
#An action made by hand, so that the tests don't need a distribution library
struct ChooseWithProbability{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::ChooseWithProbability) = rand(rng) < action.probability

#Learns an expected reward, and doesn't learn on timesteps with no observation
struct LearnsRewards <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsRewards, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsRewards, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = input.observation.reward isa NoObservation ? V₋₁ : V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = ChooseWithProbability(Vₜ)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::LearnsRewards) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)

#Has no state
struct OnlyChooses <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyChooses, parameters) = (;)
CognitiveModeling.apply_model(model::OnlyChooses, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.7)))
CognitiveModeling.attribute_names(model::OnlyChooses) = ModelAttributes(; parameters = (), states = (), observations = (), actions = :choice)

#Has no actions
struct OnlyLearns <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyLearns, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::OnlyLearns, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    return ModelOutput((;), (; expected_reward = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)))
end
CognitiveModeling.attribute_names(model::OnlyLearns) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = ())

#Has two named actions
struct TwoChoices <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TwoChoices, parameters) = (;)
CognitiveModeling.apply_model(model::TwoChoices, input::ModelInput) = ModelOutput((; first_choice = ChooseWithProbability(0.0), second_choice = ChooseWithProbability(1.0)))
CognitiveModeling.attribute_names(model::TwoChoices) = ModelAttributes(; parameters = (), states = (), observations = (), actions = (:first_choice, :second_choice))

#Has noise in its state update
struct LearnsWithNoise <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsWithNoise, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsWithNoise, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    return ModelOutput((; choice = ChooseWithProbability(0.5)), (; expected_reward = V₋₁ + rand(input.rng) - 0.5))
end
CognitiveModeling.attribute_names(model::LearnsWithNoise) = ModelAttributes(; parameters = (), states = :expected_reward, observations = (), actions = :choice)

#Changes a Vector in its state in place, and prepares its state for saving with the function it is given
struct LearnsInPlace{Tprepare_state} <: AbstractCognitiveModel
    prepare_state::Tprepare_state
end
CognitiveModeling.prepare_state_for_saving(model::LearnsInPlace, state) = model.prepare_state(state)
CognitiveModeling.initialize_state(model::LearnsInPlace, parameters) = (; values = [0.5, 0.5])
function CognitiveModeling.apply_model(model::LearnsInPlace, input::ModelInput)
    values = input.previous_state.values
    values[1] += input.parameters.learning_rate * (input.observation.reward - values[1])
    return ModelOutput((; choice = ChooseWithProbability(0.5)), input.previous_state)
end
CognitiveModeling.attribute_names(model::LearnsInPlace) = ModelAttributes(; parameters = :learning_rate, states = :values, observations = :reward, actions = :choice)

#Starts with an integer as its state, which becomes a Float64 from the second timestep
struct StartsWithInteger <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::StartsWithInteger, parameters) = (; expected_reward = 0)
function CognitiveModeling.apply_model(model::StartsWithInteger, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = input.observation.reward isa NoObservation ? V₋₁ : V₋₁ + 0.5 * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = ChooseWithProbability(0.5)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::StartsWithInteger) = ModelAttributes(; parameters = (), states = :expected_reward, observations = :reward, actions = :choice)

#Has two states, so that one of them can be selected for saving
struct LearnsAndCounts <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsAndCounts, parameters) = (; expected_reward = 0.5, n_timesteps = 0)
function CognitiveModeling.apply_model(model::LearnsAndCounts, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = input.observation.reward isa NoObservation ? V₋₁ : V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = ChooseWithProbability(Vₜ)), (; expected_reward = Vₜ, n_timesteps = input.previous_state.n_timesteps + 1))
end
CognitiveModeling.attribute_names(model::LearnsAndCounts) = ModelAttributes(; parameters = :learning_rate, states = (:expected_reward, :n_timesteps), observations = :reward, actions = :choice)

#Has an action that is never taken, and a NamedTuple of actions
struct ChoosesInGroups <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ChoosesInGroups, parameters) = (;)
function CognitiveModeling.apply_model(model::ChoosesInGroups, input::ModelInput)
    return ModelOutput((; choice = ChooseWithProbability(1.0), skipped = NoAction(), movement = (; left = ChooseWithProbability(0.0), right = ChooseWithProbability(1.0))))
end
CognitiveModeling.attribute_names(model::ChoosesInGroups) = ModelAttributes(; parameters = (), states = (), observations = (), actions = (:choice, :skipped, :movement))

#Has default parameters and names, and an initial state which depends on a parameter
struct HasDefaults <: AbstractCognitiveModel end
CognitiveModeling.default_parameters(model::HasDefaults) = (; learning_rate = 0.1, initial_value = 0.5)
CognitiveModeling.initialize_state(model::HasDefaults, parameters) = (; expected_reward = parameters.initial_value)
function CognitiveModeling.apply_model(model::HasDefaults, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    return ModelOutput((; choice = ChooseWithProbability(V₋₁)), (; expected_reward = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)))
end
function CognitiveModeling.attribute_names(model::HasDefaults)
    return ModelAttributes(; parameters = (:learning_rate, :initial_value), states = :expected_reward, observations = :reward, actions = :choice)
end

#Throws the error for invalid parameters when the learning rate is above 1
struct RejectsHighLearningRates <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RejectsHighLearningRates, parameters) = (;)
function CognitiveModeling.apply_model(model::RejectsHighLearningRates, input::ModelInput)
    input.parameters.learning_rate > 1 && throw(InvalidParametersError("The learning rate must be at most 1."))
    return ModelOutput((; choice = ChooseWithProbability(0.5)))
end
CognitiveModeling.attribute_names(model::RejectsHighLearningRates) = ModelAttributes(; parameters = :learning_rate, states = (), observations = (), actions = :choice)

#Has no names, so it can't be simulated
struct WithoutNames <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::WithoutNames, parameters) = (;)
CognitiveModeling.apply_model(model::WithoutNames, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.5)))

#Gives rewards with a fixed probability, whatever the agent does
struct RewardsWithProbability <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsWithProbability, parameters) = (;)
function CognitiveModeling.apply_model(environment::RewardsWithProbability, input::EnvironmentInput)
    return EnvironmentOutput((; reward = float(rand(input.rng) < input.parameters.reward_probability)))
end
CognitiveModeling.attribute_names(environment::RewardsWithProbability) = ModelAttributes(; parameters = :reward_probability, states = (), observations = :reward, actions = ())

#Has default parameters, and reads the agent's choice
struct RewardsWithDefaults <: AbstractEnvironmentModel end
CognitiveModeling.default_parameters(environment::RewardsWithDefaults) = (; reward_size = 1.0, bias = 0.0)
CognitiveModeling.initialize_state(environment::RewardsWithDefaults, parameters) = (; last_choice = NoAction())
function CognitiveModeling.apply_model(environment::RewardsWithDefaults, input::EnvironmentInput)
    reward = input.action.choice isa NoAction ? input.parameters.bias : input.parameters.bias + input.parameters.reward_size * input.action.choice
    return EnvironmentOutput((; reward), (; last_choice = input.action.choice))
end
CognitiveModeling.attribute_names(environment::RewardsWithDefaults) = ModelAttributes(; parameters = (:reward_size, :bias), states = :last_choice, observations = :reward, actions = :choice)

#Gives a reward and a cue, which a model doesn't have to list
struct RewardsAndCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsAndCues, parameters) = (;)
CognitiveModeling.apply_model(environment::RewardsAndCues, input::EnvironmentInput) = EnvironmentOutput((; reward = 1.0, cue = 2))
CognitiveModeling.attribute_names(environment::RewardsAndCues) = ModelAttributes(; parameters = (), states = (), observations = (:reward, :cue), actions = ())

#Gives only a cue, which a model which learns from rewards can't use
struct OnlyCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::OnlyCues, parameters) = (;)
CognitiveModeling.apply_model(environment::OnlyCues, input::EnvironmentInput) = EnvironmentOutput((; cue = 2))
CognitiveModeling.attribute_names(environment::OnlyCues) = ModelAttributes(; parameters = (), states = (), observations = :cue, actions = ())

#Has no names, so it can't be simulated
struct EnvironmentWithoutNames <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::EnvironmentWithoutNames, parameters) = (;)
CognitiveModeling.apply_model(environment::EnvironmentWithoutNames, input::EnvironmentInput) = EnvironmentOutput((; reward = 1.0))

#Gives fixed rewards, counting the timesteps in its state
struct FixedRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::FixedRewards, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::FixedRewards, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    return EnvironmentOutput((; reward = input.parameters.rewards[n_timesteps]), (; n_timesteps))
end
CognitiveModeling.attribute_names(environment::FixedRewards) = ModelAttributes(; parameters = :rewards, states = :n_timesteps, observations = :reward, actions = ())

#Changes a Vector in its state in place, and copies its state when it's saved
struct CountsInPlace <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::CountsInPlace, parameters) = (; counts = [0])
function CognitiveModeling.apply_model(environment::CountsInPlace, input::EnvironmentInput)
    input.previous_state.counts[1] += 1
    return EnvironmentOutput((; reward = 1.0), input.previous_state)
end
CognitiveModeling.prepare_state_for_saving(environment::CountsInPlace, state) = deepcopy(state)
CognitiveModeling.attribute_names(environment::CountsInPlace) = ModelAttributes(; parameters = (), states = :counts, observations = :reward, actions = ())

#Rewards an action of true, and has nothing to say before the first action
struct RewardsTrue <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsTrue, parameters) = (;)
function CognitiveModeling.apply_model(environment::RewardsTrue, input::EnvironmentInput)
    input.action.choice isa NoAction && return EnvironmentOutput((; reward = NoObservation()))
    return EnvironmentOutput((; reward = float(input.action.choice)))
end
CognitiveModeling.attribute_names(environment::RewardsTrue) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = :choice)

#Gives no observation on every 5th timestep
struct SometimesSilent <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::SometimesSilent, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::SometimesSilent, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    observation = iszero(mod(n_timesteps, 5)) ? (; reward = NoObservation()) : (; reward = 1.0)
    return EnvironmentOutput(observation, (; n_timesteps))
end
CognitiveModeling.attribute_names(environment::SometimesSilent) = ModelAttributes(; parameters = (), states = :n_timesteps, observations = :reward, actions = ())

#Saves the timestep size it gets in its state
struct SavesTimestepSize <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::SavesTimestepSize, parameters) = (; timestep_size = 0)
CognitiveModeling.apply_model(model::SavesTimestepSize, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.5)), (; timestep_size = input.timestep_size))
CognitiveModeling.attribute_names(model::SavesTimestepSize) = ModelAttributes(; parameters = (), states = :timestep_size, observations = (), actions = :choice)

#Gives the timestep size it gets as its observation
struct ObservesTimestepSize <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesTimestepSize, parameters) = (;)
CognitiveModeling.apply_model(environment::ObservesTimestepSize, input::EnvironmentInput) = EnvironmentOutput((; timestep_size = input.timestep_size))
CognitiveModeling.attribute_names(environment::ObservesTimestepSize) = ModelAttributes(; parameters = (), states = (), observations = :timestep_size, actions = ())

#Returns a tuple instead of a ModelOutput
struct ReturnsTuple <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ReturnsTuple, parameters) = (;)
CognitiveModeling.apply_model(model::ReturnsTuple, input::ModelInput) = (ChooseWithProbability(0.5), (;))
CognitiveModeling.attribute_names(model::ReturnsTuple) = ModelAttributes(; parameters = (), states = (), observations = (), actions = :choice)

#Returns the observation instead of an EnvironmentOutput
struct ReturnsObservation <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ReturnsObservation, parameters) = (;)
CognitiveModeling.apply_model(environment::ReturnsObservation, input::EnvironmentInput) = (; reward = 1.0)
CognitiveModeling.attribute_names(environment::ReturnsObservation) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = ())

#Have a method for the initial state, but none for a timestep
struct OnlyInitializesState <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyInitializesState, parameters) = (;)
CognitiveModeling.attribute_names(model::OnlyInitializesState) = ModelAttributes(; parameters = (), states = (), observations = (), actions = ())
struct EnvironmentOnlyInitializesState <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::EnvironmentOnlyInitializesState, parameters) = (;)
CognitiveModeling.attribute_names(environment::EnvironmentOnlyInitializesState) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = ())

#Have a method for a timestep, but none for the initial state, so they must be given a state
struct NeedsGivenState <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::NeedsGivenState, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.5)), input.previous_state)
CognitiveModeling.attribute_names(model::NeedsGivenState) = ModelAttributes(; parameters = (), states = :value, observations = (), actions = :choice)
struct EnvironmentNeedsGivenState <: AbstractEnvironmentModel end
CognitiveModeling.apply_model(environment::EnvironmentNeedsGivenState, input::EnvironmentInput) = EnvironmentOutput((; reward = 1.0), input.previous_state)
CognitiveModeling.attribute_names(environment::EnvironmentNeedsGivenState) = ModelAttributes(; parameters = (), states = :value, observations = :reward, actions = ())

#Rewards a lever of true with a payoff, and gives a cue: the same as RewardsTrue, under other names than the models use
struct PaysForLever <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::PaysForLever, parameters) = (;)
function CognitiveModeling.apply_model(environment::PaysForLever, input::EnvironmentInput)
    input.action.lever isa NoAction && return EnvironmentOutput((; payoff = NoObservation(), cue = 2))
    return EnvironmentOutput((; payoff = float(input.action.lever), cue = 2))
end
CognitiveModeling.attribute_names(environment::PaysForLever) = ModelAttributes(; parameters = (), states = (), observations = (:payoff, :cue), actions = :lever)

#Gives payoffs with a fixed probability, whatever the lever is, so that its outputs always have the same types
struct PaysWithProbability <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::PaysWithProbability, parameters) = (;)
function CognitiveModeling.apply_model(environment::PaysWithProbability, input::EnvironmentInput)
    return EnvironmentOutput((; payoff = float(rand(input.rng) < input.parameters.reward_probability), cue = 2))
end
CognitiveModeling.attribute_names(environment::PaysWithProbability) = ModelAttributes(; parameters = :reward_probability, states = (), observations = (:payoff, :cue), actions = :lever)

const LEARNING = (; learning_rate = 0.5)
const REWARDS = (; reward_probability = 0.8)

#The maps between the names of LearnsRewards and those of PaysForLever and PaysWithProbability
const PAYOFF_AS_REWARD = NameMap(:payoff => :reward)
const CHOICE_AS_LEVER = NameMap(:choice => :lever)

#As many names as a large model has, for checking that comparing them costs nothing
const MANY_NAMES = Tuple(Symbol(:state_, i) for i in 1:65)
const MANY_NAMES_REVERSED = reverse(MANY_NAMES)

#Shortens the calls below
function simulate_for_test(model, environment; who_starts = EnvironmentStarts(), n_steps = 50, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng = Xoshiro(1), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()), keywords...)
    return simulate(model, environment; who_starts, n_steps, cognitive_parameters, environment_parameters, rng, save_history, save_environment_history, keywords...)
end

@testset "simulate" begin
    expected_rewards(cognitive_states) = [state.expected_reward for state in cognitive_states]

    @testset "Who starts decides what the model learns from" begin
        fixed_rewards = (; rewards = [1.0, 0.0, 1.0])
        #Learns from the reward of the same timestep
        _, _, cognitive_states, environment_states = simulate_for_test(LearnsRewards(), FixedRewards(); who_starts = EnvironmentStarts(), n_steps = 3, environment_parameters = fixed_rewards)
        @test expected_rewards(cognitive_states) == [0.75, 0.375, 0.6875]
        @test environment_states[end] == (; n_timesteps = 3)
        #Learns from the reward of the timestep before, and from nothing on the first
        _, _, cognitive_states, environment_states = simulate_for_test(LearnsRewards(), FixedRewards(); who_starts = AgentStarts(), n_steps = 3, environment_parameters = fixed_rewards)
        @test expected_rewards(cognitive_states) == [0.5, 0.75, 0.375]
        @test environment_states[end] == (; n_timesteps = 3)
    end

    @testset "The environment responds to the agent's actions" begin
        #The environment sees the action of the same timestep
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); who_starts = AgentStarts(), cognitive_parameters = (;), environment_parameters = (;))
        @test [observation.reward for observation in observations] == [float(action.choice) for action in actions]
        #The environment sees the action of the timestep before, and nothing on the first
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); who_starts = EnvironmentStarts(), cognitive_parameters = (;), environment_parameters = (;))
        @test observations[1] === (; reward = NoObservation())
        @test [observation.reward for observation in observations[2:end]] == [float(action.choice) for action in actions[1:(end - 1)]]
    end

    @testset "The same seed gives the same simulation, for the model and the environment" begin
        for who_starts in (EnvironmentStarts(), AgentStarts()), (model, cognitive_parameters) in ((LearnsRewards(), LEARNING), (LearnsWithNoise(), (;)))
            first_simulation = simulate_for_test(model, RewardsWithProbability(); who_starts, cognitive_parameters, rng = Xoshiro(3))
            second_simulation = simulate_for_test(model, RewardsWithProbability(); who_starts, cognitive_parameters, rng = Xoshiro(3))
            other_seed = simulate_for_test(model, RewardsWithProbability(); who_starts, cognitive_parameters, rng = Xoshiro(4))
            @test first_simulation == second_simulation
            @test first_simulation != other_seed
        end
    end

    @testset "The return type is known and the loop doesn't allocate" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate(LearnsRewards(), RewardsWithProbability(); who_starts, n_steps, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng, save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()))
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa Tuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(who_starts, 10)
            run_simulation(who_starts, 10)
            #Doubling the timesteps only adds the storage: 8 bytes each for an observation and a state, and 1 for an action
            extra_bytes = @allocated(run_simulation(who_starts, 20_000)) - @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 18 * 10_000
        end
    end

    @testset "Outputs whose types change are all kept" begin
        #An observation that is sometimes missing
        observations, _, cognitive_states, _ = simulate_for_test(LearnsRewards(), SometimesSilent(); environment_parameters = (;))
        @test count(observation -> observation.reward isa NoObservation, observations) == 10
        @test expected_rewards(cognitive_states)[4] == expected_rewards(cognitive_states)[5]
        #A first observation that is missing
        observations, _, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); cognitive_parameters = (;), environment_parameters = (;))
        @test observations[1] === (; reward = NoObservation())
        @test observations[2].reward isa Float64
        #A state that changes from an integer to a Float64
        _, _, cognitive_states, _ = simulate_for_test(StartsWithInteger(), FixedRewards(); who_starts = AgentStarts(), n_steps = 3, cognitive_parameters = (;), environment_parameters = (; rewards = [1.0, 1.0, 1.0]))
        @test expected_rewards(cognitive_states) == [0, 0.5, 0.75]
    end

    @testset "Models and environments can leave things out" begin
        _, _, cognitive_states, environment_states = simulate_for_test(OnlyChooses(), RewardsWithProbability(); cognitive_parameters = (;))
        @test eltype(cognitive_states) === typeof((;))
        @test eltype(environment_states) === typeof((;))
        _, actions, cognitive_states, _ = simulate_for_test(OnlyLearns(), RewardsWithProbability())
        @test eltype(actions) === typeof((;))
        @test eltype(cognitive_states) <: NamedTuple
    end

    @testset "A single action keeps the name the model gave it" begin
        _, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsWithProbability(); cognitive_parameters = (;))
        @test actions isa Vector{@NamedTuple{choice::Bool}}
    end

    @testset "Each of several named actions is sampled" begin
        _, actions, _, _ = simulate_for_test(TwoChoices(), RewardsWithProbability(); cognitive_parameters = (;))
        @test all(action -> action === (; first_choice = false, second_choice = true), actions)
    end

    @testset "NoAction and a NamedTuple inside the actions are sampled on their own" begin
        _, actions, _, _ = simulate_for_test(ChoosesInGroups(), RewardsWithProbability(); cognitive_parameters = (;))
        @test all(action -> action === (; choice = true, skipped = NoAction(), movement = (; left = false, right = true)), actions)
        @test simulate!(CognitiveAgent(ChoosesInGroups(), (;)), (;)) === (; choice = true, skipped = NoAction(), movement = (; left = false, right = true))
    end

    @testset "An InvalidParametersError from a model stops the simulation, with its message" begin
        @test_throws InvalidParametersError simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 1.5))
        @test_throws "The learning rate must be at most 1." simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 1.5))
        @test simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 0.5)) isa Tuple
    end

    @testset "save_history selects what is saved of each kind" begin
        fixed_rewards = (; rewards = [1.0, 0.0, 1.0])
        run_with(; keywords...) = simulate_for_test(LearnsAndCounts(), FixedRewards(); n_steps = 3, environment_parameters = fixed_rewards, keywords...)
        #The helper saves everything, which selecting everything also does
        everything = run_with()
        @test run_with(; save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything())) == everything
        @test everything[3] == [(; expected_reward = 0.75, n_timesteps = 1), (; expected_reward = 0.375, n_timesteps = 2), (; expected_reward = 0.6875, n_timesteps = 3)]
        #Names keep those entries, and a kind that isn't given keeps nothing
        observations, actions, cognitive_states, environment_states = run_with(; save_history = SaveHistory(; states = :n_timesteps))
        @test observations == fill((;), 3) && eltype(observations) === typeof((;))
        @test actions == fill((;), 3)
        @test cognitive_states == [(; n_timesteps = 1), (; n_timesteps = 2), (; n_timesteps = 3)]
        @test environment_states == everything[4]
        #Several names keep the entries in the order given
        _, _, cognitive_states, _ = run_with(; save_history = SaveHistory(; states = (:n_timesteps, :expected_reward)))
        @test cognitive_states[1] === (; n_timesteps = 1, expected_reward = 0.75)
        #The environment's selection only affects its states
        observations, actions, cognitive_states, environment_states = run_with(; save_environment_history = SaveHistory(; observations = :reward, actions = :choice))
        @test (observations, actions, cognitive_states) == everything[1:3]
        @test environment_states == fill((;), 3)
        #Nothing at all gives empty NamedTuples of everything
        @test all(==(fill((;), 3)), run_with(; save_history = SaveHistory(), save_environment_history = SaveHistory()))
    end

    @testset "Nothing is saved by default" begin
        @test all(==(fill((;), 3)), simulate(LearnsRewards(), RewardsWithProbability(); who_starts = EnvironmentStarts(), n_steps = 3, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng = Xoshiro(1)))
    end

    @testset "An entry that is NoObservation is saved like any other, and a model without actions saves empty NamedTuples" begin
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), save_history = SaveHistory(; observations = :reward, actions = :choice))
        @test observations[1] === (; reward = NoObservation())
        @test observations[2] == (; reward = float(actions[1].choice))
        for save_history in (SaveHistory(; actions = ()), SaveHistory(SaveEverything()))
            _, actions, _, _ = simulate_for_test(OnlyLearns(), RewardsWithProbability(); n_steps = 3, save_history)
            @test actions == fill((;), 3)
        end
    end

    @testset "The action before the first timestep has every action as NoAction" begin
        @test CognitiveModeling.initial_action(LearnsRewards()) === (; choice = NoAction())
        @test CognitiveModeling.initial_action(OnlyLearns()) === (;)
        #The names are in the type, so it is inferred
        @test @inferred(CognitiveModeling.initial_action(TwoChoices())) === (; first_choice = NoAction(), second_choice = NoAction())
        @test CognitiveAgent(TwoChoices(), (;)).previous_action === (; first_choice = NoAction(), second_choice = NoAction())
        #The environment receives it on the first timestep when it starts
        observations, _, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); n_steps = 2, cognitive_parameters = (;), environment_parameters = (;))
        @test observations[1] === (; reward = NoObservation())
        #An environment has one too, with the names of the actions it lists
        @test CognitiveModeling.initial_action(RewardsTrue()) === (; choice = NoAction())
        @test CognitiveModeling.initial_action(FixedRewards()) === (;)
    end

    @testset "The observation before the first timestep has every observation as NoObservation" begin
        @test CognitiveModeling.initial_observation(RewardsWithProbability()) === (; reward = NoObservation())
        @test CognitiveModeling.initial_observation(ObservesTimestepSize()) === (; timestep_size = NoObservation())
        #The names are in the type, so it is inferred
        @test @inferred(CognitiveModeling.initial_observation(RewardsAndCues())) === (; reward = NoObservation(), cue = NoObservation())
        #A model has one too, with the names of the observations it lists
        @test CognitiveModeling.initial_observation(LearnsRewards()) === (; reward = NoObservation())
        @test CognitiveModeling.initial_observation(OnlyChooses()) === (;)
        #The model receives it on the first timestep when it starts, with the environment's names
        _, _, cognitive_states, _ = simulate_for_test(LearnsRewards(), RewardsAndCues(); who_starts = AgentStarts(), n_steps = 2, environment_parameters = (;))
        @test expected_rewards(cognitive_states) == [0.5, 0.75]
        @test Environment(RewardsAndCues(), (;)).previous_observation === (; reward = NoObservation(), cue = NoObservation())
    end

    @testset "A state isn't prepared for saving when none of it is saved" begin
        _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(state -> error("prepared anyway")), RewardsWithProbability(); save_history = SaveHistory(; observations = SaveEverything()))
        @test cognitive_states == fill((;), 50)
    end

    @testset "Selecting a name that isn't there, or from something that isn't a NamedTuple, gives an error" begin
        @test_throws FieldError simulate_for_test(LearnsAndCounts(), RewardsWithProbability(); save_history = SaveHistory(; states = :surprise))
        @test_throws "can only select names from a NamedTuple" simulate_for_test(LearnsInPlace(state -> state.values), RewardsWithProbability(); save_history = SaveHistory(; states = :values))
    end

    @testset "A selection keeps the return type known and the loop free of allocations" begin
        rng = Xoshiro(1)
        run_simulation(save_history, n_steps) = simulate(LearnsAndCounts(), RewardsWithProbability(); who_starts = EnvironmentStarts(), n_steps, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng, save_history, save_environment_history = SaveHistory())
        for save_history in (SaveHistory(; states = :n_timesteps), SaveHistory(; observations = :reward, states = (:expected_reward, :n_timesteps)), SaveHistory())
            @test @inferred(run_simulation(save_history, 10)) isa Tuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(save_history, 10)
        end
        #Saving only the integer state adds 8 bytes for each timestep
        run_simulation(SaveHistory(; states = :n_timesteps), 10)
        extra_bytes = @allocated(run_simulation(SaveHistory(; states = :n_timesteps), 20_000)) - @allocated(run_simulation(SaveHistory(; states = :n_timesteps), 10_000))
        @test extra_bytes <= 9 * 10_000
    end

    @testset "A method which copies the state keeps the saved values of a state changed in place" begin
        for prepare_state in (deepcopy, state -> (; values = copy(state.values)))
            _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(prepare_state), RewardsWithProbability())
            @test length(unique(state.values[1] for state in cognitive_states)) > 1
            @test allunique(objectid(state.values) for state in cognitive_states)
        end
    end

    @testset "A method which does nothing copies nothing" begin
        _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(identity), RewardsWithProbability())
        @test all(state -> state.values === first(cognitive_states).values, cognitive_states)
    end

    @testset "An environment can also have a method which copies its state" begin
        _, _, _, environment_states = simulate_for_test(LearnsRewards(), CountsInPlace(); environment_parameters = (;))
        @test [state.counts[1] for state in environment_states] == 1:length(environment_states)
    end

    @testset "A method can save something else than the state" begin
        _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(state -> (; first_value = state.values[1])), RewardsWithProbability())
        @test eltype(cognitive_states) === @NamedTuple{first_value::Float64}
        @test length(unique(cognitive_states)) > 1
    end

    @testset "There must be at least one timestep" begin
        @test length(simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps = 1)[2]) == 1
        @test_throws ArgumentError simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps = 0)
    end

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "Each model must define CognitiveModeling.apply_model" simulate_for_test(OnlyInitializesState(), RewardsWithProbability(); cognitive_parameters = (;))
        @test_throws "Each environment model must define CognitiveModeling.apply_model" simulate_for_test(LearnsRewards(), EnvironmentOnlyInitializesState(); environment_parameters = (;))
    end

    @testset "A missing method for the initial state gives an informative error, unless a state is given" begin
        @test_throws "Each model must define CognitiveModeling.initialize_state" simulate_for_test(NeedsGivenState(), RewardsWithProbability(); cognitive_parameters = (;))
        @test_throws "Each environment model must define CognitiveModeling.initialize_state" simulate_for_test(LearnsRewards(), EnvironmentNeedsGivenState(); environment_parameters = (;))
        _, _, cognitive_states, environment_states = simulate(NeedsGivenState(), EnvironmentNeedsGivenState(); who_starts = EnvironmentStarts(), n_steps = 2, cognitive_parameters = (;), environment_parameters = (;), previous_state = (; value = 1.0), previous_environment_state = (; value = 2.0), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()))
        @test cognitive_states == [(; value = 1.0), (; value = 1.0)]
        @test environment_states == [(; value = 2.0), (; value = 2.0)]
    end

    @testset "A timestep method which returns the wrong type gives an informative error" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test_throws "model of type ReturnsTuple must return a CognitiveModeling.ModelOutput, but it returned a value of type Tuple" simulate_for_test(ReturnsTuple(), RewardsWithProbability(); who_starts, cognitive_parameters = (;))
            @test_throws "environment model of type ReturnsObservation must return a CognitiveModeling.EnvironmentOutput, but it returned a value of type @NamedTuple" simulate_for_test(LearnsRewards(), ReturnsObservation(); who_starts, environment_parameters = (;))
        end
    end

    @testset "The model and the environment both get the size of each timestep" begin
        get_sizes(outputs) = [output.timestep_size for output in outputs]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_with_sizes(; keywords...) = simulate(SavesTimestepSize(), ObservesTimestepSize(); who_starts, n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), rng = Xoshiro(1), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()), keywords...)
            #The default is the integer 1
            observations, _, cognitive_states, _ = run_with_sizes()
            @test get_sizes(observations) == [1, 1, 1] && eltype(get_sizes(observations)) === Int
            @test get_sizes(cognitive_states) == [1, 1, 1]
            #A single number is the size of every timestep
            observations, _, cognitive_states, _ = run_with_sizes(timestep_sizes = 0.5)
            @test get_sizes(observations) == [0.5, 0.5, 0.5]
            @test get_sizes(cognitive_states) == [0.5, 0.5, 0.5]
            #A vector has one size for each timestep
            observations, _, cognitive_states, _ = run_with_sizes(timestep_sizes = [0.5, 2.0, 0.25])
            @test get_sizes(observations) == [0.5, 2.0, 0.25]
            @test get_sizes(cognitive_states) == [0.5, 2.0, 0.25]
            #Any vector can be used, like a range
            observations, _, _, _ = run_with_sizes(timestep_sizes = 1:3)
            @test get_sizes(observations) == [1, 2, 3]
            #The type of a timestep size isn't restricted
            observations, _, _, _ = run_with_sizes(timestep_sizes = (; fast = 0.1, slow = 10.0))
            @test get_sizes(observations) == fill((; fast = 0.1, slow = 10.0), 3)
            #A vector with the wrong length isn't accepted
            @test_throws "it has 2 sizes and n_steps is 3" run_with_sizes(timestep_sizes = [0.5, 2.0])
        end
    end

    @testset "Timestep sizes keep the return type known" begin
        rng = Xoshiro(1)
        run_simulation(timestep_sizes) = simulate(LearnsRewards(), RewardsWithProbability(); who_starts = EnvironmentStarts(), n_steps = 3, timestep_sizes, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng, save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()))
        for timestep_sizes in (1, 0.5, [0.5, 2.0, 0.25], 1:3)
            @test @inferred(run_simulation(timestep_sizes)) isa Tuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(timestep_sizes)
        end
    end
end

@testset "simulate!" begin
    @testset "Two simulations in a row give the same as one long one" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate(LearnsRewards(), RewardsTrue(); who_starts, n_steps = 60, cognitive_parameters = LEARNING, environment_parameters = (;), rng = Xoshiro(9), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()))

            agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
            environment = Environment(RewardsTrue(), (;); save_history = SaveHistory(SaveEverything()))
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps = 25, rng)
            second_part = simulate!(agent, environment; who_starts, n_steps = 35, rng)

            for (i, outputs) in enumerate(one_simulation)
                @test isequal(vcat(first_part[i], second_part[i]), outputs)
            end
            @test agent.state == one_simulation[3][end]
            @test agent.previous_action == one_simulation[2][end]
            @test environment.state == one_simulation[4][end]
            @test environment.previous_observation == one_simulation[1][end]
        end
    end

    @testset "Timestep sizes in two simulations in a row give the same as in one long one" begin
        sizes = [0.5, 2.0, 0.25, 1.5, 3.0]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate(SavesTimestepSize(), ObservesTimestepSize(); who_starts, n_steps = 5, timestep_sizes = sizes, cognitive_parameters = (;), environment_parameters = (;), rng = Xoshiro(9), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()))

            agent = CognitiveAgent(SavesTimestepSize(), (;); save_history = SaveHistory(SaveEverything()))
            environment = Environment(ObservesTimestepSize(), (;); save_history = SaveHistory(SaveEverything()))
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps = 2, timestep_sizes = sizes[1:2], rng)
            second_part = simulate!(agent, environment; who_starts, n_steps = 3, timestep_sizes = sizes[3:5], rng)

            for (i, outputs) in enumerate(one_simulation)
                @test isequal(vcat(first_part[i], second_part[i]), outputs)
            end
        end
    end

    @testset "Later simulations don't change the outputs of earlier ones" begin
        agent = CognitiveAgent(LearnsInPlace(deepcopy), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(RewardsWithProbability(), REWARDS)
        _, _, cognitive_states, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        last_value = cognitive_states[end].values[1]
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        @test cognitive_states[end].values[1] == last_value
    end

    @testset "simulate! continues from the state itself, whatever is saved" begin
        agent = CognitiveAgent(LearnsInPlace(state -> (; first_value = state.values[1])), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(RewardsWithProbability(), REWARDS)
        _, _, first_states, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        @test agent.state.values[1] == first_states[end].first_value
        _, _, second_states, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        @test agent.state.values[1] == second_states[end].first_value
    end

    @testset "An agent fills in the default parameters, and starts from them" begin
        agent = CognitiveAgent(HasDefaults(), (;))
        @test agent.parameters == (; learning_rate = 0.1, initial_value = 0.5)
        @test agent.state == (; expected_reward = 0.5)
        agent = CognitiveAgent(HasDefaults(), (; initial_value = 0.9))
        @test agent.parameters == (; learning_rate = 0.1, initial_value = 0.9)
        @test agent.state == (; expected_reward = 0.9)
        @test CognitiveAgent(HasDefaults(), (; learning_rate = 0.3), (; expected_reward = 0.2), (; choice = NoAction())).parameters == (; learning_rate = 0.3, initial_value = 0.5)
        #A model without defaults keeps the parameters as they are
        @test CognitiveAgent(LearnsRewards(), LEARNING).parameters === LEARNING
    end

    @testset "An agent checks the names of the parameters and the state against attribute_names" begin
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but attribute_names lists the parameters of a model of type HasDefaults as (:learning_rate, :initial_value)" CognitiveAgent(HasDefaults(), (; noise = 0.1))
        @test_throws "The state is named (:value,), but attribute_names lists the states of a model of type HasDefaults as (:expected_reward,)" CognitiveAgent(HasDefaults(), (;), (; value = 0.5), (; choice = NoAction()))
        @test_throws "must be a NamedTuple with the actions that attribute_names lists" CognitiveAgent(HasDefaults(), (;), (; expected_reward = 0.5), NoAction())
        @test_throws "must be a NamedTuple with the actions that attribute_names lists" CognitiveAgent(HasDefaults(), (;), (; expected_reward = 0.5), (; bet = true))
        #The order of the names doesn't matter
        @test CognitiveAgent(HasDefaults(), (; initial_value = 0.5, learning_rate = 0.1)).parameters.learning_rate == 0.1
        #A state that isn't a NamedTuple isn't checked
        @test CognitiveAgent(HasDefaults(), (;), [0.5], (; choice = NoAction())).state == [0.5]
        #A model without names can't be simulated or be an agent
        @test_throws "Each model must define CognitiveModeling.attribute_names" CognitiveAgent(WithoutNames(), (;))
        @test_throws "Each model must define CognitiveModeling.attribute_names" simulate_for_test(WithoutNames(), RewardsWithProbability(); cognitive_parameters = (;))
    end

    @testset "An agent and an environment need a method for the initial state, unless a state is given" begin
        @test_throws "Each model must define CognitiveModeling.initialize_state" CognitiveAgent(NeedsGivenState(), (;))
        @test_throws "Each environment model must define CognitiveModeling.initialize_state" Environment(EnvironmentNeedsGivenState(), (;))
        @test CognitiveAgent(NeedsGivenState(), (;), (; value = 1.0), (; choice = NoAction())).state == (; value = 1.0)
        @test Environment(EnvironmentNeedsGivenState(), (;), (; value = 2.0), (; reward = NoObservation())).state == (; value = 2.0)
    end

    @testset "An environment fills in the default parameters, and checks the names like an agent" begin
        environment = Environment(RewardsWithDefaults(), (;))
        @test environment.parameters == (; reward_size = 1.0, bias = 0.0)
        @test Environment(RewardsWithDefaults(), (; bias = 0.5)).parameters == (; reward_size = 1.0, bias = 0.5)
        @test Environment(RewardsWithDefaults(), (; bias = 0.5), (; last_choice = true), (; reward = 1.0)).parameters == (; reward_size = 1.0, bias = 0.5)
        @test Environment(RewardsWithProbability(), REWARDS).parameters === REWARDS
        @test_throws "The parameters are named (:reward_size, :bias, :noise), but attribute_names lists the parameters of an environment model of type RewardsWithDefaults as (:reward_size, :bias)" Environment(RewardsWithDefaults(), (; noise = 0.1))
        @test_throws "The state is named (:value,), but attribute_names lists the states of an environment model of type RewardsWithDefaults as (:last_choice,)" Environment(RewardsWithDefaults(), (;), (; value = 0.5), (; reward = NoObservation()))
        @test_throws "must be a NamedTuple with the observations that attribute_names lists" Environment(RewardsWithDefaults(), (;), (; last_choice = NoAction()), NoObservation())
        @test_throws "must be a NamedTuple with the observations that attribute_names lists" Environment(RewardsWithDefaults(), (;), (; last_choice = NoAction()), (; cue = 1))
        #A state that isn't a NamedTuple isn't checked
        @test Environment(RewardsWithDefaults(), (;), [0.5], (; reward = NoObservation())).state == [0.5]
        #An environment model without names can't be simulated or be an Environment
        @test_throws "Each environment model must define CognitiveModeling.attribute_names" Environment(EnvironmentWithoutNames(), (;))
        @test_throws "Each environment model must define CognitiveModeling.attribute_names" simulate_for_test(LearnsRewards(), EnvironmentWithoutNames(); environment_parameters = (;))
    end

    @testset "simulate fills in the default parameters, and checks the names like an agent" begin
        run_with(model, environment; keywords...) = simulate_for_test(model, environment; n_steps = 2, keywords...)
        #The defaults are used, also for the initial state, and the given values replace them
        _, _, cognitive_states, _ = run_with(HasDefaults(), RewardsWithProbability(); cognitive_parameters = (;))
        @test cognitive_states[1].expected_reward != 0.5
        _, _, cognitive_states, _ = run_with(HasDefaults(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 0.0, initial_value = 0.9))
        @test [state.expected_reward for state in cognitive_states] == [0.9, 0.9]
        observations, _, _, environment_states = run_with(OnlyChooses(), RewardsWithDefaults(); cognitive_parameters = (;), environment_parameters = (; bias = 0.5))
        @test all(observation -> observation.reward >= 0.5, observations)
        @test environment_states[end].last_choice isa Bool
        #The names of the parameters, the states, the previous action and the previous observation are checked
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but attribute_names lists the parameters of a model of type HasDefaults" run_with(HasDefaults(), RewardsWithProbability(); cognitive_parameters = (; noise = 0.1))
        @test_throws "The parameters are named (:reward_probability, :noise), but attribute_names lists the parameters of an environment model of type RewardsWithProbability" run_with(LearnsRewards(), RewardsWithProbability(); environment_parameters = (; reward_probability = 0.8, noise = 0.1))
        @test_throws "The state is named (:value,), but attribute_names lists the states of a model of type LearnsRewards" run_with(LearnsRewards(), RewardsWithProbability(); previous_state = (; value = 0.5))
        @test_throws "The state is named (:value,), but attribute_names lists the states of an environment model of type FixedRewards" run_with(LearnsRewards(), FixedRewards(); environment_parameters = (; rewards = [1.0, 0.0]), previous_environment_state = (; value = 0))
        @test_throws "The previous action is (bet = true,), but it must be a NamedTuple with the actions that attribute_names lists for a model of type LearnsRewards" run_with(LearnsRewards(), RewardsWithProbability(); previous_action = (; bet = true))
        @test_throws "The previous observation is (cue = 1,), but it must be a NamedTuple with the observations that attribute_names lists for an environment model of type RewardsWithProbability" run_with(LearnsRewards(), RewardsWithProbability(); previous_observation = (; cue = 1))
        @test_throws "must be a NamedTuple with the observations" run_with(LearnsRewards(), RewardsWithProbability(); previous_observation = NoObservation())
        #The order of the names doesn't matter, and a state that isn't a NamedTuple isn't checked
        @test run_with(HasDefaults(), RewardsWithProbability(); cognitive_parameters = (; initial_value = 0.5, learning_rate = 0.1)) isa Tuple
        @test run_with(NeedsGivenState(), RewardsWithProbability(); cognitive_parameters = (;), previous_state = [1.0]) isa Tuple
    end

    @testset "The checks of the names are decided at compile time, so they cost nothing" begin
        #Each compiles to a single statement which returns nothing, however many names there are
        folds(f, argument_types...) = length(only(code_typed(f, argument_types))[1].code) == 1
        @test folds(CognitiveModeling.check_names_fit, LearnsRewards, RewardsAndCues)
        @test folds(CognitiveModeling.check_match_attribute_names, LearnsAndCounts, typeof(LEARNING), typeof((; expected_reward = 0.5, n_timesteps = 0)), typeof((; choice = NoAction())))
        @test folds(CognitiveModeling.check_match_attribute_names, RewardsAndCues, typeof((;)), typeof((;)), typeof((; reward = NoObservation(), cue = NoObservation())))
        @test folds(CognitiveModeling.check_observation_names, LearnsRewards, typeof((; reward = 1.0, cue = 2)))
        @test folds(() -> CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED))
        @test CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED)
        @test !CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES[1:64])
        @test CognitiveModeling.has_every_name((:reward,), (:cue, :reward)) && !CognitiveModeling.has_every_name((:reward, :cue), (:reward,))
    end

    @testset "The model and the environment must fit together" begin
        #The environment must give every observation the model lists, and can give more
        @test_throws "A model of type LearnsRewards lists the observations (:reward,), but an environment model of type OnlyCues gives (:cue,). The environment must give every observation the model lists." simulate_for_test(LearnsRewards(), OnlyCues(); environment_parameters = (;))
        @test simulate_for_test(LearnsRewards(), RewardsAndCues(); environment_parameters = (;)) isa Tuple
        @test simulate_for_test(OnlyChooses(), OnlyCues(); cognitive_parameters = (;), environment_parameters = (;)) isa Tuple
        #The model must give every action the environment lists, and can give more
        @test_throws "An environment model of type RewardsTrue lists the actions (:choice,), but a model of type OnlyLearns gives (). The model must give every action the environment lists." simulate_for_test(OnlyLearns(), RewardsTrue(); environment_parameters = (;))
        @test simulate_for_test(ChoosesInGroups(), RewardsTrue(); cognitive_parameters = (;), environment_parameters = (;)) isa Tuple
        #simulate! checks it too
        @test_throws "The environment must give every observation the model lists." simulate!(CognitiveAgent(LearnsRewards(), LEARNING), Environment(OnlyCues(), (;)); who_starts = EnvironmentStarts(), n_steps = 2)
        @test_throws "The model must give every action the environment lists." simulate!(CognitiveAgent(OnlyLearns(), LEARNING), Environment(RewardsTrue(), (;)); who_starts = EnvironmentStarts(), n_steps = 2)
        @test simulate!(CognitiveAgent(LearnsRewards(), LEARNING), Environment(RewardsAndCues(), (;)); who_starts = EnvironmentStarts(), n_steps = 2) isa Tuple
    end

    @testset "Maps of names let a model and an environment with other names be used together" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #The same simulation as with an environment which uses the model's names
            same_names = simulate_for_test(LearnsRewards(), RewardsTrue(); who_starts, environment_parameters = (;), rng = Xoshiro(5))
            mapped = simulate_for_test(LearnsRewards(), PaysForLever(); who_starts, environment_parameters = (;), rng = Xoshiro(5), observation_map = PAYOFF_AS_REWARD, action_map = CHOICE_AS_LEVER)
            @test isequal(mapped, same_names)
            #The observations are returned as the model received them, without the cue, and the actions as it gave them
            @test keys(mapped[1][end]) == (:reward,)
            @test keys(mapped[2][end]) == (:choice,)
        end
        #Without the maps, the two don't fit
        @test_throws "The environment must give every observation the model lists. A NameMap as observation_map can rename them." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;))
        @test_throws "The model must give every action the environment lists. A NameMap as action_map can rename them." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = PAYOFF_AS_REWARD)
    end

    @testset "A map selects what the receiver gets" begin
        #Without a map, the model receives everything the environment gives
        observations, _, _, _ = simulate_for_test(LearnsRewards(), RewardsAndCues(); n_steps = 2, environment_parameters = (;))
        @test observations == fill((; reward = 1.0, cue = 2), 2)
        #With one, only what the map takes
        observations, _, _, _ = simulate_for_test(LearnsRewards(), RewardsAndCues(); n_steps = 2, environment_parameters = (;), observation_map = NameMap(:reward))
        @test observations == fill((; reward = 1.0), 2)
        #An environment receives only the actions the map takes, here the second of two
        observations, actions, _, _ = simulate_for_test(TwoChoices(), RewardsTrue(); n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), action_map = NameMap(:second_choice => :choice))
        @test [observation.reward for observation in observations] == [NoObservation(), 1.0, 1.0]
        @test actions == fill((; first_choice = false, second_choice = true), 3)
        observations, _, _, _ = simulate_for_test(TwoChoices(), RewardsTrue(); n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), action_map = NameMap(:first_choice => :choice))
        @test [observation.reward for observation in observations] == [NoObservation(), 0.0, 0.0]
    end

    @testset "A map must take names that are given, and give exactly the names the receiver lists" begin
        @test_throws "The observation_map takes the observations (:prize,), but an environment model of type PaysForLever gives (:payoff, :cue). A map can only take observations the environment gives." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = NameMap(:prize => :reward), action_map = CHOICE_AS_LEVER)
        @test_throws "The observation_map gives the observations (:reward, :cue), but a model of type LearnsRewards lists (:reward,). A map must give exactly the observations the model lists." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = NameMap(:payoff => :reward, :cue), action_map = CHOICE_AS_LEVER)
        @test_throws "The observation_map gives the observations (:cue,), but a model of type LearnsRewards lists (:reward,)." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = NameMap(:cue), action_map = CHOICE_AS_LEVER)
        @test_throws "The action_map takes the actions (:press,), but a model of type LearnsRewards gives (:choice,). A map can only take actions the model gives." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = PAYOFF_AS_REWARD, action_map = NameMap(:press => :lever))
        @test_throws "The action_map gives the actions (:choice,), but an environment model of type PaysForLever lists (:lever,). A map must give exactly the actions the environment lists." simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = PAYOFF_AS_REWARD, action_map = NameMap(:choice))
        #Only a NameMap is taken
        @test_throws TypeError simulate_for_test(LearnsRewards(), PaysForLever(); environment_parameters = (;), observation_map = (:payoff => :reward,))
    end

    @testset "Maps keep the return type known and the loop free of allocations, and their checks cost nothing" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate(LearnsRewards(), PaysWithProbability(); who_starts, n_steps, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng, save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()), observation_map = PAYOFF_AS_REWARD, action_map = CHOICE_AS_LEVER)
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa Tuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(who_starts, 10)
            run_simulation(who_starts, 10)
            #As without maps: 8 bytes each for an observation and a state, and 1 for an action
            extra_bytes = @allocated(run_simulation(who_starts, 20_000)) - @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 18 * 10_000
        end
        folds(f, argument_types...) = length(only(code_typed(f, argument_types))[1].code) == 1
        @test folds(CognitiveModeling.check_names_fit, LearnsRewards, PaysForLever, typeof(PAYOFF_AS_REWARD), typeof(CHOICE_AS_LEVER))
    end

    @testset "With maps, each history has the names of its own side" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(PaysForLever(), (;); save_history = SaveHistory(SaveEverything()))
        observations, actions, _, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1), observation_map = PAYOFF_AS_REWARD, action_map = CHOICE_AS_LEVER)
        #The agent's are the ones returned, under the model's names
        @test agent.history.observations[2:end] == observations
        @test agent.history.actions[2:end] == actions
        @test all(observation -> keys(observation) == (:reward,), agent.history.observations)
        @test all(action -> keys(action) == (:choice,), agent.history.actions)
        #The environment's are the same values, under its own names, with the cue the model didn't receive
        @test all(observation -> keys(observation) == (:payoff, :cue), environment.history.observations)
        @test all(action -> keys(action) == (:lever,), environment.history.actions)
        @test isequal([observation.payoff for observation in environment.history.observations], [observation.reward for observation in agent.history.observations])
        @test [action.lever for action in environment.history.actions[2:end]] == [action.choice for action in actions]
        #Where they are left for the next simulation, each side has its own names too
        @test keys(environment.previous_observation) == (:payoff, :cue)
        @test keys(agent.previous_action) == (:choice,)
        @test simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2, observation_map = PAYOFF_AS_REWARD, action_map = CHOICE_AS_LEVER) isa Tuple
        #simulate! checks the maps too
        @test_throws "A map must give exactly the observations the model lists." simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2, observation_map = NameMap(:cue), action_map = CHOICE_AS_LEVER)
    end

    @testset "An error in a timestep method gives the same errors as in simulate" begin
        agent = CognitiveAgent(ReturnsTuple(), (;))
        environment = Environment(RewardsWithProbability(), REWARDS)
        @test_throws "model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 1)
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        environment = Environment(ReturnsObservation(), (;))
        @test_throws "environment model of type ReturnsObservation must return a CognitiveModeling.EnvironmentOutput" simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 1)
    end
end

@testset "simulate! on a single observation" begin
    @testset "It updates the agent and returns the sampled action" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        action = simulate!(agent, (; reward = 1.0); rng = Xoshiro(1))
        @test action isa @NamedTuple{choice::Bool}
        @test agent.previous_action === action
        @test agent.state == (; expected_reward = 0.75)
    end

    @testset "Several in a row give the same as a simulation with the same observations" begin
        rewards = [1.0, 0.0, 1.0, 1.0]
        one_simulation = simulate(LearnsRewards(), FixedRewards(); who_starts = EnvironmentStarts(), n_steps = 4, cognitive_parameters = LEARNING, environment_parameters = (; rewards), rng = Xoshiro(9), save_history = SaveHistory(SaveEverything()))

        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        rng = Xoshiro(9)
        actions = [simulate!(agent, (; reward); rng) for reward in rewards]

        @test actions == one_simulation[2]
        @test agent.state == one_simulation[3][end]
    end

    @testset "The same seed gives the same action" begin
        action_with_seed(seed) = [simulate!(CognitiveAgent(LearnsRewards(), LEARNING), (; reward = 1.0); rng = Xoshiro(seed)) for _ in 1:20]
        @test action_with_seed(3) == action_with_seed(3)
        @test action_with_seed(3) != action_with_seed(4)
    end

    @testset "An observation can be NoObservation" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        simulate!(agent, (; reward = NoObservation()))
        @test agent.state == (; expected_reward = 0.5)
    end

    @testset "The observation must be a NamedTuple with the observations the model lists, and can have more" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        @test_throws "The observation is 1.0, but it must be a NamedTuple with the observations that attribute_names lists for a model of type LearnsRewards, (:reward,)." simulate!(agent, 1.0)
        @test_throws "must be a NamedTuple with the observations" simulate!(agent, NoObservation())
        @test_throws "must be a NamedTuple with the observations" simulate!(agent, (; cue = 1.0))
        @test simulate!(agent, (; reward = 1.0, cue = 1.0)) isa NamedTuple
        @test agent.state == (; expected_reward = 0.75)
        #A model without observations takes an empty NamedTuple
        @test simulate!(CognitiveAgent(OnlyChooses(), (;)), (;)) isa NamedTuple
        #The check costs nothing when the types are known
        JET.@test_opt target_modules = (CognitiveModeling,) CognitiveModeling.run_cognitive_model(LearnsRewards(), (; reward = 1.0, cue = 1.0), (; expected_reward = 0.5), (; choice = NoAction()), LEARNING, 1, Xoshiro(1), SaveHistory())
    end

    @testset "The model gets the size of the timestep" begin
        agent = CognitiveAgent(SavesTimestepSize(), (;))
        #The default is the integer 1
        simulate!(agent, (;))
        @test agent.state.timestep_size === 1
        simulate!(agent, (;); timestep_size = 0.5)
        @test agent.state.timestep_size === 0.5
    end

    @testset "It works without actions, and with several actions" begin
        @test simulate!(CognitiveAgent(OnlyLearns(), LEARNING), (; reward = 1.0)) === (;)
        @test simulate!(CognitiveAgent(TwoChoices(), (;)), (;)) === (; first_choice = false, second_choice = true)
    end

    @testset "An error in the timestep method gives the same errors as in simulate" begin
        @test_throws "model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate!(CognitiveAgent(ReturnsTuple(), (;)), (;))
        @test_throws "Each model must define CognitiveModeling.apply_model" simulate!(CognitiveAgent(OnlyInitializesState(), (;)), (;))
    end
end

@testset "The history of an agent" begin
    @testset "It starts where the agent starts" begin
        history = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything())).history
        @test history isa CognitiveModeling.AgentHistory
        @test history.observations == [(; reward = NoObservation())]
        @test history.actions == [(; choice = NoAction())]
        @test history.states == [(; expected_reward = 0.5)]
        #Also if the state and the previous action are given
        history = CognitiveAgent(LearnsRewards(), LEARNING, (; expected_reward = 0.25), (; choice = true); save_history = SaveHistory(SaveEverything())).history
        @test history.observations == [(; reward = NoObservation())]
        @test history.actions == [(; choice = true)]
        @test history.states == [(; expected_reward = 0.25)]
    end

    @testset "Nothing is saved by default, and a SaveHistory selects what is" begin
        @test CognitiveAgent(LearnsRewards(), LEARNING).save_history === SaveHistory()
        @test CognitiveAgent(LearnsRewards(), LEARNING, (; expected_reward = 0.25), (; choice = NoAction())).save_history === SaveHistory()
        @test Environment(RewardsWithProbability(), REWARDS).save_history === SaveHistory()
        @test Environment(RewardsWithProbability(), REWARDS, (;), (; reward = NoObservation())).save_history === SaveHistory()
        save_history = SaveHistory(; states = :n_timesteps)
        agent = CognitiveAgent(LearnsAndCounts(), LEARNING; save_history)
        @test agent.save_history === save_history
        @test CognitiveAgent(LearnsAndCounts(), LEARNING, (; expected_reward = 0.25, n_timesteps = 0), (; choice = NoAction()); save_history).save_history === save_history
        #The history starts with what is selected
        @test agent.history.observations == [(;)]
        @test agent.history.actions == [(;)]
        @test agent.history.states == [(; n_timesteps = 0)]
        #It gets what is selected on a single observation and with an environment, which selects its own states
        simulate!(agent, (; reward = 1.0))
        environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0]); save_history = SaveHistory())
        @test environment.save_history === SaveHistory()
        _, _, cognitive_states, environment_states = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test agent.history.observations == fill((;), 4)
        @test agent.history.actions == fill((;), 4)
        @test agent.history.states == [(; n_timesteps = 0), (; n_timesteps = 1), (; n_timesteps = 2), (; n_timesteps = 3)]
        @test cognitive_states == [(; n_timesteps = 2), (; n_timesteps = 3)]
        @test environment_states == [(;), (;)]
        #The agent and the environment are still updated in full
        @test agent.state.n_timesteps == 3 && agent.state.expected_reward != 0.5
        @test environment.state == (; n_timesteps = 2)
    end

    @testset "A history can only select observations and actions that the model lists" begin
        @test_throws "The SaveHistory selects the observations (:reward,), but attribute_names lists the observations of a model of type OnlyChooses as (). A history can only select observations and actions that the model lists." CognitiveAgent(OnlyChooses(), (;); save_history = SaveHistory(; observations = :reward))
        @test_throws "The SaveHistory selects the actions (:bet,), but attribute_names lists the actions of a model of type LearnsRewards as (:choice,)." CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(; actions = :bet))
        @test_throws "The SaveHistory selects the actions (:choice,), but attribute_names lists the actions of an environment model of type FixedRewards as ()." Environment(FixedRewards(), (; rewards = [1.0]); save_history = SaveHistory(; actions = :choice))
        @test_throws "attribute_names lists the observations of an environment model of type FixedRewards as (:reward,)" Environment(FixedRewards(), (; rewards = [1.0]); save_history = SaveHistory(; observations = :cue))
        #The states aren't checked, since prepare_state_for_saving can save anything
        @test CognitiveAgent(LearnsInPlace(state -> (; first_value = state.values[1])), LEARNING; save_history = SaveHistory(; states = :first_value)).history.states == [(; first_value = 0.5)]
        #Everything can always be selected, and an observation with more than the model lists is saved in full
        agent = CognitiveAgent(OnlyChooses(), (;); save_history = SaveHistory(SaveEverything()))
        simulate!(agent, (; cue = 1.0))
        @test agent.history.observations == [(;), (; cue = 1.0)]
        #A selected observation is kept from an observation with more
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(; observations = :reward))
        simulate!(agent, (; reward = 1.0, cue = 1.0))
        @test agent.history.observations == [(; reward = NoObservation()), (; reward = 1.0)]
    end

    @testset "It adds what simulate! returns" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
            environment = Environment(RewardsWithProbability(), REWARDS)
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps = 3, rng)
            second_part = simulate!(agent, environment; who_starts, n_steps = 2, rng)

            @test agent.history.observations == vcat([(; reward = NoObservation())], first_part[1], second_part[1])
            @test agent.history.actions == vcat((; choice = NoAction()), first_part[2], second_part[2])
            @test agent.history.states == vcat((; expected_reward = 0.5), first_part[3], second_part[3])
        end
    end

    @testset "It adds what happens on a single observation" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        first_action = simulate!(agent, (; reward = 1.0))
        second_action = simulate!(agent, (; reward = NoObservation()))

        @test agent.history.observations == [(; reward = NoObservation()), (; reward = 1.0), (; reward = NoObservation())]
        @test agent.history.actions == [(; choice = NoAction()), first_action, second_action]
        @test agent.history.states == [(; expected_reward = 0.5), (; expected_reward = 0.75), (; expected_reward = 0.75)]
    end

    @testset "A single observation and an environment can be used on the same agent" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(RewardsWithProbability(), REWARDS)
        simulate!(agent, (; reward = 1.0))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        simulate!(agent, (; reward = 0.0))
        @test length(agent.history.observations) == length(agent.history.actions) == length(agent.history.states) == 6
        @test agent.history.states[end] == agent.state
    end

    @testset "It stays empty by default" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        environment = Environment(RewardsWithProbability(), REWARDS)
        simulate!(agent, (; reward = 1.0))
        _, _, cognitive_states, environment_states = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test isempty(agent.history.observations)
        @test isempty(agent.history.actions)
        @test isempty(agent.history.states)
        #The agent is still updated, and the environment's states are still returned
        @test agent.state != (; expected_reward = 0.5)
        @test cognitive_states == fill((;), 3)
        @test eltype(environment_states) === typeof((;))
    end

    @testset "It keeps the values of a state changed in place, with a method which copies it" begin
        for prepare_state in (deepcopy, state -> (; values = copy(state.values)))
            agent = CognitiveAgent(LearnsInPlace(prepare_state), LEARNING; save_history = SaveHistory(SaveEverything()))
            environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0]))
            simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
            simulate!(agent, (; reward = 1.0))
            #The initial state is copied by the same method
            @test [state.values[1] for state in agent.history.states] == [0.5, 0.75, 0.375, 0.6875]
            @test allunique(objectid(state.values) for state in agent.history.states)
        end
    end

    @testset "It saves what the method gives, also of the initial state" begin
        agent = CognitiveAgent(LearnsInPlace(state -> (; first_value = state.values[1])), LEARNING; save_history = SaveHistory(SaveEverything()))
        simulate!(agent, (; reward = 1.0))
        @test agent.history.states == [(; first_value = 0.5), (; first_value = 0.75)]
    end

    @testset "The initial state is the same object as the agent's, with a method which does nothing" begin
        agent = CognitiveAgent(LearnsInPlace(identity), LEARNING; save_history = SaveHistory(SaveEverything()))
        @test agent.history.states[1].values === agent.state.values
    end

    @testset "Later simulations don't change it" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(RewardsWithProbability(), REWARDS)
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        states_so_far = copy(agent.history.states)
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test agent.history.states[1:4] == states_so_far
    end

    @testset "The environment has a history too, with the same observations and actions" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0, 1.0]); save_history = SaveHistory(SaveEverything()))
        @test environment.history isa CognitiveModeling.EnvironmentHistory
        @test environment.history.observations == [(; reward = NoObservation())]
        #It starts with the actions it lists, which this environment has none of
        @test environment.history.actions == [(;)]
        @test Environment(RewardsTrue(), (;); save_history = SaveHistory(SaveEverything())).history.actions == [(; choice = NoAction())]
        @test environment.history.states == [(; n_timesteps = 0)]
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        @test environment.history.observations == agent.history.observations
        @test environment.history.actions[2:end] == agent.history.actions[2:end]
        @test environment.history.states == [(; n_timesteps = 0), (; n_timesteps = 1), (; n_timesteps = 2), (; n_timesteps = 3)]
        #Its selection is its own
        environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0, 1.0]); save_history = SaveHistory(; states = :n_timesteps))
        simulate!(CognitiveAgent(LearnsRewards(), LEARNING), environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test environment.history.observations == fill((;), 3)
        @test environment.history.actions == fill((;), 3)
        @test environment.history.states == [(; n_timesteps = 0), (; n_timesteps = 1), (; n_timesteps = 2)]
        #It stays empty by default
        environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0, 1.0]))
        simulate!(CognitiveAgent(LearnsRewards(), LEARNING), environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test isempty(environment.history.observations) && isempty(environment.history.actions) && isempty(environment.history.states)
    end
end

@testset "reset!" begin
    @testset "An agent goes back to where a new one starts, with a new history" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything()))
        environment = Environment(RewardsTrue(), (;))
        first_run = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 5, rng = Xoshiro(2))
        history = agent.history
        @test reset!(agent) === agent
        @test agent.state === (; expected_reward = 0.5)
        @test agent.previous_action === (; choice = NoAction())
        @test agent.parameters === LEARNING
        #The history is a new one, and the one from before is as it was
        @test agent.history !== history
        @test length(history.states) == 6
        @test agent.history.observations == [(; reward = NoObservation())]
        @test agent.history.actions == [(; choice = NoAction())]
        @test agent.history.states == [(; expected_reward = 0.5)]
        #With the environment reset too, the same simulation runs again
        reset!(environment)
        @test isequal(simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 5, rng = Xoshiro(2)), first_run)
    end

    @testset "An agent keeps its parameters and its selection, and gets a new state object" begin
        agent = CognitiveAgent(HasDefaults(), (; learning_rate = 0.3); save_history = SaveHistory(; states = SaveEverything()))
        parameters = agent.parameters
        simulate!(agent, (; reward = 1.0))
        reset!(agent)
        @test agent.parameters === parameters
        @test agent.save_history === SaveHistory(; states = SaveEverything())
        @test agent.history.observations == [(;)] && length(agent.history.states) == 1
        #A state that is changed in place is made again
        agent = CognitiveAgent(LearnsInPlace(deepcopy), LEARNING)
        values = agent.state.values
        simulate!(agent, (; reward = 1.0))
        reset!(agent)
        @test agent.state.values == [0.5, 0.5]
        @test agent.state.values !== values
        #A history that saves nothing stays empty
        @test isempty(agent.history.observations) && isempty(agent.history.actions) && isempty(agent.history.states)
    end

    @testset "An agent that was given a state of its own goes back to the initial state" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING, (; expected_reward = 0.25), (; choice = true))
        reset!(agent)
        @test agent.state === (; expected_reward = 0.5)
        @test agent.previous_action === (; choice = NoAction())
    end

    @testset "An environment goes back to where a new one starts, with a new history" begin
        environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0, 1.0]); save_history = SaveHistory(SaveEverything()))
        simulate!(CognitiveAgent(LearnsRewards(), LEARNING), environment; who_starts = EnvironmentStarts(), n_steps = 2)
        history = environment.history
        @test reset!(environment) === environment
        @test environment.state === (; n_timesteps = 0)
        @test environment.previous_observation === (; reward = NoObservation())
        @test environment.history !== history
        @test length(history.states) == 3
        @test environment.history.observations == [(; reward = NoObservation())]
        @test environment.history.actions == [(;)]
        @test environment.history.states == [(; n_timesteps = 0)]
        #It can then be simulated from the start again
        observations, _, _, _ = simulate!(CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory(SaveEverything())), environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test observations == [(; reward = 1.0), (; reward = 0.0), (; reward = 1.0)]
    end
end
