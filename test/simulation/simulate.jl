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
    Vₜ = input.observation isa NoObservation ? V₋₁ : V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
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
    return ModelOutput(NoAction(), (; expected_reward = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)))
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
    Vₜ = input.observation isa NoObservation ? V₋₁ : V₋₁ + 0.5 * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = ChooseWithProbability(0.5)), (; expected_reward = Vₜ))
end

#Has two states, so that one of them can be selected for saving
struct LearnsAndCounts <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsAndCounts, parameters) = (; expected_reward = 0.5, n_timesteps = 0)
function CognitiveModeling.apply_model(model::LearnsAndCounts, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = input.observation isa NoObservation ? V₋₁ : V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
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

#Gives rewards with a fixed probability, whatever the agent does
struct RewardsWithProbability <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsWithProbability, parameters) = (;)
function CognitiveModeling.apply_model(environment::RewardsWithProbability, input::EnvironmentInput)
    return EnvironmentOutput((; reward = float(rand(input.rng) < input.parameters.reward_probability)))
end

#Gives fixed rewards, counting the timesteps in its state
struct FixedRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::FixedRewards, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::FixedRewards, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    return EnvironmentOutput((; reward = input.parameters.rewards[n_timesteps]), (; n_timesteps))
end

#Changes a Vector in its state in place, and copies its state when it's saved
struct CountsInPlace <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::CountsInPlace, parameters) = (; counts = [0])
function CognitiveModeling.apply_model(environment::CountsInPlace, input::EnvironmentInput)
    input.previous_state.counts[1] += 1
    return EnvironmentOutput((; reward = 1.0), input.previous_state)
end
CognitiveModeling.prepare_state_for_saving(environment::CountsInPlace, state) = deepcopy(state)

#Rewards an action of true, and has nothing to say before the first action
struct RewardsTrue <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsTrue, parameters) = (;)
function CognitiveModeling.apply_model(environment::RewardsTrue, input::EnvironmentInput)
    input.action isa NoAction && return EnvironmentOutput(NoObservation())
    return EnvironmentOutput((; reward = float(input.action.choice)))
end

#Gives no observation on every 5th timestep
struct SometimesSilent <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::SometimesSilent, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::SometimesSilent, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    observation = iszero(mod(n_timesteps, 5)) ? NoObservation() : (; reward = 1.0)
    return EnvironmentOutput(observation, (; n_timesteps))
end

#Saves the timestep size it gets in its state
struct SavesTimestepSize <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::SavesTimestepSize, parameters) = (; timestep_size = 0)
CognitiveModeling.apply_model(model::SavesTimestepSize, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.5)), (; timestep_size = input.timestep_size))
CognitiveModeling.attribute_names(model::SavesTimestepSize) = ModelAttributes(; parameters = (), states = :timestep_size, observations = (), actions = :choice)

#Gives the timestep size it gets as its observation
struct ObservesTimestepSize <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesTimestepSize, parameters) = (;)
CognitiveModeling.apply_model(environment::ObservesTimestepSize, input::EnvironmentInput) = EnvironmentOutput((; timestep_size = input.timestep_size))

#Returns a tuple instead of a ModelOutput
struct ReturnsTuple <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ReturnsTuple, parameters) = (;)
CognitiveModeling.apply_model(model::ReturnsTuple, input::ModelInput) = (ChooseWithProbability(0.5), (;))
CognitiveModeling.attribute_names(model::ReturnsTuple) = ModelAttributes(; parameters = (), states = (), observations = (), actions = :choice)

#Returns the observation instead of an EnvironmentOutput
struct ReturnsObservation <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ReturnsObservation, parameters) = (;)
CognitiveModeling.apply_model(environment::ReturnsObservation, input::EnvironmentInput) = (; reward = 1.0)

#Have a method for the initial state, but none for a timestep
struct OnlyInitializesState <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyInitializesState, parameters) = (;)
CognitiveModeling.attribute_names(model::OnlyInitializesState) = ModelAttributes(; parameters = (), states = (), observations = (), actions = ())
struct EnvironmentOnlyInitializesState <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::EnvironmentOnlyInitializesState, parameters) = (;)

#Have a method for a timestep, but none for the initial state, so they must be given a state
struct NeedsGivenState <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::NeedsGivenState, input::ModelInput) = ModelOutput((; choice = ChooseWithProbability(0.5)), input.previous_state)
CognitiveModeling.attribute_names(model::NeedsGivenState) = ModelAttributes(; parameters = (), states = :value, observations = (), actions = :choice)
struct EnvironmentNeedsGivenState <: AbstractEnvironmentModel end
CognitiveModeling.apply_model(environment::EnvironmentNeedsGivenState, input::EnvironmentInput) = EnvironmentOutput((; reward = 1.0), input.previous_state)

const LEARNING = (; learning_rate = 0.5)
const REWARDS = (; reward_probability = 0.8)

#Shortens the calls below
function simulate_for_test(model, environment; who_starts = EnvironmentStarts(), n_steps = 50, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng = Xoshiro(1), keywords...)
    return simulate(model, environment; who_starts, n_steps, cognitive_parameters, environment_parameters, rng, keywords...)
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
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); who_starts = AgentStarts())
        @test [observation.reward for observation in observations] == [float(action.choice) for action in actions]
        #The environment sees the action of the timestep before, and nothing on the first
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); who_starts = EnvironmentStarts())
        @test observations[1] === NoObservation()
        @test [observation.reward for observation in observations[2:end]] == [float(action.choice) for action in actions[1:(end - 1)]]
    end

    @testset "The same seed gives the same simulation, for the model and the environment" begin
        for who_starts in (EnvironmentStarts(), AgentStarts()), model in (LearnsRewards(), LearnsWithNoise())
            first_simulation = simulate_for_test(model, RewardsWithProbability(); who_starts, rng = Xoshiro(3))
            second_simulation = simulate_for_test(model, RewardsWithProbability(); who_starts, rng = Xoshiro(3))
            other_seed = simulate_for_test(model, RewardsWithProbability(); who_starts, rng = Xoshiro(4))
            @test first_simulation == second_simulation
            @test first_simulation != other_seed
        end
    end

    @testset "The return type is known and the loop doesn't allocate" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate(LearnsRewards(), RewardsWithProbability(); who_starts, n_steps, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng)
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
        observations, _, cognitive_states, _ = simulate_for_test(LearnsRewards(), SometimesSilent())
        @test count(observation -> observation isa NoObservation, observations) == 10
        @test expected_rewards(cognitive_states)[4] == expected_rewards(cognitive_states)[5]
        #A first observation that is missing
        observations, _, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue())
        @test observations[1] === NoObservation()
        @test observations[2] isa NamedTuple
        #A state that changes from an integer to a Float64
        _, _, cognitive_states, _ = simulate_for_test(StartsWithInteger(), FixedRewards(); who_starts = AgentStarts(), n_steps = 3, environment_parameters = (; rewards = [1.0, 1.0, 1.0]))
        @test expected_rewards(cognitive_states) == [0, 0.5, 0.75]
    end

    @testset "Models and environments can leave things out" begin
        _, _, cognitive_states, environment_states = simulate_for_test(OnlyChooses(), RewardsWithProbability())
        @test eltype(cognitive_states) === typeof((;))
        @test eltype(environment_states) === typeof((;))
        _, actions, cognitive_states, _ = simulate_for_test(OnlyLearns(), RewardsWithProbability())
        @test eltype(actions) === NoAction
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
        @test simulate!(CognitiveAgent(ChoosesInGroups(), (;)), NoObservation()) === (; choice = true, skipped = NoAction(), movement = (; left = false, right = true))
    end

    @testset "An InvalidParametersError from a model stops the simulation, with its message" begin
        @test_throws InvalidParametersError simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 1.5))
        @test_throws "The learning rate must be at most 1." simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 1.5))
        @test simulate_for_test(RejectsHighLearningRates(), RewardsWithProbability(); cognitive_parameters = (; learning_rate = 0.5)) isa Tuple
    end

    @testset "save_history selects what is saved of each kind" begin
        fixed_rewards = (; rewards = [1.0, 0.0, 1.0])
        run_with(; keywords...) = simulate_for_test(LearnsAndCounts(), FixedRewards(); n_steps = 3, environment_parameters = fixed_rewards, keywords...)
        #Everything is saved by default
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

    @testset "NoObservation and NoAction are saved as they are, whatever is selected" begin
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), save_history = SaveHistory(; observations = :reward, actions = :choice))
        @test observations[1] === NoObservation()
        @test observations[2] == (; reward = float(actions[1].choice))
        for save_history in (SaveHistory(; actions = :choice), SaveHistory(SaveEverything()))
            _, actions, _, _ = simulate_for_test(OnlyLearns(), RewardsWithProbability(); n_steps = 3, save_history)
            @test actions == fill(NoAction(), 3)
        end
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
        _, _, _, environment_states = simulate_for_test(LearnsRewards(), CountsInPlace())
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
        @test_throws "Each model must define CognitiveModeling.apply_model" simulate_for_test(OnlyInitializesState(), RewardsWithProbability())
        @test_throws "Each environment model must define CognitiveModeling.apply_model" simulate_for_test(LearnsRewards(), EnvironmentOnlyInitializesState())
    end

    @testset "A missing method for the initial state gives an informative error, unless a state is given" begin
        @test_throws "Each model must define CognitiveModeling.initialize_state" simulate_for_test(NeedsGivenState(), RewardsWithProbability())
        @test_throws "Each environment model must define CognitiveModeling.initialize_state" simulate_for_test(LearnsRewards(), EnvironmentNeedsGivenState())
        _, _, cognitive_states, environment_states = simulate(NeedsGivenState(), EnvironmentNeedsGivenState(); who_starts = EnvironmentStarts(), n_steps = 2, cognitive_parameters = (;), environment_parameters = (;), previous_state = (; value = 1.0), previous_environment_state = (; value = 2.0))
        @test cognitive_states == [(; value = 1.0), (; value = 1.0)]
        @test environment_states == [(; value = 2.0), (; value = 2.0)]
    end

    @testset "A timestep method which returns the wrong type gives an informative error" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test_throws "model of type ReturnsTuple must return a CognitiveModeling.ModelOutput, but it returned a value of type Tuple" simulate_for_test(ReturnsTuple(), RewardsWithProbability(); who_starts)
            @test_throws "environment model of type ReturnsObservation must return a CognitiveModeling.EnvironmentOutput, but it returned a value of type @NamedTuple" simulate_for_test(LearnsRewards(), ReturnsObservation(); who_starts)
        end
    end

    @testset "The model and the environment both get the size of each timestep" begin
        get_sizes(outputs) = [output.timestep_size for output in outputs]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_with_sizes(; keywords...) = simulate(SavesTimestepSize(), ObservesTimestepSize(); who_starts, n_steps = 3, cognitive_parameters = (;), environment_parameters = (;), rng = Xoshiro(1), keywords...)
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
        run_simulation(timestep_sizes) = simulate(LearnsRewards(), RewardsWithProbability(); who_starts = EnvironmentStarts(), n_steps = 3, timestep_sizes, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng)
        for timestep_sizes in (1, 0.5, [0.5, 2.0, 0.25], 1:3)
            @test @inferred(run_simulation(timestep_sizes)) isa Tuple
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(timestep_sizes)
        end
    end
end

@testset "simulate!" begin
    @testset "Two simulations in a row give the same as one long one" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate(LearnsRewards(), RewardsTrue(); who_starts, n_steps = 60, cognitive_parameters = LEARNING, environment_parameters = (;), rng = Xoshiro(9))

            agent = CognitiveAgent(LearnsRewards(), LEARNING)
            environment = Environment(RewardsTrue(), (;))
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
            one_simulation = simulate(SavesTimestepSize(), ObservesTimestepSize(); who_starts, n_steps = 5, timestep_sizes = sizes, cognitive_parameters = (;), environment_parameters = (;), rng = Xoshiro(9))

            agent = CognitiveAgent(SavesTimestepSize(), (;))
            environment = Environment(ObservesTimestepSize(), (;))
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps = 2, timestep_sizes = sizes[1:2], rng)
            second_part = simulate!(agent, environment; who_starts, n_steps = 3, timestep_sizes = sizes[3:5], rng)

            for (i, outputs) in enumerate(one_simulation)
                @test isequal(vcat(first_part[i], second_part[i]), outputs)
            end
        end
    end

    @testset "Later simulations don't change the outputs of earlier ones" begin
        agent = CognitiveAgent(LearnsInPlace(deepcopy), LEARNING)
        environment = Environment(RewardsWithProbability(), REWARDS)
        _, _, cognitive_states, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        last_value = cognitive_states[end].values[1]
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3, rng = Xoshiro(1))
        @test cognitive_states[end].values[1] == last_value
    end

    @testset "simulate! continues from the state itself, whatever is saved" begin
        agent = CognitiveAgent(LearnsInPlace(state -> (; first_value = state.values[1])), LEARNING)
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
        @test CognitiveAgent(HasDefaults(), (; learning_rate = 0.3), (; expected_reward = 0.2), NoAction()).parameters == (; learning_rate = 0.3, initial_value = 0.5)
        #A model without defaults keeps the parameters as they are
        @test CognitiveAgent(LearnsRewards(), LEARNING).parameters === LEARNING
    end

    @testset "An agent checks the names of the parameters and the state against attribute_names" begin
        @test_throws "The parameters are named (:learning_rate, :initial_value, :noise), but attribute_names lists the parameters of a model of type HasDefaults as (:learning_rate, :initial_value)" CognitiveAgent(HasDefaults(), (; noise = 0.1))
        @test_throws "The state is named (:value,), but attribute_names lists the states of a model of type HasDefaults as (:expected_reward,)" CognitiveAgent(HasDefaults(), (;), (; value = 0.5), NoAction())
        #The order of the names doesn't matter
        @test CognitiveAgent(HasDefaults(), (; initial_value = 0.5, learning_rate = 0.1)).parameters.learning_rate == 0.1
        #A state that isn't a NamedTuple isn't checked
        @test CognitiveAgent(HasDefaults(), (;), [0.5], NoAction()).state == [0.5]
        #A model without names can't be an agent, though it can be simulated
        @test_throws "Each model must define CognitiveModeling.attribute_names" CognitiveAgent(StartsWithInteger(), (;))
        @test simulate_for_test(StartsWithInteger(), RewardsWithProbability(); cognitive_parameters = (;)) isa Tuple
    end

    @testset "An agent and an environment need a method for the initial state, unless a state is given" begin
        @test_throws "Each model must define CognitiveModeling.initialize_state" CognitiveAgent(NeedsGivenState(), (;))
        @test_throws "Each environment model must define CognitiveModeling.initialize_state" Environment(EnvironmentNeedsGivenState(), (;))
        @test CognitiveAgent(NeedsGivenState(), (;), (; value = 1.0), NoAction()).state == (; value = 1.0)
        @test Environment(EnvironmentNeedsGivenState(), (;), (; value = 2.0), NoObservation()).state == (; value = 2.0)
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
        one_simulation = simulate(LearnsRewards(), FixedRewards(); who_starts = EnvironmentStarts(), n_steps = 4, cognitive_parameters = LEARNING, environment_parameters = (; rewards), rng = Xoshiro(9))

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

    @testset "The observation can be left out" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        simulate!(agent, NoObservation())
        @test agent.state == (; expected_reward = 0.5)
    end

    @testset "The model gets the size of the timestep" begin
        agent = CognitiveAgent(SavesTimestepSize(), (;))
        #The default is the integer 1
        simulate!(agent, NoObservation())
        @test agent.state.timestep_size === 1
        simulate!(agent, NoObservation(); timestep_size = 0.5)
        @test agent.state.timestep_size === 0.5
    end

    @testset "It works without actions, and with several actions" begin
        @test simulate!(CognitiveAgent(OnlyLearns(), LEARNING), (; reward = 1.0)) === NoAction()
        @test simulate!(CognitiveAgent(TwoChoices(), (;)), NoObservation()) === (; first_choice = false, second_choice = true)
    end

    @testset "An error in the timestep method gives the same errors as in simulate" begin
        @test_throws "model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate!(CognitiveAgent(ReturnsTuple(), (;)), NoObservation())
        @test_throws "Each model must define CognitiveModeling.apply_model" simulate!(CognitiveAgent(OnlyInitializesState(), (;)), NoObservation())
    end
end

@testset "The history of an agent" begin
    @testset "It starts where the agent starts" begin
        history = CognitiveAgent(LearnsRewards(), LEARNING).history
        @test history isa CognitiveModeling.AgentHistory
        @test history.observations == [NoObservation()]
        @test history.actions == [NoAction()]
        @test history.states == [(; expected_reward = 0.5)]
        #Also if the state and the previous action are given
        history = CognitiveAgent(LearnsRewards(), LEARNING, (; expected_reward = 0.25), (; choice = true)).history
        @test history.observations == [NoObservation()]
        @test history.actions == [(; choice = true)]
        @test history.states == [(; expected_reward = 0.25)]
    end

    @testset "Everything is saved by default, and a SaveHistory selects what is" begin
        @test CognitiveAgent(LearnsRewards(), LEARNING).save_history === SaveHistory(SaveEverything())
        @test CognitiveAgent(LearnsRewards(), LEARNING, (; expected_reward = 0.25), NoAction()).save_history === SaveHistory(SaveEverything())
        @test Environment(RewardsWithProbability(), REWARDS).save_history === SaveHistory(SaveEverything())
        @test Environment(RewardsWithProbability(), REWARDS, (;), NoObservation()).save_history === SaveHistory(SaveEverything())
        save_history = SaveHistory(; states = :n_timesteps)
        agent = CognitiveAgent(LearnsAndCounts(), LEARNING; save_history)
        @test agent.save_history === save_history
        @test CognitiveAgent(LearnsAndCounts(), LEARNING, (; expected_reward = 0.25, n_timesteps = 0), NoAction(); save_history).save_history === save_history
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

    @testset "An observation that isn't a NamedTuple can't have names selected" begin
        agent = CognitiveAgent(OnlyChooses(), (;); save_history = SaveHistory(; observations = :reward))
        @test_throws "can only select names from a NamedTuple" simulate!(agent, 1.0)
        @test simulate!(CognitiveAgent(OnlyChooses(), (;); save_history = SaveHistory(SaveEverything())), 1.0) isa NamedTuple
    end

    @testset "It adds what simulate! returns" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            agent = CognitiveAgent(LearnsRewards(), LEARNING)
            environment = Environment(RewardsWithProbability(), REWARDS)
            rng = Xoshiro(9)
            first_part = simulate!(agent, environment; who_starts, n_steps = 3, rng)
            second_part = simulate!(agent, environment; who_starts, n_steps = 2, rng)

            @test agent.history.observations == vcat(NoObservation(), first_part[1], second_part[1])
            @test agent.history.actions == vcat(NoAction(), first_part[2], second_part[2])
            @test agent.history.states == vcat((; expected_reward = 0.5), first_part[3], second_part[3])
        end
    end

    @testset "It adds what happens on a single observation" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        first_action = simulate!(agent, (; reward = 1.0))
        second_action = simulate!(agent, NoObservation())

        @test agent.history.observations == [NoObservation(), (; reward = 1.0), NoObservation()]
        @test agent.history.actions == [NoAction(), first_action, second_action]
        @test agent.history.states == [(; expected_reward = 0.5), (; expected_reward = 0.75), (; expected_reward = 0.75)]
    end

    @testset "A single observation and an environment can be used on the same agent" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        environment = Environment(RewardsWithProbability(), REWARDS)
        simulate!(agent, (; reward = 1.0))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        simulate!(agent, (; reward = 0.0))
        @test length(agent.history.observations) == length(agent.history.actions) == length(agent.history.states) == 6
        @test agent.history.states[end] == agent.state
    end

    @testset "It stays empty when nothing is selected" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING; save_history = SaveHistory())
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
            agent = CognitiveAgent(LearnsInPlace(prepare_state), LEARNING)
            environment = Environment(FixedRewards(), (; rewards = [1.0, 0.0]))
            simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
            simulate!(agent, (; reward = 1.0))
            #The initial state is copied by the same method
            @test [state.values[1] for state in agent.history.states] == [0.5, 0.75, 0.375, 0.6875]
            @test allunique(objectid(state.values) for state in agent.history.states)
        end
    end

    @testset "It saves what the method gives, also of the initial state" begin
        agent = CognitiveAgent(LearnsInPlace(state -> (; first_value = state.values[1])), LEARNING)
        simulate!(agent, (; reward = 1.0))
        @test agent.history.states == [(; first_value = 0.5), (; first_value = 0.75)]
    end

    @testset "The initial state is the same object as the agent's, with a method which does nothing" begin
        agent = CognitiveAgent(LearnsInPlace(identity), LEARNING)
        @test agent.history.states[1].values === agent.state.values
    end

    @testset "Later simulations don't change it" begin
        agent = CognitiveAgent(LearnsRewards(), LEARNING)
        environment = Environment(RewardsWithProbability(), REWARDS)
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        states_so_far = copy(agent.history.states)
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test agent.history.states[1:4] == states_so_far
    end
end
