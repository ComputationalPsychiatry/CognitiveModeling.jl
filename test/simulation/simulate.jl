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
    return ModelOutput(ChooseWithProbability(Vₜ), (; expected_reward = Vₜ))
end

#Has no state
struct OnlyChooses <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::OnlyChooses, input::ModelInput) = ModelOutput(ChooseWithProbability(0.7))

#Has no actions
struct OnlyLearns <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyLearns, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::OnlyLearns, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    return ModelOutput(NoAction(), (; expected_reward = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)))
end

#Has two named actions
struct TwoChoices <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::TwoChoices, input::ModelInput) = ModelOutput((; first_choice = ChooseWithProbability(0.0), second_choice = ChooseWithProbability(1.0)))

#Has noise in its state update
struct LearnsWithNoise <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::LearnsWithNoise, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::LearnsWithNoise, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    return ModelOutput(ChooseWithProbability(0.5), (; expected_reward = V₋₁ + rand(input.rng) - 0.5))
end

#Changes a Vector in its state in place, and saves its state with the function it is given
struct LearnsInPlace{Tsave_state} <: AbstractCognitiveModel
    save_state::Tsave_state
end
CognitiveModeling.initialize_state(model::LearnsInPlace, parameters) = (; values = [0.5, 0.5])
function CognitiveModeling.apply_model(model::LearnsInPlace, input::ModelInput)
    values = input.previous_state.values
    values[1] += input.parameters.learning_rate * (input.observation.reward - values[1])
    return ModelOutput(ChooseWithProbability(0.5), input.previous_state; save_state = model.save_state)
end

#Starts with an integer as its state, which becomes a Float64 from the second timestep
struct StartsWithInteger <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::StartsWithInteger, parameters) = (; expected_reward = 0)
function CognitiveModeling.apply_model(model::StartsWithInteger, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = input.observation isa NoObservation ? V₋₁ : V₋₁ + 0.5 * (input.observation.reward - V₋₁)
    return ModelOutput(ChooseWithProbability(0.5), (; expected_reward = Vₜ))
end

#Gives rewards with a fixed probability, whatever the agent does
struct RewardsWithProbability <: AbstractEnvironmentModel end
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
    return EnvironmentOutput((; reward = 1.0), input.previous_state; save_state = deepcopy)
end

#Rewards an action of true, and has nothing to say before the first action
struct RewardsTrue <: AbstractEnvironmentModel end
function CognitiveModeling.apply_model(environment::RewardsTrue, input::EnvironmentInput)
    input.action isa NoAction && return EnvironmentOutput(NoObservation())
    return EnvironmentOutput((; reward = float(input.action.action)))
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
CognitiveModeling.apply_model(model::SavesTimestepSize, input::ModelInput) = ModelOutput(ChooseWithProbability(0.5), (; timestep_size = input.timestep_size))

#Gives the timestep size it gets as its observation
struct ObservesTimestepSize <: AbstractEnvironmentModel end
CognitiveModeling.apply_model(environment::ObservesTimestepSize, input::EnvironmentInput) = EnvironmentOutput((; timestep_size = input.timestep_size))

#Returns a tuple instead of a ModelOutput
struct ReturnsTuple <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::ReturnsTuple, input::ModelInput) = (ChooseWithProbability(0.5), (;))

#Returns the observation instead of an EnvironmentOutput
struct ReturnsObservation <: AbstractEnvironmentModel end
CognitiveModeling.apply_model(environment::ReturnsObservation, input::EnvironmentInput) = (; reward = 1.0)

const LEARNING = (; learning_rate = 0.5)
const REWARDS = (; reward_probability = 0.8)

#Shortens the calls below
function simulate_for_test(model, environment; who_starts = EnvironmentStarts(), n_steps = 50, cognitive_parameters = LEARNING, environment_parameters = REWARDS, rng = Xoshiro(1))
    return simulate(model, environment; who_starts, n_steps, cognitive_parameters, environment_parameters, rng)
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
        @test [observation.reward for observation in observations] == [float(action.action) for action in actions]
        #The environment sees the action of the timestep before, and nothing on the first
        observations, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsTrue(); who_starts = EnvironmentStarts())
        @test observations[1] === NoObservation()
        @test [observation.reward for observation in observations[2:end]] == [float(action.action) for action in actions[1:(end - 1)]]
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

    @testset "A single action is named action" begin
        _, actions, _, _ = simulate_for_test(OnlyChooses(), RewardsWithProbability(); cognitive_parameters = (;))
        @test actions isa Vector{@NamedTuple{action::Bool}}
    end

    @testset "Each of several named actions is sampled" begin
        _, actions, _, _ = simulate_for_test(TwoChoices(), RewardsWithProbability(); cognitive_parameters = (;))
        @test all(action -> action === (; first_choice = false, second_choice = true), actions)
    end

    @testset "A function which copies the state keeps the saved values of a state changed in place" begin
        for save_state in (deepcopy, state -> (; values = copy(state.values)))
            _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(save_state), RewardsWithProbability())
            @test length(unique(state.values[1] for state in cognitive_states)) > 1
            @test allunique(objectid(state.values) for state in cognitive_states)
        end
    end

    @testset "Without a function, nothing is copied" begin
        _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(identity), RewardsWithProbability())
        @test all(state -> state.values === first(cognitive_states).values, cognitive_states)
    end

    @testset "An environment can also give a function which copies its state" begin
        _, _, _, environment_states = simulate_for_test(LearnsRewards(), CountsInPlace())
        @test [state.counts[1] for state in environment_states] == 1:length(environment_states)
    end

    @testset "A function can save something else than the state" begin
        _, _, cognitive_states, _ = simulate_for_test(LearnsInPlace(state -> (; first_value = state.values[1])), RewardsWithProbability())
        @test eltype(cognitive_states) === @NamedTuple{first_value::Float64}
        @test length(unique(cognitive_states)) > 1
    end

    @testset "There must be at least one timestep" begin
        @test length(simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps = 1)[2]) == 1
        @test_throws ArgumentError simulate_for_test(LearnsRewards(), RewardsWithProbability(); n_steps = 0)
    end

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "Each model must define" simulate_for_test(ModelWithoutMethods(), RewardsWithProbability())
        @test_throws "Each environment model must define" simulate_for_test(LearnsRewards(), EnvironmentWithoutMethods())
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
end
