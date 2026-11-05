using CommonRLInterface: CommonRLInterface
using Random: AbstractRNG

## Environments and models for the tests ##
#A two-armed bandit where the first arm pays, in episodes of three pulls, which counts how often it is cloned
mutable struct CountingBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
    clones::Int
end
CommonRLInterface.reset!(env::CountingBandit) = (env.pulls = 0; nothing)
CommonRLInterface.actions(env::CountingBandit) = 1:2
CommonRLInterface.observe(env::CountingBandit) = env.pulls
CommonRLInterface.act!(env::CountingBandit, arm) = (env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(env::CountingBandit) = env.pulls >= 3
CommonRLInterface.clone(env::CountingBandit) = (env.clones += 1; CountingBandit(env.pulls, env.clones))

#The same bandit without clone, so that it is copied with deepcopy
mutable struct PlainBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(env::PlainBandit) = (env.pulls = 0; nothing)
CommonRLInterface.actions(env::PlainBandit) = 1:2
CommonRLInterface.observe(env::PlainBandit) = env.pulls
CommonRLInterface.act!(env::PlainBandit, arm) = (env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(env::PlainBandit) = env.pulls >= 3

#An action that is always the same
struct Always{T}
    value::T
end
Base.rand(rng::AbstractRNG, action::Always) = action.value

#Pulls the arm in its parameters, and keeps the last reward in its state
struct PullsArm <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsArm, parameters) = (; last_reward = NoObservation())
function CognitiveModeling.apply_model(model::PullsArm, input::ModelInput)
    return ModelOutput((; arm = Always(input.parameters.arm)), (; last_reward = input.observation.reward))
end
CognitiveModeling.attribute_names(model::PullsArm) = ModelAttributes(; parameters = :arm, states = :last_reward, observations = (:observation, :reward, :terminated), actions = :arm)

#Has two actions, of which the environment takes the one named, and lists only the reward
struct PullsTwoArms <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsTwoArms, parameters) = (;)
CognitiveModeling.apply_model(model::PullsTwoArms, input::ModelInput) = ModelOutput((; first = Always(1), second = Always(2)))
CognitiveModeling.attribute_names(model::PullsTwoArms) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = (:first, :second))

#Shortens the calls below
function run_bandit(env, model = PullsArm(); action_name = :arm, n_steps = 7, cognitive_parameters = (; arm = 1), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()), keywords...)
    return simulate(model, CommonRLEnvironment(env; action_name); who_starts = EnvironmentStarts(), n_steps, cognitive_parameters, environment_parameters = (;), save_history, save_environment_history, keywords...)
end

@testset "CommonRLInterface extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingCommonRLInterfaceExt) isa Module

    @testset "The name of the action is required, and kept in the type with the names" begin
        environment = CommonRLEnvironment(PlainBandit(0); action_name = :arm)
        @test environment isa CommonRLEnvironment{PlainBandit, :arm}
        @test CognitiveModeling.attribute_names(environment) === ModelAttributes(; parameters = (), states = (), observations = (:observation, :reward, :terminated), actions = :arm)
        @test_throws UndefKeywordError CommonRLEnvironment(PlainBandit(0))
        @test occursin("Actions: arm", repr("text/plain", environment))
    end

    @testset "The environment object is reset and used as the state" begin
        env = CountingBandit(2, 0)
        @test CognitiveModeling.initialize_state(CommonRLEnvironment(env; action_name = :arm), (;)) === env
        @test env.pulls == 0
    end

    @testset "The reward and the end of the episode are in the observation" begin
        observations, actions, cognitive_states, _ = run_bandit(CountingBandit(0, 0))
        #The first observation has no reward, since nothing has been done yet
        @test observations[1] === (; observation = 0, reward = NoObservation(), terminated = false)
        @test observations[2] == (; observation = 1, reward = 1.0, terminated = false)
        @test observations[4] == (; observation = 3, reward = 1.0, terminated = true)
        #The episode after a terminated one starts from the initial observation, again without a reward
        @test observations[5] === (; observation = 0, reward = NoObservation(), terminated = false)
        @test observations[6] == (; observation = 1, reward = 1.0, terminated = false)
        #The model gets the actions it took and the rewards
        @test all(action -> action === (; arm = 1), actions)
        @test [state.last_reward for state in cognitive_states] == [NoObservation(), 1.0, 1.0, 1.0, NoObservation(), 1.0, 1.0]
        #The second arm doesn't pay
        observations, _, _, _ = run_bandit(CountingBandit(0, 0); cognitive_parameters = (; arm = 2))
        @test observations[2] == (; observation = 1, reward = 0.0, terminated = false)
    end

    @testset "The saved states are copies, with clone when the environment has it and deepcopy otherwise" begin
        env = CountingBandit(0, 0)
        _, _, _, environment_states = run_bandit(env)
        @test all(state -> state isa CountingBandit, environment_states)
        @test allunique(objectid(state) for state in environment_states)
        @test [state.pulls for state in environment_states] == [0, 1, 2, 3, 0, 1, 2]
        @test env.clones == 7
        env = PlainBandit(0)
        _, _, _, environment_states = run_bandit(env)
        @test allunique(objectid(state) for state in environment_states)
        @test [state.pulls for state in environment_states] == [0, 1, 2, 3, 0, 1, 2]
    end

    @testset "Nothing is copied when the environment's states aren't saved" begin
        env = CountingBandit(0, 0)
        _, _, _, environment_states = run_bandit(env; save_environment_history = SaveHistory())
        @test env.clones == 0
        @test environment_states == fill((;), 7)
    end

    @testset "simulate! continues with the same environment object" begin
        env = CountingBandit(0, 0)
        environment = Environment(CommonRLEnvironment(env; action_name = :arm), (;))
        @test environment.previous_observation === (; observation = NoObservation(), reward = NoObservation(), terminated = NoObservation())
        agent = CognitiveAgent(PullsArm(), (; arm = 1); save_history = SaveHistory(SaveEverything()))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test environment.state === env
        @test env.pulls == 1
        observations, _, _, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test [observation.observation for observation in observations] == [2, 3, 0]
    end

    @testset "The action named is the one given to act!, and the model must have it" begin
        observations, _, _, _ = run_bandit(CountingBandit(0, 0), PullsTwoArms(); action_name = :second, cognitive_parameters = (;))
        @test observations[2] == (; observation = 1, reward = 0.0, terminated = false)
        observations, _, _, _ = run_bandit(CountingBandit(0, 0), PullsTwoArms(); action_name = :first, cognitive_parameters = (;))
        @test observations[2] == (; observation = 1, reward = 1.0, terminated = false)
        @test_throws "lists the actions (:lever,), but a model of type PullsTwoArms gives (:first, :second). The model must give every action the environment lists." run_bandit(CountingBandit(0, 0), PullsTwoArms(); action_name = :lever, cognitive_parameters = (;))
    end
end
