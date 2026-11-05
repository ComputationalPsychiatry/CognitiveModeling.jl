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
CognitiveModeling.initialize_state(model::PullsArm, parameters) = (; last_reward = missing)
function CognitiveModeling.apply_model(model::PullsArm, input::ModelInput)
    return ModelOutput((; arm = Always(input.parameters.arm)), (; last_reward = input.observation.reward))
end
CognitiveModeling.attribute_names(model::PullsArm) = ModelAttributes(; parameters = :arm, states = :last_reward, observations = (:observation, :reward, :terminated), actions = :arm)

#Has two actions, which the environment can't take
struct PullsTwoArms <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsTwoArms, parameters) = (;)
CognitiveModeling.apply_model(model::PullsTwoArms, input::ModelInput) = ModelOutput((; first = Always(1), second = Always(2)))

#Shortens the calls below
function run_bandit(env, model = PullsArm(); n_steps = 7, cognitive_parameters = (; arm = 1), keywords...)
    return simulate(model, CommonRLEnvironment(env); who_starts = EnvironmentStarts(), n_steps, cognitive_parameters, environment_parameters = (;), keywords...)
end

@testset "CommonRLInterface extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingCommonRLInterfaceExt) isa Module

    @testset "The environment object is reset and used as the state" begin
        env = CountingBandit(2, 0)
        @test CognitiveModeling.initialize_state(CommonRLEnvironment(env), (;)) === env
        @test env.pulls == 0
    end

    @testset "The reward and the end of the episode are in the observation" begin
        observations, actions, cognitive_states, _ = run_bandit(CountingBandit(0, 0))
        #The first observation has no reward, since nothing has been done yet
        @test isequal(observations[1], (; observation = 0, reward = missing, terminated = false))
        @test observations[2] == (; observation = 1, reward = 1.0, terminated = false)
        @test observations[4] == (; observation = 3, reward = 1.0, terminated = true)
        #The episode after a terminated one starts from the initial observation, again without a reward
        @test isequal(observations[5], (; observation = 0, reward = missing, terminated = false))
        @test observations[6] == (; observation = 1, reward = 1.0, terminated = false)
        #The model gets the actions it took and the rewards
        @test all(action -> action === (; arm = 1), actions)
        @test isequal([state.last_reward for state in cognitive_states], [missing, 1.0, 1.0, 1.0, missing, 1.0, 1.0])
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
        environment = Environment(CommonRLEnvironment(env), (;))
        agent = CognitiveAgent(PullsArm(), (; arm = 1))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test environment.state === env
        @test env.pulls == 1
        observations, _, _, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)
        @test [observation.observation for observation in observations] == [2, 3, 0]
    end

    @testset "A model with several actions gives an informative error" begin
        @test_throws "must have exactly one action, but it has 2: first, second" run_bandit(CountingBandit(0, 0), PullsTwoArms())
    end
end
