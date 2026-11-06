using CommonRLInterface: CommonRLInterface
using Random: AbstractRNG

## Environments and models for the tests ##
#A two-armed bandit where the first arm pays, in episodes of three pulls, which counts how often it is cloned
mutable struct CountingBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
    clones::Int
end
CommonRLInterface.reset!(rl_env::CountingBandit) = (rl_env.pulls = 0; nothing)
CommonRLInterface.actions(rl_env::CountingBandit) = 1:2
CommonRLInterface.observe(rl_env::CountingBandit) = rl_env.pulls
CommonRLInterface.act!(rl_env::CountingBandit, arm) = (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(rl_env::CountingBandit) = rl_env.pulls >= 3
CommonRLInterface.clone(rl_env::CountingBandit) = (rl_env.clones += 1; CountingBandit(rl_env.pulls, rl_env.clones))

#The same bandit without clone, so that it is copied with deepcopy
mutable struct PlainBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(rl_env::PlainBandit) = (rl_env.pulls = 0; nothing)
CommonRLInterface.actions(rl_env::PlainBandit) = 1:2
CommonRLInterface.observe(rl_env::PlainBandit) = rl_env.pulls
CommonRLInterface.act!(rl_env::PlainBandit, arm) = (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(rl_env::PlainBandit) = rl_env.pulls >= 3

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

#Has two actions, of which a map passes one on to the environment, and lists only the reward
struct PullsTwoArms <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsTwoArms, parameters) = (;)
CognitiveModeling.apply_model(model::PullsTwoArms, input::ModelInput) = ModelOutput((; first = Always(1), second = Always(2)))
CognitiveModeling.attribute_names(model::PullsTwoArms) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = (:first, :second))

#Shortens the calls below
function run_bandit(rl_env, model = PullsArm(); action_name = :arm, n_steps = 4, cognitive_parameters = (; arm = 1), save_history = SaveHistory(SaveEverything()), save_environment_history = SaveHistory(SaveEverything()), keywords...)
    return simulate(model, CommonRLEnvironment(rl_env; action_name); who_starts = EnvironmentStarts(), n_steps, cognitive_parameters, environment_parameters = (;), save_history, save_environment_history, keywords...)
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
        rl_env = CountingBandit(2, 0)
        @test CognitiveModeling.initialize_state(CommonRLEnvironment(rl_env; action_name = :arm), (;)) === rl_env
        @test rl_env.pulls == 0
    end

    @testset "The reward and the end of the episode are in the observation" begin
        observations, actions, cognitive_states, _ = run_bandit(CountingBandit(0, 0))
        #The first observation has no reward, since nothing has been done yet
        @test observations[1] === (; observation = 0, reward = NoObservation(), terminated = false)
        @test observations[2] == (; observation = 1, reward = 1.0, terminated = false)
        @test observations[4] == (; observation = 3, reward = 1.0, terminated = true)
        #The model gets the actions it took and the rewards
        @test all(action -> action === (; arm = 1), actions)
        @test [state.last_reward for state in cognitive_states] == [NoObservation(), 1.0, 1.0, 1.0]
        #The second arm doesn't pay
        observations, _, _, _ = run_bandit(CountingBandit(0, 0); cognitive_parameters = (; arm = 2))
        @test observations[2] == (; observation = 1, reward = 0.0, terminated = false)
    end

    @testset "A simulation stops when the episode ends, and a new simulation starts a new one" begin
        rl_env = CountingBandit(0, 0)
        environment = CommonRLEnvironment(rl_env; action_name = :arm)
        #The stopping criterion is whether the episode has ended
        @test CognitiveModeling.stopping_criterion(environment, rl_env, (;)) === false
        observations, actions, cognitive_states, environment_states, _ = run_bandit(rl_env; n_steps = 10)
        @test length(observations) == length(actions) == length(cognitive_states) == length(environment_states) == 4
        @test observations[end].terminated
        @test rl_env.pulls == 3
        @test CognitiveModeling.stopping_criterion(environment, rl_env, (;)) === true
        #A new simulation resets the environment, and runs the episode again
        observations, _, _, _ = run_bandit(rl_env)
        @test [observation.observation for observation in observations] == [0, 1, 2, 3]
        #An episode that has ended isn't started again by an action given to it by hand
        @test_throws "The episode of the environment of type CountingBandit from CommonRLInterface.jl has ended, so it can't take the action arm." CognitiveModeling.apply_model(environment, EnvironmentInput((; arm = 1), rl_env, observations[end], (;)))
        @test rl_env.pulls == 3
    end

    @testset "Before the first action, and when the model takes none, the environment stays where it is" begin
        rl_env = CountingBandit(0, 0)
        environment = CommonRLEnvironment(rl_env; action_name = :arm)
        for no_action in (BeforeFirstTimestep(), NoAction())
            output = CognitiveModeling.apply_model(environment, EnvironmentInput((; arm = no_action), rl_env, CognitiveModeling.initial_observation(environment), (;)))
            @test output.observation === (; observation = 0, reward = NoObservation(), terminated = false)
            @test rl_env.pulls == 0
        end
    end

    @testset "The saved states are copies, with clone when the environment has it and deepcopy otherwise" begin
        rl_env = CountingBandit(0, 0)
        _, _, _, environment_states = run_bandit(rl_env)
        @test all(state -> state isa CountingBandit, environment_states)
        @test allunique(objectid(state) for state in environment_states)
        @test [state.pulls for state in environment_states] == [0, 1, 2, 3]
        @test rl_env.clones == 4
        rl_env = PlainBandit(0)
        _, _, _, environment_states = run_bandit(rl_env)
        @test allunique(objectid(state) for state in environment_states)
        @test [state.pulls for state in environment_states] == [0, 1, 2, 3]
    end

    @testset "Nothing is copied when the environment's states aren't saved" begin
        rl_env = CountingBandit(0, 0)
        _, _, _, environment_states = run_bandit(rl_env; save_environment_history = SaveHistory())
        @test rl_env.clones == 0
        @test environment_states == fill((;), 4)
    end

    @testset "simulate! continues with the same environment object" begin
        rl_env = CountingBandit(0, 0)
        environment = Environment(CommonRLEnvironment(rl_env; action_name = :arm), (;))
        @test environment.previous_observation === (; observation = BeforeFirstTimestep(), reward = BeforeFirstTimestep(), terminated = BeforeFirstTimestep())
        agent = CognitiveAgent(PullsArm(), (; arm = 1); save_history = SaveHistory(SaveEverything()))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test environment.state === rl_env
        @test rl_env.pulls == 1
        observations, _, _, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
        @test [observation.observation for observation in observations] == [2, 3]
        #The episode has ended, so it can't continue until the environment is reset
        @test_throws "The stopping criterion of an environment model of type CommonRLEnvironment{CountingBandit, :arm} is already met before the first timestep" simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 1)
        #After a reset of both, a call that asks for more timesteps than the episode has stops with it
        CognitiveModeling.reset!(environment)
        CognitiveModeling.reset!(agent)
        @test length(simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 10)[1]) == 4
    end

    @testset "reset! of the agent and the environment starts a new episode with the same environment object" begin
        rl_env = CountingBandit(0, 0)
        environment = Environment(CommonRLEnvironment(rl_env; action_name = :arm), (;))
        agent = CognitiveAgent(PullsArm(), (; arm = 1); save_history = SaveHistory(; observations = SaveEverything()))
        simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 4)
        @test rl_env.pulls == 3
        CognitiveModeling.reset!(environment)
        CognitiveModeling.reset!(agent)
        @test environment.state === rl_env
        @test rl_env.pulls == 0
        observations, _, _, _ = simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 4)
        @test [observation.observation for observation in observations] == [0, 1, 2, 3]
        @test observations[1].reward === NoObservation()
    end

    @testset "Maps of names pass on what a model with other names lists" begin
        #PullsTwoArms lists only the reward, and has two actions, of which a map passes one on as the arm
        observations, _, _, _ = run_bandit(CountingBandit(0, 0), PullsTwoArms(); cognitive_parameters = (;), observation_map = NameMap(:reward), action_map = NameMap(:second => :arm))
        @test observations[2] == (; reward = 0.0)
        observations, _, _, _ = run_bandit(CountingBandit(0, 0), PullsTwoArms(); cognitive_parameters = (;), observation_map = NameMap(:reward), action_map = NameMap(:first => :arm))
        @test observations[2] == (; reward = 1.0)
        #Without the maps, the two don't fit
        @test_throws "A cognitive model of type PullsTwoArms lists the observations (:reward,), but an environment model of type CommonRLEnvironment{CountingBandit, :arm} gives (:observation, :reward, :terminated). Without a map, the two must list the same observations." run_bandit(CountingBandit(0, 0), PullsTwoArms(); cognitive_parameters = (;))
        @test_throws "Without a map, the two must list the same actions." run_bandit(CountingBandit(0, 0), PullsTwoArms(); cognitive_parameters = (;), observation_map = NameMap(:reward))
    end
end
