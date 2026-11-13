## Environments and models for the tests ##
#A two-armed bandit where the first arm pays, in episodes of three pulls, which counts how often it is cloned
mutable struct CountingBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
    clones::Int
end
CommonRLInterface.reset!(rl_env::CountingBandit) = (rl_env.pulls=0; nothing)
CommonRLInterface.actions(rl_env::CountingBandit) = 1:2
CommonRLInterface.observe(rl_env::CountingBandit) = rl_env.pulls
function CommonRLInterface.act!(rl_env::CountingBandit, arm)
    return (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
end
CommonRLInterface.terminated(rl_env::CountingBandit) = rl_env.pulls >= 3
function CommonRLInterface.clone(rl_env::CountingBandit)
    return (rl_env.clones += 1; CountingBandit(rl_env.pulls, rl_env.clones))
end

#The same bandit without clone, so that it is copied with deepcopy
mutable struct PlainBandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(rl_env::PlainBandit) = (rl_env.pulls=0; nothing)
CommonRLInterface.actions(rl_env::PlainBandit) = 1:2
CommonRLInterface.observe(rl_env::PlainBandit) = rl_env.pulls
CommonRLInterface.act!(rl_env::PlainBandit, arm) = (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(rl_env::PlainBandit) = rl_env.pulls >= 3

#A bandit whose observation is its own vector of how often each arm was pulled, which act! changes in place
mutable struct TallyBandit <: CommonRLInterface.AbstractEnv
    tallies::Vector{Int}
end
CommonRLInterface.reset!(rl_env::TallyBandit) = (rl_env.tallies.=0; nothing)
CommonRLInterface.actions(rl_env::TallyBandit) = 1:2
CommonRLInterface.observe(rl_env::TallyBandit) = rl_env.tallies
function CommonRLInterface.act!(rl_env::TallyBandit, arm)
    return (rl_env.tallies[arm] += 1; arm == 1 ? 1.0 : 0.0)
end
CommonRLInterface.terminated(rl_env::TallyBandit) = sum(rl_env.tallies) >= 3

#Pulls the arm in its parameters, and keeps the last reward in its state. The parameter can't be called arm as well, since a
#model uses a name for one attribute
struct PullsArm <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::PullsArm, parameters)
    return (; last_reward=NoValue())
end
function CognitiveModeling.apply_timestep_model(model::PullsArm, context::ModelContext)
    return ModelOutput(
        (; arm=Deterministic(context.parameters.arm_to_pull)),
        (; last_reward=context.input.reward),
    )
end
function CognitiveModeling.attribute_names(model::PullsArm)
    return ModelAttributes(;
        parameters=:arm_to_pull,
        states=:last_reward,
        inputs=(:observation, :reward, :terminated),
        outputs=:arm,
    )
end

#Has two actions, of which a map passes one on to the environment, and only the reward as an input
struct PullsTwoArms <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsTwoArms, parameters) = (;)
function CognitiveModeling.apply_timestep_model(model::PullsTwoArms, context::ModelContext)
    return ModelOutput((; first=Deterministic(1), second=Deterministic(2)))
end
function CognitiveModeling.attribute_names(model::PullsTwoArms)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:reward, outputs=(:first, :second)
    )
end

#Shortens the calls below, saving everything but timestep 0, so that element t is timestep t
function run_bandit(
    rl_env,
    model=PullsArm();
    action_name=:arm,
    n_steps=4,
    cognitive_parameters=(; arm_to_pull=1),
    cognitive_save_history=SaveHistory(
        SaveEverything(); save_initial_timestep_values=false
    ),
    environment_save_history=SaveHistory(
        SaveEverything(); save_initial_timestep_values=false
    ),
    keywords...,
)
    return simulate(
        model,
        CommonRLEnvironment(rl_env; action_name);
        who_starts=EnvironmentStarts(),
        n_steps,
        cognitive_parameters,
        environment_parameters=(;),
        cognitive_save_history,
        environment_save_history,
        keywords...,
    )
end

@testset "CommonRLEnvironment" begin
    @testset "The name of the action is required, and kept in the type with the names" begin
        environment = CommonRLEnvironment(PlainBandit(0); action_name=:arm)
        @test environment isa CommonRLEnvironment{PlainBandit,:arm}
        #The action it receives is its input, and the observations it gives are its outputs
        @test CognitiveModeling.attribute_names(environment) === ModelAttributes(;
            parameters=(),
            states=:rl_env,
            inputs=:arm,
            outputs=(:observation, :reward, :terminated),
        )
        @test_throws UndefKeywordError CommonRLEnvironment(PlainBandit(0))
        #Only an environment of CommonRLInterface.jl can be wrapped
        @test_throws MethodError CommonRLEnvironment(0; action_name=:arm)
        @test occursin("Inputs: arm", repr("text/plain", environment))
        @test occursin(
            "Outputs: observation, reward, terminated", repr("text/plain", environment)
        )
    end

    @testset "The environment object is reset and used as the state, in a NamedTuple" begin
        rl_env = CountingBandit(2, 0)
        @test CognitiveModeling.initialize_state(
            CommonRLEnvironment(rl_env; action_name=:arm), (;)
        ) === (; rl_env)
        @test rl_env.pulls == 0
    end

    @testset "The reward and the end of the episode are in the observation" begin
        simulation = run_bandit(CountingBandit(0, 0))
        observations = simulation.observations
        #The first observation has no reward, since nothing has been done yet
        @test observations[1] === (; observation=0, reward=NoValue(), terminated=false)
        @test observations[2] == (; observation=1, reward=1.0, terminated=false)
        @test observations[4] == (; observation=3, reward=1.0, terminated=true)
        #The model gets the actions it took and the rewards
        @test all(action -> action === (; arm=1), simulation.actions)
        @test [state.last_reward for state in simulation.states] == [NoValue(), 1.0, 1.0, 1.0]
        #The environment's own history has the same observations, as it gave them, and the actions it received
        @test simulation.environment.observations == observations
        @test simulation.environment.actions == simulation.actions
        @test simulation.environment.observation_distributions[2] === (;
            observation=Deterministic(1),
            reward=Deterministic(1.0),
            terminated=Deterministic(false),
        )
        #The second arm doesn't pay
        observations =
            run_bandit(CountingBandit(0, 0); cognitive_parameters=(; arm_to_pull=2)).observations
        @test observations[2] == (; observation=1, reward=0.0, terminated=false)
    end

    @testset "A simulation stops when the episode ends, and a new simulation starts a new one" begin
        rl_env = CountingBandit(0, 0)
        environment = CommonRLEnvironment(rl_env; action_name=:arm)
        #The stopping criterion is whether the episode has ended
        @test CognitiveModeling.stopping_criterion(environment, (; rl_env), (;)) === false
        simulation = run_bandit(rl_env; n_steps=10)
        @test length(simulation.observations) ==
            length(simulation.actions) ==
            length(simulation.states) ==
            length(simulation.environment.states) ==
            4
        @test simulation.observations[end].terminated
        @test rl_env.pulls == 3
        @test CognitiveModeling.stopping_criterion(environment, (; rl_env), (;)) === true
        #A new simulation resets the environment, and runs the episode again
        observations = run_bandit(rl_env).observations
        @test [observation.observation for observation in observations] == [0, 1, 2, 3]
        #An episode that has ended isn't started again by an action given to it by hand
        @test_throws "The episode of the environment of type CountingBandit from CommonRLInterface.jl has ended, so it can't take the action arm. A simulation stops by itself when the episode ends. A new call of simulate starts a new episode, and so does reset! of an InstantiatedModel." CognitiveModeling.apply_timestep_model(
            environment, ModelContext((; arm=1), (; rl_env), observations[end], (;))
        )
        @test rl_env.pulls == 3
    end

    @testset "Before the first action, and when the model takes none, the environment stays where it is" begin
        rl_env = CountingBandit(0, 0)
        environment = CommonRLEnvironment(rl_env; action_name=:arm)
        for no_action in (BeforeFirstTimestep(), NoValue())
            output = CognitiveModeling.apply_timestep_model(
                environment,
                ModelContext(
                    (; arm=no_action),
                    (; rl_env),
                    CognitiveModeling.output_before_first_timestep(environment),
                    (;),
                ),
            )
            @test output.output_distribution === (;
                observation=Deterministic(0),
                reward=NoValue(),
                terminated=Deterministic(false),
            )
            @test rl_env.pulls == 0
        end
        #Without an action after the episode has ended, it still says that the episode has ended
        rl_env = CountingBandit(3, 0)
        output = CognitiveModeling.apply_timestep_model(
            CommonRLEnvironment(rl_env; action_name=:arm),
            ModelContext((; arm=NoValue()), (; rl_env), (;), (;)),
        )
        @test output.output_distribution.terminated === Deterministic(true)
        @test rl_env.pulls == 3
    end

    @testset "What the environment gives after an action is passed on as it is, with nothing sampled" begin
        rl_env = CountingBandit(0, 0)
        environment = CommonRLEnvironment(rl_env; action_name=:arm)
        output = CognitiveModeling.apply_timestep_model(
            environment,
            ModelContext(
                (; arm=1),
                (; rl_env),
                CognitiveModeling.output_before_first_timestep(environment),
                (;),
            ),
        )
        @test output.output_distribution === (;
            observation=Deterministic(1),
            reward=Deterministic(1.0),
            terminated=Deterministic(false),
        )
        @test output.new_state === (; rl_env)
    end

    @testset "An observation that the environment changes in place is copied, so that what was saved stays as it was" begin
        rl_env = TallyBandit([0, 0])
        simulation = run_bandit(rl_env)
        received = [observations.observation for observations in simulation.observations]
        given = [
            observations.observation for observations in simulation.environment.observations
        ]
        @test received == given == [[0, 0], [1, 0], [2, 0], [3, 0]]
        @test all(tallies -> tallies !== rl_env.tallies, received)
    end

    @testset "The saved states are copies, with clone when the environment has it and deepcopy otherwise" begin
        rl_env = CountingBandit(0, 0)
        environment_states = run_bandit(rl_env).environment.states
        @test all(state -> state.rl_env isa CountingBandit, environment_states)
        @test allunique(objectid(state.rl_env) for state in environment_states)
        @test [state.rl_env.pulls for state in environment_states] == [0, 1, 2, 3]
        @test rl_env.clones == 4
        #The state can be selected by its name
        selected = SaveHistory(; states=:rl_env, save_initial_timestep_values=false)
        environment_states =
            run_bandit(CountingBandit(0, 0); environment_save_history=selected).environment.states
        @test [state.rl_env.pulls for state in environment_states] == [0, 1, 2, 3]
        rl_env = PlainBandit(0)
        environment_states = run_bandit(rl_env).environment.states
        @test allunique(objectid(state.rl_env) for state in environment_states)
        @test [state.rl_env.pulls for state in environment_states] == [0, 1, 2, 3]
    end

    @testset "Timestep 0 has the environment where the episode starts, copied like the rest" begin
        rl_env = CountingBandit(0, 0)
        everything = SaveHistory(SaveEverything())
        simulation = run_bandit(
            rl_env; cognitive_save_history=everything, environment_save_history=everything
        )
        @test simulation.observations[1] === (;
            observation=BeforeFirstTimestep(),
            reward=BeforeFirstTimestep(),
            terminated=BeforeFirstTimestep(),
        )
        @test simulation.actions[1] === (; arm=BeforeFirstTimestep())
        @test [state.rl_env.pulls for state in simulation.environment.states] == [0, 0, 1, 2, 3]
        @test rl_env.clones == 5
    end

    @testset "Nothing is copied when the environment's states aren't saved" begin
        rl_env = CountingBandit(0, 0)
        simulation = run_bandit(rl_env; environment_save_history=SaveHistory())
        @test rl_env.clones == 0
        @test simulation.environment.states == fill((;), 4)
    end

    @testset "simulate! continues with the same environment object" begin
        rl_env = CountingBandit(0, 0)
        environment = InstantiatedModel(CommonRLEnvironment(rl_env; action_name=:arm), (;))
        @test environment.previous_output === (;
            observation=BeforeFirstTimestep(),
            reward=BeforeFirstTimestep(),
            terminated=BeforeFirstTimestep(),
        )
        agent = InstantiatedModel(
            PullsArm(), (; arm_to_pull=1); save_history=SaveHistory(SaveEverything())
        )
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2)
        @test environment.state === (; rl_env)
        @test rl_env.pulls == 1
        observations =
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=2).observations
        @test [observation.observation for observation in observations] == [2, 3]
        #The episode has ended, so it can't continue until the environment is reset
        @test_throws "The stopping criterion of an environment model of type CommonRLEnvironment{CountingBandit, :arm} is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=1
        )
        #After a reset of both, a call that asks for more timesteps than the episode has stops with it
        CognitiveModeling.reset!(environment)
        CognitiveModeling.reset!(agent)
        @test length(
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10).observations,
        ) == 4
    end

    @testset "reset! of the agent and the environment starts a new episode with the same environment object" begin
        rl_env = CountingBandit(0, 0)
        environment = InstantiatedModel(CommonRLEnvironment(rl_env; action_name=:arm), (;))
        agent = InstantiatedModel(
            PullsArm(),
            (; arm_to_pull=1);
            save_history=SaveHistory(; inputs=SaveEverything()),
        )
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=4)
        @test rl_env.pulls == 3
        CognitiveModeling.reset!(environment)
        CognitiveModeling.reset!(agent)
        @test environment.state === (; rl_env)
        @test rl_env.pulls == 0
        observations =
            simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=4).observations
        @test [observation.observation for observation in observations] == [0, 1, 2, 3]
        @test observations[1].reward === NoValue()
    end

    @testset "Maps of names pass on what a model with other names has in its attributes" begin
        #PullsTwoArms has only the reward as an input, and two actions, of which a map passes one on as the arm
        simulation = run_bandit(
            CountingBandit(0, 0),
            PullsTwoArms();
            cognitive_parameters=(;),
            observation_map=NameMap(:reward),
            action_map=NameMap(:second => :arm),
        )
        @test simulation.observations[2] == (; reward=0.0)
        #The environment's history has the observations as it gave them, and the action as it received it
        @test simulation.environment.observations[2] ==
            (; observation=1, reward=0.0, terminated=false)
        @test simulation.environment.actions[1] === (; arm=2)
        observations = run_bandit(
            CountingBandit(0, 0),
            PullsTwoArms();
            cognitive_parameters=(;),
            observation_map=NameMap(:reward),
            action_map=NameMap(:first => :arm),
        ).observations
        @test observations[2] == (; reward=1.0)
        #Without the maps, the two don't fit
        @test_throws "The attributes of a cognitive model of type PullsTwoArms have the observations (:reward,) as inputs, but an environment model of type CommonRLEnvironment{CountingBandit, :arm} gives (:observation, :reward, :terminated). Without a map, the two must have the same observations in their attributes." run_bandit(
            CountingBandit(0, 0), PullsTwoArms(); cognitive_parameters=(;)
        )
        @test_throws "A cognitive model of type PullsTwoArms gives the actions (:first, :second), but the attributes of an environment model of type CommonRLEnvironment{CountingBandit, :arm} have (:arm,) as inputs. Without a map, the two must have the same actions in their attributes." run_bandit(
            CountingBandit(0, 0),
            PullsTwoArms();
            cognitive_parameters=(;),
            observation_map=NameMap(:reward),
        )
    end
end
