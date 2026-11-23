#A model with two observations, of which the cue shifts what it learns from the reward
struct LearnsRewardsWithCues <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsRewardsWithCues, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::LearnsRewardsWithCues, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ +
        context.parameters.learning_rate * context.input.cue * (context.input.reward - V₋₁)
    end
    return ModelOutput((; choice=Deterministic(Vₜ > 0.5)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::LearnsRewardsWithCues)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=:expected_reward,
        inputs=(:reward, :cue),
        outputs=:choice,
    )
end

#A model which only observes rewards, and takes no actions
struct ObservesRewards <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ObservesRewards, parameters) = (;)
function CognitiveModeling.timestep_model(model::ObservesRewards, context::ModelContext)
    return ModelOutput((;))
end
function CognitiveModeling.attribute_names(model::ObservesRewards)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=())
end

#Simulates a model with an environment of fixed observations, saving everything but timestep 0. The environment takes no actions, so the map passes none on
function simulate_fixed_observations(environment; model=Baseline(ByHand()), keywords...)
    return simulate_session(
        model,
        environment;
        who_starts=EnvironmentStarts(),
        n_steps=10,
        cognitive_parameters=BASELINE_PARAMETERS,
        environment_parameters=(;),
        action_map=NameMap(),
        cognitive_save_history=EVERY_TIMESTEP,
        environment_save_history=EVERY_TIMESTEP,
        keywords...,
    )
end

@testset "FixedObservationsEnvironment" begin
    rewards = (; reward=[1.0, 0.0, 1.0])

    @testset "It keeps its observations, with the observations and its count of timesteps as its attributes" begin
        environment = FixedObservationsEnvironment(rewards)
        @test environment isa AbstractEnvironmentModel
        @test environment.observations === rewards
        @test CognitiveModeling.attribute_names(environment) === ModelAttributes(;
            parameters=(), states=:n_observations_given, inputs=(), outputs=:reward
        )
        several = FixedObservationsEnvironment((; reward=[1.0, 0.0], cue=[1, 2]))
        @test CognitiveModeling.attribute_names(several) === ModelAttributes(;
            parameters=(), states=:n_observations_given, inputs=(), outputs=(:reward, :cue)
        )
        @test environment isa FixedObservationsEnvironment{typeof(rewards)}
        @test CognitiveModeling.initialize_state(environment, (;)) ===
            (; n_observations_given=0)
        @test @inferred(CognitiveModeling.attribute_names(several)) isa ModelAttributes
    end

    @testset "It gives the next entry of each observation on every timestep" begin
        environment = FixedObservationsEnvironment(rewards)
        output = CognitiveModeling.timestep_model(
            environment, ModelContext((;), (; n_observations_given=1), (; reward=1.0), (;))
        )
        @test output ===
            ModelOutput((; reward=Deterministic(0.0)), (; n_observations_given=2))
        several = FixedObservationsEnvironment((; reward=[1.0, 0.0], cue=[:left, :right]))
        @test CognitiveModeling.timestep_model(
            several,
            ModelContext(
                (;),
                (; n_observations_given=0),
                (; reward=BeforeFirstTimestep(), cue=BeforeFirstTimestep()),
                (;),
            ),
        ) === ModelOutput(
            (; reward=Deterministic(1.0), cue=Deterministic(:left)),
            (; n_observations_given=1),
        )
    end

    @testset "A simulation goes through the observations and stops after the last, whoever starts" begin
        environment = FixedObservationsEnvironment(rewards)
        simulation = simulate_fixed_observations(environment)
        @test simulation.observations == [(; reward=1.0), (; reward=0.0), (; reward=1.0)]
        @test simulation.states == [
            (; expected_reward=0.75), (; expected_reward=0.375), (; expected_reward=0.6875)
        ]
        @test simulation.environment.states == [
            (; n_observations_given=1),
            (; n_observations_given=2),
            (; n_observations_given=3),
        ]
        @test simulation.environment.observation_distributions == [
            (; reward=Deterministic(1.0)),
            (; reward=Deterministic(0.0)),
            (; reward=Deterministic(1.0)),
        ]
        #When the agent starts, it learns from each reward on the timestep after it
        agent_first = simulate_fixed_observations(environment; who_starts=AgentStarts())
        @test agent_first.states ==
            [(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.375)]
        @test length(agent_first.environment.states) == 3
        #Fewer timesteps than there are observations gives only the first ones
        @test simulate_fixed_observations(environment; n_steps=2).observations ==
            [(; reward=1.0), (; reward=0.0)]
        #The same environment starts from its first entry in every simulation
        @test simulate_fixed_observations(environment).states == simulation.states
    end

    @testset "It takes no actions, which the simulation is told with an empty map" begin
        environment = FixedObservationsEnvironment(rewards)
        simulation = simulate_fixed_observations(environment)
        @test simulation.environment.actions == [(;), (;), (;)]
        #Without the map, the actions of the model aren't those of the environment, which has none
        @test_throws "Without a map, the two must have the same actions in their attributes. A NameMap as action_map can select and rename them." simulate_fixed_observations(
            environment; action_map=nothing
        )
        #A model without actions needs no map
        @test length(
            simulate_session(
                ObservesRewards(),
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=10,
                cognitive_parameters=(;),
                environment_parameters=(;),
                cognitive_save_history=EVERY_TIMESTEP,
            ).observations,
        ) == 3
    end

    @testset "An entry that marks no value is given as it is" begin
        environment = FixedObservationsEnvironment((;
            reward=Union{NoValue,Float64}[1.0, NoValue(), 1.0]
        ))
        simulation = simulate_fixed_observations(environment)
        @test simulation.observations ==
            [(; reward=1.0), (; reward=NoValue()), (; reward=1.0)]
        @test simulation.states == [
            (; expected_reward=0.75), (; expected_reward=0.75), (; expected_reward=0.875)
        ]
        @test CognitiveModeling.fixed_observation(1.0) === Deterministic(1.0)
        @test CognitiveModeling.fixed_observation(NoValue()) === NoValue()
        @test CognitiveModeling.fixed_observation(missing) === Deterministic(missing)
    end

    @testset "Several observations are given together" begin
        environment = FixedObservationsEnvironment((;
            reward=[1.0, 0.0, 1.0], cue=[1.0, 0.0, 0.5]
        ))
        simulation = simulate_fixed_observations(
            environment;
            model=LearnsRewardsWithCues(),
            cognitive_parameters=(; learning_rate=0.5),
        )
        @test simulation.observations ==
            [(; reward=1.0, cue=1.0), (; reward=0.0, cue=0.0), (; reward=1.0, cue=0.5)]
        @test simulation.states == [
            (; expected_reward=0.75), (; expected_reward=0.75), (; expected_reward=0.8125)
        ]
    end

    @testset "A timestep after the last entry is an error, when the stopping criterion isn't checked" begin
        environment = FixedObservationsEnvironment(rewards)
        @test !CognitiveModeling.stopping_criterion(
            environment, (; n_observations_given=2), (;)
        )
        @test CognitiveModeling.stopping_criterion(
            environment, (; n_observations_given=3), (;)
        )
        @test CognitiveModeling.n_fixed_timesteps(environment) == 3
        #A single timestep, which doesn't check the stopping criterion, can ask for one more
        @test_throws ErrorException(
            "The FixedObservationsEnvironment has given the observations of all its 3 timesteps, so it has none for another timestep. A simulation stops by itself after the last one, while a single timestep of simulate_timestep doesn't check that. A new call of simulate_session starts from the first again, and so does reset! of an InstantiatedModel.",
        ) simulate_timestep(
            environment, (;); parameters=(;), previous_state=(; n_observations_given=3)
        )
        @test length(simulate_fixed_observations(environment; n_steps=3).states) == 3
    end

    @testset "An instantiated one is simulated in steps, and starts from its first entry after reset!" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(FixedObservationsEnvironment(rewards), (;))
        simulate_steps!(n_steps) = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps,
            action_map=NameMap(),
        )
        simulate_steps!(2)
        @test environment.state == (; n_observations_given=2)
        @test agent.state == (; expected_reward=0.375)
        simulate_steps!(10)
        @test environment.state == (; n_observations_given=3)
        @test agent.state == (; expected_reward=0.6875)
        #It has ended, so it can't be simulated again before it is reset, and it stays as it is
        @test_throws "is already met before the first timestep, so no timestep can run" simulate_steps!(
            1
        )
        @test environment.state == (; n_observations_given=3) &&
            agent.state == (; expected_reward=0.6875)
        reset!(environment)
        @test environment.state == (; n_observations_given=0)
        simulate_steps!(1)
        @test environment.state == (; n_observations_given=1)
    end

    @testset "The observations and the action names are checked when it is made" begin
        @test_throws ArgumentError(
            "The observations of a FixedObservationsEnvironment must be a NamedTuple with a vector for each observation, such as (; reward = [1.0, 0.0, 1.0]), but they are NamedTuple().",
        ) FixedObservationsEnvironment((;))
        @test_throws ArgumentError(
            "The observations of a FixedObservationsEnvironment must be a NamedTuple with a vector for each observation, such as (; reward = [1.0, 0.0, 1.0]), but they are (reward = 1.0,).",
        ) FixedObservationsEnvironment((; reward=1.0))
        @test_throws DimensionMismatch(
            "The observations of a FixedObservationsEnvironment must have an entry each for every timestep, but (:reward, :cue) have (1, 2) entries.",
        ) FixedObservationsEnvironment((; reward=[1.0], cue=[1, 2]))
        #A name is used for one attribute only, as in every model, so no observation can have the name of the state
        @test_throws "The name n_observations_given is used for both a state and an output" FixedObservationsEnvironment((;
            n_observations_given=[1.0]
        ))
        @test_throws MethodError FixedObservationsEnvironment([1.0, 0.0, 1.0])
    end

    @testset "The REPL shows its observations and its attributes" begin
        @test repr("text/plain", FixedObservationsEnvironment(rewards)) == """
            -- Environment model --
            Type: FixedObservationsEnvironment
            Fields: observations = (reward = [1.0, 0.0, 1.0],)
            Parameters: none
            States: n_observations_given
            Inputs: none
            Outputs: reward"""
    end

    @testset "A timestep is inferred and doesn't allocate, also with an observation that isn't always given" begin
        environment = FixedObservationsEnvironment((;
            reward=[1.0, 0.0, 1.0], cue=[1, 2, 3]
        ))
        context = ModelContext((;), (; n_observations_given=1), (; reward=1.0, cue=1), (;))
        @test @inferred(CognitiveModeling.timestep_model(environment, context)) isa
            ModelOutput
        CognitiveModeling.timestep_model(environment, context)
        @test @allocated(CognitiveModeling.timestep_model(environment, context)) == 0
    end
end
