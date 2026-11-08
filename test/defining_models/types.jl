using Random: Xoshiro, default_rng

@testset "Types" begin
    observation = (; reward=1.0)
    state = (; expected_reward=0.5)
    parameters = (; learning_rate=0.1)
    previous_action = (; choice=NoAction())

    @testset "ModelInput sets the timestep size to 1 if it isn't specified" begin
        @test ModelInput(observation, state, previous_action, parameters).timestep_size ===
            1
        @test ModelInput(observation, state, previous_action, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "EnvironmentInput sets the timestep size to 1 if it isn't specified" begin
        @test EnvironmentInput(previous_action, state, observation, parameters).timestep_size ===
            1
        @test EnvironmentInput(previous_action, state, observation, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "The inputs use the default generator if it isn't specified" begin
        @test ModelInput(observation, state, previous_action, parameters).rng ===
            default_rng()
        @test EnvironmentInput(previous_action, state, observation, parameters).rng ===
            default_rng()
        rng = Xoshiro(1)
        @test ModelInput(observation, state, previous_action, parameters, 1, rng).rng ===
            rng
        @test EnvironmentInput(previous_action, state, observation, parameters, 1, rng).rng ===
            rng
    end

    @testset "ModelInput says who starts, and unknown if it isn't specified" begin
        @test ModelInput(observation, state, previous_action, parameters).who_starts ===
            UnknownWhoStarts()
        @test ModelInput(
            observation, state, previous_action, parameters, 1, default_rng(), AgentStarts()
        ).who_starts === AgentStarts()
        @test ModelInput(
            observation,
            state,
            previous_action,
            parameters,
            1,
            default_rng(),
            EnvironmentStarts(),
        ).who_starts === EnvironmentStarts()
        #It is in the type, so a model can choose by it at compile time
        @test typeof(
            ModelInput(
                observation,
                state,
                previous_action,
                parameters,
                1,
                default_rng(),
                AgentStarts(),
            ),
        ).parameters[end] === AgentStarts
        @test isbitstype(UnknownWhoStarts)
        @test EnvironmentInput(previous_action, state, observation, parameters).who_starts ===
            UnknownWhoStarts()
        @test EnvironmentInput(
            previous_action,
            state,
            observation,
            parameters,
            1,
            default_rng(),
            EnvironmentStarts(),
        ).who_starts === EnvironmentStarts()
    end

    @testset "ModelOutput only takes named action distributions" begin
        @test ModelOutput((; choice=0.7)).action_distribution === (; choice=0.7)
        @test ModelOutput((;)).action_distribution === (;)
        @test ModelOutput((; choice=NoAction())).action_distribution ===
            (; choice=NoAction())
        @test_throws MethodError ModelOutput(NoAction())
        @test_throws MethodError ModelOutput(0.7)
        @test_throws MethodError ModelOutput((0.7, 0.5))
    end

    @testset "ModelOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput((; choice=0.7)).new_state === (;)
        @test ModelOutput((; choice=0.7), state).new_state === state
    end

    @testset "EnvironmentOutput only takes named observation distributions" begin
        @test EnvironmentOutput((; reward=Deterministic(1.0))).observation_distribution ===
            (; reward=Deterministic(1.0))
        @test EnvironmentOutput((;)).observation_distribution === (;)
        @test EnvironmentOutput((; reward=NoObservation())).observation_distribution ===
            (; reward=NoObservation())
        @test_throws MethodError EnvironmentOutput(NoObservation())
        @test_throws MethodError EnvironmentOutput(Deterministic(1.0))
    end

    @testset "EnvironmentOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test EnvironmentOutput(observation).new_state === (;)
        @test EnvironmentOutput(observation, state).new_state === state
    end

    @testset "The constructors are type stable" begin
        @test @inferred(ModelInput(observation, state, previous_action, parameters)) isa
            ModelInput
        @test @inferred(
            ModelInput(observation, state, previous_action, parameters, 0.5)
        ) isa ModelInput
        @test @inferred(
            EnvironmentInput(previous_action, state, observation, parameters)
        ) isa EnvironmentInput
        @test @inferred(
            EnvironmentInput(previous_action, state, observation, parameters, 0.5)
        ) isa EnvironmentInput
        @test @inferred(ModelOutput((; choice=0.7))) isa ModelOutput
        @test @inferred(ModelOutput((; choice=0.7), state)) isa ModelOutput
        @test @inferred(ModelOutput((;))) isa ModelOutput
        @test @inferred(EnvironmentOutput(observation)) isa EnvironmentOutput
        @test @inferred(EnvironmentOutput(observation, state)) isa EnvironmentOutput
    end

    @testset "The types hold their values directly, so they don't allocate" begin
        @test isbitstype(
            typeof(ModelInput(observation, state, previous_action, parameters))
        )
        @test isbitstype(typeof(ModelOutput((; choice=0.7))))
        @test isbitstype(typeof(EnvironmentOutput((; reward=Deterministic(1.0)))))
    end

    @testset "Sampling from a Deterministic gives its value" begin
        rng = Xoshiro(1)
        @test rand(rng, Deterministic(1.0)) === 1.0
        @test rand(rng, Deterministic(:left)) === :left
        @test @inferred(rand(rng, Deterministic(2))) === 2
        #The generator isn't used, and the default one works too
        @test rng == Xoshiro(1)
        @test rand(Deterministic(true)) === true
        #The value is kept as it is, without a copy
        values = [0.2, 0.7]
        @test Deterministic(values).value === values
        @test rand(rng, Deterministic(values)) === values
    end

    @testset "Cognitive models and environment models share a supertype" begin
        @test AbstractCognitiveModel <: CognitiveModeling.AbstractModel
        @test AbstractEnvironmentModel <: CognitiveModeling.AbstractModel
    end

    @testset "ModelAttributes keeps the names of each kind in a tuple" begin
        attributes = ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=(:expected_reward,),
            observations=(:reward,),
            actions=(:choice,),
        )
        @test attributes.parameters === (:learning_rate, :noise)
        @test attributes.states === (:expected_reward,)
        @test attributes.observations === (:reward,)
        @test attributes.actions === (:choice,)
        @test propertynames(attributes) == (:parameters, :states, :observations, :actions)
        @test_throws FieldError attributes.names
    end

    @testset "ModelAttributes puts a single name or a vector of names in a tuple" begin
        attributes = ModelAttributes(;
            parameters=:learning_rate,
            states=[:expected_reward, :surprise],
            observations=(),
            actions=(),
        )
        @test attributes.parameters === (:learning_rate,)
        @test attributes.states === (:expected_reward, :surprise)
        @test attributes.observations === ()
    end

    @testset "ModelAttributes needs all four kinds, and only takes names" begin
        @test_throws UndefKeywordError ModelAttributes(;
            parameters=(:learning_rate,), states=(), observations=()
        )
        @test_throws MethodError ModelAttributes(;
            parameters=("learning_rate",), states=(), observations=(), actions=()
        )
        @test_throws MethodError ModelAttributes(;
            parameters=(:learning_rate, 1), states=(), observations=(), actions=()
        )
    end

    @testset "ModelAttributes keeps the names in the type, so constructing it from literals is inferred" begin
        from_literals() = ModelAttributes(;
            parameters=(:learning_rate,),
            states=:expected_reward,
            observations=(),
            actions=(:choice,),
        )
        @test Base.return_types(from_literals, Tuple{})[1] ===
            ModelAttributes{(:learning_rate,),(:expected_reward,),(),(:choice,)}
        @test sizeof(from_literals()) == 0
    end

    @testset "InvalidParametersError has a default message, and takes another" begin
        @test InvalidParametersError() isa Exception
        @test sprint(showerror, InvalidParametersError()) ==
            "InvalidParametersError: Model parameters are not valid, possibly dependent on observations, actions and states."
        @test sprint(
            showerror, InvalidParametersError("The learning rate must be between 0 and 1.")
        ) == "InvalidParametersError: The learning rate must be between 0 and 1."
        @test InvalidParametersError("The learning rate must be between 0 and 1.").message ==
            "The learning rate must be between 0 and 1."
        @test_throws InvalidParametersError throw(InvalidParametersError())
    end
end
