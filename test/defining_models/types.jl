using Random: Xoshiro, default_rng

@testset "Types" begin
    observation = (; reward=1.0)
    state = (; expected_reward=0.5)
    parameters = (; learning_rate=0.1)
    previous_action = (; choice=NoValue())

    @testset "ModelContext sets the timestep size to 1 if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).timestep_size ===
            1
        @test ModelContext(observation, state, previous_action, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "ModelContext for an environment model has the action as its input and the observation as its previous output" begin
        environment_context = ModelContext(previous_action, state, observation, parameters)
        @test environment_context.input === previous_action
        @test environment_context.previous_output === observation
        @test environment_context.timestep_size === 1
        @test ModelContext(previous_action, state, observation, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "ModelContext uses the default generator if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).rng ===
            default_rng()
        @test ModelContext(previous_action, state, observation, parameters).rng ===
            default_rng()
        rng = Xoshiro(1)
        @test ModelContext(observation, state, previous_action, parameters, 1, rng).rng ===
            rng
        @test ModelContext(previous_action, state, observation, parameters, 1, rng).rng ===
            rng
    end

    @testset "ModelContext says who starts, and unknown if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).who_starts ===
            UnknownWhoStarts()
        @test ModelContext(
            observation, state, previous_action, parameters, 1, default_rng(), AgentStarts()
        ).who_starts === AgentStarts()
        @test ModelContext(
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
            ModelContext(
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
        @test ModelContext(previous_action, state, observation, parameters).who_starts ===
            UnknownWhoStarts()
        @test ModelContext(
            previous_action,
            state,
            observation,
            parameters,
            1,
            default_rng(),
            EnvironmentStarts(),
        ).who_starts === EnvironmentStarts()
    end

    @testset "ModelContext takes its arguments by name too, with the same defaults" begin
        rng = Xoshiro(1)
        @test ModelContext(;
            input=observation,
            previous_state=state,
            previous_output=previous_action,
            parameters,
            timestep_size=0.5,
            rng,
            who_starts=AgentStarts(),
        ) === ModelContext(
            observation, state, previous_action, parameters, 0.5, rng, AgentStarts()
        )
        by_name = ModelContext(;
            input=observation,
            previous_state=state,
            previous_output=previous_action,
            parameters,
        )
        @test by_name.timestep_size === 1
        @test by_name.rng === default_rng()
        @test by_name.who_starts === UnknownWhoStarts()
        @test_throws UndefKeywordError ModelContext(;
            input=observation, previous_state=state, parameters
        )
    end

    @testset "ModelOutput only takes named output distributions" begin
        @test ModelOutput((; choice=0.7)).output_distribution === (; choice=0.7)
        @test ModelOutput((;)).output_distribution === (;)
        @test ModelOutput((; choice=NoValue())).output_distribution === (; choice=NoValue())
        @test_throws MethodError ModelOutput(NoValue())
        @test_throws MethodError ModelOutput(0.7)
        @test_throws MethodError ModelOutput((0.7, 0.5))
    end

    @testset "ModelOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput((; choice=0.7)).new_state === (;)
        @test ModelOutput((; choice=0.7), state).new_state === state
    end

    @testset "ModelOutput only takes a named state, also for a single object" begin
        @test ModelOutput((; choice=0.7), (; values=[0.5, 0.5])).new_state.values ==
            [0.5, 0.5]
        @test_throws MethodError ModelOutput((; choice=0.7), 0.5)
        @test_throws MethodError ModelOutput((; choice=0.7), [0.5, 0.5])
        @test_throws MethodError ModelOutput((; choice=0.7), nothing)
    end

    @testset "ModelOutput for an environment model only takes named observation distributions" begin
        @test ModelOutput((; reward=Deterministic(1.0))).output_distribution ===
            (; reward=Deterministic(1.0))
        @test ModelOutput((; reward=NoValue())).output_distribution === (; reward=NoValue())
        @test_throws MethodError ModelOutput(NoValue())
        @test_throws MethodError ModelOutput(Deterministic(1.0))
    end

    @testset "ModelOutput for an environment model sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput(observation).new_state === (;)
        @test ModelOutput(observation, state).new_state === state
    end

    @testset "The constructors are type stable" begin
        @test @inferred(ModelContext(observation, state, previous_action, parameters)) isa
            ModelContext
        @test @inferred(
            ModelContext(observation, state, previous_action, parameters, 0.5)
        ) isa ModelContext
        @test @inferred(ModelContext(previous_action, state, observation, parameters)) isa
            ModelContext
        @test @inferred(
            ModelContext(previous_action, state, observation, parameters, 0.5)
        ) isa ModelContext
        @test @inferred(
            ModelContext(;
                input=observation,
                previous_state=state,
                previous_output=previous_action,
                parameters,
                timestep_size=0.5,
            )
        ) isa ModelContext
        @test @inferred(ModelOutput((; choice=0.7))) isa ModelOutput
        @test @inferred(ModelOutput((; choice=0.7), state)) isa ModelOutput
        @test @inferred(ModelOutput((;))) isa ModelOutput
        @test @inferred(ModelOutput(observation)) isa ModelOutput
        @test @inferred(ModelOutput(observation, state)) isa ModelOutput
    end

    @testset "The types hold their values directly, so they don't allocate" begin
        @test isbitstype(
            typeof(ModelContext(observation, state, previous_action, parameters))
        )
        @test isbitstype(typeof(ModelOutput((; choice=0.7))))
        @test isbitstype(typeof(ModelOutput((; reward=Deterministic(1.0)))))
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
        @test AbstractCognitiveModel <: CognitiveModeling.AbstractTimestepModel
        @test AbstractEnvironmentModel <: CognitiveModeling.AbstractTimestepModel
    end

    @testset "ModelAttributes keeps the names of each kind in a tuple" begin
        attributes = ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=(:expected_reward,),
            inputs=(:reward,),
            outputs=(:choice,),
        )
        @test attributes.parameters === (:learning_rate, :noise)
        @test attributes.states === (:expected_reward,)
        @test attributes.inputs === (:reward,)
        @test attributes.outputs === (:choice,)
        @test propertynames(attributes) == (:parameters, :states, :inputs, :outputs)
        @test_throws FieldError attributes.names
    end

    @testset "ModelAttributes puts a single name or a vector of names in a tuple" begin
        attributes = ModelAttributes(;
            parameters=:learning_rate,
            states=[:expected_reward, :surprise],
            inputs=(),
            outputs=(),
        )
        @test attributes.parameters === (:learning_rate,)
        @test attributes.states === (:expected_reward, :surprise)
        @test attributes.inputs === ()
    end

    @testset "ModelAttributes needs all four kinds, and only takes names" begin
        @test_throws UndefKeywordError ModelAttributes(;
            parameters=(:learning_rate,), states=(), inputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters=("learning_rate",), states=(), inputs=(), outputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters=(:learning_rate, 1), states=(), inputs=(), outputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters="learning_rate", states=(), inputs=(), outputs=()
        )
    end

    @testset "ModelAttributes only takes a name once, within a kind and across the kinds" begin
        #A name used for two kinds says which two
        @test_throws "The name choice is used for both a state and an output, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=:learning_rate, states=:choice, inputs=:reward, outputs=:choice
        )
        @test_throws "The name reward is used for both a parameter and an input, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=:reward, states=(), inputs=:reward, outputs=:choice
        )
        @test_throws "The name arm is used for both an input and an output, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=(), states=(), inputs=:arm, outputs=:arm
        )
        #A name used twice within a kind shows the names of that kind
        @test_throws "The states are named (:a, :a), but a model can only use a name for one attribute." ModelAttributes(;
            parameters=(), states=(:a, :a), inputs=(), outputs=()
        )
        @test_throws "The parameters are named (:a, :b, :a), but a model can only use a name for one attribute." ModelAttributes(;
            parameters=[:a, :b, :a], states=(), inputs=(), outputs=()
        )
        @test_throws ArgumentError ModelAttributes(;
            parameters=(), states=(), inputs=(:reward, :reward), outputs=()
        )
        #Distinct names pass, however many kinds have names
        @test ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=:expected_reward,
            inputs=:reward,
            outputs=:choice,
        ) isa ModelAttributes
        @test ModelAttributes(; parameters=(), states=(), inputs=(), outputs=()) isa
            ModelAttributes
        @test CognitiveModeling.check_unique_attribute_names(
            ModelAttributes(; parameters=:a, states=:b, inputs=:c, outputs=:d)
        ) === nothing
    end

    @testset "ModelAttributes keeps the names in the type, so constructing it from literals is inferred" begin
        from_literals() = ModelAttributes(;
            parameters=(:learning_rate,),
            states=:expected_reward,
            inputs=(),
            outputs=(:choice,),
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
