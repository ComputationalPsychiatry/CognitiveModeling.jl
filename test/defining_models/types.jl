using Random: Xoshiro, default_rng

@testset "Types" begin
    observation = (; reward = 1.0)
    state = (; expected_reward = 0.5)
    parameters = (; learning_rate = 0.1)

    @testset "ModelInput sets the timestep size to 1 if it isn't specified" begin
        @test ModelInput(observation, state, NoAction(), parameters).timestep_size === 1
        @test ModelInput(observation, state, NoAction(), parameters, 0.5).timestep_size === 0.5
    end

    @testset "EnvironmentInput sets the timestep size to 1 if it isn't specified" begin
        @test EnvironmentInput(NoAction(), state, observation, parameters).timestep_size === 1
        @test EnvironmentInput(NoAction(), state, observation, parameters, 0.5).timestep_size === 0.5
    end

    @testset "The inputs use the default generator if it isn't specified" begin
        @test ModelInput(observation, state, NoAction(), parameters).rng === default_rng()
        @test EnvironmentInput(NoAction(), state, observation, parameters).rng === default_rng()
        rng = Xoshiro(1)
        @test ModelInput(observation, state, NoAction(), parameters, 1, rng).rng === rng
        @test EnvironmentInput(NoAction(), state, observation, parameters, 1, rng).rng === rng
    end

    @testset "ModelOutput only takes named action distributions, or no action" begin
        @test ModelOutput((; choice = 0.7)).action_distribution === (; choice = 0.7)
        @test ModelOutput(NoAction()).action_distribution === NoAction()
        @test_throws MethodError ModelOutput(0.7)
        @test_throws MethodError ModelOutput((0.7, 0.5))
    end

    @testset "ModelOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput((; choice = 0.7)).new_state === (;)
        @test ModelOutput((; choice = 0.7), state).new_state === state
    end

    @testset "EnvironmentOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test EnvironmentOutput(observation).new_state === (;)
        @test EnvironmentOutput(observation, state).new_state === state
    end

    @testset "The constructors are type stable" begin
        @test @inferred(ModelInput(observation, state, NoAction(), parameters)) isa ModelInput
        @test @inferred(ModelInput(observation, state, NoAction(), parameters, 0.5)) isa ModelInput
        @test @inferred(EnvironmentInput(NoAction(), state, observation, parameters)) isa EnvironmentInput
        @test @inferred(EnvironmentInput(NoAction(), state, observation, parameters, 0.5)) isa EnvironmentInput
        @test @inferred(ModelOutput((; choice = 0.7))) isa ModelOutput
        @test @inferred(ModelOutput((; choice = 0.7), state)) isa ModelOutput
        @test @inferred(ModelOutput(NoAction())) isa ModelOutput
        @test @inferred(EnvironmentOutput(observation)) isa EnvironmentOutput
        @test @inferred(EnvironmentOutput(observation, state)) isa EnvironmentOutput
    end

    @testset "The types hold their values directly, so they don't allocate" begin
        @test isbitstype(typeof(ModelInput(observation, state, NoAction(), parameters)))
        @test isbitstype(typeof(ModelOutput((; choice = 0.7))))
    end

    @testset "ModelAttributes keeps the names of each kind in a tuple" begin
        attributes = ModelAttributes(; parameters = (:learning_rate, :noise), states = (:expected_reward,), observations = (:reward,), actions = (:choice,))
        @test attributes.parameters === (:learning_rate, :noise)
        @test attributes.states === (:expected_reward,)
        @test attributes.observations === (:reward,)
        @test attributes.actions === (:choice,)
    end

    @testset "ModelAttributes puts a single name or a vector of names in a tuple" begin
        attributes = ModelAttributes(; parameters = :learning_rate, states = [:expected_reward, :surprise], observations = (), actions = ())
        @test attributes.parameters === (:learning_rate,)
        @test attributes.states === (:expected_reward, :surprise)
        @test attributes.observations === ()
    end

    @testset "ModelAttributes needs all four kinds, and only takes names" begin
        @test_throws UndefKeywordError ModelAttributes(; parameters = (:learning_rate,), states = (), observations = ())
        @test_throws MethodError ModelAttributes(; parameters = ("learning_rate",), states = (), observations = (), actions = ())
        @test_throws MethodError ModelAttributes(; parameters = (:learning_rate, 1), states = (), observations = (), actions = ())
    end

    @testset "ModelAttributes is type stable from tuples and symbols" begin
        @test @inferred(ModelAttributes(; parameters = (:learning_rate,), states = :expected_reward, observations = (), actions = (:choice,))) isa ModelAttributes
    end

    @testset "InvalidParametersError has a default message, and takes another" begin
        @test InvalidParametersError() isa Exception
        @test sprint(showerror, InvalidParametersError()) == "InvalidParametersError: Model parameters are not valid, possibly dependent on observations, actions and states."
        @test sprint(showerror, InvalidParametersError("The learning rate must be between 0 and 1.")) == "InvalidParametersError: The learning rate must be between 0 and 1."
        @test InvalidParametersError("The learning rate must be between 0 and 1.").message == "The learning rate must be between 0 and 1."
        @test_throws InvalidParametersError throw(InvalidParametersError())
    end
end
