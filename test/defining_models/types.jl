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

    @testset "ModelOutput puts a single action distribution in a NamedTuple" begin
        @test ModelOutput(0.7).action_distribution === (; action = 0.7)
        @test ModelOutput((; choice = 0.7)).action_distribution === (; choice = 0.7)
        @test ModelOutput(NoAction()).action_distribution === NoAction()
    end

    @testset "ModelOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput(0.7).new_state === (;)
        @test ModelOutput(0.7, state).new_state === state
    end

    @testset "EnvironmentOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test EnvironmentOutput(observation).new_state === (;)
        @test EnvironmentOutput(observation, state).new_state === state
    end

    @testset "ModelOutput saves the state as it is unless a function is given" begin
        @test ModelOutput(0.7).save_state === identity
        @test ModelOutput(0.7, state).save_state === identity
        @test ModelOutput(0.7, state; save_state = deepcopy).save_state === deepcopy
        @test ModelOutput((; choice = 0.7), state; save_state = deepcopy).save_state === deepcopy
        @test ModelOutput(NoAction(), state; save_state = deepcopy).save_state === deepcopy
    end

    @testset "EnvironmentOutput saves the state as it is unless a function is given" begin
        @test EnvironmentOutput(observation).save_state === identity
        @test EnvironmentOutput(observation, state).save_state === identity
        @test EnvironmentOutput(observation, state; save_state = deepcopy).save_state === deepcopy
    end

    @testset "The constructors are type stable" begin
        @test @inferred(ModelInput(observation, state, NoAction(), parameters)) isa ModelInput
        @test @inferred(ModelInput(observation, state, NoAction(), parameters, 0.5)) isa ModelInput
        @test @inferred(EnvironmentInput(NoAction(), state, observation, parameters)) isa EnvironmentInput
        @test @inferred(EnvironmentInput(NoAction(), state, observation, parameters, 0.5)) isa EnvironmentInput
        @test @inferred(ModelOutput(0.7)) isa ModelOutput
        @test @inferred(ModelOutput(0.7, state)) isa ModelOutput
        @test @inferred(ModelOutput((; choice = 0.7))) isa ModelOutput
        @test @inferred(ModelOutput(NoAction())) isa ModelOutput
        @test @inferred(ModelOutput(0.7, state; save_state = deepcopy)) isa ModelOutput
        @test @inferred(EnvironmentOutput(observation)) isa EnvironmentOutput
        @test @inferred(EnvironmentOutput(observation, state; save_state = deepcopy)) isa EnvironmentOutput
    end

    @testset "The types hold their values directly, so they don't allocate" begin
        @test isbitstype(typeof(ModelInput(observation, state, NoAction(), parameters)))
        @test isbitstype(typeof(ModelOutput(0.7)))
    end
end
