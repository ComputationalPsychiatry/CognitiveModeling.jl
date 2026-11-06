@testset "SaveHistory" begin
    @testset "It keeps each selection as a tuple of names, or as the marker for everything" begin
        save_history = SaveHistory(; observations = SaveEverything(), actions = (:choice,), states = (:expected_reward, :surprise))
        @test save_history.observations === SaveEverything()
        @test save_history.actions === (:choice,)
        @test save_history.states === (:expected_reward, :surprise)
        @test propertynames(save_history) == (:observations, :actions, :states)
        @test_throws FieldError save_history.parameters
    end

    @testset "A single name or a vector of names is put in a tuple" begin
        save_history = SaveHistory(; actions = :choice, states = [:expected_reward, :surprise])
        @test save_history.actions === (:choice,)
        @test save_history.states === (:expected_reward, :surprise)
    end

    @testset "A kind that isn't given saves nothing, and SaveEverything() saves everything" begin
        @test SaveHistory().observations === SaveHistory().actions === SaveHistory().states === ()
        @test SaveHistory(; states = :expected_reward).observations === ()
        everything = SaveHistory(SaveEverything())
        @test everything.observations === everything.actions === everything.states === SaveEverything()
        @test everything === SaveHistory(; observations = SaveEverything(), actions = SaveEverything(), states = SaveEverything())
    end

    @testset "It only takes names" begin
        @test_throws MethodError SaveHistory(; states = "expected_reward")
        @test_throws MethodError SaveHistory(; states = (:expected_reward, 1))
        @test_throws MethodError SaveHistory(; states = true)
    end

    @testset "The selections are in the type, so constructing it from literals is inferred, and holding it costs nothing" begin
        from_literals() = SaveHistory(; actions = :choice, states = (:expected_reward, :surprise))
        @test Base.return_types(from_literals, Tuple{})[1] === SaveHistory{(), (:choice,), (:expected_reward, :surprise)}
        @test @inferred(SaveHistory(SaveEverything())) isa SaveHistory
        @test @inferred(SaveHistory()) isa SaveHistory
        @test isbitstype(typeof(SaveHistory(; states = :expected_reward)))
        @test sizeof(SaveHistory(; states = (:expected_reward, :surprise))) == 0
    end

    @testset "The REPL shows the selections on one line each" begin
        @test repr("text/plain", SaveHistory(; actions = :choice, states = (:expected_reward, :surprise))) == """
            SaveHistory
            Observations: none
            Actions: choice
            States: expected_reward, surprise"""
        @test repr("text/plain", SaveHistory(SaveEverything())) == """
            SaveHistory
            Observations: everything
            Actions: everything
            States: everything"""
    end

    @testset "select_names keeps what is selected" begin
        state = (; expected_reward = 0.5, surprise = 0.1, n_timesteps = 3)
        @test CognitiveModeling.select_names(state, Val(SaveEverything())) === state
        @test CognitiveModeling.select_names(state, Val(())) === (;)
        @test CognitiveModeling.select_names(state, Val((:surprise, :expected_reward))) === (; surprise = 0.1, expected_reward = 0.5)
        @test CognitiveModeling.select_names((; reward = NoObservation(), cue = 1), Val((:reward,))) === (; reward = NoObservation())
        @test CognitiveModeling.select_names(NoObservation(), Val(())) === (;)
        @test CognitiveModeling.select_names([0.5], Val(())) === (;)
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(NoObservation(), Val((:reward,)))
        @test_throws FieldError CognitiveModeling.select_names(state, Val((:nope,)))
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names([0.5], Val((:expected_reward,)))
        #The selections of a SaveHistory come as Vals, so that the names are known at compile time
        save_history = SaveHistory(; observations = SaveEverything(), states = :expected_reward)
        @test CognitiveModeling.observation_selection(save_history) === Val(SaveEverything())
        @test CognitiveModeling.action_selection(save_history) === Val(())
        @test CognitiveModeling.state_selection(save_history) === Val((:expected_reward,))
    end

    @testset "The sizes of the timesteps are kept with a history, one for each timestep, when anything is saved" begin
        saves_states = SaveHistory(; states = :expected_reward)
        @test CognitiveModeling.saved_timestep_sizes(1, 3, saves_states) == [1, 1, 1]
        @test CognitiveModeling.saved_timestep_sizes(0.5, 2, SaveHistory(SaveEverything())) == [0.5, 0.5]
        #A vector of sizes is kept as it is
        sizes = [0.5, 2.0, 0.25]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, saves_states) === sizes
        #Nothing is kept when nothing is saved
        @test CognitiveModeling.saved_timestep_sizes(1, 3, SaveHistory()) == Int[]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory()) == Float64[]
        @test isempty(CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory())) && length(sizes) == 3
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, saves_states)) isa Vector{Float64}
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, SaveHistory())) isa Vector{Float64}
    end

    @testset "saves_nothing is only true when no kind has a selection" begin
        @test CognitiveModeling.saves_nothing(SaveHistory())
        @test !CognitiveModeling.saves_nothing(SaveHistory(; states = :expected_reward))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; observations = SaveEverything()))
    end
end
