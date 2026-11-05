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
        @test CognitiveModeling.select_names(NoObservation(), Val((:reward,))) === NoObservation()
        @test CognitiveModeling.select_names(NoAction(), Val((:choice,))) === NoAction()
        @test CognitiveModeling.select_names(NoAction(), Val(())) === (;)
        @test CognitiveModeling.select_names([0.5], Val(())) === (;)
        @test_throws FieldError CognitiveModeling.select_names(state, Val((:nope,)))
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names([0.5], Val((:expected_reward,)))
        #The selections of a SaveHistory come as Vals, so that the names are known at compile time
        save_history = SaveHistory(; observations = SaveEverything(), states = :expected_reward)
        @test CognitiveModeling.observation_selection(save_history) === Val(SaveEverything())
        @test CognitiveModeling.action_selection(save_history) === Val(())
        @test CognitiveModeling.state_selection(save_history) === Val((:expected_reward,))
    end

    @testset "saves_nothing is only true when no kind has a selection" begin
        @test CognitiveModeling.saves_nothing(SaveHistory())
        @test !CognitiveModeling.saves_nothing(SaveHistory(; states = :expected_reward))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; observations = SaveEverything()))
    end
end
