@testset "Reading the selection of a save history" begin
    @testset "The selection of each kind is read as a Val, so that the names are known at compile time" begin
        save_history = SaveHistory(; inputs=SaveEverything(), states=:expected_reward)
        @test CognitiveModeling.input_selection(save_history) === Val(SaveEverything())
        @test CognitiveModeling.output_selection(save_history) === Val(())
        @test CognitiveModeling.state_selection(save_history) === Val((:expected_reward,))
        @test CognitiveModeling.output_distribution_selection(
            SaveHistory(; output_distributions=:choice)
        ) === Val((:choice,))
    end

    @testset "Whether the initial timestep values are saved is read from the type" begin
        @test CognitiveModeling.saves_initial_timestep_values(SaveHistory())
        @test !CognitiveModeling.saves_initial_timestep_values(
            SaveHistory(; save_initial_timestep_values=false)
        )
    end

    @testset "saves_nothing is only true when no kind has a selection" begin
        @test CognitiveModeling.saves_nothing(SaveHistory())
        @test !CognitiveModeling.saves_nothing(SaveHistory(; states=:expected_reward))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; inputs=SaveEverything()))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; output_distributions=:choice))
    end
end
