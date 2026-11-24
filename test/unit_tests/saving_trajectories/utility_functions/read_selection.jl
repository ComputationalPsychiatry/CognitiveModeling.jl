@testset "Reading the selection of a SaveTrajectories" begin
    @testset "The selection of each kind is read as a Val, so that the names are known at compile time" begin
        save_trajectories = SaveTrajectories(;
            inputs=SaveEverything(), states=:expected_reward
        )
        @test CognitiveModeling.input_selection(save_trajectories) === Val(SaveEverything())
        @test CognitiveModeling.output_selection(save_trajectories) === Val(())
        @test CognitiveModeling.state_selection(save_trajectories) ===
            Val((:expected_reward,))
        @test CognitiveModeling.output_distribution_selection(
            SaveTrajectories(; output_distributions=:choice)
        ) === Val((:choice,))
    end

    @testset "Whether the initial timestep values are saved is read from the type" begin
        @test CognitiveModeling.saves_initial_timestep_values(SaveTrajectories())
        @test !CognitiveModeling.saves_initial_timestep_values(
            SaveTrajectories(; save_initial_timestep_values=false)
        )
    end

    @testset "saves_nothing is only true when no kind has a selection" begin
        @test CognitiveModeling.saves_nothing(SaveTrajectories())
        @test !CognitiveModeling.saves_nothing(SaveTrajectories(; states=:expected_reward))
        @test !CognitiveModeling.saves_nothing(SaveTrajectories(; inputs=SaveEverything()))
        @test !CognitiveModeling.saves_nothing(
            SaveTrajectories(; output_distributions=:choice)
        )
    end
end
