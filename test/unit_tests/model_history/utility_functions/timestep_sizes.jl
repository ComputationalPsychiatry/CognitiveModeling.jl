@testset "The sizes of the timesteps that are saved" begin
    @testset "The sizes of the timesteps are kept with a history, one for each timestep, when anything is saved" begin
        saves_states = SaveHistory(;
            states=:expected_reward, save_initial_timestep_values=false
        )
        @test CognitiveModeling.saved_timestep_sizes(1, 3, saves_states) == [1, 1, 1]
        @test CognitiveModeling.saved_timestep_sizes(
            0.5, 2, SaveHistory(SaveEverything(); save_initial_timestep_values=false)
        ) == [0.5, 0.5]
        #A vector of sizes is kept as it is, in a vector of its own
        sizes = [0.5, 2.0, 0.25]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, saves_states) == sizes
        #With the initial timestep values, the sizes start with BeforeFirstTimestep for timestep 0
        @test isequal(
            CognitiveModeling.saved_timestep_sizes(
                sizes, 3, SaveHistory(; states=:expected_reward)
            ),
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25],
        )
        @test isequal(
            CognitiveModeling.saved_timestep_sizes(1, 2, SaveHistory(SaveEverything())),
            [BeforeFirstTimestep(), 1, 1],
        )
        #Nothing is kept when nothing is saved
        @test CognitiveModeling.saved_timestep_sizes(1, 3, SaveHistory()) == Int[]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory()) == Float64[]
        @test isempty(CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory())) &&
            length(sizes) == 3
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, saves_states)) isa
            Vector{Float64}
        @test @inferred(
            CognitiveModeling.saved_timestep_sizes(
                0.5, 2, SaveHistory(; states=:expected_reward)
            )
        ) isa Vector{Union{BeforeFirstTimestep,Float64}}
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, SaveHistory())) isa
            Vector{Float64}
    end

    @testset "One size is repeated for every timestep, and a vector of sizes is kept as it is" begin
        sizes = [0.5, 2.0, 0.25]
        @test CognitiveModeling.each_timestep_size(sizes, 3) === sizes
        @test CognitiveModeling.each_timestep_size(0.5, 3) == [0.5, 0.5, 0.5]
    end

    @testset "The sizes that are returned are cut to the timesteps that were simulated, after timestep 0 when it is saved" begin
        sizes = [0.5, 2.0, 0.25]
        @test CognitiveModeling.returned_timestep_sizes(sizes, 2, Val(false)) == [0.5, 2.0]
        @test isequal(
            CognitiveModeling.returned_timestep_sizes(sizes, 2, Val(true)),
            [BeforeFirstTimestep(), 0.5, 2.0],
        )
        #A single size is returned as it is
        @test CognitiveModeling.returned_timestep_sizes(0.5, 2, Val(true)) === 0.5
    end
end
