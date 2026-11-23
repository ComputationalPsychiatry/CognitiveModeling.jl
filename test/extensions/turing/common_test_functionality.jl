#The Turing.jl extension of the code that the suites share: every pair of models simulates with the distributions of
#Distributions.jl, which the shared inference cases score in sessions of every pair
@testset "The models of the tests with the distributions of Distributions.jl" begin
    @test Base.get_extension(CommonTestFunctionality, :CommonTestFunctionalityTuringExt) isa
        Module
    @testset "$(pair.name), $(nameof(typeof(who_starts)))" for pair in MODEL_PAIRS,
        who_starts in (EnvironmentStarts(), AgentStarts())

        simulation = simulate_pair(pair, DistributionsJl(), who_starts)
        @test simulation.n_timesteps >= 1
        @test length(simulation.actions) == simulation.n_timesteps + 1
    end
end
