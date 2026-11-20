using CognitiveModeling
using Test
using Enzyme: Enzyme, Reverse
using FiniteDifferences: central_fdm, grad
using Reactant: Reactant, @jit

#The example models and the simulation cases that the suites share, and the sessions run by hand
include(joinpath(@__DIR__, "..", "..", "example_models", "ExampleModels.jl"))
using .ExampleModels
include(joinpath(@__DIR__, "..", "..", "test_cases", "simulation_cases.jl"))
include("../session_by_hand.jl")

# The test cases read their parameters from a vector one number at a time. The package doesn't need this
Reactant.allowscalar(true)

# Checks that a function of a vector gives the same number when Reactant compiles it, and that its compiled gradient is
# that of finite differences. Reactant differentiates through Enzyme
function test_compiled(f, x)
    x_reactant = Reactant.to_rarray(x)

    @testset "Compiled function" begin
        @test Float64(@jit f(x_reactant)) ≈ f(x) rtol = 1.0e-8
    end

    @testset "Compiled gradient" begin
        gradient_function(y) = Enzyme.gradient(Reverse, f, y)[1]
        reference = grad(central_fdm(5, 1), f, x)[1]
        @test Array(@jit gradient_function(x_reactant)) ≈ reference rtol = 1.0e-6
    end
end

@testset "Reactant" begin
    #Every simulation case with a gradient that Reactant can compile, with the means of the distributions, as the sum of
    #every number that it saved
    @testset "$(case.name)" for case in filter(
        case -> case.reactant && case.differentiated, simulation_cases(Means())
    )
        run = case.run
        test_compiled(x -> sum_of_reals(run(x)), case.x)
    end

    # The functions for use within the model of an inference backend, in sessions that are run by hand
    @testset "Sessions run by hand" begin
        @testset "A session run by hand has the log density that is worked out on paper" begin
            # With a learning rate of 0.5, the expected reward is 0.75 after the reward 1 and 0.375 after the reward 0,
            # so with an inverse temperature of 2 the two choices have the probabilities below
            logistic(value) = 1 / (1 + exp(-value))
            session_info = SessionInfo(
                LearnsRewards{ChoiceByHand}(),
                (; reward=[1.0, 0.0]),
                (; choice=[true, false]);
                who_starts=EnvironmentStarts(),
            )
            @test log_density_by_hand(
                session_by_hand,
                session_info,
                (; learning_rate=0.5, inverse_temperature=2.0),
                (;),
                (),
                0.0,
            ) ≈ log(logistic(0.5)) + log(1 - logistic(-0.25))
        end

        @testset "$(case.name)" for case in SESSION_BY_HAND_CASES
            test_compiled(case.f, case.x)
        end
    end
end
