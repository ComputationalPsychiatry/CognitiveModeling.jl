using CognitiveModeling
using Test
using Enzyme: Enzyme, Reverse
using FiniteDifferences: central_fdm, grad
using Reactant: Reactant, @jit

include("../test_cases.jl")

@testset "Reactant" begin
    @testset "$(case.name)" for case in TEST_CASES
        (; f, x) = case
        x_reactant = Reactant.to_rarray(x)

        @testset "Compiled function" begin
            @test Float64(@jit f(x_reactant)) ≈ f(x) rtol = 1.0e-8
        end

        # Reactant differentiates through Enzyme
        @testset "Compiled gradient" begin
            gradient_function(y) = Enzyme.gradient(Reverse, f, y)[1]
            reference = grad(central_fdm(5, 1), f, x)[1]
            @test Array(@jit gradient_function(x_reactant)) ≈ reference rtol = 1.0e-6
        end
    end
end
