using CognitiveModeling
using Test
using Enzyme: Enzyme, Reverse
using Reactant: Reactant, @jit

# Placeholder until the package has real models to check
placeholder_function(x) = sum(sin, x)

@testset "Reactant" begin
    x = [0.3, 2.0]
    x_reactant = Reactant.to_rarray(x)

    @testset "Compiled function" begin
        @test Float64(@jit placeholder_function(x_reactant)) ≈ placeholder_function(x)
    end

    # Reactant differentiates through Enzyme
    @testset "Compiled gradient" begin
        gradient_function(y) = Enzyme.gradient(Reverse, placeholder_function, y)[1]
        @test Array(@jit gradient_function(x_reactant)) ≈ cos.(x)
    end
end
