using CognitiveModeling
using Test
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Forward, Reverse, set_runtime_activity
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff

# Every form of every supported backend. Zygote is left out on purpose, because the
# package uses for loops.
const BACKENDS = [
    "ForwardDiff" => AutoForwardDiff(),
    "ForwardDiff, chunk size 1" => AutoForwardDiff(; chunksize = 1),
    "ReverseDiff" => AutoReverseDiff(),
    "ReverseDiff, compiled tape" => AutoReverseDiff(; compile = true),
    "Enzyme forward" => AutoEnzyme(; mode = Forward),
    "Enzyme reverse" => AutoEnzyme(; mode = Reverse),
    "Enzyme forward, runtime activity" => AutoEnzyme(; mode = set_runtime_activity(Forward)),
    "Enzyme reverse, runtime activity" => AutoEnzyme(; mode = set_runtime_activity(Reverse)),
    "Mooncake reverse" => AutoMooncake(),
    "Mooncake forward" => AutoMooncakeForward(),
]

# Placeholder until the package has real models to check
placeholder_function(x) = sum(sin, x)

@testset "AD backends" begin
    x = [0.3, 2.0]
    reference = cos.(x)

    @testset "$name" for (name, backend) in BACKENDS
        @test DI.gradient(placeholder_function, backend, x) ≈ reference

        # Prepare at a different point, so that stale compiled tapes get caught
        preparation = DI.prepare_gradient(placeholder_function, backend, x .+ 0.25)
        @test DI.gradient(placeholder_function, preparation, backend, x) ≈ reference
    end
end
