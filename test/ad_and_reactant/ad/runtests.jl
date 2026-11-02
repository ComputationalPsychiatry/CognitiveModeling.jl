using CognitiveModeling
using Test
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Forward, Reverse, set_runtime_activity
using FiniteDifferences: central_fdm, grad
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff

include("../test_cases.jl")

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

@testset "AD backends" begin
    @testset "$(case.name)" for case in TEST_CASES
        (; f, x) = case
        reference = grad(central_fdm(5, 1), f, x)[1]

        @testset "$name" for (name, backend) in BACKENDS
            @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6

            # Prepare at a different point, so that stale compiled tapes get caught
            preparation = DI.prepare_gradient(f, backend, x .+ 0.25)
            @test DI.gradient(f, preparation, backend, x) ≈ reference rtol = 1.0e-6
        end
    end
end
