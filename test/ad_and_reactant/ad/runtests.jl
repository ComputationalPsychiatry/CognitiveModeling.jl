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
# package uses for loops. The Turing.jl models have their own suite in
# test/extensions/turing/ad, so that the package is tested without Turing.jl here.
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

# Gradients that are wrong or throw because of a bug in a backend, which the package doesn't
# work around. Their tests are marked broken, so that the suite passes while the bug is there,
# and a fix shows up as an unexpected pass. Each entry has the bug, the backends and the
# test cases it hits. The documentation page on automatic differentiation lists the same
# bugs, so update both together.
const KNOWN_FAILURES = [
    (;
        bug = "Enzyme reverse gives wrong gradients, with no error, through a loop that draws from a Xoshiro",
        backends = ["Enzyme reverse", "Enzyme reverse, runtime activity"],
        cases = ["simulate with sampling, agent starts"],
    ),
    (;
        bug = "Enzyme with runtime activity gives wrong gradients (forward) or throws (reverse) when a state read from an untyped field starts as a constant",
        backends = ["Enzyme forward, runtime activity", "Enzyme reverse, runtime activity"],
        cases = [
            "simulate! with a history, environment starts",
            "simulate! with a history, agent starts",
            "simulate! saving only the states, environment starts",
            "simulate! saving only the states, agent starts",
            "reading the history of an agent, environment starts",
            "reading the history of an agent, agent starts",
        ],
    ),
    (;
        bug = "ReverseDiff gives zero gradients, with no error, through deepcopy of a container of tracked numbers; fixed on its master branch on 2026-11-06, in pull request 329, and not yet released after v1.18.7",
        backends = ["ReverseDiff", "ReverseDiff, compiled tape"],
        cases = [
            "simulate with a state saved with deepcopy, environment starts",
            "simulate with a state saved with deepcopy, agent starts",
            "reading the history of an agent with a state saved with deepcopy",
        ],
    ),
]

is_known_failure(case_name, backend_name) = any(failure -> backend_name in failure.backends && case_name in failure.cases, KNOWN_FAILURES)

@testset "AD backends" begin
    @testset "Every known failure names test cases and backends that exist" begin
        for failure in KNOWN_FAILURES
            @test all(name -> any(case -> case.name == name, TEST_CASES), failure.cases)
            @test all(name -> any(backend -> first(backend) == name, BACKENDS), failure.backends)
        end
    end

    @testset "$(case.name)" for case in TEST_CASES
        (; f, x) = case
        reference = grad(central_fdm(5, 1), f, x)[1]

        @testset "$name" for (name, backend) in BACKENDS
            if is_known_failure(case.name, name)
                # Broken until the bug is fixed, which then shows up as an unexpected pass
                @test_broken DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6
                @test_broken DI.gradient(f, DI.prepare_gradient(f, backend, x .+ 0.25), backend, x) ≈ reference rtol = 1.0e-6
            else
                @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6

                # Prepare at a different point, so that stale compiled tapes get caught
                preparation = DI.prepare_gradient(f, backend, x .+ 0.25)
                @test DI.gradient(f, preparation, backend, x) ≈ reference rtol = 1.0e-6
            end
        end
    end
end
