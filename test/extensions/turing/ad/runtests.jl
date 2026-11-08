using CognitiveModeling
using Test
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Const, Forward, Reverse, set_runtime_activity
using FiniteDifferences: central_fdm, grad
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff
using Turing

include("turing_cases.jl")

# Every form of every supported backend, as in test/ad_and_reactant/ad/runtests.jl. The functions are marked constant
# for Enzyme, as Turing.jl does, because the derivative data is in x alone: the function holds a
# VarInfo which is written during evaluation, which Enzyme otherwise refuses as not read-only.
const BACKENDS = [
    "ForwardDiff" => AutoForwardDiff(),
    "ForwardDiff, chunk size 1" => AutoForwardDiff(; chunksize=1),
    "ReverseDiff" => AutoReverseDiff(),
    "ReverseDiff, compiled tape" => AutoReverseDiff(; compile=true),
    "Enzyme forward" => AutoEnzyme(; mode=Forward, function_annotation=Const),
    "Enzyme reverse" => AutoEnzyme(; mode=Reverse, function_annotation=Const),
    "Enzyme forward, runtime activity" =>
        AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const),
    "Enzyme reverse, runtime activity" =>
        AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
    "Mooncake reverse" => AutoMooncake(),
    "Mooncake forward" => AutoMooncakeForward(),
]

# Gradients that are wrong or throw because of a bug or a limitation in a backend, which the
# package doesn't work around. Their tests are marked broken, so that the suite passes while the
# bug is there, and a fix shows up as an unexpected pass. Each entry has the bug, the backends and
# the test cases it hits. The documentation page on automatic differentiation lists the same
# bugs, so update both together.
const KNOWN_FAILURES = [
    (;
        bug="Enzyme without runtime activity throws an EnzymeRuntimeActivityError on a Turing.jl model of a session with an imputed action, an environment, or a model which samples within a timestep, and on the full model, which Turing.jl documents as needing runtime activity. A plain session passes",
        backends=["Enzyme forward", "Enzyme reverse"],
        cases=[
            "turing session with imputed actions",
            "turing session with an environment",
            "turing session with imputed observations",
            "turing session with noise in each timestep",
            "turing full model with three sessions and independent priors",
        ],
    ),
    (;
        bug="Enzyme reverse throws an IllegalTypeAnalysisException, with and without runtime activity, for a session whose observations are NoObservation on some timesteps, so that a row of the data is one of two NamedTuples",
        backends=["Enzyme reverse", "Enzyme reverse, runtime activity"],
        cases=[
            "turing session with a reward that isn't always given, environment starts",
            "turing session with a reward that isn't always given, agent starts",
        ],
    ),
    (;
        bug="Enzyme forward throws the same IllegalTypeAnalysisException when the agent starts, so that the observation of one of two types is carried from one timestep to the next, and is right when the environment starts",
        backends=["Enzyme forward", "Enzyme forward, runtime activity"],
        cases=["turing session with a reward that isn't always given, agent starts"],
    ),
]

function is_known_failure(case_name, backend_name)
    return any(
        failure -> backend_name in failure.backends && case_name in failure.cases,
        KNOWN_FAILURES,
    )
end

@testset "AD backends on Turing.jl models" begin
    @testset "Every known failure names test cases and backends that exist" begin
        for failure in KNOWN_FAILURES
            @test all(name -> any(case -> case.name == name, TURING_CASES), failure.cases)
            @test all(
                name -> any(backend -> first(backend) == name, BACKENDS), failure.backends
            )
        end
    end

    @testset "$(case.name)" for case in TURING_CASES
        (; f, x) = case
        reference = grad(central_fdm(5, 1), f, x)[1]

        @testset "$name" for (name, backend) in BACKENDS
            if is_known_failure(case.name, name)
                # Broken until the bug is fixed, which then shows up as an unexpected pass
                @test_broken DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6
                @test_broken DI.gradient(
                    f, DI.prepare_gradient(f, backend, x .+ 0.25), backend, x
                ) ≈ reference rtol = 1.0e-6
            else
                @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6

                # Prepare at a different point, so that stale compiled tapes get caught
                preparation = DI.prepare_gradient(f, backend, x .+ 0.25)
                @test DI.gradient(f, preparation, backend, x) ≈ reference rtol = 1.0e-6
            end
        end
    end
end
