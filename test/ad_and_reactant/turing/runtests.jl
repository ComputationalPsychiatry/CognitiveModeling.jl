using CognitiveModeling
using CognitiveModeling: turing_behavioral_session, turing_multiple_behavioral_sessions
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
            "turing session with an environment which doesn't run, since nothing in the data is missing",
            "turing session with imputed observations",
            "turing session with noise in each timestep",
            "turing session with parameter domains and an environment",
            "turing full model with parameter domains",
            "turing full model with three sessions and independent priors",
            "turing full model with three parameters",
            "turing full model with an environment parameter shared by the sessions",
            "turing full model with parameters from the distributions of groups",
            "turing full model with a composite of centered group parameters and independent ones",
            "turing full model with parameters from regressions",
            "turing full model with a regression with a family distribution",
            "turing full model with a vector parameter for each session, from a Dirichlet prior",
            "turing full model with an output whose values are NamedTuples, made from two columns",
            "turing full model where only the first and the third session impute missing actions",
            "turing full model where only the first and the third session run their environment",
            "turing full model of sessions of different lengths, where only the second and the third impute missing actions",
            "turing full model with an output whose values are vectors, made from two columns",
            "turing full model with observations, actions, states and parameters that are vectors",
            "turing full model with a missing action that is a vector, imputed",
        ],
    ),
    (;
        bug="Enzyme forward throws an EnzymeNoDerivativeError, with and without runtime activity, for the log density of a normal distribution over vectors whose mean or covariance depends on what is differentiated, since it has no forward rule for LAPACK's triangular solve, trtrs, which Enzyme.jl issue 1991 reports. Reverse mode is right",
        backends=["Enzyme forward", "Enzyme forward, runtime activity"],
        cases=[
            "turing full model with an output whose values are vectors, made from two columns",
            "turing full model with observations, actions, states and parameters that are vectors",
        ],
    ),
    (;
        bug="Enzyme with runtime activity throws an IllegalTypeAnalysisException for a session with a vector action that is missing on one timestep and imputed: the imputed vector is a view of the vector of parameters, a SubArray, which meets the recorded actions, each a Vector{Float64}, in the loop over the timesteps, where Enzyme's type analysis can't merge the two. A missing action that is a number is right, since it is imputed as a Float64, the type of the recorded ones",
        backends=["Enzyme forward, runtime activity", "Enzyme reverse, runtime activity"],
        cases=["turing full model with a missing action that is a vector, imputed"],
    ),
    (;
        bug="Mooncake forward throws an UndefRefError on a full model where only some sessions have latent variables. Preparing its cache, it reads every element of the arrays that the differentiated function holds, here through the closure over the LogDensityFunction, and DynamicPPL's GrowableArray has elements that aren't assigned for the sessions without latent variables. Turing.jl's samplers, which differentiate through LogDensityFunction(model; adtype), aren't affected",
        backends=["Mooncake forward"],
        cases=[
            "turing full model where only the first and the third session impute missing actions",
            "turing full model where only the first and the third session run their environment",
            "turing full model of sessions of different lengths, where only the second and the third impute missing actions",
            "turing full model with a missing action that is a vector, imputed",
        ],
    ),
    (;
        bug="Enzyme reverse throws an IllegalTypeAnalysisException, with and without runtime activity, for a session whose observations are NoValue on some timesteps, so that a row of the data is one of two NamedTuples",
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
