using CognitiveModeling
using Test
using ADTypes
using DifferentiationInterface: DifferentiationInterface as DI
using Enzyme: Const, Forward, Reverse, set_runtime_activity
using FiniteDifferences: central_fdm, grad
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using ReverseDiff: ReverseDiff

using Random: AbstractRNG, Xoshiro

#The example models and the simulation cases that the suites share
include(joinpath(@__DIR__, "..", "..", "example_models", "ExampleModels.jl"))
using .ExampleModels
include(joinpath(@__DIR__, "..", "..", "test_cases", "simulation_cases.jl"))

#The cases of the suite, each differentiated as the sum of every real number that it saved: every simulation case with the
#means of the distributions, which is deterministic, so that the gradient goes all the way round the loop, and those of
#the baseline and of the models that sample within a timestep with the distributions written by hand, which sample from
#a seeded Xoshiro
const SAMPLED_CASES = (
    "Baseline, environment starts",
    "Baseline, agent starts",
    "SamplingWithinTimestep, environment starts",
    "SamplingWithinTimestep, agent starts",
    "Baseline with SamplingWithinTimestepEnvironment, environment starts",
    "Baseline with SamplingWithinTimestepEnvironment, agent starts",
)
#Each case with the function that is differentiated, which holds only the case's own function
function with_sum(case)
    run = case.run
    return merge(case, (; f=x -> sum_of_reals(run(x))))
end
#The cases that reproduce a bug in a backend which the shared cases don't hit
include("bug_reproducers.jl")

const TEST_CASES = [
    map(with_sum, filter(case -> case.differentiated, simulation_cases(Means())));
    [
        merge(with_sum(case), (; name="$(case.name), sampled")) for
        case in simulation_cases(ByHand()) if case.name in SAMPLED_CASES
    ];
    BUG_REPRODUCERS
]

# Every form of every supported backend. Zygote is left out on purpose, because the
# package uses for loops. The Turing.jl models have their own suite in
# test/ad_and_reactant/turing, so that the package is tested without Turing.jl here. The
# functions are marked constant for Enzyme, because the derivative data is in x alone: each
# case's function holds the models, the data and the forms of the parameters, which aren't
# differentiated, some of them in vectors, which Enzyme otherwise refuses as not read-only.
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

#The backends that this run tests: every one when the suite is run without arguments, and otherwise those whose names
#start with one of the arguments, such as Enzyme or "Enzyme forward", so that CI can test the backends in jobs of their
#own: julia --project=<this folder> <this folder>/runtests.jl ForwardDiff ReverseDiff
const SELECTED_BACKENDS = filter(BACKENDS) do (name, backend)
    return isempty(ARGS) || any(prefix -> startswith(name, prefix), ARGS)
end
isempty(SELECTED_BACKENDS) && throw(
    ArgumentError(
        "No backend's name starts with any of $(ARGS). The backends are $(first.(BACKENDS)).",
    ),
)

# Gradients that are wrong or throw because of a bug or a limitation in a backend, which the
# package doesn't work around. Their tests are marked broken, so that the suite passes while the
# bug is there, and a fix shows up as an unexpected pass. Each entry has the bug, the backends and
# the test cases it hits, and an entry whose gradient is only wrong when it is prepared at another
# point has only_when_prepared_elsewhere = true. The documentation page on automatic
# differentiation lists the same bugs, so update both together.
const KNOWN_FAILURES = [
    (;
        bug="Enzyme reverse gives wrong gradients, with no error, through a loop that draws from a Xoshiro",
        backends=["Enzyme reverse", "Enzyme reverse, runtime activity"],
        cases=["Choices and rewards sampled from a Xoshiro for 30 timesteps, agent starts"],
    ),
    (;
        bug="Enzyme with runtime activity gives wrong gradients (forward) or throws (reverse) when a state read from an untyped field starts as a constant, as an agent's does",
        backends=["Enzyme forward, runtime activity", "Enzyme reverse, runtime activity"],
        cases=[
            "An agent and an environment, $starts" for
            starts in ("environment starts", "agent starts")
        ] ∪ [
            "An agent and an environment saving the states only, $starts" for
            starts in ("environment starts", "agent starts")
        ] ∪ [
            "An agent and an environment simulated twice, $starts" for
            starts in ("environment starts", "agent starts")
        ] ∪ [
            "An agent and an environment reset before the second simulation, $starts" for
            starts in ("environment starts", "agent starts")
        ],
    ),
    (;
        bug="Enzyme without runtime activity throws an EnzymeRuntimeActivityError when the observations of a FixedObservationsEnvironment have two types, such as a reward that is NoValue on some timesteps. It is right with runtime activity, and with observations of one type",
        backends=["Enzyme forward"],
        cases=[
            "Baseline with fixed observations that aren't always given, environment starts",
            "Baseline with fixed observations that aren't always given, agent starts",
        ],
    ),
    (;
        bug="Enzyme reverse, with and without runtime activity, throws a TypeError, that a RefValue was given where a value was expected, or an EnzymeNoTypeError, when an output of a model is NoValue on some timesteps, so that it has two types in the loop over the timesteps",
        backends=["Enzyme reverse", "Enzyme reverse, runtime activity"],
        cases=[
            "OutputNotEveryTimestep, environment starts",
            "OutputNotEveryTimestep, agent starts",
            "Baseline with OutputNotEveryTimestepEnvironment, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment, agent starts",
            "Baseline with OutputNotEveryTimestepEnvironment saving the states only, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment saving the states only, agent starts",
            "Baseline with NoOutputBeforeFirstInputEnvironment, environment starts",
            "Baseline with fixed observations that aren't always given, environment starts",
            "Baseline with fixed observations that aren't always given, agent starts",
        ],
    ),
    (;
        bug="Enzyme forward, with and without runtime activity, gives wrong gradients, with no error, when the cognitive model's action is NoValue on some timesteps and the environment starts. It is right when the agent starts",
        backends=["Enzyme forward", "Enzyme forward, runtime activity"],
        cases=["OutputNotEveryTimestep, environment starts"],
    ),
    (;
        bug="Enzyme throws an IllegalTypeAnalysisException in every form when a state changes type within a simulation, an integer at the start and a number after the first timestep, and on the model with every feature, whose choice is NoValue on some timesteps among outputs of other kinds",
        backends=[
            "Enzyme forward",
            "Enzyme reverse",
            "Enzyme forward, runtime activity",
            "Enzyme reverse, runtime activity",
        ],
        cases=[
            "StateChangesType, environment starts",
            "StateChangesType, agent starts",
            "EveryFeatureWithoutSampling, environment starts",
            "EveryFeatureWithoutSampling, agent starts",
            "EveryFeatureWithoutSampling as an agent with an environment, environment starts",
            "EveryFeatureWithoutSampling as an agent with an environment, agent starts",
            "EveryFeatureWithoutSampling continuing from a given state and action, environment starts",
            "EveryFeatureWithoutSampling continuing from a given state and action, agent starts",
        ],
    ),
    (;
        bug="ReverseDiff's compiled tape records the path through the code at the point where it is compiled, and the tests compile it at another point too, where a simulation that stops once a value passes a threshold, whose parameters turn invalid on another timestep, or that samples its choices and rewards, takes another path, so the tape gives the gradient of that path, with no error. This is a limitation of compiled tapes, which the page on automatic differentiation describes among its pitfalls",
        backends=["ReverseDiff, compiled tape"],
        only_when_prepared_elsewhere=true,
        cases=[
            "WithStoppingCriterion, environment starts",
            "WithStoppingCriterion, agent starts",
            "WithStoppingCriterion until its criterion is met, environment starts",
            "WithStoppingCriterion until its criterion is met, agent starts",
            "WithInvalidParameters from an expected reward above the highest valid one, environment starts",
            "Baseline, environment starts, sampled",
            "Baseline, agent starts, sampled",
            "SamplingWithinTimestep, environment starts, sampled",
            "SamplingWithinTimestep, agent starts, sampled",
            "Baseline with SamplingWithinTimestepEnvironment, environment starts, sampled",
            "Baseline with SamplingWithinTimestepEnvironment, agent starts, sampled",
        ],
    ),
]

#The known failure of a case and a backend, or nothing
function known_failure(case_name, backend_name)
    index = findfirst(
        failure -> backend_name in failure.backends && case_name in failure.cases,
        KNOWN_FAILURES,
    )
    return index === nothing ? nothing : KNOWN_FAILURES[index]
end

@testset "AD backends" begin
    @testset "Every known failure names test cases and backends that exist" begin
        for failure in KNOWN_FAILURES
            @test all(name -> any(case -> case.name == name, TEST_CASES), failure.cases)
            @test all(
                name -> any(backend -> first(backend) == name, BACKENDS), failure.backends
            )
        end
    end

    @testset "$(case.name)" for case in TEST_CASES
        (; f, x) = case
        reference = grad(central_fdm(5, 1), f, x)[1]

        @testset "$name" for (name, backend) in SELECTED_BACKENDS
            failure = known_failure(case.name, name)
            if failure === nothing
                @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6

                # Prepare at a different point, so that stale compiled tapes get caught
                preparation = DI.prepare_gradient(f, backend, x .+ 0.25)
                @test DI.gradient(f, preparation, backend, x) ≈ reference rtol = 1.0e-6
            elseif get(failure, :only_when_prepared_elsewhere, false)
                # Right when prepared at the point itself, and broken when prepared at another
                @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6
                @test_broken DI.gradient(
                    f, DI.prepare_gradient(f, backend, x .+ 0.25), backend, x
                ) ≈ reference rtol = 1.0e-6
            else
                # Broken until the bug is fixed, which then shows up as an unexpected pass
                @test_broken DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6
                @test_broken DI.gradient(
                    f, DI.prepare_gradient(f, backend, x .+ 0.25), backend, x
                ) ≈ reference rtol = 1.0e-6
            end
        end
    end
end
