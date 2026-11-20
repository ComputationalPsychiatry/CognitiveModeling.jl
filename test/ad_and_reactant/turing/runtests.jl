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
# bug is there, and a fix shows up as an unexpected pass. A bug that kills the process, which no
# test can catch, says kills_the_process = true, and its tests are skipped, so that a fix doesn't
# show up, and the bug must be checked by hand. Each entry has the bug, the backends and the test
# cases it hits. The documentation page on automatic differentiation lists the same bugs, so
# update both together.
const KNOWN_FAILURES = [
    (;
        bug="Enzyme without runtime activity throws an EnzymeRuntimeActivityError on every full model, and on many sessions: those with imputed values, of models whose outputs are numbers from a normal distribution, vectors or NamedTuples, of a model which uses the sizes of the timesteps, and with an environment which samples within a timestep or which has names of its own. Turing.jl documents that Enzyme needs runtime activity for its models. The sessions of the baseline and of most models whose outputs are choices pass",
        backends=["Enzyme forward", "Enzyme reverse"],
        cases=[
            "ManyAttributes with a missing confidence, environment starts",
            "ManyAttributes with a missing cue observed, environment starts",
            "VectorAttributes with its environment, environment starts",
            "VectorAttributes with a missing ratings, environment starts",
            "NamedTupleAttributes with its environment, environment starts",
            "ContinuousOutput with its environment, environment starts",
            "ContinuousOutput with a missing rating, environment starts",
            "ContinuousOutput with a missing reward observed, environment starts",
            "UsesTimestepSize with its environment, environment starts",
            "Baseline with SamplingWithinTimestepEnvironment with its environment, environment starts",
            "Baseline with SamplingWithinTimestepEnvironment with a missing reward observed, environment starts",
            "Baseline with OtherNamesEnvironment with its environment, environment starts",
            "ContinuousOutput, sessions, environment starts",
            "Independent parameters, without missing values",
            "Independent parameters, agent starts, without missing values",
            "Identical parameters, without missing values",
            "A parameter fixed by keyword, without missing values",
            "Hierarchical parameters, centered and non-centered, without missing values",
            "A regression with every kind of term, without missing values",
            "Composite parameters, without missing values",
            "Environment parameters estimated, without missing values",
            "A parameter that is a vector, without missing values",
            "Every feature with a regression, without missing values",
            "Invalid parameters caught, without missing values",
            "A parameter that is a vector",
            "Every feature with a regression",
        ],
    ),
    (;
        bug="Enzyme throws an IllegalTypeAnalysisException, in every form, on a session whose observation is a vector which is missing on one timestep and imputed: the imputed vector is a view of the vector of parameters, a SubArray, which meets the recorded vectors, each a Vector{Float64}, in the loop over the timesteps. Enzyme.jl issue 4106 reports it",
        backends=[
            "Enzyme forward",
            "Enzyme reverse",
            "Enzyme forward, runtime activity",
            "Enzyme reverse, runtime activity",
        ],
        cases=["VectorAttributes with a missing rewards observed, environment starts"],
    ),
    (;
        bug="Enzyme throws an IllegalTypeAnalysisException, in every form, when a value of one of two types goes from one timestep to the next: on the model with every feature, whose choice is NoValue on some timesteps among outputs of other kinds, and on sessions where the agent starts, so that an observation which is a number or imputed is carried to the next timestep. Without runtime activity, the full models throw the EnzymeRuntimeActivityError above instead. It is related to Enzyme.jl issue 4031, on a value that is one of two NamedTuples",
        backends=[
            "Enzyme forward",
            "Enzyme reverse",
            "Enzyme forward, runtime activity",
            "Enzyme reverse, runtime activity",
        ],
        cases=[
            "EveryFeature with its environment, environment starts",
            "EveryFeature with a missing rating, environment starts",
            "EveryFeature with its environment, agent starts",
            "EveryFeatureWithoutSampling with its environment, environment starts",
            "ContinuousOutput, sessions, agent starts",
            "Every feature with a regression, without missing values",
            "Every feature with a regression",
        ],
    ),
    (;
        bug="Enzyme reverse, with and without runtime activity, throws a TypeError, that a RefValue was given where a value was expected, when an output of a model is NoValue on some timesteps, so that it has two types in the loop over the timesteps. Enzyme.jl issue 4044 reports it",
        backends=["Enzyme reverse", "Enzyme reverse, runtime activity"],
        cases=[
            "OutputNotEveryTimestep with its environment, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment with its environment, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment with its environment, agent starts",
        ],
    ),
    (;
        bug="Enzyme forward with runtime activity throws an EnzymeNoDerivativeError for the log density of a normal distribution over vectors whose covariance depends on what is differentiated, since it has no forward rule for LAPACK's triangular solve, trtrs, which Enzyme.jl issue 1991 reports. Without runtime activity, the EnzymeRuntimeActivityError above comes first. Reverse mode is right",
        backends=["Enzyme forward, runtime activity"],
        cases=[
            "VectorAttributes with its environment, environment starts",
            "A parameter that is a vector, without missing values",
        ],
    ),
    (;
        bug="Enzyme reverse with runtime activity throws an LLVMException, \"Referring to an instruction in another function!\", when it links its code for the full model with a parameter that is a vector, in a new process and after another form of Enzyme failed on the model, though it was right in a process that had differentiated many other models before it. Not reported yet",
        backends=["Enzyme reverse, runtime activity"],
        cases=["A parameter that is a vector, without missing values"],
    ),
    (;
        bug="Enzyme forward, with and without runtime activity, kills the process, with an illegal insertion in its type analysis, \"UNREACHABLE executed at TypeTree.h:364\", when an output of a model is NoValue on some timesteps. The message is that of Enzyme.jl issue 3476",
        backends=["Enzyme forward", "Enzyme forward, runtime activity"],
        kills_the_process=true,
        cases=[
            "OutputNotEveryTimestep with its environment, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment with its environment, environment starts",
            "Baseline with OutputNotEveryTimestepEnvironment with its environment, agent starts",
            "Baseline with NoOutputBeforeFirstInputEnvironment with its environment, environment starts",
        ],
    ),
    (;
        bug="Enzyme with runtime activity, forward and reverse, kills the process with the same illegal insertion on a session with an imputed value among several outputs, such as a missing confidence beside a choice, or a missing vector of ratings, and on the full model with a parameter that is a vector and a missing rating",
        backends=["Enzyme forward, runtime activity", "Enzyme reverse, runtime activity"],
        kills_the_process=true,
        cases=[
            "ManyAttributes with a missing confidence, environment starts",
            "ManyAttributes with a missing cue observed, environment starts",
            "VectorAttributes with a missing ratings, environment starts",
            "A parameter that is a vector",
        ],
    ),
    (;
        bug="Enzyme forward with runtime activity kills the process with the same illegal insertion on a session with an environment whose outputs have names of their own, which maps of names connect to the cognitive model",
        backends=["Enzyme forward, runtime activity"],
        kills_the_process=true,
        cases=[
            "Baseline with OtherNamesEnvironment with its environment, environment starts"
        ],
    ),
    (;
        bug="ReverseDiff, also with a compiled tape, throws an InexactError on sessions of a model whose actions are numbers from a normal distribution, when the agent starts: the first action of a session, before any reward, has a mean that is a constant, so that its distribution has another type of tracked numbers than the later ones, and Distributions.jl's product_distribution makes their means a TrackedReal{Real}, whose log ReverseDiff can't take. Not reported yet",
        backends=["ReverseDiff", "ReverseDiff, compiled tape"],
        cases=["ContinuousOutput, sessions, agent starts"],
    ),
    (;
        bug="Mooncake forward throws an UndefRefError on several sessions where only some have latent variables. Preparing its cache, it reads every element of the arrays that the differentiated function holds, here through the closure over the LogDensityFunction, and DynamicPPL's GrowableArray has elements that aren't assigned for the sessions without latent variables. Mooncake.jl issue 1476 reports it. Turing.jl's samplers, which differentiate through LogDensityFunction(model; adtype), aren't affected",
        backends=["Mooncake forward"],
        cases=[
            "ContinuousOutput, sessions, environment starts",
            "ContinuousOutput, sessions, agent starts",
        ],
    ),
    (;
        bug="Mooncake can't differentiate the model which catches invalid parameters, since it runs the sessions inside a try block: reverse mode doesn't support try and catch, a limitation that its documentation lists, and forward mode throws an ArgumentError, that expressions of type :the_exception aren't supported",
        backends=["Mooncake reverse", "Mooncake forward"],
        cases=["Invalid parameters caught, without missing values"],
    ),
]

#The known failure of a case and a backend, or nothing. One that kills the process is looked for first, so that it is
#never run
function known_failure(case_name, backend_name)
    matches(failure) = backend_name in failure.backends && case_name in failure.cases
    index = findfirst(
        failure -> matches(failure) && get(failure, :kills_the_process, false),
        KNOWN_FAILURES,
    )
    index === nothing && (index = findfirst(matches, KNOWN_FAILURES))
    return index === nothing ? nothing : KNOWN_FAILURES[index]
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
            failure = known_failure(case.name, name)
            if failure === nothing
                # With an absolute tolerance too, since some cases have a gradient of zero, which finite differences give
                # up to rounding: a model without outputs, and models which guess or don't learn, whose parameters
                # don't change the log density
                @test DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6 atol = 1.0e-10

                # Prepare at a different point, so that stale compiled tapes get caught
                preparation = DI.prepare_gradient(f, backend, x .+ 0.25)
                @test DI.gradient(f, preparation, backend, x) ≈ reference rtol = 1.0e-6 atol =
                    1.0e-10
            elseif get(failure, :kills_the_process, false)
                # Not run, since the backend would end the suite
                @test_skip DI.gradient(f, backend, x) ≈ reference rtol = 1.0e-6
                @test_skip DI.gradient(
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
