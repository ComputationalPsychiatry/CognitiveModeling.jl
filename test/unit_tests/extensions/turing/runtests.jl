using CognitiveModeling
using CognitiveModeling:
    timestep_model, initialize_state, turing_timestep_model, turing_initialize_state
#The Turing.jl models of the sessions, which are public but not exported, since session_model and the other functions
#that every inference backend has make them
using CognitiveModeling:
    turing_behavioral_session, turing_multiple_behavioral_sessions, turing_full_model
using Test
using JET: JET
using Random: Random, Xoshiro
using Turing

#The example models that the suites share, with their methods for Turing.jl
const EXAMPLE_MODELS_DIRECTORY = joinpath(@__DIR__, "..", "..", "..", "example_models")
include(joinpath(EXAMPLE_MODELS_DIRECTORY, "ExampleModels.jl"))
using .ExampleModels
include(joinpath(EXAMPLE_MODELS_DIRECTORY, "turing_methods.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "test_cases", "inference_cases.jl"))

#Whether a report is about a local variable being undefined. DynamicPPL's ~ expands to code which reads
#its left-hand side before assigning it, on a branch that is never taken, so JET reports every ~ this way
function is_about_undefined_local(report)
    return report isa JET.UndefVarErrorReport && report.var isa Symbol
end

#The two parts of the suite, which CI runs in jobs of their own: features, the tests of each feature of the extension,
#with JET and the example models, and shared_cases, what holds for every inference case that the suites share. The suite
#runs the parts it is given as arguments, and both without arguments:
#    julia --project=test/unit_tests/extensions/turing test/unit_tests/extensions/turing/runtests.jl features
const PARTS = isempty(ARGS) ? ["features", "shared_cases"] : ARGS
if !all(part -> part in ("features", "shared_cases"), PARTS)
    throw(
        ArgumentError(
            "The parts of the suite are features and shared_cases, but $(ARGS) was given."
        ),
    )
end

#The tests of the Turing.jl extension, in an environment of their own, since Turing.jl holds FlexiChains.jl at an older
#version than the main suite tests. Automatic differentiation of its models is tested in test/ad_and_reactant/turing,
#which has its own environment and its own runner
@testset "Turing extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
    @test extension isa Module

    if "features" in PARTS
        #JET's test_package doesn't look inside extensions, so the extension is checked on its own. The reports about the
        #left-hand sides of ~ are left out, and everything else must be clean
        @testset "JET" begin
            result = JET.report_package(
                extension; toplevel_logger=nothing, target_modules=(extension,)
            )
            reports = filter(!is_about_undefined_local, JET.get_reports(result))
            @test isempty(reports)
            isempty(reports) || show(stdout, result)
        end

        #The example models with the distributions of Distributions.jl and their methods for Turing.jl
        include(joinpath(EXAMPLE_MODELS_DIRECTORY, "turing_runtests.jl"))

        include("extension_defaults.jl")
        include("turing_behavioral_session.jl")
        include("full_model.jl")
        include("submodels.jl")
        include("parameter_models.jl")
        include("get_session_trajectories.jl")
        include("predict_session_trajectories.jl")
        include("decondition_sessions.jl")
        include("condition_sessions.jl")
        include("discrete_and_continuous_variables.jl")
        include("vector_parameters.jl")
        include("multivariate_attributes.jl")
        include("outputs_of_any_type.jl")
    end

    #What holds for every inference case that the suites share
    "shared_cases" in PARTS && include("inference_cases.jl")
end
