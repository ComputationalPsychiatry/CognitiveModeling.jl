using CognitiveModeling
using CognitiveModeling: apply_model, initialize_state, turing_apply_model, turing_initialize_state
using Test
using JET: JET
using Random: Xoshiro
using Turing

#Whether a report is about a local variable being undefined. DynamicPPL's ~ expands to code which reads
#its left-hand side before assigning it, on a branch that is never taken, so JET reports every ~ this way
is_about_undefined_local(report) = report isa JET.UndefVarErrorReport && report.var isa Symbol

#The tests of the Turing.jl extension, in an environment of their own, since Turing.jl holds FlexiChains.jl at an older
#version than the main suite tests. Automatic differentiation of its models is tested in the ad folder, which has its own
#environment and its own runner
@testset "Turing extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own. The reports about the
    #left-hand sides of ~ are left out, and everything else must be clean
    @testset "JET" begin
        result = JET.report_package(extension; toplevel_logger = nothing, target_modules = (extension,))
        reports = filter(!is_about_undefined_local, JET.get_reports(result))
        @test isempty(reports)
        isempty(reports) || show(stdout, result)
    end

    include("extension_defaults.jl")
    include("turing_behavioural_session.jl")
    include("turing_full_model.jl")
    include("session_values_chain.jl")
end
