using JET
using TOML

#Whether a report is about a local variable being undefined. DynamicPPL's ~ expands to code which reads
#its left-hand side before assigning it, on a branch that is never taken, so JET reports every ~ this way
is_about_undefined_local(report) = report isa JET.UndefVarErrorReport && report.var isa Symbol

@testset "JET" begin
    JET.test_package(CognitiveModeling; target_modules = (CognitiveModeling,))

    # test_package does not look inside extensions, so check each one separately
    project = TOML.parsefile(joinpath(pkgdir(CognitiveModeling), "Project.toml"))
    @testset "$name" for name in keys(project["extensions"])
        extension = Base.get_extension(CognitiveModeling, Symbol(name))
        if name == "CognitiveModelingTuringExt"
            #The reports about the left-hand sides of ~ are left out, and everything else must be clean
            result = JET.report_package(extension; toplevel_logger = nothing, target_modules = (extension,))
            reports = filter(!is_about_undefined_local, JET.get_reports(result))
            @test isempty(reports)
            isempty(reports) || show(stdout, result)
        else
            JET.test_package(extension; target_modules = (extension,))
        end
    end
end
