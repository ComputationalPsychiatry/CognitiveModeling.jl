using JET
using TOML

@testset "JET" begin
    JET.test_package(CognitiveModeling; target_modules = (CognitiveModeling,))

    # test_package does not look inside extensions, so check each one separately.
    # Reactant has its own test environment, so its extension isn't loaded here
    project = TOML.parsefile(joinpath(pkgdir(CognitiveModeling), "Project.toml"))
    @testset "$name" for name in setdiff(keys(project["extensions"]), ["CognitiveModelingReactantExt"])
        extension = Base.get_extension(CognitiveModeling, Symbol(name))
        JET.test_package(extension; target_modules = (extension,))
    end
end
