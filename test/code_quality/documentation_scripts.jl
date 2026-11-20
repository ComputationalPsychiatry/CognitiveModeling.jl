using TOML: TOML

#Every script of the documentation has its settings in docs/how_scripts_are_run.toml, with values that the build and the
#tests know, and a folder for its tests in test/documentation with the checks of its results. A tutorial that the build
#doesn't run has an environment and a runner of its own there, and one that it runs has neither, since the runner in the
#documentation's environment runs it
@testset "Every script of the documentation has its settings and its tests" begin
    root = joinpath(@__DIR__, "..", "..")
    how_scripts_are_run = TOML.parsefile(joinpath(root, "docs", "how_scripts_are_run.toml"))
    for (kind, folder) in (("tutorials", "literate"), ("examples", "examples"))
        @testset "$kind" begin
            scripts = sort(filter(endswith(".jl"), readdir(joinpath(root, "docs", folder))))
            @test sort(collect(keys(how_scripts_are_run[kind]))) == scripts
            for name in scripts
                settings = how_scripts_are_run[kind][name]
                tests = joinpath(root, "test", "documentation", kind, splitext(name)[1])
                @test settings["test_tier"] in ("fast", "long", "none")
                @test isfile(joinpath(tests, "results.jl"))
                if kind == "tutorials"
                    @test settings["run_in_build"] in ("always", "for_releases", "never")
                    has_own_environment = settings["run_in_build"] == "never"
                    @test isfile(joinpath(tests, "Project.toml")) == has_own_environment
                    @test isfile(joinpath(tests, "runtests.jl")) == has_own_environment
                end
            end
        end
    end
end
