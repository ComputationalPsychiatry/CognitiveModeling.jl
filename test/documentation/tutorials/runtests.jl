using Test

#The tutorials that the documentation's build runs, each run here in the documentation's environment and followed by the
#checks of its results, in the folder named after it. A tutorial that the build doesn't run has an environment and a
#runner of its own in its folder. Run it with the tier as its argument, fast when it has none:
#    julia --project=docs test/documentation/tutorials/runtests.jl fast
include(joinpath(@__DIR__, "..", "how_scripts_are_run.jl"))
const TIER = tier_to_test(ARGS)

@testset "Tutorials in the documentation's environment, $TIER tier" begin
    for name in sort(collect(keys(HOW_SCRIPTS_ARE_RUN["tutorials"])))
        settings = HOW_SCRIPTS_ARE_RUN["tutorials"][name]
        settings["run_in_build"] == "never" && continue
        settings["test_tier"] == TIER || continue
        @testset "$name" begin
            run_script_and_check_results(
                joinpath(DOCS_DIRECTORY, "literate", name),
                joinpath(@__DIR__, splitext(name)[1], "results.jl"),
            )
        end
    end
end
