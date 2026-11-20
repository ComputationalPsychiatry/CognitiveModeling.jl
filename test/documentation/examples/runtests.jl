using Test

#The examples in docs/examples, which the documentation's build doesn't run, each run to the end here and followed by
#the checks of its results, in the folder named after it, so that a change of the package which breaks one fails here.
#This runner takes the tier of the tests as its argument, fast without one, and runs the examples of that tier in
#docs/how_scripts_are_run.toml, as in
#    julia --project=test/documentation/examples test/documentation/examples/runtests.jl fast
include(joinpath(@__DIR__, "..", "how_scripts_are_run.jl"))
const TIER = tier_to_test(ARGS)

@testset "Examples in docs/examples, $TIER tier" begin
    for name in sort(collect(keys(HOW_SCRIPTS_ARE_RUN["examples"])))
        HOW_SCRIPTS_ARE_RUN["examples"][name]["test_tier"] == TIER || continue
        @testset "$name" begin
            run_script_and_check_results(
                joinpath(DOCS_DIRECTORY, "examples", name),
                joinpath(@__DIR__, splitext(name)[1], "results.jl"),
            )
        end
    end
end
