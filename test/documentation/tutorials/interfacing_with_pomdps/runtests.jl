using Test

#The tutorial on POMDPs.jl, in an environment of its own, since the documentation shows its code without running it, to
#keep the documentation's environment small. The tutorial doesn't simulate its models, so its results.jl does. This
#runner takes the tier of the tests as its argument, fast without one, which must be the tutorial's tier in
#docs/how_scripts_are_run.toml, as in
#    julia --project=test/documentation/tutorials/interfacing_with_pomdps test/documentation/tutorials/interfacing_with_pomdps/runtests.jl fast
include(joinpath(@__DIR__, "..", "..", "how_scripts_are_run.jl"))
check_tier("tutorials", "interfacing_with_pomdps.jl", ARGS)

@testset "POMDPs tutorial" begin
    run_script_and_check_results(
        joinpath(DOCS_DIRECTORY, "literate", "interfacing_with_pomdps.jl"),
        joinpath(@__DIR__, "results.jl"),
    )
end
