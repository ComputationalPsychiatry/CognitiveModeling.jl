using Test

#The tutorial on the models of SequentialSamplingModels.jl, in an environment of its own, since the documentation shows
#its code without running it, to keep the documentation's environment small. This runner takes the tier of the tests as
#its argument, fast without one, which must be the tutorial's tier in docs/how_scripts_are_run.toml, as in
#    julia --project=test/documentation/tutorials/interfacing_with_sequential_sampling_models test/documentation/tutorials/interfacing_with_sequential_sampling_models/runtests.jl fast
include(joinpath(@__DIR__, "..", "..", "how_scripts_are_run.jl"))
check_tier("tutorials", "interfacing_with_sequential_sampling_models.jl", ARGS)

@testset "SequentialSamplingModels tutorial" begin
    run_script_and_check_results(
        joinpath(
            DOCS_DIRECTORY, "literate", "interfacing_with_sequential_sampling_models.jl"
        ),
        joinpath(@__DIR__, "results.jl"),
    )
end
