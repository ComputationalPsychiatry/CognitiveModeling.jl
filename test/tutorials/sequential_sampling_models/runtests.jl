using Test

#The tutorial on the models of SequentialSamplingModels.jl, in an environment of its own, since the documentation shows
#its code without running it, to keep the documentation's environment small
@testset "SequentialSamplingModels tutorial" begin
    literate_dir = joinpath(@__DIR__, "..", "..", "..", "docs", "literate")
    include(joinpath(literate_dir, "interfacing_with_sequential_sampling_models.jl"))

    #The tutorial simulates and fits its model, and what they gave is checked here
    include("results.jl")
end
