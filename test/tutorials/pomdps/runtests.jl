using Test

#The tutorial on POMDPs.jl, in an environment of its own, since the documentation shows its code without running it, to
#keep the documentation's environment small
@testset "POMDPs tutorial" begin
    literate_dir = joinpath(@__DIR__, "..", "..", "..", "docs", "literate")
    include(joinpath(literate_dir, "interfacing_with_pomdps.jl"))

    #The tutorial doesn't simulate its models, so they are simulated here
    include("simulations.jl")
end
