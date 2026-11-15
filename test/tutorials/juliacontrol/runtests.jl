using Test

#The tutorial on the packages of JuliaControl, in an environment of its own, since the documentation shows its code
#without running it, to keep the documentation's environment small
@testset "JuliaControl tutorial" begin
    literate_dir = joinpath(@__DIR__, "..", "..", "..", "docs", "literate")
    include(joinpath(literate_dir, "interfacing_with_juliacontrol.jl"))

    #The tutorial simulates and fits its models, and what they gave is checked here
    include("results.jl")
end
