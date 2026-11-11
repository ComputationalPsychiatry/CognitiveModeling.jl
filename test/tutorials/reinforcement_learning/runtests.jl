using Test

#The tutorial on ReinforcementLearning.jl, in an environment of its own, since the documentation shows its code without
#running it, because ReinforcementLearning.jl is too large for the documentation's environment
@testset "Reinforcement learning tutorial" begin
    literate_dir = joinpath(@__DIR__, "..", "..", "..", "docs", "literate")
    include(joinpath(literate_dir, "interfacing_with_reinforcement_learning.jl"))

    #The tutorial doesn't simulate its models, so they are simulated here
    include("simulations.jl")
end
