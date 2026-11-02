using CognitiveModeling
using Test
using Turing

@testset "CognitiveModeling.jl" begin
    include("aqua.jl")
    include("jet.jl")
    include("premade_models.jl")
    include("ext_turing.jl")
end
