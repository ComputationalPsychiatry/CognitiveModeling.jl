using CognitiveModeling
using Test

#Packages that trigger extensions
using Turing
using CommonRLInterface

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("defining_models/runtests.jl")
    include("simulation/runtests.jl")
    include("inference/runtests.jl")
    include("extensions/runtests.jl")
end
