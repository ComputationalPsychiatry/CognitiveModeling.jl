using CognitiveModeling
using CognitiveModeling: EnvironmentInput, EnvironmentOutput
using Test

#Packages that trigger extensions
using Turing

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("defining_models/runtests.jl")
    include("simulation/runtests.jl")
    include("extensions/runtests.jl")
end
