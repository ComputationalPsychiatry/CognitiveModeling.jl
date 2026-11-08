using CognitiveModeling
using Test

#Packages that trigger extensions. CommonRLInterface.jl exports a reset! of its own, so only its name is brought in.
#The Turing.jl extension is tested in an environment of its own, in extensions/turing
using CommonRLInterface: CommonRLInterface
using FlexiChains: FlexiChains

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("defining_models/runtests.jl")
    include("simulation/runtests.jl")
    include("inference/runtests.jl")
    include("extensions/runtests.jl")
end
