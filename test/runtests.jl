using CognitiveModeling
using Test

#Packages that trigger extensions. CommonRLInterface.jl exports a reset! of its own, so only its name is brought in.
#The Turing.jl extension is tested in an environment of its own, in unit_tests/extensions/turing
using CommonRLInterface: CommonRLInterface
using FlexiChains: FlexiChains
using Tables: Tables

#The example models that the suites share
include("example_models/ExampleModels.jl")
using .ExampleModels

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("example_models/runtests.jl")
    include("unit_tests/runtests.jl")
end
