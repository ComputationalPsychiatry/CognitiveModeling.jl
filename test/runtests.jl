using CognitiveModeling
using Test

#Packages that trigger extensions. CommonRLInterface.jl exports a reset! of its own, so only its name is brought in.
#The Turing.jl extension is tested in an environment of its own, in unit_tests/extensions/turing
using CommonRLInterface: CommonRLInterface
using FlexiChains: FlexiChains
using Tables: Tables

#The code that the suites share, the package in common/: the models of the tests, with their pairs, and the cases
using CommonTestFunctionality

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("common/test/runtests.jl")
    include("unit_tests/runtests.jl")
end
