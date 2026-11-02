using CognitiveModeling
using Test

# Load the packages that trigger extensions, so that the code quality checks can see the extensions
using Turing

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("extensions/runtests.jl")
    include("premade_models/runtests.jl")
end
