using CognitiveModeling
using CognitiveModeling: ModelInput, ModelOutput, EnvironmentInput, EnvironmentOutput
using Test

# Load the packages that trigger extensions, so that the code quality checks can see the extensions
using Turing

@testset "CognitiveModeling.jl" begin
    include("code_quality/runtests.jl")
    include("defining_models/runtests.jl")
    include("simulation/runtests.jl")
    include("extensions/runtests.jl")
end
