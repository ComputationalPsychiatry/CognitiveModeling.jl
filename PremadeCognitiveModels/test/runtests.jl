using PremadeCognitiveModels
using Test

@testset "PremadeCognitiveModels.jl" begin
    include("code_quality/runtests.jl")
    include("models/runtests.jl")
end
