using CognitiveModeling: apply_model, turing_apply_model
using Random: Xoshiro

#The tests of the Turing.jl extension which run with the main suite. Automatic differentiation
#of its models is tested in the ad folder, which has its own environment and its own runner
@testset "Turing extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt) isa Module

    include("turing_apply_model.jl")
    include("turing_single_session.jl")
end
