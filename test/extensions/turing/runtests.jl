using CognitiveModeling: apply_model, initialize_state, turing_apply_model, turing_initialize_state
using Random: Xoshiro

#The tests of the Turing.jl extension which run with the main suite. Automatic differentiation
#of its models is tested in the ad folder, which has its own environment and its own runner
@testset "Turing extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt) isa Module

    include("extension_defaults.jl")
    include("turing_behavioural_session.jl")
    include("turing_full_model.jl")
end
