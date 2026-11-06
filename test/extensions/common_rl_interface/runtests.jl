using CommonRLInterface: CommonRLInterface

#The tests of the CommonRLInterface.jl extension, which run with the main suite
@testset "CommonRLInterface extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingCommonRLInterfaceExt) isa Module

    include("common_rl_environment.jl")
end
