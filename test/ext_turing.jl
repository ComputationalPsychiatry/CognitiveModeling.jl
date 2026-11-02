@testset "Turing extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt) isa Module
end
