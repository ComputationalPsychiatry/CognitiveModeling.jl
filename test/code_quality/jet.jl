using JET

@testset "JET" begin
    JET.test_package(PremadeCognitiveModels; target_modules=(PremadeCognitiveModels,))
end
