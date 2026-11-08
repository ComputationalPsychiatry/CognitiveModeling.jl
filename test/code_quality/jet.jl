using JET

#The extensions are checked by the runners of their own tests, since test_package doesn't look inside them
@testset "JET" begin
    JET.test_package(CognitiveModeling; target_modules = (CognitiveModeling,))
end
