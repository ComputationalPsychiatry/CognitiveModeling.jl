using JET

@testset "JET" begin
    JET.test_package(CognitiveModeling; target_modules = (CognitiveModeling,))

    # test_package does not look inside extensions, so check them separately
    turing_ext = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
    JET.test_package(turing_ext; target_modules = (turing_ext,))
end
