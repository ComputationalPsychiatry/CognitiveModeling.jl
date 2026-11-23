using CommonRLInterface: CommonRLInterface
using JET: JET

#The tests of the CommonRLInterface.jl extension, which run with the main suite
@testset "CommonRLInterface extension" begin
    extension = Base.get_extension(
        CognitiveModeling, :CognitiveModelingCommonRLInterfaceExt
    )
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own
    @testset "JET" begin
        JET.test_package(extension; target_modules=(extension,))
    end

    include("common_rl_environment.jl")
end
