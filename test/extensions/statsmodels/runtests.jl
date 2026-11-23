using StatsModels: @formula
using JET: JET

#The tests of the StatsModels.jl extension, which run with the main suite
@testset "StatsModels extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingStatsModelsExt)
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own
    @testset "JET" begin
        JET.test_package(extension; target_modules=(extension,))
    end

    include("hierarchical_regression_parameters.jl")
end
