using Tables: Tables
using JET: JET

#The tests of the Tables.jl extension, which run with the main suite
@testset "Tables extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTablesExt)
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own
    @testset "JET" begin
        JET.test_package(extension; target_modules=(extension,))
    end

    include("column_table.jl")
end
