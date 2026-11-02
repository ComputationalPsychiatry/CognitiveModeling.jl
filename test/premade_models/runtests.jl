# Runs the tests of all premade models. Each model gets its own test file in this folder,
# and new files are picked up automatically. Also add each model to
# ../ad_and_reactant/test_cases.jl, so that the AD and Reactant tests check it.
@testset "Premade models" begin
    @test PremadeModels isa Module

    for (root, _, files) in walkdir(@__DIR__), file in files
        if endswith(file, ".jl") && file != "runtests.jl"
            include(joinpath(root, file))
        end
    end
end
