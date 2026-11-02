using CognitiveModeling
using Test

# Load the packages that trigger extensions, so that the code quality checks can see the extensions
using Turing

# Include every test file in a folder of the test suite, so new files only need to be added there
function include_folder(folder)
    for (root, _, files) in walkdir(joinpath(@__DIR__, folder)), file in filter(endswith(".jl"), files)
        include(joinpath(root, file))
    end
    return nothing
end

@testset "CognitiveModeling.jl" begin
    @testset "Code quality" begin
        include_folder("code_quality")
    end
    @testset "Extensions" begin
        include_folder("extensions")
    end
    @testset "Premade models" begin
        include_folder("premade_models")
    end
end
