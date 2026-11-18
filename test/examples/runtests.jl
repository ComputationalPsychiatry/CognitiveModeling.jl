using Test

#The examples in docs/examples, which the documentation doesn't run, each run to the end in a module of its own, so
#that a change of the package which breaks one fails here
@testset "Examples in docs/examples" begin
    examples_dir = joinpath(@__DIR__, "..", "..", "docs", "examples")
    for file_name in sort(filter(endswith(".jl"), readdir(examples_dir)))
        @testset "$file_name" begin
            example = Module(Symbol(splitext(file_name)[1]))
            Base.include(example, joinpath(examples_dir, file_name))
            @test true
        end
    end
end
