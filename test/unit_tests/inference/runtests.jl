@testset "Inference" begin
    include("types.jl")
    include("functions_to_call.jl")
    include("parameter_models.jl")
    include("utility_functions/runtests.jl")
    include("workflow_and_postprocessing/runtests.jl")
end
