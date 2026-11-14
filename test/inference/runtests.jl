@testset "Inference" begin
    include("types.jl")
    include("functions_to_call.jl")
    include("session_parameter_models.jl")
    include("prepare_full_model.jl")
    include("preparing_sessions/runtests.jl")
    include("running_sessions/runtests.jl")
    include("display_sessions.jl")
end
