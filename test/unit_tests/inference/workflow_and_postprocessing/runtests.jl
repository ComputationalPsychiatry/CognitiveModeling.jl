#The tests of what is done with the sessions and the models of a fit, with the models and the data of types.jl
@testset "Workflow and postprocessing" begin
    include("decondition_sessions.jl")
    include("condition_sessions.jl")
end
