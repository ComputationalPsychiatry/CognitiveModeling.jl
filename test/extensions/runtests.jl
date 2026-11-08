#The tests of the Turing.jl extension are in turing/, with an environment and a runner of their own
@testset "Extensions" begin
    include("common_rl_interface/runtests.jl")
    include("flexichains/runtests.jl")
end
