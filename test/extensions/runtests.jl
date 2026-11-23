#The tests of each extension, in a folder named after it. Those of the Turing.jl extension are in turing/, with an
#environment and a runner of their own, and its automatic differentiation in turing/ad, also with its own
@testset "Extensions" begin
    include("common_rl_interface/runtests.jl")
    include("flexichains/runtests.jl")
    include("statsmodels/runtests.jl")
    include("tables/runtests.jl")
end
