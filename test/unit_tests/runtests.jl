#The tests of each part of the package, in folders that follow src/. The tests of the extensions are in test/extensions
@testset "Unit tests" begin
    include("defining_models/runtests.jl")
    include("saving_trajectories/runtests.jl")
    include("simulation/runtests.jl")
    include("inference/runtests.jl")
end
