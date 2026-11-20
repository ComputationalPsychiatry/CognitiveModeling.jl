#The tests of each part of the package, in folders that follow src/, with the extensions in extensions/
@testset "Unit tests" begin
    include("defining_models/runtests.jl")
    include("model_history/runtests.jl")
    include("simulation/runtests.jl")
    include("inference/runtests.jl")
    include("extensions/runtests.jl")
end
