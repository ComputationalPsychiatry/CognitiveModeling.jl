@testset "Simulation" begin
    include("types.jl")
    include("simulate.jl")
    include("fixed_observations_environment.jl")
    include("display_name_maps_and_instantiated_models.jl")

    #What holds for every simulation case that the suites share
    include("simulation_cases.jl")
end
