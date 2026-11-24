@testset "Utility functions" begin
    include("read_selection.jl")
    include("select_values_to_save.jl")
    include("check_selection_names.jl")
    include("timestep_sizes.jl")
    include("storage.jl")
    include("display_save_trajectories.jl")
    include("display_session_trajectories.jl")
end
