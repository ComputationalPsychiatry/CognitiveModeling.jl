@testset "Preparing models" begin
    include("prepare_full_model.jl")
    include("prepare_session_data.jl")
    include("split_table_into_sessions.jl")
    include("include_environment.jl")
    include("check_session_identifiers.jl")
    include("read_session_columns.jl")
    include("display_sessions.jl")
end
