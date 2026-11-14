@testset "Preparing sessions" begin
    include("prepare_session_data.jl")
    include("split_table_into_sessions.jl")
    include("include_environment.jl")
    include("check_session_identifiers.jl")
    include("read_session_columns.jl")
end
