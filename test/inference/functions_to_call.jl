@testset "Functions that users call to read the sessions" begin
    sessions_info = MultipleSessionsInfo(
        SessionCounter(),
        COUNTER_TABLE;
        observation_columns=:outcome,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    )

    @testset "The sessions are counted, and the index of a session is found by its identifier" begin
        @test CognitiveModeling.n_sessions(sessions_info) == 2
        @test CognitiveModeling.session_index(sessions_info, "S2") == 2
        @test_throws "There is no session with the identifier \"S3\". The identifiers are [\"S1\", \"S2\"]." CognitiveModeling.session_index(
            sessions_info, "S3"
        )
    end

    @testset "The symbol name of a session is its identifier, or the name and the value of each of several columns" begin
        @test CognitiveModeling.session_id_to_symbol("S1") === :S1
        @test CognitiveModeling.session_id_to_symbol(2) === Symbol("2")
        @test CognitiveModeling.session_id_to_symbol((; id="S1", block=2)) ===
            :id_S1__block_2
    end

    @testset "The values of a column are read for each session, from the table the sessions were split from" begin
        @test CognitiveModeling.session_column_values(sessions_info, :dt) ==
            [[1.0, 0.5], [1.0, 2.0, 1.0]]
        @test_throws "The table that the sessions were split from has no column age." CognitiveModeling.session_column_values(
            sessions_info, :age
        )
        #Sessions that were given already prepared have no table to read
        prepared = MultipleSessionsInfo(sessions_info.sessions, sessions_info.session_ids)
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column dt." CognitiveModeling.session_column_values(
            prepared, :dt
        )
    end
end
