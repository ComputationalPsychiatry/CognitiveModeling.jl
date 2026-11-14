@testset "Reading the columns that a session parameter model uses" begin
    reader = "GroupedHierarchicalSessionParameters groups probability by"

    @testset "The value of a column is read for every session, in the order of the sessions" begin
        sessions_info = grouped_sessions()
        @test CognitiveModeling.read_one_value_of_each_session(
            sessions_info, :diagnosis, reader
        ) == ["clinical", "control", "clinical", "control"]
        @test CognitiveModeling.read_one_value_of_each_session(
            sessions_info, :site, reader
        ) == [1, 1, 2, 2]
        #The error starts with what reads the column
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column wave, which has the values [1, 2] in the session \"S1\". Each session must have one value of every column that a session parameter model reads." CognitiveModeling.read_one_value_of_each_session(
            sessions_info,
            :wave,
            "HierarchicalRegressionSessionParameters predicts probability from",
        )
    end

    @testset "The groups of the sessions are found from the table, in the order of their first session" begin
        sessions_info = grouped_sessions()
        @test CognitiveModeling.group_sessions(sessions_info, (:diagnosis,), reader) == (;
            group_of_each_session=[1, 2, 1, 2],
            n_groups=2,
            group_symbol_names=[:clinical, :control],
        )
        #With several columns, a group is a combination of their values
        @test CognitiveModeling.group_sessions(
            sessions_info, (:diagnosis, :site), reader
        ) == (;
            group_of_each_session=[1, 2, 3, 4],
            n_groups=4,
            group_symbol_names=[
                :diagnosis_clinical__site_1,
                :diagnosis_control__site_1,
                :diagnosis_clinical__site_2,
                :diagnosis_control__site_2,
            ],
        )
        #Without columns, every session is in one group, also when the sessions have no table
        @test CognitiveModeling.group_sessions(sessions_info, (), reader) == (;
            group_of_each_session=[1, 1, 1, 1],
            n_groups=1,
            group_symbol_names=[:all_sessions],
        )
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test CognitiveModeling.group_sessions(without_table, (), reader).n_groups == 1
    end

    @testset "A session must have one value of each column that is read, and the table must have the column" begin
        sessions_info = grouped_sessions()
        @test_throws "GroupedHierarchicalSessionParameters groups probability by the column wave, which has the values [1, 2] in the session \"S1\". Each session must have one value of every column that a session parameter model reads." CognitiveModeling.group_sessions(
            sessions_info, (:wave,), reader
        )
        with_missing = grouped_sessions(
            merge(GROUPS_TABLE, (; site=[1, 1, 1, 1, missing, 2]))
        )
        @test_throws "GroupedHierarchicalSessionParameters groups probability by the column site, which has the values Union{Missing, Int64}[missing] in the session \"S3\"." CognitiveModeling.group_sessions(
            with_missing, (:site,), reader
        )
        @test_throws "The table that the sessions were split from has no column age. Its columns are (:id, :diagnosis, :site, :wave, :outcome, :choice)." CognitiveModeling.group_sessions(
            sessions_info, (:age,), reader
        )
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column diagnosis." CognitiveModeling.group_sessions(
            without_table, (:diagnosis,), reader
        )
    end
end
