@testset "Including the environment in a session, or leaving it out" begin
    @testset "With sampled environment parameters, every session with an environment runs it" begin
        #S1 has nothing missing, so it doesn't run its environment until the parameters are sampled
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !sessions_info.sessions[1].includes_environment
        @test CognitiveModeling.include_environments_if_estimated(
            NoParameters(), sessions_info
        ) === sessions_info
        @test CognitiveModeling.include_environments_if_estimated(
            FixedSessionParameters((; scale=2.0)), sessions_info
        ) === sessions_info
        for sampling in (
            IdenticalSessionParameters((; scale=1)),
            IndependentSessionParameters((; scale=1)),
            ParametersModelOfOwn(),
        )
            sampled = CognitiveModeling.include_environments_if_estimated(
                sampling, sessions_info
            )
            @test all(session -> session.includes_environment, sampled.sessions)
            @test sampled.session_ids == sessions_info.session_ids
            @test sampled.sessions[1].observations ===
                sessions_info.sessions[1].observations
        end
        #Sessions without an environment still have none to run
        without = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test !any(
            session -> session.includes_environment,
            CognitiveModeling.include_environments_if_estimated(
                IdenticalSessionParameters((; scale=1)), without
            ).sessions,
        )
    end
end
