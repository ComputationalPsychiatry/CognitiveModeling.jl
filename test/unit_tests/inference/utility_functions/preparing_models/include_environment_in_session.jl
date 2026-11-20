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
            FixedParameters((; scale=2.0)), sessions_info
        ) === sessions_info
        for sampling in (
            IdenticalParameters((; scale=1)),
            IndependentParameters((; scale=1)),
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
                IdenticalParameters((; scale=1)), without
            ).sessions,
        )
    end

    @testset "A session that is made again keeps whether its parameters are checked against their domains" begin
        for check_parameter_domains in (true, false)
            session_info = SessionInfo(
                SessionCounter(),
                COUNTER_REWARDS,
                COUNTER_CHOICES;
                who_starts=EnvironmentStarts(),
                environment=GivesReward(),
                check_parameter_domains,
            )
            including = CognitiveModeling.remake_session_including_environment(session_info)
            without = CognitiveModeling.remake_session_without_environment(session_info)
            @test including.check_parameter_domains === check_parameter_domains
            @test without.check_parameter_domains === check_parameter_domains
        end
    end
end
