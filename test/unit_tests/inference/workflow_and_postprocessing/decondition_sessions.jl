@testset "decondition_sessions" begin
    @testset "Every action is simulated, and the observations stay those in the data without an environment" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, missing, true]);
            who_starts=AgentStarts(),
            timestep_sizes=[1.0, 2.0, 1.0],
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            impute_missing_actions=false,
        )
        predictive = decondition_sessions(session_info)
        @test predictive.actions == (; choice=fill(SimulatedMissingValue(), 3))
        @test predictive.observations === session_info.observations
        @test isempty(predictive.actions_to_condition_on) &&
            !predictive.includes_environment
        #The observations are still data, but the session doesn't score them, so only the actions say where the data continues, and none of them is data
        @test predictive.last_timestep_with_observation == 3 &&
            predictive.last_timestep_with_action == 0
        @test !CognitiveModeling.data_continues_after(predictive, 0)
        #Everything else is kept
        @test predictive.who_starts === AgentStarts() && predictive.n_timesteps == 3
        @test predictive.timestep_sizes == [1.0, 2.0, 1.0]
        @test predictive.cognitive_save_history === session_info.cognitive_save_history
        @test predictive.previous_observation === session_info.previous_observation
        @test predictive.check_parameter_domains === true
        #A session that doesn't check its parameters against their domains still doesn't
        unchecked = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            check_parameter_domains=false,
        )
        @test decondition_sessions(unchecked).check_parameter_domains === false
    end

    @testset "With an environment, every observation is simulated too, so the environment runs" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        predictive = decondition_sessions(session_info)
        @test predictive.observations == (; reward=fill(SimulatedMissingValue(), 3))
        @test predictive.includes_environment && predictive.environment === GivesReward()
        #With no values in the data, a stopping criterion can end the copy on any timestep without making the parameters invalid, as in a simulation
        @test predictive.last_timestep_with_observation == 0 &&
            predictive.last_timestep_with_action == 0
        @test !CognitiveModeling.data_continues_after(predictive, 0)
    end

    @testset "Several sessions are each copied, with their data identifiers" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        predictive = decondition_sessions(sessions_info)
        @test predictive.session_ids == sessions_info.session_ids
        @test [session.n_timesteps for session in predictive.sessions] == [2, 3]
        @test all(
            session -> all(==(SimulatedMissingValue()), session.actions.choice),
            predictive.sessions,
        )
    end
end
