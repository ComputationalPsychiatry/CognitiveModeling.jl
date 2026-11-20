@testset "Displaying sessions" begin
    @testset "The REPL shows a summary of a session" begin
        @test repr(
            "text/plain",
            SessionInfo(
                SessionCounter(),
                COUNTER_REWARDS,
                COUNTER_CHOICES;
                who_starts=EnvironmentStarts(),
            ),
        ) == """
SessionInfo
Cognitive model: SessionCounter
Environment: none
Timesteps: 3
Observations: reward
Actions: choice
Who starts: environment"""
        @test repr(
            "text/plain",
            SessionInfo(
                SessionCounter(),
                (; reward=[1.0, MissingValue(), 1.0]),
                COUNTER_CHOICES;
                who_starts=AgentStarts(),
                environment=GivesReward(),
            ),
        ) == """
SessionInfo
Cognitive model: SessionCounter
Environment: GivesReward
Timesteps: 3
Observations: reward
Actions: choice
Who starts: agent"""
    end

    @testset "The REPL shows a summary of several sessions" begin
        @test repr(
            "text/plain",
            MultipleSessionsInfo(
                SessionCounter(),
                COUNTER_TABLE;
                observation_columns=:outcome => :reward,
                action_columns=:choice,
                session_columns=:id,
                who_starts=EnvironmentStarts(),
            ),
        ) == """
MultipleSessionsInfo
Sessions: 2
Cognitive model: SessionCounter
Environment: none
Observations: reward
Actions: choice"""
    end
end
