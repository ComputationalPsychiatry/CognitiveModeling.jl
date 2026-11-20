@testset "Reading what a session is set up with" begin
    without_environment = SessionInfo(
        SessionCounter(), COUNTER_REWARDS, COUNTER_CHOICES; who_starts=EnvironmentStarts()
    )
    with_environment = SessionInfo(
        SessionCounter(),
        COUNTER_REWARDS,
        COUNTER_CHOICES;
        who_starts=EnvironmentStarts(),
        environment=GivesReward(),
    )

    @testset "A session has an environment when it was given one, also when it doesn't include it" begin
        @test !CognitiveModeling.has_environment(without_environment)
        @test CognitiveModeling.has_environment(with_environment) &&
            !with_environment.includes_environment
        @test CognitiveModeling.has_environment(
            MultipleSessionsInfo([with_environment], ["A"])
        )
        @test !CognitiveModeling.has_environment(
            MultipleSessionsInfo([without_environment], ["A"])
        )
    end

    @testset "Environment parameters can only be given for sessions with an environment" begin
        @test CognitiveModeling.check_environment_parameters(
            with_environment, (; scale=2.0)
        ) === nothing
        @test CognitiveModeling.check_environment_parameters(without_environment, (;)) ===
            nothing
        @test_throws "Environment parameters were given, (scale = 2.0,), but the session has no environment." CognitiveModeling.check_environment_parameters(
            without_environment, (; scale=2.0)
        )
        #Several sessions are checked by their first
        @test CognitiveModeling.check_environment_parameters(
            MultipleSessionsInfo([with_environment], ["A"]), (; scale=[2.0])
        ) === nothing
        @test_throws "Environment parameters were given, (scale = [2.0],), but the session has no environment." CognitiveModeling.check_environment_parameters(
            MultipleSessionsInfo([without_environment], ["A"]), (; scale=[2.0])
        )
    end

    @testset "The sessions save something when a save history of any of them selects anything" begin
        @test !CognitiveModeling.sessions_save_anything(
            MultipleSessionsInfo([with_environment, with_environment], ["A", "B"])
        )
        saving_states = SessionInfo(
            with_environment; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        )
        @test CognitiveModeling.sessions_save_anything(
            MultipleSessionsInfo([saving_states], ["A"])
        )
        saving_rewards = SessionInfo(
            with_environment; environment_save_history=SaveHistory(; outputs=:reward)
        )
        @test CognitiveModeling.sessions_save_anything(
            MultipleSessionsInfo([saving_rewards], ["A"])
        )
    end

    @testset "The last timestep with data is the last one where any column has a value that isn't missing" begin
        @test CognitiveModeling.last_timestep_with_data((;
            reward=[1.0, MissingValue(), MissingValue()],
            choice=[true, false, MissingValue()],
        )) == 2
        #A marker for nothing happening counts as data
        @test CognitiveModeling.last_timestep_with_data((;
            choice=[true, MissingValue(), NoValue()]
        )) == 3
        @test CognitiveModeling.last_timestep_with_data((;
            choice=[MissingValue(), MissingValue()]
        )) == 0
    end
end
