@testset "Displaying the trajectories of a session" begin
    #The trajectories of a session of two timesteps which saved its expected rewards and its rewards, with what the environment saved, the parameters' validity and the last timestep as given
    function trajectories_to_display(;
        environment=(;
            states=fill((;), 3),
            observations=[(; reward=BeforeFirstTimestep()), (; reward=1.0), (; reward=0.0)],
            actions=fill((;), 3),
            observation_distributions=fill((;), 3),
        ),
        parameters_valid=true,
        last_timestep=nothing,
    )
        return SessionTrajectories(
            [(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.6)],
            [(; reward=BeforeFirstTimestep()), (; reward=1.0), (; reward=0.0)],
            fill((;), 3),
            fill((;), 3),
            environment,
            1,
            parameters_valid,
            2,
            last_timestep,
        )
    end

    @testset "The REPL shows the timesteps, the names of what was saved of each kind, and whether the parameters stayed valid" begin
        @test repr("text/plain", trajectories_to_display()) == """
SessionTrajectories
Timesteps: 2, after timestep 0
States: expected_reward
Observations: reward
Actions: none
Action distributions: none
Environment states: none
Environment observations: reward
Environment actions: none
Environment observation distributions: none
Parameters valid: true"""
    end

    @testset "Trajectories without timestep 0 show the timesteps alone" begin
        without_timestep_0 = SessionTrajectories(
            [(; expected_reward=0.75), (; expected_reward=0.6)],
            fill((;), 2),
            fill((;), 2),
            fill((;), 2),
            NoEnvironment(),
            [1, 1],
            ValidByDefault(),
            2,
            nothing,
        )
        #A session without an environment says so, and the default of models that can't set parameters_valid shows as valid
        @test repr("text/plain", without_timestep_0) == """
SessionTrajectories
Timesteps: 2
States: expected_reward
Observations: none
Actions: none
Action distributions: none
Environment: none
Parameters valid: true"""
    end

    @testset "Invalid parameters point to the last timestep, when it is there" begin
        last_timestep = (;
            timestep=3,
            parameters_valid=false,
            previous_state=(; expected_reward=0.6),
            state=(; expected_reward=1.2),
            environment=(; parameters_valid=true, previous_state=(;), state=(;)),
        )
        @test endswith(
            repr(
                "text/plain",
                trajectories_to_display(; parameters_valid=false, last_timestep),
            ),
            "Parameters valid: false (see last_timestep)",
        )
        #A fitted session has no last timestep
        @test endswith(
            repr("text/plain", trajectories_to_display(; parameters_valid=false)),
            "Parameters valid: false",
        )
    end

    @testset "On one line, they show the number of timesteps" begin
        @test repr(trajectories_to_display()) == "SessionTrajectories (2 timesteps)"
        one_timestep = SessionTrajectories(
            [(;)], [(;)], [(;)], [(;)], NoEnvironment(), 1, true, 1, nothing
        )
        @test repr(one_timestep) == "SessionTrajectories (1 timestep)"
        #As inside a vector
        @test occursin(
            "SessionTrajectories (2 timesteps)", repr([trajectories_to_display()])
        )
    end

    @testset "Trajectories made by hand with other values are still shown" begin
        @test repr("text/plain", SessionTrajectories(1, 2, 3, 4, 5, 6, 7, 8, 9)) == """
SessionTrajectories
Timesteps: 8
States: none
Observations: none
Actions: none
Action distributions: none
Environment: none
Parameters valid: 7"""
    end
end
