@testset "What condition_sessions is made of" begin
    @testset "with_data prepares a session again with other data and its own settings" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=fill(SimulatedMissingValue(), 3));
            who_starts=AgentStarts(),
            timestep_sizes=[1.0, 2.0, 1.0],
            check_parameter_domains=false,
        )
        with_choices = CognitiveModeling.with_data(
            session_info,
            session_info.observations,
            (; choice=[true, false, true]);
            check_stopping_criterion=nothing,
        )
        @test with_choices.actions.choice == [true, false, true]
        @test with_choices.actions_to_condition_on.choice == [true, false, true]
        @test with_choices.timestep_sizes == [1.0, 2.0, 1.0]
        @test with_choices.who_starts === AgentStarts()
        @test with_choices.check_parameter_domains === false
        #The stopping criterion is checked from the data the copy has, after its last timestep with data
        @test with_choices.first_timestep_to_check_stopping == 4
        #Data with other timesteps is given their sizes
        with_two_timesteps = CognitiveModeling.with_data(
            session_info,
            map(column -> column[1:2], session_info.observations),
            (; choice=[true, false]);
            check_stopping_criterion=nothing,
            timestep_sizes=[1.0, 2.0],
        )
        @test with_two_timesteps.n_timesteps == 2 &&
            with_two_timesteps.timestep_sizes == [1.0, 2.0]
    end

    @testset "cut_after_last_timestep_with_data leaves out the timesteps after the last one with data" begin
        observations = (;
            reward=Any[1.0, 0.0, SimulatedMissingValue(), SimulatedMissingValue()]
        )
        actions = (;
            choice=Any[
                true,
                SimulatedMissingValue(),
                SimulatedMissingValue(),
                SimulatedMissingValue(),
            ],
        )
        #The second timestep is the last with a value, the reward, so the action without a value on it stays
        data = CognitiveModeling.cut_after_last_timestep_with_data(
            observations, actions, [0.5, 1.0, 2.0, 4.0]
        )
        @test data.observations == (; reward=Any[1.0, 0.0])
        @test data.actions == (; choice=Any[true, SimulatedMissingValue()])
        @test data.timestep_sizes == [0.5, 1.0]
        #One size for every timestep is kept as it is
        @test CognitiveModeling.cut_after_last_timestep_with_data(
            observations, actions, 0.5
        ).timestep_sizes === 0.5
        #A value that is imputed isn't data either, and any other marker is, since it says that the timestep happened
        with_markers = (;
            choice=Any[true, NoValue(), ImputedMissingValue(), SimulatedMissingValue()]
        )
        @test CognitiveModeling.cut_after_last_timestep_with_data(
            (; reward=fill(SimulatedMissingValue(), 4)), with_markers, 1
        ).actions == (; choice=Any[true, NoValue()])
        #Data with a value on its last timestep, and data without any value, keep every timestep
        full = (; choice=Any[true, false, true, false])
        @test CognitiveModeling.cut_after_last_timestep_with_data(
            observations, full, [0.5, 1.0, 2.0, 4.0]
        ) == (; observations, actions=full, timestep_sizes=[0.5, 1.0, 2.0, 4.0])
        nothing_given = (; choice=fill(SimulatedMissingValue(), 4))
        uncut = CognitiveModeling.cut_after_last_timestep_with_data(
            (; reward=fill(SimulatedMissingValue(), 4)), nothing_given, [0.5, 1.0, 2.0, 4.0]
        )
        @test uncut.actions === nothing_given && length(uncut.timestep_sizes) == 4
    end

    @testset "with_common_element_types gives the columns of a name the types of their entries in every session" begin
        columns_of_sessions = [
            (; choice=Any[true, SimulatedMissingValue()], bet=Any[1.0, 2.0]),
            (; choice=Any[false, true], bet=Any[3, 4.0]),
        ]
        typed = CognitiveModeling.with_common_element_types(columns_of_sessions)
        @test all(
            columns -> columns.choice isa Vector{Union{SimulatedMissingValue,Bool}}, typed
        )
        @test all(columns -> columns.bet isa Vector{Union{Int,Float64}}, typed)
        @test typed[1].choice == [true, SimulatedMissingValue()] && typed[2].bet == [3, 4.0]
        #Columns whose entries have one type in every session get that type
        @test only(
            CognitiveModeling.with_common_element_types([(; choice=Any[true, false])])
        ).choice isa Vector{Bool}
    end
end
