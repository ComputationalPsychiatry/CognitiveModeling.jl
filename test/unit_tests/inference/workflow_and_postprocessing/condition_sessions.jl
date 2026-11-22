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

@testset "condition_sessions on sessions" begin
    #The sessions of COUNTER_TABLE, read with their column settings, in which a missing choice is imputed
    sessions_info = MultipleSessionsInfo(
        SessionCounter(),
        COUNTER_TABLE;
        observation_columns=:outcome => :reward,
        action_columns=:choice,
        session_columns=:id,
        timestep_sizes=:dt,
        who_starts=EnvironmentStarts(),
        cognitive_save_history=SaveHistory(; states=:n_timesteps),
    )

    @testset "MultipleSessionsInfo keeps the settings that read its table, and copies of it keep them too" begin
        settings = sessions_info.column_settings
        @test settings.observation_columns === (:outcome => :reward) &&
            settings.action_columns === :choice &&
            settings.session_columns === :id &&
            settings.timestep_sizes === :dt
        @test isequal(settings.missing_observation_values, (missing,)) &&
            settings.impute_missing_actions === true
        @test decondition_sessions(sessions_info).column_settings === settings
        @test MultipleSessionsInfo(sessions_info; cognitive_save_history=SaveHistory()).column_settings ===
            settings
        #Sessions that are given already prepared have none
        given = MultipleSessionsInfo(sessions_info.sessions, sessions_info.session_ids)
        @test given.column_settings === nothing
    end

    @testset "A table with the same sessions is read with those settings, and the sessions keep their other settings" begin
        other_data = (;
            id=["S1", "S2", "S2", "S2"],
            outcome=[0.0, 1.0, 0.0, 1.0],
            choice=[false, true, missing, false],
            dt=[2.0, 1.0, 1.0, 0.5],
        )
        conditioned = condition_sessions(sessions_info, other_data)
        @test conditioned.session_ids == ["S1", "S2"]
        @test map(session -> session.n_timesteps, conditioned.sessions) == [1, 3]
        @test conditioned.sessions[1].observations.reward == [0.0]
        @test conditioned.sessions[2].observations.reward == [1.0, 0.0, 1.0]
        #A missing value is imputed, as when the sessions were made, and the sizes of the timesteps are read from the column
        @test isequal(
            conditioned.sessions[2].actions.choice, [true, ImputedMissingValue(), false]
        )
        @test conditioned.sessions[2].timestep_sizes == [1.0, 1.0, 0.5]
        @test conditioned.sessions[1].who_starts === EnvironmentStarts() &&
            conditioned.sessions[1].cognitive_save_history ===
              SaveHistory(; states=:n_timesteps)
        @test conditioned.column_settings === sessions_info.column_settings &&
            conditioned.table === other_data
        #check_stopping_criterion is given to the sessions
        @test all(
            session -> session.check_stopping_criterion === false,
            condition_sessions(sessions_info, other_data; check_stopping_criterion=false).sessions,
        )
    end

    @testset "A table with other sessions, and sessions without column settings, are errors" begin
        other_sessions = (;
            id=["S2", "S1"], outcome=[1.0, 0.0], choice=[true, false], dt=[1.0, 1.0]
        )
        @test_throws "The table has the sessions [\"S2\", \"S1\"], but the model has [\"S1\", \"S2\"]" condition_sessions(
            sessions_info, other_sessions
        )
        given = MultipleSessionsInfo(sessions_info.sessions, sessions_info.session_ids)
        @test_throws "The sessions were given already prepared, without the column settings that read them from a table" condition_sessions(
            given, COUNTER_TABLE
        )
    end

    @testset "A single session takes observations and actions, in the forms SessionInfo takes them" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            check_parameter_domains=false,
        )
        conditioned = condition_sessions(
            session_info, (; reward=[0.0, 1.0]), [false, missing]
        )
        @test conditioned.n_timesteps == 2
        @test conditioned.observations.reward == [0.0, 1.0]
        @test conditioned.actions.choice[2] === ImputedMissingValue()
        @test conditioned.who_starts === AgentStarts() &&
            conditioned.check_parameter_domains === false
        #Other sizes of the timesteps can be given, as for data with other timesteps
        sized = SessionInfo(session_info; cognitive_save_history=SaveHistory())
        @test condition_sessions(
            sized, [0.0, 1.0], [false, true]; timestep_sizes=[0.5, 2.0]
        ).timestep_sizes == [0.5, 2.0]
    end
end
