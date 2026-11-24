@testset "condition_sessions on a Turing.jl model" begin
    #The deconditioned full model of sessions of one, two and five timesteps, and a draw from its prior, whose choices
    #are simulated with the draw's parameters
    sessions_info = sessions_for_test(
        WithDefaultParameters(DistributionsJl()), DIFFERENT_LENGTHS_TABLE
    )
    imputing = full_model(
        TuringBackend(), sessions_info, IndependentParameters(FULL_PRIORS)
    )
    deconditioned = decondition_sessions(imputing)
    draw = @test_logs match_mode = :any rand(Xoshiro(1), deconditioned)
    simulated_choices = Turing.DynamicPPL.subset(draw, [@varname(simulated_sessions)])
    parameter_names = [
        @varname(learning_rate.sessions), @varname(inverse_temperature.sessions)
    ]
    #Two sets of parameters to evaluate the models at
    some_parameters = Dict(
        @varname(learning_rate.sessions) => [0.2, 0.3, 0.4],
        @varname(inverse_temperature.sessions) => [1.0, 2.0, 3.0],
    )
    other_parameters = Dict(
        @varname(learning_rate.sessions) => [0.9, 0.5, 0.1],
        @varname(inverse_temperature.sessions) => [4.0, 0.5, 1.5],
    )

    @testset "The simulated choices are the data of the copy, which has only the parameters as variables" begin
        conditioned = condition_sessions(deconditioned, simulated_choices)
        @test conditioned.f === deconditioned.f
        for (i, n_timesteps) in enumerate((1, 2, 5))
            @test conditioned.args.sessions_info.sessions[i].actions.choice == [
                draw[@varname(simulated_sessions[i].timesteps[t].choice.simulated)] for
                t in 1:n_timesteps
            ]
        end
        draws = @test_logs match_mode = :any sample(
            Xoshiro(2), conditioned, Prior(), 3; progress=false
        )
        @test issetequal(Turing.FlexiChains.parameters(draws), parameter_names)
        #Their data isn't simulated any more, so the sessions are named as those of a fit again
        extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
        @test extension.name_of_session(conditioned.args.sessions_info, 2) ==
            @varname(sessions[2])
    end

    @testset "It gives the same log joint as Turing.jl's condition on the same variables" begin
        conditioned = condition_sessions(deconditioned, simulated_choices)
        turing_conditioned = condition(deconditioned, simulated_choices)
        for parameters in (some_parameters, other_parameters)
            @test logjoint(conditioned, parameters) ≈
                logjoint(turing_conditioned, parameters)
        end
    end

    @testset "A value that has no place in the data is an error, which names it" begin
        #The draw's parameters aren't data
        @test_throws "the sessions have none for inverse_temperature.sessions, learning_rate.sessions. Such variables are named as in simulated_sessions[1].timesteps[1].choice.simulated" condition_sessions(
            deconditioned, draw
        )
        #Nor is a choice at a timestep that the session doesn't have, or one that is data already
        @test_throws "none for simulated_sessions[1].timesteps[2].choice.simulated" condition_sessions(
            deconditioned,
            Dict(@varname(simulated_sessions[1].timesteps[2].choice.simulated) => true),
        )
        #Nor is a choice under the name of the fit's sessions
        @test_throws "none for sessions[1].timesteps[1].choice.simulated" condition_sessions(
            deconditioned, Dict(@varname(sessions[1].timesteps[1].choice.simulated) => true)
        )
        @test_throws "none for sessions[2].timesteps[1].choice.imputed" condition_sessions(
            imputing, Dict(@varname(sessions[2].timesteps[1].choice.imputed) => true)
        )
        #Only the first five are named
        @test_throws "and 2 more" condition_sessions(
            deconditioned,
            Dict(
                @varname(simulated_sessions[1].timesteps[t].choice.simulated) => true for
                t in 2:8
            ),
        )
    end

    @testset "Choices without a value stay simulated, and the sessions keep one type" begin
        values_of_second_session = Dict(
            @varname(simulated_sessions[2].timesteps[t].choice.simulated) =>
                draw[@varname(simulated_sessions[2].timesteps[t].choice.simulated)] for
            t in 1:2
        )
        conditioned = condition_sessions(deconditioned, values_of_second_session)
        sessions = conditioned.args.sessions_info.sessions
        @test all(==(SimulatedMissingValue()), sessions[1].actions.choice)
        @test all(==(SimulatedMissingValue()), sessions[3].actions.choice)
        @test sessions[2].actions.choice == [
            draw[@varname(simulated_sessions[2].timesteps[t].choice.simulated)] for t in 1:2
        ]
        @test all(
            session -> eltype(session.actions.choice) == Union{SimulatedMissingValue,Bool},
            sessions,
        )
        draws = @test_logs match_mode = :any sample(
            Xoshiro(3), conditioned, Prior(), 3; progress=false
        )
        #Their data is still simulated, so the sessions keep their names
        @test issetequal(
            Turing.FlexiChains.parameters(draws),
            [
                parameter_names
                @varname(simulated_sessions[1].timesteps[1].choice.simulated)
                [
                    @varname(simulated_sessions[3].timesteps[t].choice.simulated) for
                    t in 1:5
                ]
            ],
        )
    end

    @testset "A choice that a fit imputes takes a value too, which makes it data" begin
        #S2's choice at timestep 2 and S3's at timestep 5 are imputed in the fit
        conditioned = condition_sessions(
            imputing, Dict(@varname(sessions[3].timesteps[5].choice.imputed) => true)
        )
        sessions = conditioned.args.sessions_info.sessions
        @test sessions[3].actions.choice[5] === true
        @test sessions[2].actions.choice[2] === ImputedMissingValue()
    end

    @testset "With an environment, the simulated observations are data too, and the scores of the environment are what differs from Turing.jl's condition" begin
        deconditioned_with_environment = decondition_sessions(
            some_sessions_running_environment()
        )
        draw_with_environment = @test_logs match_mode = :any rand(
            Xoshiro(4), deconditioned_with_environment
        )
        simulated_data = Turing.DynamicPPL.subset(
            draw_with_environment, [@varname(simulated_sessions)]
        )
        conditioned = condition_sessions(deconditioned_with_environment, simulated_data)
        sessions = conditioned.args.sessions_info.sessions
        @test sessions[3].observations.reward == [
            draw_with_environment[@varname(
                simulated_sessions[3].timesteps[t].environment.reward.simulated
            )] for t in 1:4
        ]
        #Nothing within the sessions is left to sample, so they leave out their environment, whose mean is fixed, and
        #the log joint differs from that of Turing.jl's condition by the scores of the rewards, which are the same for
        #every value of the parameters
        @test !any(session -> session.includes_environment, sessions)
        turing_conditioned = condition(deconditioned_with_environment, simulated_data)
        differences = [
            logjoint(conditioned, parameters) - logjoint(turing_conditioned, parameters) for
            parameters in (some_parameters, other_parameters)
        ]
        @test !(differences[1] ≈ 0) && differences[1] ≈ differences[2]
    end

    @testset "Each session ends after its last timestep with data, as a simulation that a model stopped" begin
        deconditioned_with_environment = decondition_sessions(
            some_sessions_running_environment()
        )
        draw_with_environment = @test_logs match_mode = :any rand(
            Xoshiro(4), deconditioned_with_environment
        )
        #The values of a draw in which S3 stopped after two of its four timesteps
        simulated_data = Turing.DynamicPPL.subset(
            draw_with_environment, [@varname(simulated_sessions)]
        )
        stopped_early = Dict(
            name => simulated_data[name] for name in keys(simulated_data) if
            !any(t -> startswith(string(name), "simulated_sessions[3].timesteps[$t]."), 3:4)
        )
        conditioned = condition_sessions(deconditioned_with_environment, stopped_early)
        cut_sessions = conditioned.args.sessions_info
        #S3 has the two timesteps with values, and the others every timestep
        @test map(session -> session.n_timesteps, cut_sessions.sessions) == [3, 2, 2]
        @test cut_sessions.sessions[3].actions.choice == [
            draw_with_environment[@varname(
                simulated_sessions[3].timesteps[t].choice.simulated
            )] for t in 1:2
        ]
        @test cut_sessions.sessions[3].observations.reward == [
            draw_with_environment[@varname(
                simulated_sessions[3].timesteps[t].environment.reward.simulated
            )] for t in 1:2
        ]
        #Its rows in the table are those of the timesteps it has left
        deconditioned_rows = deconditioned_with_environment.args.sessions_info.session_rows
        @test cut_sessions.session_rows ==
            [deconditioned_rows[1], deconditioned_rows[2], deconditioned_rows[3][1:2]]
        #So nothing within the sessions is left to sample, and their data continues to their last timestep
        @test issetequal(keys(rand(Xoshiro(5), conditioned)), parameter_names)
        @test cut_sessions.sessions[3].n_timesteps == 2 &&
            cut_sessions.sessions[3].last_timestep_with_action == 2
        #A value on a later timestep keeps the timesteps before it, whose other values stay simulated
        only_a_late_choice = Dict(
            @varname(simulated_sessions[3].timesteps[3].choice.simulated) =>
                draw_with_environment[@varname(
                    simulated_sessions[3].timesteps[3].choice.simulated
                )],
        )
        with_late_choice = condition_sessions(
            deconditioned_with_environment, only_a_late_choice
        ).args.sessions_info.sessions[3]
        @test with_late_choice.n_timesteps == 3
        @test with_late_choice.actions.choice[1:2] == fill(SimulatedMissingValue(), 2)
        #The sizes of the timesteps are cut with the data
        session_info = SessionInfo(
            WithDefaultParameters(DistributionsJl()),
            (; reward=[1.0, 0.0, 1.0]),
            (; choice=[true, false, true]);
            who_starts=EnvironmentStarts(),
            environment=RewardsAroundMean(),
            timestep_sizes=[0.5, 1.0, 2.0],
        )
        deconditioned_session = decondition_sessions(
            session_model(
                TuringBackend(),
                session_info,
                (; learning_rate=0.3, inverse_temperature=2.0),
                (; mean_reward=0.6),
            ),
        )
        first_timestep = Dict(
            @varname(simulated_session.timesteps[1].choice.simulated) => true,
            @varname(simulated_session.timesteps[1].environment.reward.simulated) => 0.4,
        )
        cut_session =
            condition_sessions(deconditioned_session, first_timestep).args.session_info
        @test cut_session.n_timesteps == 1 && cut_session.timestep_sizes == [0.5]
        @test cut_session.actions.choice == [true] &&
            cut_session.observations.reward == [0.4]
    end

    @testset "An output of the environment that the map doesn't pass on has no place in the data" begin
        session_info = SessionInfo(
            WithDefaultParameters(DistributionsJl()),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            environment=NoisyPayoffsAndCues(),
            observation_map=NameMap(:payoff => :reward),
            action_map=NameMap(:choice => :pull),
        )
        deconditioned_session = decondition_sessions(
            session_model(
                TuringBackend(),
                session_info,
                (; learning_rate=0.3, inverse_temperature=2.0),
            ),
        )
        session_draw = @test_logs match_mode = :any rand(Xoshiro(5), deconditioned_session)
        @test_throws "none for simulated_session.timesteps[1].environment.cue.simulated, simulated_session.timesteps[2].environment.cue.simulated. Such variables are named as in simulated_session.timesteps[1].choice.simulated" condition_sessions(
            deconditioned_session, session_draw
        )
        #Without the cues, the payoffs are the rewards in the data, and the session then leaves out its environment
        without_cues = Dict(
            name => session_draw[name] for
            name in keys(session_draw) if !occursin("cue", string(name))
        )
        conditioned = condition_sessions(deconditioned_session, without_cues)
        @test conditioned.args.session_info.observations.reward == [
            session_draw[@varname(
                simulated_session.timesteps[t].environment.payoff.simulated
            )] for t in 1:2
        ]
        @test !conditioned.args.session_info.includes_environment
    end

    @testset "A single session takes the values of its variables, which are under simulated_session" begin
        single = decondition_sessions(
            session_model(
                TuringBackend(),
                sessions_info.sessions[3],
                (; learning_rate=0.3, inverse_temperature=2.0),
            ),
        )
        single_draw = @test_logs match_mode = :any rand(Xoshiro(6), single)
        conditioned_single = condition_sessions(single, single_draw)
        @test conditioned_single.args.session_info.actions.choice == [
            single_draw[@varname(simulated_session.timesteps[t].choice.simulated)] for
            t in 1:5
        ]
        @test isempty(keys(rand(Xoshiro(7), conditioned_single)))
        #It keeps the prefix that decondition_sessions gave it
        partly = condition_sessions(
            single,
            Dict(
                @varname(simulated_session.timesteps[1].choice.simulated) =>
                    single_draw[@varname(simulated_session.timesteps[1].choice.simulated)],
            ),
        )
        @test issetequal(
            keys(rand(Xoshiro(8), partly)),
            [@varname(simulated_session.timesteps[t].choice.simulated) for t in 2:5],
        )
    end

    @testset "The values can be given as pairs, a table of several sessions, or the data of a single session" begin
        #Pairs, as Turing.jl's condition takes them, give the same as a Dict
        first_choice = @varname(simulated_sessions[1].timesteps[1].choice.simulated)
        third_choice = @varname(simulated_sessions[3].timesteps[2].choice.simulated)
        from_pairs = condition_sessions(
            deconditioned, first_choice => true, third_choice => false
        )
        from_dict = condition_sessions(
            deconditioned, Dict(first_choice => true, third_choice => false)
        )
        @test from_pairs.args.sessions_info.sessions[1].actions.choice == [true]
        @test isequal(
            map(session -> session.actions, from_pairs.args.sessions_info.sessions),
            map(session -> session.actions, from_dict.args.sessions_info.sessions),
        )
        #A table with the simulated choices, read as the sessions were read, gives the same log joint as the values
        simulated_table = (;
            DIFFERENT_LENGTHS_TABLE...,
            choice=[
                draw[@varname(simulated_sessions[i].timesteps[t].choice.simulated)] for
                (i, n_timesteps) in enumerate((1, 2, 5)) for t in 1:n_timesteps
            ],
        )
        from_table = condition_sessions(deconditioned, simulated_table)
        for parameters in (some_parameters, other_parameters)
            @test logjoint(from_table, parameters) ≈
                logjoint(condition_sessions(deconditioned, simulated_choices), parameters)
        end
        #A table with another session is an error, as is a table for a single session
        @test_throws "The table has the sessions" condition_sessions(
            deconditioned, (; id=["S9"], reward=[1.0], choice=[true])
        )
        single = session_model(
            TuringBackend(),
            sessions_info.sessions[3],
            (; learning_rate=0.3, inverse_temperature=2.0),
        )
        @test_throws "gives several sessions the data in a table, but the model has a single session" condition_sessions(
            single, simulated_table
        )
        #A single session takes its observations and actions, also with other timesteps
        conditioned_single = condition_sessions(
            decondition_sessions(single), (; reward=[1.0, 0.0]), (; choice=[true, false])
        )
        @test conditioned_single.args.session_info.actions.choice == [true, false]
        @test conditioned_single.args.session_info.n_timesteps == 2
        @test_throws "gives a single session its data, but the model has several sessions" condition_sessions(
            deconditioned, (; reward=[1.0]), (; choice=[true])
        )
    end

    @testset "A model of one's own is an error, which says what condition_sessions takes" begin
        Turing.@model own_conditioned_model() = (x ~ Normal(); return x)
        @test_throws "condition_sessions(model, values) takes a model made by full_model" condition_sessions(
            own_conditioned_model(), Dict(@varname(x) => 1.0)
        )
    end
end
