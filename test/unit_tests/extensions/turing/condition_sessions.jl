@testset "condition_sessions on a Turing.jl model" begin
    #The deconditioned full model of sessions of one, two and five timesteps, and a draw from its prior, whose choices
    #are simulated with the draw's parameters
    sessions_info = sessions_for_test(
        WithDefaultParameters(DistributionsJl()), DIFFERENT_LENGTHS_TABLE
    )
    imputing = full_model(
        TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info
    )
    deconditioned = decondition_sessions(imputing)
    draw = @test_logs match_mode = :any rand(Xoshiro(1), deconditioned)
    simulated_choices = Turing.DynamicPPL.subset(draw, [@varname(session)])
    parameter_names = [
        @varname(learning_rate.session), @varname(inverse_temperature.session)
    ]
    #Two sets of parameters to evaluate the models at
    some_parameters = Dict(
        @varname(learning_rate.session) => [0.2, 0.3, 0.4],
        @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
    )
    other_parameters = Dict(
        @varname(learning_rate.session) => [0.9, 0.5, 0.1],
        @varname(inverse_temperature.session) => [4.0, 0.5, 1.5],
    )

    @testset "The simulated choices are the data of the copy, which has only the parameters as variables" begin
        conditioned = condition_sessions(deconditioned, simulated_choices)
        @test conditioned.f === deconditioned.f
        for (i, n_timesteps) in enumerate((1, 2, 5))
            @test conditioned.args.sessions_info.sessions[i].actions.choice == [
                draw[@varname(session[i].timestep[t].choice.simulated)] for
                t in 1:n_timesteps
            ]
        end
        draws = @test_logs match_mode = :any sample(
            Xoshiro(2), conditioned, Prior(), 3; progress=false
        )
        @test issetequal(Turing.FlexiChains.parameters(draws), parameter_names)
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
        @test_throws "the sessions have none for inverse_temperature.session, learning_rate.session. Such variables are named as in session[1].timestep[1].choice.simulated" condition_sessions(
            deconditioned, draw
        )
        #Nor is a choice at a timestep that the session doesn't have, or one that is data already
        @test_throws "none for session[1].timestep[2].choice.simulated" condition_sessions(
            deconditioned, Dict(@varname(session[1].timestep[2].choice.simulated) => true)
        )
        @test_throws "none for session[2].timestep[1].choice.imputed" condition_sessions(
            imputing, Dict(@varname(session[2].timestep[1].choice.imputed) => true)
        )
        #Only the first five are named
        @test_throws "and 2 more" condition_sessions(
            deconditioned,
            Dict(@varname(session[1].timestep[t].choice.simulated) => true for t in 2:8),
        )
    end

    @testset "Choices without a value stay simulated, and the sessions keep one type" begin
        values_of_second_session = Dict(
            @varname(session[2].timestep[t].choice.simulated) =>
                draw[@varname(session[2].timestep[t].choice.simulated)] for t in 1:2
        )
        conditioned = condition_sessions(deconditioned, values_of_second_session)
        sessions = conditioned.args.sessions_info.sessions
        @test all(==(SimulatedValue()), sessions[1].actions.choice)
        @test all(==(SimulatedValue()), sessions[3].actions.choice)
        @test sessions[2].actions.choice ==
            [draw[@varname(session[2].timestep[t].choice.simulated)] for t in 1:2]
        @test all(
            session -> eltype(session.actions.choice) == Union{SimulatedValue,Bool},
            sessions,
        )
        draws = @test_logs match_mode = :any sample(
            Xoshiro(3), conditioned, Prior(), 3; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(draws),
            [
                parameter_names
                @varname(session[1].timestep[1].choice.simulated)
                [@varname(session[3].timestep[t].choice.simulated) for t in 1:5]
            ],
        )
    end

    @testset "A choice that a fit imputes takes a value too, which makes it data" begin
        #S2's choice at timestep 2 and S3's at timestep 5 are imputed in the fit
        conditioned = condition_sessions(
            imputing, Dict(@varname(session[3].timestep[5].choice.imputed) => true)
        )
        sessions = conditioned.args.sessions_info.sessions
        @test sessions[3].actions.choice[5] === true
        @test sessions[2].actions.choice[2] === MissingValue()
    end

    @testset "With an environment, the simulated observations are data too, and the scores of the environment are what differs from Turing.jl's condition" begin
        deconditioned_with_environment = decondition_sessions(
            some_sessions_running_environment()
        )
        draw_with_environment = @test_logs match_mode = :any rand(
            Xoshiro(4), deconditioned_with_environment
        )
        simulated_data = Turing.DynamicPPL.subset(
            draw_with_environment, [@varname(session)]
        )
        conditioned = condition_sessions(deconditioned_with_environment, simulated_data)
        sessions = conditioned.args.sessions_info.sessions
        @test sessions[3].observations.reward == [
            draw_with_environment[@varname(
                session[3].timestep[t].environment.reward.simulated
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
        @test_throws "none for timestep[1].environment.cue.simulated, timestep[2].environment.cue.simulated. Such variables are named as in timestep[1].choice.simulated" condition_sessions(
            deconditioned_session, session_draw
        )
        #Without the cues, the payoffs are the rewards in the data, and the session then leaves out its environment
        without_cues = Dict(
            name => session_draw[name] for
            name in keys(session_draw) if !occursin("cue", string(name))
        )
        conditioned = condition_sessions(deconditioned_session, without_cues)
        @test conditioned.args.session_info.observations.reward == [
            session_draw[@varname(timestep[t].environment.payoff.simulated)] for t in 1:2
        ]
        @test !conditioned.args.session_info.includes_environment
    end

    @testset "A single session takes the values of its variables" begin
        single = decondition_sessions(
            session_model(
                TuringBackend(),
                sessions_info.sessions[3],
                (; learning_rate=0.3, inverse_temperature=2.0),
            ),
        )
        single_draw = @test_logs match_mode = :any rand(Xoshiro(6), single)
        @test condition_sessions(single, single_draw).args.session_info.actions.choice ==
            [single_draw[@varname(timestep[t].choice.simulated)] for t in 1:5]
    end

    @testset "check_stopping_criterion is given to the sessions" begin
        conditioned = condition_sessions(
            deconditioned, simulated_choices; check_stopping_criterion=false
        )
        @test all(
            session -> session.check_stopping_criterion === false,
            conditioned.args.sessions_info.sessions,
        )
    end

    @testset "A model of one's own is an error, which says what condition_sessions takes" begin
        Turing.@model own_conditioned_model() = (x ~ Normal(); return x)
        @test_throws "condition_sessions(model, values) takes a model made by full_model" condition_sessions(
            own_conditioned_model(), Dict(@varname(x) => 1.0)
        )
    end
end
