@testset "decondition and predict" begin
    #A full model of sessions of one, two and five timesteps, whose fit imputes a choice at timestep 2 of S2 and at timestep 5 of S3
    sessions_info = sessions_for_test(
        WithDefaultParameters(DistributionsJl()), DIFFERENT_LENGTHS_TABLE
    )
    imputing = full_model(
        TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info
    )
    parameter_names = [
        @varname(learning_rate.session), @varname(inverse_temperature.session)
    ]
    #The variable of every choice of every session
    every_choice = [
        @varname(session[i].timestep[t].choice.imputed) for
        (i, n_timesteps) in enumerate((1, 2, 5)) for t in 1:n_timesteps
    ]

    @testset "decondition makes every choice a variable, which a prior predictive check samples" begin
        deconditioned = decondition(imputing)
        #The sessions are those that decondition_sessions makes, and the rest of the model is as it was
        @test deconditioned.f === imputing.f
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            deconditioned.args.sessions_info.sessions,
        )
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(1), deconditioned, Prior(), 5; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(prior_predictions),
            [parameter_names; every_choice],
        )
    end

    @testset "With an environment, decondition makes every observation a variable too, in every session" begin
        #S2 doesn't run its environment in the fit, since none of its rewards is missing
        deconditioned = decondition(some_sessions_running_environment())
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(2), deconditioned, Prior(), 5; progress=false
        )
        every_reward = [
            @varname(session[i].timestep[t].environment.reward.imputed) for
            (i, n_timesteps) in enumerate((3, 2, 4)) for t in 1:n_timesteps
        ]
        @test every_reward ⊆ Turing.FlexiChains.parameters(prior_predictions)
    end

    @testset "decondition removes what condition gave the model too, and with variables given, Turing.jl's own method deconditions only those" begin
        conditioned = condition(
            imputing, @varname(learning_rate.session) => [0.2, 0.3, 0.4]
        )
        @test isempty(keys(Turing.DynamicPPL.conditioned(decondition(conditioned))))
        partly = decondition(conditioned, @varname(learning_rate.session))
        @test partly.args.sessions_info === imputing.args.sessions_info
        @test isempty(keys(Turing.DynamicPPL.conditioned(partly)))
    end

    @testset "decondition deconditions a single session, several sessions, and a model inside catch_invalid_parameters" begin
        single = session_model(
            TuringBackend(),
            sessions_info.sessions[3],
            (; learning_rate=0.3, inverse_temperature=2.0),
        )
        @test all(==(MissingValue()), decondition(single).args.session_info.actions.choice)
        several = multiple_sessions_model(
            TuringBackend(),
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0]),
        )
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            decondition(several).args.sessions_info.sessions,
        )
        caught = catch_invalid_parameters(TuringBackend(), imputing)
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            decondition(caught).args.model.args.sessions_info.sessions,
        )
    end

    @testset "A model of one's own is deconditioned by Turing.jl's own method" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test isempty(
            keys(Turing.DynamicPPL.conditioned(decondition(own_model() | (; x=1.0))))
        )
    end
end
