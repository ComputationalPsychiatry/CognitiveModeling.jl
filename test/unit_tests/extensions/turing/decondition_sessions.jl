@testset "decondition_sessions on a Turing.jl model" begin
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

    @testset "Every choice is a variable of the copy, which a prior predictive check samples" begin
        deconditioned = decondition_sessions(imputing)
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

    @testset "With an environment, every observation is a variable of the copy too, in every session" begin
        #S2 doesn't run its environment in the fit, since none of its rewards is missing
        deconditioned = decondition_sessions(some_sessions_running_environment())
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(2), deconditioned, Prior(), 5; progress=false
        )
        every_reward = [
            @varname(session[i].timestep[t].environment.reward.imputed) for
            (i, n_timesteps) in enumerate((3, 2, 4)) for t in 1:n_timesteps
        ]
        @test every_reward ⊆ Turing.FlexiChains.parameters(prior_predictions)
    end

    @testset "What condition gave the model stays, and Turing.jl's own decondition removes it and leaves the sessions as they are" begin
        conditioned = condition(
            imputing, @varname(learning_rate.session) => [0.2, 0.3, 0.4]
        )
        deconditioned = decondition_sessions(conditioned)
        @test @varname(learning_rate.session) in
            keys(Turing.DynamicPPL.conditioned(deconditioned))
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            deconditioned.args.sessions_info.sessions,
        )
        unconditioned = decondition(conditioned)
        @test isempty(keys(Turing.DynamicPPL.conditioned(unconditioned)))
        @test unconditioned.args.sessions_info === imputing.args.sessions_info
    end

    @testset "Turing.jl's condition conditions a variable within the deconditioned sessions, such as a simulated choice" begin
        conditioned_choice = @varname(session[3].timestep[5].choice.imputed)
        conditioned = condition(decondition_sessions(imputing), conditioned_choice => true)
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(3), conditioned, Prior(), 5; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(prior_predictions),
            [parameter_names; setdiff(every_choice, [conditioned_choice])],
        )
    end

    @testset "A single session, several sessions, and a model inside catch_invalid_parameters are deconditioned" begin
        single = session_model(
            TuringBackend(),
            sessions_info.sessions[3],
            (; learning_rate=0.3, inverse_temperature=2.0),
        )
        @test all(
            ==(MissingValue()),
            decondition_sessions(single).args.session_info.actions.choice,
        )
        several = multiple_sessions_model(
            TuringBackend(),
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0]),
        )
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            decondition_sessions(several).args.sessions_info.sessions,
        )
        caught = catch_invalid_parameters(TuringBackend(), imputing)
        @test all(
            session -> all(==(MissingValue()), session.actions.choice),
            decondition_sessions(caught).args.model.args.sessions_info.sessions,
        )
    end

    @testset "check_stopping_criterion is given to the sessions" begin
        deconditioned = decondition_sessions(imputing; check_stopping_criterion=false)
        @test all(
            session -> session.check_stopping_criterion === false,
            deconditioned.args.sessions_info.sessions,
        )
    end

    @testset "Turing.jl's predict simulates every choice that was data with the parameters of the chain, and keeps the choices that the fit imputed" begin
        #S3's choice at timestep 3 is imputed in the fit
        fitted = full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_for_test()
        )
        prior_chain = sample(Xoshiro(4), fitted, Prior(), 20; progress=false)
        predictions = @test_logs match_mode = :any predict(
            Xoshiro(5), decondition_sessions(fitted), prior_chain
        )
        choices_of_full_table = [
            @varname(session[i].timestep[t].choice.imputed) for
            (i, n_timesteps) in enumerate((3, 2, 4)) for t in 1:n_timesteps
        ]
        @test issetequal(
            Turing.FlexiChains.parameters(predictions),
            [parameter_names; choices_of_full_table],
        )
        for name in parameter_names
            @test vec(predictions[name]) == vec(prior_chain[name])
        end
        imputed = @varname(session[3].timestep[3].choice.imputed)
        @test vec(predictions[imputed]) == vec(prior_chain[imputed])
    end

    @testset "Turing.jl's predict can't read a fit that imputed values at different timesteps in different sessions" begin
        prior_chain = sample(Xoshiro(6), imputing, Prior(), 5; progress=false)
        #FlexiChains.jl rebuilds such a sample wrongly, which DynamicPPL.jl pull request 1542 fixes; this marks it until Turing.jl uses a release with the fix
        @test_broken try
            predict(Xoshiro(7), decondition_sessions(imputing), prior_chain)
            true
        catch error
            error isa BoundsError || rethrow()
            false
        end
    end

    @testset "A model of one's own is an error, which says what decondition_sessions takes" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "decondition_sessions(model) takes a model made by full_model" decondition_sessions(
            own_model()
        )
    end
end
