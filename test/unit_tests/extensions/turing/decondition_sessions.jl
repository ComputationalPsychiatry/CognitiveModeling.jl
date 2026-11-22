#Gives the reward of NoisyRewards under another name, with a cue which is random and which the map doesn't pass on,
#so that the cue is a latent variable on every timestep
struct NoisyPayoffsAndCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::NoisyPayoffsAndCues, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::NoisyPayoffsAndCues, context::ModelContext
)
    return ModelOutput((; payoff=Normal(0.5, 1.0), cue=Normal(0.0, 1.0)))
end
function CognitiveModeling.attribute_names(environment::NoisyPayoffsAndCues)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:pull, outputs=(:payoff, :cue)
    )
end

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
    #The variable of every choice of every session, which is named apart from the choices that a fit imputes
    every_choice = [
        @varname(session[i].timestep[t].choice.simulated) for
        (i, n_timesteps) in enumerate((1, 2, 5)) for t in 1:n_timesteps
    ]

    @testset "Every choice is a variable of the copy, which a prior predictive check samples" begin
        deconditioned = decondition_sessions(imputing)
        #The sessions are those that decondition_sessions makes, and the rest of the model is as it was
        @test deconditioned.f === imputing.f
        @test all(
            session -> all(==(SimulatedValue()), session.actions.choice),
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
            @varname(session[i].timestep[t].environment.reward.simulated) for
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
            session -> all(==(SimulatedValue()), session.actions.choice),
            deconditioned.args.sessions_info.sessions,
        )
        unconditioned = decondition(conditioned)
        @test isempty(keys(Turing.DynamicPPL.conditioned(unconditioned)))
        @test unconditioned.args.sessions_info === imputing.args.sessions_info
    end

    @testset "Turing.jl's condition conditions a variable within the deconditioned sessions, such as a simulated choice" begin
        conditioned_choice = @varname(session[3].timestep[5].choice.simulated)
        conditioned = condition(decondition_sessions(imputing), conditioned_choice => true)
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(3), conditioned, Prior(), 5; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(prior_predictions),
            [parameter_names; setdiff(every_choice, [conditioned_choice])],
        )
    end

    @testset "A single session and several sessions are deconditioned" begin
        single = session_model(
            TuringBackend(),
            sessions_info.sessions[3],
            (; learning_rate=0.3, inverse_temperature=2.0),
        )
        @test all(
            ==(SimulatedValue()),
            decondition_sessions(single).args.session_info.actions.choice,
        )
        several = multiple_sessions_model(
            TuringBackend(),
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0]),
        )
        @test all(
            session -> all(==(SimulatedValue()), session.actions.choice),
            decondition_sessions(several).args.sessions_info.sessions,
        )
    end

    @testset "check_stopping_criterion is given to the sessions" begin
        deconditioned = decondition_sessions(imputing; check_stopping_criterion=false)
        @test all(
            session -> session.check_stopping_criterion === false,
            deconditioned.args.sessions_info.sessions,
        )
    end

    @testset "Turing.jl's predict simulates every choice with the parameters of the chain, also those that the fit imputed" begin
        #S3's choice at timestep 3 is imputed in the fit
        fitted = full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_for_test()
        )
        prior_chain = sample(Xoshiro(4), fitted, Prior(), 20; progress=false)
        predictions = @test_logs match_mode = :any predict(
            Xoshiro(5), decondition_sessions(fitted), prior_chain
        )
        choices_of_full_table = [
            @varname(session[i].timestep[t].choice.simulated) for
            (i, n_timesteps) in enumerate((3, 2, 4)) for t in 1:n_timesteps
        ]
        @test issetequal(
            Turing.FlexiChains.parameters(predictions),
            [parameter_names; choices_of_full_table],
        )
        for name in parameter_names
            @test vec(predictions[name]) == vec(prior_chain[name])
        end
        #The choice that the fit imputed is simulated again, so it isn't the fit's in every sample
        @test vec(predictions[@varname(session[3].timestep[3].choice.simulated)]) !=
            vec(prior_chain[@varname(session[3].timestep[3].choice.imputed)])
    end

    @testset "An output of the environment that the map doesn't pass on is simulated too, named apart from the fit's" begin
        #The missing reward makes the fit run the environment, which imputes the cue on every timestep
        session_info = SessionInfo(
            WithDefaultParameters(DistributionsJl()),
            (; reward=[1.0, missing, 0.0]),
            (; choice=[true, false, true]);
            who_starts=EnvironmentStarts(),
            environment=NoisyPayoffsAndCues(),
            observation_map=NameMap(:payoff => :reward),
            action_map=NameMap(:choice => :pull),
        )
        fitted = session_model(
            TuringBackend(), session_info, (; learning_rate=0.3, inverse_temperature=2.0)
        )
        fit_draws = sample(Xoshiro(8), fitted, Prior(), 2; progress=false)
        @test issetequal(
            Turing.FlexiChains.parameters(fit_draws),
            [
                @varname(timestep[2].environment.payoff.imputed)
                [@varname(timestep[t].environment.cue.imputed) for t in 1:3]
            ],
        )
        simulated_draws = sample(
            Xoshiro(9), decondition_sessions(fitted), Prior(), 2; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(simulated_draws),
            [
                [@varname(timestep[t].environment.payoff.simulated) for t in 1:3]
                [@varname(timestep[t].environment.cue.simulated) for t in 1:3]
                [@varname(timestep[t].choice.simulated) for t in 1:3]
            ],
        )
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
