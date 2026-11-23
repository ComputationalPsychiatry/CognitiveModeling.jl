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
        @varname(learning_rate.sessions), @varname(inverse_temperature.sessions)
    ]
    #The variable of every choice of every session, which is named apart from the choices that a fit imputes, in sessions named apart from the fit's
    every_choice = [
        @varname(simulated_sessions[i].timesteps[t].choice.simulated) for
        (i, n_timesteps) in enumerate((1, 2, 5)) for t in 1:n_timesteps
    ]

    @testset "Every choice is a variable of the copy, which a prior predictive check samples" begin
        deconditioned = decondition_sessions(imputing)
        #The sessions are those that decondition_sessions makes, and the rest of the model is as it was
        @test deconditioned.f === imputing.f
        @test all(
            session -> all(==(SimulatedMissingValue()), session.actions.choice),
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
            @varname(simulated_sessions[i].timesteps[t].environment.reward.simulated) for
            (i, n_timesteps) in enumerate((3, 2, 4)) for t in 1:n_timesteps
        ]
        @test every_reward ⊆ Turing.FlexiChains.parameters(prior_predictions)
    end

    @testset "What condition gave the model stays, and Turing.jl's own decondition removes it and leaves the sessions as they are" begin
        conditioned = condition(
            imputing, @varname(learning_rate.sessions) => [0.2, 0.3, 0.4]
        )
        deconditioned = decondition_sessions(conditioned)
        @test @varname(learning_rate.sessions) in
            keys(Turing.DynamicPPL.conditioned(deconditioned))
        @test all(
            session -> all(==(SimulatedMissingValue()), session.actions.choice),
            deconditioned.args.sessions_info.sessions,
        )
        unconditioned = decondition(conditioned)
        @test isempty(keys(Turing.DynamicPPL.conditioned(unconditioned)))
        @test unconditioned.args.sessions_info === imputing.args.sessions_info
    end

    @testset "Turing.jl's condition conditions a variable within the deconditioned sessions, such as a simulated choice" begin
        conditioned_choice = @varname(simulated_sessions[3].timesteps[5].choice.simulated)
        conditioned = condition(decondition_sessions(imputing), conditioned_choice => true)
        prior_predictions = @test_logs match_mode = :any sample(
            Xoshiro(3), conditioned, Prior(), 5; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(prior_predictions),
            [parameter_names; setdiff(every_choice, [conditioned_choice])],
        )
    end

    @testset "A single session and several sessions are deconditioned, under names apart from those of a fit" begin
        single = session_model(
            TuringBackend(),
            sessions_info.sessions[3],
            (; learning_rate=0.3, inverse_temperature=2.0),
        )
        @test all(
            ==(SimulatedMissingValue()),
            decondition_sessions(single).args.session_info.actions.choice,
        )
        #A single session has its variables under simulated_session, also when it is deconditioned again
        single_names = [
            @varname(simulated_session.timesteps[t].choice.simulated) for t in 1:5
        ]
        @test issetequal(
            keys(rand(Xoshiro(10), decondition_sessions(single))), single_names
        )
        @test issetequal(
            keys(rand(Xoshiro(10), decondition_sessions(decondition_sessions(single)))),
            single_names,
        )
        several = multiple_sessions_model(
            TuringBackend(),
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0]),
        )
        @test all(
            session -> all(==(SimulatedMissingValue()), session.actions.choice),
            decondition_sessions(several).args.sessions_info.sessions,
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
            @varname(simulated_sessions[i].timesteps[t].choice.simulated) for
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
        @test vec(
            predictions[@varname(simulated_sessions[3].timesteps[3].choice.simulated)]
        ) != vec(prior_chain[@varname(sessions[3].timesteps[3].choice.imputed)])
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
                @varname(timesteps[2].environment.payoff.imputed)
                [@varname(timesteps[t].environment.cue.imputed) for t in 1:3]
            ],
        )
        simulated_draws = sample(
            Xoshiro(9), decondition_sessions(fitted), Prior(), 2; progress=false
        )
        @test issetequal(
            Turing.FlexiChains.parameters(simulated_draws),
            [
                [
                    @varname(simulated_session.timesteps[t].environment.payoff.simulated)
                    for t in 1:3
                ]
                [
                    @varname(simulated_session.timesteps[t].environment.cue.simulated) for
                    t in 1:3
                ]
                [@varname(simulated_session.timesteps[t].choice.simulated) for t in 1:3]
            ],
        )
    end

    @testset "Turing.jl's predict reads a fit that imputed values at different timesteps in different sessions" begin
        #The deconditioned sessions are named apart from the fit's, so their variables aren't read from the fit's sessions
        prior_chain = sample(Xoshiro(6), imputing, Prior(), 5; progress=false)
        predictions = @test_logs match_mode = :any predict(
            Xoshiro(7), decondition_sessions(imputing), prior_chain
        )
        @test issetequal(
            Turing.FlexiChains.parameters(predictions), [parameter_names; every_choice]
        )
    end

    @testset "Turing.jl's predict draws again what a model samples itself, at its start and within a timestep" begin
        #SampledInitialState samples an offset of its start, and SamplingWithinTimestep a noise on every timestep
        for (model_name, variable_of_fit, variable_of_prediction) in (
            (
                "SampledInitialState",
                @varname(sessions[2].initial_cognitive_state.initial_offset),
                @varname(simulated_sessions[2].initial_cognitive_state.initial_offset),
            ),
            (
                "SamplingWithinTimestep",
                @varname(sessions[2].timesteps[1].noise),
                @varname(simulated_sessions[2].timesteps[1].noise),
            ),
        )
            pair = pair_named(model_name)
            sampling = full_model(
                TuringBackend(),
                FixedParameters(pair.cognitive_parameters),
                sessions_for_test(pair.model(DistributionsJl()), FULL_TABLE),
            )
            prior_chain = sample(Xoshiro(11), sampling, Prior(), 5; progress=false)
            predictions = @test_logs match_mode = :any predict(
                Xoshiro(12), decondition_sessions(sampling), prior_chain
            )
            @test vec(predictions[variable_of_prediction]) !=
                vec(prior_chain[variable_of_fit])
        end
    end

    @testset "A model of one's own is an error, which says what decondition_sessions takes" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "decondition_sessions(model) takes a model made by full_model" decondition_sessions(
            own_model()
        )
    end
end
