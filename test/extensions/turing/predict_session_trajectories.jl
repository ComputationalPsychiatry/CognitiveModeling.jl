@testset "predict_session_trajectories" begin
    #The full model of the sessions of the table, fitted as in the tests of get_session_trajectories
    sessions_info = sessions_for_test(; impute_missing_actions=false)
    model = full_model(TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info)
    chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)

    @testset "predict_session_trajectories simulates every action again with the parameters of each sample" begin
        save_cognitive_trajectories = SaveTrajectories(; inputs=:reward, outputs=:choice)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(2), model, chain; save_cognitive_trajectories
        )
        @test Turing.FlexiChains.iter_indices(simulated) ==
            Turing.FlexiChains.iter_indices(chain)
        #Without an environment, the rewards are those in the data, and the choices are simulated, every one of them
        @test all(
            ==([BeforeFirstTimestep(), 1.0, 1.0, 0.0, 1.0]),
            vec(simulated[@varname(S3.reward)]),
        )
        simulated_choices = vec(simulated[@varname(S3.choice)])
        @test all(
            choices -> all(choice -> choice isa Bool, choices[2:end]), simulated_choices
        )
        @test length(unique(simulated_choices)) > 1
    end

    @testset "With an environment, the observations are simulated too, in response to the simulated actions" begin
        #RewardsPreviousChoice gives the previous choice as the reward, and 0.5 before the first
        with_environment = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(; environment=RewardsPreviousChoice()),
        )
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(3),
            with_environment,
            chain;
            save_cognitive_trajectories=SaveTrajectories(; inputs=:reward, outputs=:choice),
            save_environment_trajectories=SaveTrajectories(;
                states=:n_timesteps, outputs=:reward
            ),
        )
        rewards = vec(simulated[@varname(S1.reward)])
        choices = vec(simulated[@varname(S1.choice)])
        @test all(
            sample -> rewards[sample][2:end] == [0.5; float.(choices[sample][2:(end - 1)])],
            eachindex(rewards),
        )
        #The environment's record is under environment: the rewards it gave are the same, and its state counts the timesteps
        @test vec(simulated[@varname(S1.environment.reward)]) == rewards
        @test all(==([(0); 1:3]), vec(simulated[@varname(S1.environment.n_timesteps)]))
        #A chain from the prior gives a prior predictive check
        prior_chain = sample(Xoshiro(4), with_environment, Prior(), 10; progress=false)
        prior_simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(5),
            with_environment,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(),
        )
        @test Turing.FlexiChains.iter_indices(prior_simulated) ==
            Turing.FlexiChains.iter_indices(prior_chain)
    end

    @testset "Values imputed in several sessions, at different timesteps, are simulated again" begin
        table = (;
            FULL_TABLE...,
            choice=Union{Missing,Bool}[
                missing, false, true, false, true, true, true, false, missing
            ],
        )
        imputing = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(WithDefaultParameters(DistributionsJl()), table),
        )
        prior_chain = sample(Xoshiro(6), imputing, Prior(), 5; progress=false)
        #predict_session_trajectories simulates the imputed choices again, like the others, from the parameters of each sample
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(7),
            imputing,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(; outputs=:choice),
        )
        @test all(
            choices -> all(choice -> choice isa Bool, choices[2:end]),
            vec(simulated[@varname(S3.choice)]),
        )
    end

    @testset "A start that a model samples is drawn again too, as the sessions are named apart from the fit's" begin
        #SampledInitialState samples an offset of the expected reward it starts from, which the chain of a fit holds
        pair = pair_named("SampledInitialState")
        sampling = full_model(
            TuringBackend(),
            FixedParameters(pair.cognitive_parameters),
            sessions_for_test(pair.model(DistributionsJl()), FULL_TABLE),
        )
        prior_chain = sample(Xoshiro(13), sampling, Prior(), 5; progress=false)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(14),
            sampling,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:expected_reward),
        )
        starts_of_fit =
            pair.cognitive_parameters.initial_value .+
            vec(prior_chain[@varname(sessions[2].initial_cognitive_state.initial_offset)])
        predicted_starts = map(first, vec(simulated[@varname(S2.expected_reward)]))
        @test !(predicted_starts ≈ starts_of_fit)
    end

    @testset "Sessions of different lengths are simulated with their own timesteps, with an environment too" begin
        different_lengths = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(
                WithDefaultParameters(DistributionsJl()), DIFFERENT_LENGTHS_TABLE
            ),
        )
        prior_chain = sample(Xoshiro(8), different_lengths, Prior(), 5; progress=false)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(9),
            different_lengths,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(; outputs=:choice),
        )
        #Every choice of each session is simulated, on each of its own timesteps
        for (name, n_timesteps) in
            ((@varname(S1.choice), 1), (@varname(S2.choice), 2), (@varname(S3.choice), 5))
            @test all(
                choices ->
                    length(choices) == n_timesteps + 1 &&
                    all(choice -> choice isa Bool, choices[2:end]),
                vec(simulated[name]),
            )
        end

        #With an environment, the rewards are simulated on each timestep of each session too, from the simulated choices
        with_environment = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(
                WithDefaultParameters(DistributionsJl()),
                DIFFERENT_LENGTHS_TABLE;
                environment=RewardsPreviousChoice(),
            ),
        )
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(10),
            with_environment,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(; inputs=:reward, outputs=:choice),
        )
        rewards = vec(simulated[@varname(S3.reward)])
        choices = vec(simulated[@varname(S3.choice)])
        @test all(sample_rewards -> length(sample_rewards) == 6, rewards)
        @test all(
            sample -> rewards[sample][2:end] == [0.5; float.(choices[sample][2:(end - 1)])],
            eachindex(rewards),
        )
    end

    @testset "A fit where only some sessions run their environment is simulated with the environment of every session" begin
        some_running = some_sessions_running_environment()
        prior_chain = sample(Xoshiro(11), some_running, Prior(), 5; progress=false)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(12),
            some_running,
            prior_chain;
            save_cognitive_trajectories=SaveTrajectories(; inputs=:reward),
        )

        #Every session observes rewards that its environment gives for each of its timesteps, S2 too, which didn't run it in the fit, so that its rewards aren't those in the data
        for (name, n_timesteps) in
            ((@varname(S1.reward), 3), (@varname(S2.reward), 2), (@varname(S3.reward), 4))
            @test all(rewards -> length(rewards) == n_timesteps + 1, vec(simulated[name]))
        end
        @test all(
            rewards -> rewards[2:end] != [0.0, 1.0], vec(simulated[@varname(S2.reward)])
        )
    end

    @testset "A chain that isn't a FlexiChain is an error, with or without a random number generator" begin
        not_a_chain = (; learning_rate=[0.3])
        @test_throws "predict_session_trajectories(model, chain) takes a FlexiChain, but the chain is of type NamedTuple." predict_session_trajectories(
            model, not_a_chain
        )
        @test_throws "predict_session_trajectories(model, chain) takes a FlexiChain, but the chain is of type NamedTuple." predict_session_trajectories(
            Xoshiro(1), model, not_a_chain
        )
    end

    @testset "A model of one's own is an error" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "predict_session_trajectories(model, chain; save_cognitive_trajectories) takes a model made by full_model" predict_session_trajectories(
            own_model(), chain; save_cognitive_trajectories=SaveTrajectories()
        )
    end
end
