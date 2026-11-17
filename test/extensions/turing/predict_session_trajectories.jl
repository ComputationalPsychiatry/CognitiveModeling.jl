@testset "predict_session_trajectories" begin
    #The full model of the sessions of the table, fitted as in the tests of get_session_trajectories
    sessions_info = sessions_for_test(; impute_missing_actions=false)
    model = turing_full_model(IndependentParameters(FULL_PRIORS), sessions_info)
    chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)

    @testset "predict_session_trajectories simulates every action again with the parameters of each sample" begin
        cognitive_save_history = SaveHistory(; inputs=:reward, outputs=:choice)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(2), model, chain; cognitive_save_history
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
        with_environment = turing_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(; environment=RewardsPreviousChoice()),
        )
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(3),
            with_environment,
            chain;
            cognitive_save_history=SaveHistory(; inputs=:reward, outputs=:choice),
            environment_save_history=SaveHistory(; states=:n_timesteps, outputs=:reward),
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
            Xoshiro(5), with_environment, prior_chain; cognitive_save_history=SaveHistory()
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
        imputing = turing_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(SessionRescorlaWagner(), table),
        )
        prior_chain = sample(Xoshiro(6), imputing, Prior(), 5; progress=false)
        #predict_session_trajectories simulates the imputed choices again, like the others, from the parameters of each sample
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(7),
            imputing,
            prior_chain;
            cognitive_save_history=SaveHistory(; outputs=:choice),
        )
        @test all(
            choices -> all(choice -> choice isa Bool, choices[2:end]),
            vec(simulated[@varname(S3.choice)]),
        )
        #The variables it leaves out of the chain are those within the timesteps
        extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
        @test extension.is_within_timestep(@varname(session[3].timestep[4].choice.imputed))
        @test extension.is_within_timestep(@varname(timestep[2].environment.reward.imputed))
        @test !extension.is_within_timestep(@varname(learning_rate.session))
        @test !extension.is_within_timestep(
            @varname(session[1].initial_cognitive_state.offset)
        )
    end

    @testset "Sessions of different lengths are simulated with their own timesteps, with an environment too" begin
        different_lengths = turing_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(SessionRescorlaWagner(), DIFFERENT_LENGTHS_TABLE),
        )
        prior_chain = sample(Xoshiro(8), different_lengths, Prior(), 5; progress=false)
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(9),
            different_lengths,
            prior_chain;
            cognitive_save_history=SaveHistory(; outputs=:choice),
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
        with_environment = turing_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(
                SessionRescorlaWagner(),
                DIFFERENT_LENGTHS_TABLE;
                environment=RewardsPreviousChoice(),
            ),
        )
        simulated = @test_logs match_mode = :any predict_session_trajectories(
            Xoshiro(10),
            with_environment,
            prior_chain;
            cognitive_save_history=SaveHistory(; inputs=:reward, outputs=:choice),
        )
        rewards = vec(simulated[@varname(S3.reward)])
        choices = vec(simulated[@varname(S3.choice)])
        @test all(sample_rewards -> length(sample_rewards) == 6, rewards)
        @test all(
            sample -> rewards[sample][2:end] == [0.5; float.(choices[sample][2:(end - 1)])],
            eachindex(rewards),
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
        @test_throws "predict_session_trajectories(model, chain; cognitive_save_history) takes a model made by turing_full_model" predict_session_trajectories(
            own_model(), chain; cognitive_save_history=SaveHistory()
        )
    end
end
