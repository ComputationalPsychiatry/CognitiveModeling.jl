#The expected rewards of a session by hand, from timestep 0, when the environment starts
function expected_rewards_by_hand(rewards, learning_rate)
    expected_rewards = [0.5]
    for reward in rewards
        push!(
            expected_rewards,
            expected_rewards[end] + learning_rate * (reward - expected_rewards[end]),
        )
    end
    return expected_rewards
end

@testset "session_values_chain" begin
    #A full model of the sessions of the table, where S3's missing choice isn't imputed, so that NUTS samples the parameters alone
    sessions_info = sessions_for_test(; impute_missing_actions=false)
    model = turing_full_model(IndependentSessionParameters(FULL_PRIORS), sessions_info)
    chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)

    @testset "The chain from a fitted full model has the saved values of each session, numbered as the fit, and the parameters are a chain of their own" begin
        session_values = session_values_chain(
            model,
            chain;
            cognitive_save_history=SaveHistory(; outputs=:choice, states=:expected_reward),
        )
        @test Turing.FlexiChains.iter_indices(session_values) ==
            Turing.FlexiChains.iter_indices(chain)
        @test Turing.FlexiChains.get_name.(keys(session_values)) == [
            @varname(S1.choice),
            @varname(S1.expected_reward),
            @varname(S2.choice),
            @varname(S2.expected_reward),
            @varname(S3.choice),
            @varname(S3.expected_reward),
        ]
        parameters = session_parameters_chain(model, chain)
        @test Turing.FlexiChains.iter_indices(parameters) ==
            Turing.FlexiChains.iter_indices(chain)
        @test Turing.FlexiChains.get_name.(keys(parameters)) == [
            @varname(S1.learning_rate),
            @varname(S1.inverse_temperature),
            @varname(S2.learning_rate),
            @varname(S2.inverse_temperature),
            @varname(S3.learning_rate),
            @varname(S3.inverse_temperature),
        ]
        @test vec(parameters[@varname(S2.learning_rate)]) == vec(
            map(learning_rates -> learning_rates[2], chain[@varname(learning_rate.session)])
        )
        #The saved states are those of each sample's parameters, from timestep 0
        rows = findall(==("S1"), FULL_TABLE.id)
        @test session_values[@varname(S1.expected_reward)][5] ≈ expected_rewards_by_hand(
            FULL_TABLE.reward[rows], parameters[@varname(S1.learning_rate)][5]
        )
        @test session_values[@varname(S1.choice)][5] ==
            [BeforeFirstTimestep(), true, false, true]
        #S3 saves NoAction() for its missing choice, so its saved values have another type than the other sessions'
        @test session_values[@varname(S3.choice)][5] ==
            [BeforeFirstTimestep(), true, true, NoAction(), true]
        #The two chains merge into one
        merged = merge(session_values, parameters)
        @test @varname(S1.choice) in Turing.FlexiChains.parameters(merged) &&
            @varname(S1.learning_rate) in Turing.FlexiChains.parameters(merged)
        #A model that isn't a full model gives no parameters
        @test_throws "session_parameters_chain(model, chain) takes a model made by turing_full_model" session_parameters_chain(
            turing_multiple_behavioral_sessions(
                sessions_info,
                (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[2.0, 2.0, 2.0]),
            ),
            chain,
        )
    end

    @testset "The same works for several sessions and a single session, also inside the wrapper for invalid parameters" begin
        cognitive_save_history = SaveHistory(; states=:expected_reward)
        sessions = turing_multiple_behavioral_sessions(
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[2.0, 2.0, 2.0]),
        )
        @test Turing.FlexiChains.get_name.(
            keys(session_values_chain(sessions, chain; cognitive_save_history))
        ) == [
            @varname(S1.expected_reward),
            @varname(S2.expected_reward),
            @varname(S3.expected_reward),
        ]
        session_values = session_values_chain(
            turing_catch_invalid_parameters(session_for_test()),
            chain;
            cognitive_save_history,
        )
        @test Turing.FlexiChains.get_name.(keys(session_values)) ==
            [@varname(expected_reward)]
        @test session_values[@varname(expected_reward)][1] ≈ expected_rewards_by_hand(
            SESSION_REWARDS.reward, SESSION_PARAMETERS.learning_rate
        )
    end

    @testset "A fit with values imputed in several sessions, at different timesteps, can be read" begin
        #FlexiChains.jl can't rebuild such a sample when the session parameters are a variable for each session, so this guards their naming
        table = (;
            FULL_TABLE...,
            choice=Union{Missing,Bool}[
                missing, false, true, false, true, true, true, false, missing
            ],
        )
        imputing = turing_full_model(
            IndependentSessionParameters(FULL_PRIORS),
            sessions_for_test(SessionRescorlaWagner(), table),
        )
        prior_chain = sample(Xoshiro(6), imputing, Prior(), 5; progress=false)
        session_values = @test_logs match_mode = :any session_values_chain(
            imputing, prior_chain; cognitive_save_history=SaveHistory(; outputs=:choice)
        )
        @test length(session_values[@varname(S3.choice)][1]) == 5
        #predict_session_values simulates the imputed choices again, like the others, from the parameters of each sample
        simulated = @test_logs match_mode = :any predict_session_values(
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

    @testset "A model of one's own is an error" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "session_values_chain(model, chain; cognitive_save_history) takes a model made by turing_full_model" session_values_chain(
            own_model(), chain; cognitive_save_history=SaveHistory()
        )
        @test_throws "predict_session_values(model, chain; cognitive_save_history) takes a model made by turing_full_model" predict_session_values(
            own_model(), chain; cognitive_save_history=SaveHistory()
        )
    end

    @testset "predict_session_values simulates every action again with the parameters of each sample" begin
        cognitive_save_history = SaveHistory(; inputs=:reward, outputs=:choice)
        simulated = @test_logs match_mode = :any predict_session_values(
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
            IndependentSessionParameters(FULL_PRIORS),
            sessions_for_test(; environment=RewardsPreviousChoice()),
        )
        simulated = @test_logs match_mode = :any predict_session_values(
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
        prior_simulated = @test_logs match_mode = :any predict_session_values(
            Xoshiro(5), with_environment, prior_chain; cognitive_save_history=SaveHistory()
        )
        @test Turing.FlexiChains.iter_indices(prior_simulated) ==
            Turing.FlexiChains.iter_indices(prior_chain)
    end
end
