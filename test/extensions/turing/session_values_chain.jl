#The expected rewards of a session by hand, from timestep 0, when the environment starts
function expected_rewards_by_hand(rewards, learning_rate)
    expected_rewards = [0.5]
    for reward in rewards
        push!(expected_rewards, expected_rewards[end] + learning_rate * (reward - expected_rewards[end]))
    end
    return expected_rewards
end

@testset "session_values_chain" begin
    #A full model of the sessions of the table, where S3's missing choice isn't imputed, so that NUTS samples the parameters alone
    sessions_info = sessions_for_test(; impute_missing_actions = false)
    model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info)
    chain = sample(Xoshiro(1), model, NUTS(), 20; progress = false)

    @testset "The chain from a fitted full model has the parameters and the saved values of each session, numbered as the fit" begin
        session_values = session_values_chain(model, chain; save_history = SaveHistory(; actions = :choice, states = :expected_reward))
        @test Turing.FlexiChains.iter_indices(session_values) == Turing.FlexiChains.iter_indices(chain)
        @test vec(session_values[@varname(S2.learning_rate)]) == vec(map(learning_rates -> learning_rates[2], chain[@varname(learning_rate.session)]))
        #The saved states are those of each sample's parameters, from timestep 0
        rows = findall(==("S1"), FULL_TABLE.id)
        @test session_values[@varname(S1.expected_reward)][5] ≈ expected_rewards_by_hand(FULL_TABLE.reward[rows], session_values[@varname(S1.learning_rate)][5])
        @test session_values[@varname(S1.actions.choice)][5] == [BeforeFirstTimestep(), true, false, true]
        #S3 saves NoAction() for its missing choice, so its saved values have another type than the other sessions'
        @test session_values[@varname(S3.actions.choice)][5] == [BeforeFirstTimestep(), true, true, NoAction(), true]
    end

    @testset "The same works for several sessions and a single session, also inside the wrapper for invalid parameters" begin
        save_history = SaveHistory(; states = :expected_reward)
        sessions = turing_multiple_behavioural_sessions(sessions_info, (; learning_rate = [0.2, 0.3, 0.4], inverse_temperature = [2.0, 2.0, 2.0]))
        @test Turing.FlexiChains.get_name.(keys(session_values_chain(sessions, chain; save_history))) == [@varname(S1.expected_reward), @varname(S2.expected_reward), @varname(S3.expected_reward)]
        session_values = session_values_chain(turing_catch_invalid_parameters(session_for_test()), chain; save_history)
        @test Turing.FlexiChains.get_name.(keys(session_values)) == [@varname(expected_reward)]
        @test session_values[@varname(expected_reward)][1] ≈ expected_rewards_by_hand(SESSION_REWARDS.reward, SESSION_PARAMETERS.learning_rate)
    end

    @testset "A model of one's own is an error" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "session_values_chain(model, chain; save_history) takes a model made by turing_full_model" session_values_chain(own_model(), chain; save_history = SaveHistory())
    end
end
