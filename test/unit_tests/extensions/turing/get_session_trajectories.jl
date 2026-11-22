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

#Three sessions of one, two and five timesteps, where the choices on the last timestep of S2 and of S3 are missing and imputed, so that the variables of a session sit at timesteps which the sessions before it don't have
const DIFFERENT_LENGTHS_TABLE = (;
    id=["S1", "S2", "S2", "S3", "S3", "S3", "S3", "S3"],
    reward=[1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    choice=Union{Missing,Bool}[true, false, missing, true, true, false, true, missing],
)

#A full model of the sessions of TWO_MISSING_REWARDS_TABLE, where S1 and S3 run their environment, whose mean is fixed, to impute a reward each, and S2 doesn't run it
function some_sessions_running_environment()
    return full_model(
        TuringBackend(),
        IndependentParameters(FULL_PRIORS),
        sessions_for_test(
            WithDefaultParameters(DistributionsJl()),
            TWO_MISSING_REWARDS_TABLE;
            environment=RewardsAroundMean(),
            impute_missing_actions=false,
        );
        fixed_environment_parameters=(; mean_reward=0.6),
    )
end

@testset "get_session_trajectories" begin
    #A full model of the sessions of the table, where S3's missing choice isn't imputed, so that NUTS samples the parameters alone
    sessions_info = sessions_for_test(; impute_missing_actions=false)
    model = full_model(TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info)
    chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)

    @testset "The chain from a fitted full model has the saved trajectories of each session, numbered as the fit, and the parameters are a chain of their own" begin
        session_trajectories = get_session_trajectories(
            model,
            chain;
            cognitive_save_history=SaveHistory(; outputs=:choice, states=:expected_reward),
        )
        @test Turing.FlexiChains.iter_indices(session_trajectories) ==
            Turing.FlexiChains.iter_indices(chain)
        @test Turing.FlexiChains.get_name.(keys(session_trajectories)) == [
            @varname(S1.choice),
            @varname(S1.expected_reward),
            @varname(S2.choice),
            @varname(S2.expected_reward),
            @varname(S3.choice),
            @varname(S3.expected_reward),
        ]
        parameters = get_session_parameters(model, chain)
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
        @test session_trajectories[@varname(S1.expected_reward)][5] ≈
            expected_rewards_by_hand(
            FULL_TABLE.reward[rows], parameters[@varname(S1.learning_rate)][5]
        )
        @test session_trajectories[@varname(S1.choice)][5] ==
            [BeforeFirstTimestep(), true, false, true]
        #S3 saves UnimputedMissingValue() for its missing choice, so its saved trajectories have another type than the other sessions'
        @test session_trajectories[@varname(S3.choice)][5] ==
            [BeforeFirstTimestep(), true, true, UnimputedMissingValue(), true]
        #The two chains merge into one
        merged = merge(session_trajectories, parameters)
        @test @varname(S1.choice) in Turing.FlexiChains.parameters(merged) &&
            @varname(S1.learning_rate) in Turing.FlexiChains.parameters(merged)
        #A model of several sessions gives the parameters it was given, the same for every sample
        given = get_session_parameters(
            turing_multiple_behavioral_sessions(
                sessions_info,
                (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[2.0, 2.0, 2.0]),
            ),
            chain,
        )
        @test Turing.FlexiChains.get_name.(keys(given)) ==
            Turing.FlexiChains.get_name.(keys(parameters))
        @test all(==(0.3), vec(given[@varname(S2.learning_rate)]))
    end

    @testset "The same works for several sessions and a single session" begin
        cognitive_save_history = SaveHistory(; states=:expected_reward)
        sessions = turing_multiple_behavioral_sessions(
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[2.0, 2.0, 2.0]),
        )
        @test Turing.FlexiChains.get_name.(
            keys(get_session_trajectories(sessions, chain; cognitive_save_history))
        ) == [
            @varname(S1.expected_reward),
            @varname(S2.expected_reward),
            @varname(S3.expected_reward),
        ]
        session_trajectories = get_session_trajectories(
            session_for_test(), chain; cognitive_save_history
        )
        @test Turing.FlexiChains.get_name.(keys(session_trajectories)) ==
            [@varname(expected_reward)]
        @test session_trajectories[@varname(expected_reward)][1] ≈ expected_rewards_by_hand(
            SESSION_REWARDS.reward, SESSION_PARAMETERS.learning_rate
        )
        #A single session gives its parameters without a session in their names
        @test Turing.FlexiChains.get_name.(
            keys(get_session_parameters(session_for_test(), chain))
        ) == [@varname(learning_rate), @varname(inverse_temperature)]
    end

    @testset "A model of one's own around the sessions gives the parameters it passes to them, and what they saved, to both chains" begin
        #A hierarchical model written by hand: the learning rates of the sessions come from a population on the logit scale, and the sessions share one inverse temperature
        Turing.@model function own_hierarchical_model(sessions_info)
            n = CognitiveModeling.n_sessions(sessions_info)
            mean_logit ~ Normal(0, 1)
            sd_logit ~ Exponential(0.5)
            standardized ~ filldist(Normal(), n)
            learning_rate = to_unit_interval.(mean_logit .+ sd_logit .* standardized)
            inverse_temperature ~ LogNormal(0, 0.5)
            sessions ~ to_submodel(
                turing_multiple_behavioral_sessions(
                    sessions_info,
                    (; learning_rate, inverse_temperature=fill(inverse_temperature, n)),
                ),
                false,
            )
            return sessions
        end
        hierarchical_model = own_hierarchical_model(sessions_info)
        own_chain = sample(Xoshiro(8), hierarchical_model, Prior(), 10; progress=false)
        parameters = get_session_parameters(
            returned(hierarchical_model, own_chain), sessions_info
        )
        @test Turing.FlexiChains.get_name.(keys(parameters)) == [
            @varname(S1.learning_rate),
            @varname(S1.inverse_temperature),
            @varname(S2.learning_rate),
            @varname(S2.inverse_temperature),
            @varname(S3.learning_rate),
            @varname(S3.inverse_temperature),
        ]
        #The learning rates are the transformed values, which the chain itself doesn't have
        @test parameters[@varname(S2.learning_rate)][4] ≈ to_unit_interval(
            own_chain[@varname(mean_logit)][4] +
            own_chain[@varname(sd_logit)][4] * own_chain[@varname(standardized)][4][2],
        )
        @test vec(parameters[@varname(S3.inverse_temperature)]) ==
            vec(own_chain[@varname(inverse_temperature)])
        #The trajectories come from the same model with sessions that save them, and merge with the parameters
        saving_model = own_hierarchical_model(
            MultipleSessionsInfo(
                sessions_info; cognitive_save_history=SaveHistory(; states=:expected_reward)
            ),
        )
        trajectories = get_session_trajectories(
            returned(saving_model, own_chain), sessions_info
        )
        rows = findall(==("S1"), FULL_TABLE.id)
        @test trajectories[@varname(S1.expected_reward)][4] ≈ expected_rewards_by_hand(
            FULL_TABLE.reward[rows], parameters[@varname(S1.learning_rate)][4]
        )
        merged = merge(trajectories, parameters)
        @test @varname(S1.expected_reward) in Turing.FlexiChains.parameters(merged) &&
            @varname(S1.learning_rate) in Turing.FlexiChains.parameters(merged)
        #The form with a model takes only the package's models, which have the session info to name the parameters by
        @test_throws "get_session_parameters(model, chain) takes a model made by full_model, multiple_sessions_model or session_model" get_session_parameters(
            hierarchical_model, own_chain
        )

        #A single session in a model of one's own, whose learning rate is sampled on the logit scale
        Turing.@model function own_single_session_model(session_info)
            learning_rate_logit ~ Normal(0, 1)
            session ~ to_submodel(
                prefix(
                    turing_behavioral_session(
                        session_info,
                        (;
                            learning_rate=to_unit_interval(learning_rate_logit),
                            inverse_temperature=2.0,
                        ),
                    ),
                    @varname(session),
                ),
                false,
            )
            return session
        end
        single_model = own_single_session_model(
            SessionInfo(
                WithDefaultParameters(DistributionsJl()),
                SESSION_REWARDS,
                SESSION_CHOICES;
                who_starts=EnvironmentStarts(),
                cognitive_save_history=SaveHistory(; states=:expected_reward),
            ),
        )
        single_chain = sample(Xoshiro(9), single_model, Prior(), 10; progress=false)
        returned_values = returned(single_model, single_chain)
        single_parameters = get_session_parameters(returned_values)
        @test Turing.FlexiChains.get_name.(keys(single_parameters)) ==
            [@varname(learning_rate), @varname(inverse_temperature)]
        @test vec(single_parameters[@varname(learning_rate)]) ≈
            to_unit_interval.(vec(single_chain[@varname(learning_rate_logit)]))
        single_trajectories = get_session_trajectories(returned_values)
        @test single_trajectories[@varname(expected_reward)][3] ≈ expected_rewards_by_hand(
            SESSION_REWARDS.reward, single_parameters[@varname(learning_rate)][3]
        )
        @test Turing.FlexiChains.get_name.(
            keys(merge(single_trajectories, single_parameters))
        ) == [
            @varname(expected_reward),
            @varname(learning_rate),
            @varname(inverse_temperature),
        ]
    end

    @testset "A fit with values imputed in several sessions, at different timesteps, can be read" begin
        #FlexiChains.jl can't rebuild such a sample when the session parameters are a variable for each session, so this guards their naming
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
        session_trajectories = @test_logs match_mode = :any get_session_trajectories(
            imputing, prior_chain; cognitive_save_history=SaveHistory(; outputs=:choice)
        )
        @test length(session_trajectories[@varname(S3.choice)][1]) == 5
    end

    @testset "Sessions of different lengths each keep their own timesteps, also with values imputed at timesteps that the sessions before them don't have" begin
        different_lengths = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(
                WithDefaultParameters(DistributionsJl()), DIFFERENT_LENGTHS_TABLE
            ),
        )
        prior_chain = sample(Xoshiro(8), different_lengths, Prior(), 5; progress=false)
        session_trajectories = @test_logs match_mode = :any get_session_trajectories(
            different_lengths,
            prior_chain;
            cognitive_save_history=SaveHistory(; states=:expected_reward, outputs=:choice),
        )
        #Each entry has timestep 0 and the timesteps of its own session, for every sample
        for (name, n_timesteps) in (
            (@varname(S1.expected_reward), 1),
            (@varname(S2.expected_reward), 2),
            (@varname(S3.expected_reward), 5),
        )
            @test all(
                rewards -> length(rewards) == n_timesteps + 1,
                vec(session_trajectories[name]),
            )
        end
        #The imputed choices are read like the recorded ones
        @test all(
            choices -> length(choices) == 3 && choices[3] isa Bool,
            vec(session_trajectories[@varname(S2.choice)]),
        )
        @test all(
            choices -> length(choices) == 6 && choices[6] isa Bool,
            vec(session_trajectories[@varname(S3.choice)]),
        )
        #The parameters of the sessions are read in the same way
        parameters = get_session_parameters(different_lengths, prior_chain)
        @test vec(parameters[@varname(S3.learning_rate)]) ==
            map(rates -> rates[3], vec(prior_chain[@varname(learning_rate.session)]))
    end

    @testset "A fit where only some sessions run their environment can be read, also with the record of every session's environment" begin
        some_running = some_sessions_running_environment()
        prior_chain = sample(Xoshiro(9), some_running, Prior(), 5; progress=false)
        session_trajectories = @test_logs match_mode = :any get_session_trajectories(
            some_running,
            prior_chain;
            cognitive_save_history=SaveHistory(; inputs=:reward),
            environment_save_history=SaveHistory(; outputs=:reward),
        )

        #On their second timestep, S1 and S3 observed the rewards imputed in the fit, and S2 observed those in the data
        @test map(rewards -> rewards[3], vec(session_trajectories[@varname(S1.reward)])) ==
            vec(prior_chain[@varname(session[1].timestep[2].environment.reward.imputed)])
        @test map(rewards -> rewards[3], vec(session_trajectories[@varname(S3.reward)])) ==
            vec(prior_chain[@varname(session[3].timestep[2].environment.reward.imputed)])
        @test all(
            rewards -> rewards[2:end] == [0.0, 1.0],
            vec(session_trajectories[@varname(S2.reward)]),
        )

        #Every session's environment, which runs for its record, S2's too, gave the same rewards
        for (environment_name, cognitive_name) in (
            (@varname(S1.environment.reward), @varname(S1.reward)),
            (@varname(S2.environment.reward), @varname(S2.reward)),
            (@varname(S3.environment.reward), @varname(S3.reward)),
        )
            @test vec(session_trajectories[environment_name]) ==
                vec(session_trajectories[cognitive_name])
        end

        #The parameters of the sessions are read in the same way
        parameters = get_session_parameters(some_running, prior_chain)
        @test vec(parameters[@varname(S2.inverse_temperature)]) == map(
            temperatures -> temperatures[2],
            vec(prior_chain[@varname(inverse_temperature.session)]),
        )
    end

    @testset "A model of one's own is an error" begin
        Turing.@model own_model() = (x ~ Normal(); return x)
        @test_throws "get_session_trajectories(model, chain; cognitive_save_history) takes a model made by full_model" get_session_trajectories(
            own_model(), chain; cognitive_save_history=SaveHistory()
        )
    end

    @testset "A chain that isn't a FlexiChain is an error" begin
        not_a_chain = (; learning_rate=[0.3])
        @test_throws "get_session_trajectories(model, chain) takes a FlexiChain, but the chain is of type NamedTuple." get_session_trajectories(
            model,
            not_a_chain;
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        @test_throws "get_session_parameters(model, chain) takes a FlexiChain, but the chain is of type NamedTuple." get_session_parameters(
            model, not_a_chain
        )
    end
end
