## Models for the tests ##
#A population model of its own: one learning rate for every session, and an inverse temperature for each
struct SharedLearningRate <: AbstractPopulationModel end
Turing.@model function CognitiveModeling.turing_population_model(population_model::SharedLearningRate, sessions_info::MultipleSessionsInfo)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ filldist(LogNormal(0, 0.5), CognitiveModeling.n_sessions(sessions_info))
    return (; learning_rate = fill(learning_rate, CognitiveModeling.n_sessions(sessions_info)), inverse_temperature)
end

#A downstream model of its own: a measurement for each session regressed on its last expected reward
struct MeasurementRegression{T} <: AbstractDownstreamModel
    measurements::T
end
Turing.@model function CognitiveModeling.turing_downstream_model(downstream_model::MeasurementRegression, saved_behavioural_values, cognitive_parameters, sessions_info::MultipleSessionsInfo)
    slope ~ Normal(0, 1)
    last_expected_rewards = [saved.states[end].expected_reward for saved in saved_behavioural_values]
    downstream_model.measurements ~ MvNormal(slope .* last_expected_rewards, 1.0)
    return slope
end

#The data of three sessions, as a table, where one choice wasn't recorded
const FULL_TABLE = (;
    id = ["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3", "S3"],
    reward = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    choice = [true, false, true, false, true, true, true, missing, true],
)
const FULL_PRIORS = (; learning_rate = Beta(2, 2), inverse_temperature = LogNormal(0, 0.5))

#The sessions of the table, prepared once for the tests
function sessions_for_test(model = SessionRescorlaWagner(), table = FULL_TABLE; keywords...)
    return MultipleSessionsInfo(model, table; observation_columns = :reward, action_columns = :choice, session_columns = :id, who_starts = EnvironmentStarts(), keywords...)
end

#The log density of the sessions of the table by hand, with the parameters of each session
function sessions_by_hand(table, learning_rates, inverse_temperatures)
    log_density = 0.0
    for (i, id) in enumerate(unique(table.id))
        rows = findall(==(id), table.id)
        choices = [choice === missing ? MissingValue() : choice for choice in table.choice[rows]]
        log_density += session_by_hand(table.reward[rows], choices, learning_rates[i], inverse_temperatures[i], EnvironmentStarts())[1]
    end
    return log_density
end

@testset "turing_multiple_behavioural_sessions" begin
    @testset "Each session runs with its own parameters under session[i], and the log density is the sum over the sessions" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        learning_rates = [0.2, 0.3, 0.4]
        inverse_temperatures = [1.0, 2.0, 3.0]
        model = turing_multiple_behavioural_sessions(sessions_info, (; learning_rate = learning_rates, inverse_temperature = inverse_temperatures))
        @test logjoint(model, (;)) ≈ sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        @test isempty(keys(rand(Xoshiro(1), model)))
        #It returns what each session returns, in order
        saved = returned(model, (;))
        @test saved isa Vector && length(saved) == 3
        @test [length(session.states) for session in saved] == [4, 3, 5]
        @test saved[2].states[2].expected_reward ≈ 0.5 + 0.3 * (0.0 - 0.5)
    end

    @testset "A missing action in one session is a latent under that session, and leaves the others as they are" begin
        model = turing_multiple_behavioural_sessions(sessions_for_test(), (; learning_rate = [0.3, 0.3, 0.3], inverse_temperature = [2.0, 2.0, 2.0]))
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test collect(keys(values)) == [@varname(session[3].timestep[3].choice.imputed)]
        imputed = Dict(@varname(session[3].timestep[3].choice.imputed) => false)
        with_false = (; FULL_TABLE..., choice = [true, false, true, false, true, true, true, false, true])
        @test (@test_logs match_mode = :any logjoint(model, imputed)) ≈ sessions_by_hand(with_false, fill(0.3, 3), fill(2.0, 3))
    end

    @testset "Sessions whose saved values have different types are returned in one vector which holds them all" begin
        #S3's missing choice isn't imputed, so S3 saves NoAction() there, which the other sessions don't
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; actions = :choice))
        model = turing_multiple_behavioural_sessions(sessions_info, (; learning_rate = [0.2, 0.3, 0.4], inverse_temperature = [1.0, 2.0, 3.0]))
        saved = returned(model, (;))
        @test saved[1].actions == [(; choice = BeforeFirstTimestep()), (; choice = true), (; choice = false), (; choice = true)]
        @test saved[3].actions == [(; choice = BeforeFirstTimestep()), (; choice = true), (; choice = true), (; choice = NoAction()), (; choice = true)]
        @test logjoint(model, (;)) ≈ sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
    end

    @testset "The parameters must have one value for each session" begin
        model = turing_multiple_behavioural_sessions(sessions_for_test(; impute_missing_actions = false), (; learning_rate = [0.3, 0.3], inverse_temperature = [2.0, 2.0]))
        @test_throws BoundsError logjoint(model, (;))
    end

    @testset "Environment parameters are shared by the sessions, and can only be given when the sessions have an environment" begin
        table = (; FULL_TABLE..., reward = [1.0, missing, 1.0, 0.0, 1.0, 1.0, missing, 0.0, 1.0])
        sessions_info = sessions_for_test(SessionRescorlaWagner(), table; environment = RewardsPreviousChoice(), impute_missing_actions = false, save_history = SaveHistory(; observations = SaveEverything()))
        parameters = (; learning_rate = fill(0.3, 3), inverse_temperature = fill(2.0, 3))
        saved = returned(turing_multiple_behavioural_sessions(sessions_info, parameters, (; scale = 2.0)), (;))
        #RewardsPreviousChoice gives the previous choice scaled, so the second reward of S1 is 2.0 after a choice of true
        @test saved[1].observations[3] == (; reward = 2.0) && saved[3].observations[3] == (; reward = 2.0)
        #Without environment parameters, its defaults are used
        @test returned(turing_multiple_behavioural_sessions(sessions_info, parameters), (;))[1].observations[3] == (; reward = 1.0)
        @test_throws "Environment parameters were given, (scale = 2.0,), but the session has no environment." logjoint(turing_multiple_behavioural_sessions(sessions_for_test(; impute_missing_actions = false), parameters, (; scale = 2.0)), (;))
    end
end

@testset "turing_full_model" begin
    @testset "IndependentSessions samples each parameter for every session from its prior, named after the parameter" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false)
        model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info)
        values = rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([@varname(learning_rate.session), @varname(inverse_temperature.session)])
        @test length(values[@varname(learning_rate.session)]) == 3
        #The log density is the priors and the sessions with those parameters
        learning_rates = [0.2, 0.3, 0.4]
        inverse_temperatures = [1.0, 2.0, 3.0]
        given = Dict(@varname(learning_rate.session) => learning_rates, @varname(inverse_temperature.session) => inverse_temperatures)
        expected_prior = sum(logpdf.(Beta(2, 2), learning_rates)) + sum(logpdf.(LogNormal(0, 0.5), inverse_temperatures))
        @test logprior(model, given) ≈ expected_prior
        @test loglikelihood(model, given) ≈ sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        @test logjoint(model, given) ≈ expected_prior + sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
    end

    @testset "It returns the parameters of every session, what each session saved, and what the downstream model returned" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info)
        given = Dict(@varname(learning_rate.session) => [0.2, 0.3, 0.4], @varname(inverse_temperature.session) => [1.0, 2.0, 3.0])
        result = returned(model, given)
        @test keys(result) == (:cognitive_parameters, :saved_behavioural_values, :downstream_values)
        @test result.cognitive_parameters == (; learning_rate = [0.2, 0.3, 0.4], inverse_temperature = [1.0, 2.0, 3.0])
        @test length(result.saved_behavioural_values) == 3
        @test result.saved_behavioural_values[2].states[2].expected_reward ≈ 0.5 + 0.3 * (0.0 - 0.5)
        @test result.downstream_values === nothing
    end

    @testset "With include_submodel_prefixes, the population and downstream models have their variables under their names" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; downstream_model = MeasurementRegression([0.4, 0.6, 0.5]), include_submodel_prefixes = true)
        @test Set(keys(rand(Xoshiro(1), model))) == Set([@varname(population_model.learning_rate.session), @varname(population_model.inverse_temperature.session), @varname(downstream_model.slope)])
        without = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; downstream_model = MeasurementRegression([0.4, 0.6, 0.5]))
        @test Set(keys(rand(Xoshiro(1), without))) == Set([@varname(learning_rate.session), @varname(inverse_temperature.session), @varname(slope)])
        #The log densities are the same with the names changed
        given = Dict(@varname(population_model.learning_rate.session) => [0.2, 0.3, 0.4], @varname(population_model.inverse_temperature.session) => [1.0, 2.0, 3.0], @varname(downstream_model.slope) => 0.8)
        given_without = Dict(@varname(learning_rate.session) => [0.2, 0.3, 0.4], @varname(inverse_temperature.session) => [1.0, 2.0, 3.0], @varname(slope) => 0.8)
        @test logjoint(model, given) ≈ logjoint(without, given_without)
    end

    @testset "A population model of its own gives the parameters in the same form" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false)
        model = turing_full_model(SharedLearningRate(), sessions_info)
        @test Set(keys(rand(Xoshiro(1), model))) == Set([@varname(learning_rate), @varname(inverse_temperature)])
        given = Dict(@varname(learning_rate) => 0.3, @varname(inverse_temperature) => [1.0, 2.0, 3.0])
        @test loglikelihood(model, given) ≈ sessions_by_hand(FULL_TABLE, fill(0.3, 3), [1.0, 2.0, 3.0])
        @test returned(model, given).cognitive_parameters.learning_rate == fill(0.3, 3)
    end

    @testset "A downstream model of its own gets what the sessions saved" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        measurements = [0.4, 0.6, 0.5]
        model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; downstream_model = MeasurementRegression(measurements))
        @test @varname(slope) in keys(rand(Xoshiro(1), model))
        given = Dict(@varname(learning_rate.session) => [0.2, 0.3, 0.4], @varname(inverse_temperature.session) => [1.0, 2.0, 3.0], @varname(slope) => 0.8)
        result = returned(model, given)
        @test result.downstream_values == 0.8
        last_expected_rewards = [session.states[end].expected_reward for session in result.saved_behavioural_values]
        @test loglikelihood(model, given) ≈ sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0]) + logpdf(MvNormal(0.8 .* last_expected_rewards, 1.0), measurements)
    end

    @testset "The environment parameters and the kinds of the parts are checked" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false)
        @test_throws "Environment parameters were given, (scale = 2.0,), but the session has no environment." logjoint(turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; environment_parameters = (; scale = 2.0)), Dict(@varname(learning_rate.session) => fill(0.3, 3), @varname(inverse_temperature.session) => fill(2.0, 3)))
        @test_throws "The population model must be a subtype of AbstractPopulationModel" rand(Xoshiro(1), turing_full_model(FULL_PRIORS, sessions_info))
        @test_throws "The downstream model must be a subtype of AbstractDownstreamModel" rand(Xoshiro(1), turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; downstream_model = :none))
    end

    @testset "It can be sampled, and returned gives the parameters and the states of every session for each sample" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        model = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info)
        chain = sample(Xoshiro(1), model, NUTS(), 50; progress = false)
        @test size(chain, 1) == 50
        @test Set(Turing.FlexiChains.parameters(chain)) == Set([@varname(learning_rate.session), @varname(inverse_temperature.session)])
        results = returned(model, chain)
        @test length(results) == 50
        @test length(results[1].cognitive_parameters.learning_rate) == 3
        @test [length(session.states) for session in results[1].saved_behavioural_values] == [4, 3, 5]
        #With the missing choice imputed, its latent is sampled too, by a sampler for discrete variables
        with_imputed = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_for_test())
        chain = @test_logs match_mode = :any sample(Xoshiro(1), with_imputed, Gibbs((@varname(learning_rate), @varname(inverse_temperature)) => NUTS(), @varname(session) => PG(5)), 20; progress = false)
        @test @varname(session[3].timestep[3].choice.imputed) in Turing.FlexiChains.parameters(chain)
    end

    @testset "turing_catch_invalid_parameters wraps the sessions, so one rejected session rejects the whole sample" begin
        table = (; FULL_TABLE..., choice = [0.2, 0.4, 0.6, 0.8, 1.0, 0.2, 0.4, 0.6, 0.8])
        sessions_info = sessions_for_test(RejectsHighLearningRate(), table; save_history = SaveHistory(; states = :expected_reward))
        parameters = (; learning_rate = [0.3, 1.5, 0.3])
        @test_throws "The learning rate must be at most 1." logjoint(turing_multiple_behavioural_sessions(sessions_info, parameters), (;))
        caught = turing_catch_invalid_parameters(turing_multiple_behavioural_sessions(sessions_info, parameters))
        @test logjoint(caught, (;)) == -Inf && returned(caught, (;)) === nothing
        #Valid parameters are unaffected
        valid = turing_catch_invalid_parameters(turing_multiple_behavioural_sessions(sessions_info, (; learning_rate = fill(0.3, 3))))
        @test logjoint(valid, (;)) ≈ logjoint(turing_multiple_behavioural_sessions(sessions_info, (; learning_rate = fill(0.3, 3))), (;))
        @test length(returned(valid, (;))) == 3
        #The full model puts it around the sessions with the keyword, and returns nothing for a rejected sample without running the downstream model
        model = turing_full_model(IndependentSessions((; learning_rate = Uniform(0, 2))), sessions_info; catch_invalid_parameters = true, downstream_model = MeasurementRegression([0.4, 0.6, 0.5]))
        rejected = Dict(@varname(learning_rate.session) => [0.3, 1.5, 0.3], @varname(slope) => 0.8)
        @test logjoint(model, rejected) == -Inf
        @test returned(model, rejected) === nothing
        accepted = Dict(@varname(learning_rate.session) => fill(0.3, 3), @varname(slope) => 0.8)
        @test isfinite(logjoint(model, accepted)) && returned(model, accepted).downstream_values == 0.8
        #A sampler starts when most first draws are rejected, since the parameters are sampled outside the try block
        without_downstream = turing_full_model(IndependentSessions((; learning_rate = Uniform(0, 2))), sessions_info; catch_invalid_parameters = true)
        for seed in 1:3
            chain = sample(Xoshiro(seed), without_downstream, NUTS(), 10; progress = false)
            @test size(chain, 1) == 10 && all(result -> result isa NamedTuple, returned(without_downstream, chain))
        end
    end

    @testset "ParallelSessions runs the sessions over the threads, and needs the model to be marked thread-safe" begin
        sessions_info = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; states = :expected_reward))
        parameters = (; learning_rate = [0.2, 0.3, 0.4], inverse_temperature = [1.0, 2.0, 3.0])
        in_sequence = turing_multiple_behavioural_sessions(sessions_info, parameters)
        in_parallel = turing_multiple_behavioural_sessions(sessions_info, parameters; parallelisation = ParallelSessions())
        @test_throws "The sessions run in parallel, so the model that is evaluated must be marked thread-safe with setthreadsafe(model, true) from DynamicPPL." logjoint(in_parallel, (;))
        threadsafe = Turing.DynamicPPL.setthreadsafe(in_parallel, true)
        @test logjoint(threadsafe, (;)) ≈ logjoint(in_sequence, (;))
        @test returned(threadsafe, (;))[2].states == returned(in_sequence, (;))[2].states
        #Sessions whose saved values have different types give what they give in sequence, since S3 saves NoAction() for its missing choice
        saving_actions = sessions_for_test(; impute_missing_actions = false, save_history = SaveHistory(; actions = :choice))
        in_parallel_saving_actions = Turing.DynamicPPL.setthreadsafe(turing_multiple_behavioural_sessions(saving_actions, parameters; parallelisation = ParallelSessions()), true)
        @test returned(in_parallel_saving_actions, (;)) == returned(turing_multiple_behavioural_sessions(saving_actions, parameters), (;))
        #The full model passes it on
        full = Turing.DynamicPPL.setthreadsafe(turing_full_model(IndependentSessions(FULL_PRIORS), sessions_info; parallelisation = ParallelSessions()), true)
        given = Dict(@varname(learning_rate.session) => [0.2, 0.3, 0.4], @varname(inverse_temperature.session) => [1.0, 2.0, 3.0])
        @test loglikelihood(full, given) ≈ sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
    end

    @testset "A chain from a model that saves nothing gives the states through returned with a copy that saves" begin
        unsaved_sessions = sessions_for_test(; impute_missing_actions = false)
        saving_sessions = MultipleSessionsInfo(unsaved_sessions; save_history = SaveHistory(; states = :expected_reward))
        unsaved = turing_full_model(IndependentSessions(FULL_PRIORS), unsaved_sessions)
        saving = turing_full_model(IndependentSessions(FULL_PRIORS), saving_sessions)
        chain = sample(Xoshiro(2), unsaved, NUTS(), 20; progress = false)
        results = returned(saving, chain)
        @test length(results) == 20
        @test [length(session.states) for session in results[1].saved_behavioural_values] == [4, 3, 5]
        @test returned(unsaved, chain)[1].saved_behavioural_values[1].states == fill((;), 4)
        #The first state of each session is the initial one, and the parameters are the chain's
        learning_rates = results[1].cognitive_parameters.learning_rate
        @test results[1].saved_behavioural_values[1].states[2].expected_reward ≈ 0.5 + learning_rates[1] * (1.0 - 0.5)
    end

    @testset "Evaluating the sessions only allocates the distributions they keep, also with a column that can hold a MissingValue" begin
        function table_with(n_timesteps)
            return (; id = repeat([1, 2], inner = n_timesteps), reward = repeat([1.0, 0.0], n_timesteps), choice = Vector{Union{Missing, Bool}}(repeat([true, false], n_timesteps)))
        end
        model_with(n_timesteps) = turing_full_model(IndependentSessions(FULL_PRIORS), sessions_for_test(SessionRescorlaWagner(), table_with(n_timesteps)))
        given = Dict(@varname(learning_rate.session) => [0.2, 0.3], @varname(inverse_temperature.session) => [1.0, 2.0])
        short_model = model_with(5_000)
        long_model = model_with(10_000)
        logjoint(short_model, given)
        logjoint(long_model, given)
        #Doubling the timesteps only adds the vectors of the distributions of the recorded actions, about 23 bytes per timestep
        @test @allocated(logjoint(long_model, given)) - @allocated(logjoint(short_model, given)) <= 40 * 10_000
    end
end
