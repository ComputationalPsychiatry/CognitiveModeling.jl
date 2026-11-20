## Models for the tests ##
#A parameter model of its own: one learning rate for every session, and an inverse temperature for each
struct SharedLearningRate <: AbstractParametersModel end
Turing.@model function CognitiveModeling.turing_parameters_model(
    parameters_model::SharedLearningRate, sessions_info::MultipleSessionsInfo
)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~
    filldist(LogNormal(0, 0.5), CognitiveModeling.n_sessions(sessions_info))
    return (;
        learning_rate=fill(learning_rate, CognitiveModeling.n_sessions(sessions_info)),
        inverse_temperature,
    )
end

#A downstream model of its own: a measurement for each session regressed on its last expected reward
struct MeasurementRegression{T} <: AbstractDownstreamModel
    measurements::T
end
Turing.@model function CognitiveModeling.turing_downstream_model(
    downstream_model::MeasurementRegression,
    saved_trajectories,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    slope ~ Normal(0, 1)
    last_expected_rewards = [
        saved.states[end].expected_reward for saved in saved_trajectories
    ]
    downstream_model.measurements ~ MvNormal(slope .* last_expected_rewards, 1.0)
    return slope
end

#An environment which gives a reward around a mean which is a parameter, so that the rewards in the data inform the mean
struct RewardsAroundMean <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsAroundMean, parameters) = (;)
function CognitiveModeling.run_timestep(
    environment::RewardsAroundMean, context::ModelContext
)
    return ModelOutput((; reward=Normal(context.parameters.mean_reward, 1.0)))
end
function CognitiveModeling.attribute_names(environment::RewardsAroundMean)
    return ModelAttributes(;
        parameters=:mean_reward, states=(), inputs=:choice, outputs=:reward
    )
end

#An environment which gives a reward with a probability which it calls the learning rate, like the cognitive model does
struct RewardsWithLearningRate <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsWithLearningRate, parameters) = (;)
function CognitiveModeling.run_timestep(
    environment::RewardsWithLearningRate, context::ModelContext
)
    return ModelOutput((; reward=Bernoulli(context.parameters.learning_rate)))
end
function CognitiveModeling.attribute_names(environment::RewardsWithLearningRate)
    return ModelAttributes(;
        parameters=:learning_rate, states=(), inputs=:choice, outputs=:reward
    )
end

#The data of three sessions, as a table, where one choice wasn't recorded
const FULL_TABLE = (;
    id=["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3", "S3"],
    reward=[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    choice=[true, false, true, false, true, true, true, missing, true],
)
const FULL_PRIORS = (; learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5))

#The same, where S1 and S3 each miss a reward, so that with an environment only they run it
const TWO_MISSING_REWARDS_TABLE = (;
    FULL_TABLE..., reward=[1.0, missing, 1.0, 0.0, 1.0, 1.0, missing, 0.0, 1.0]
)

#Maps the real line to the values between 0 and 1, as a transformation for a learning rate
to_unit_interval(x) = 1 / (1 + exp(-x))

#The three sessions of FULL_TABLE with a column that groups them: S1 and S3 in one group, and S2 in another
const GROUPED_FULL_TABLE = merge(
    FULL_TABLE, (; group=["a", "a", "a", "b", "b", "a", "a", "a", "a"])
)
const GROUP_PRIORS = (; mean=Normal(0, 1), sd=Exponential(1))

#The sessions of the table, prepared once for the tests
function sessions_for_test(
    model=WithDefaultParameters(DistributionsJl()), table=FULL_TABLE; keywords...
)
    return MultipleSessionsInfo(
        model,
        table;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

#A full model of sessions which save trajectories, without a downstream model, which warns when it is made
const SAVING_WARNING = (
    :warn,
    "The sessions save trajectories, which adds a cost to each gradient, but there is no downstream model to use them. A fit is fastest with sessions that save nothing, and get_session_trajectories(model, chain; cognitive_save_history, environment_save_history) then gives the trajectories of every sample afterwards.",
)
function saving_full_model(arguments...; keywords...)
    return @test_logs SAVING_WARNING full_model(TuringBackend(), arguments...; keywords...)
end

#The log density of the sessions of the table by hand, with the parameters of each session
function sessions_by_hand(table, learning_rates, inverse_temperatures)
    log_density = 0.0
    for (i, id) in enumerate(unique(table.id))
        rows = findall(==(id), table.id)
        choices = [
            choice === missing ? MissingValue() : choice for choice in table.choice[rows]
        ]
        log_density += session_by_hand(
            table.reward[rows],
            choices,
            learning_rates[i],
            inverse_temperatures[i],
            EnvironmentStarts(),
        )[1]
    end
    return log_density
end

@testset "turing_multiple_behavioral_sessions" begin
    @testset "Each session runs with its own parameters under session[i], and the log density is the sum over the sessions" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        learning_rates = [0.2, 0.3, 0.4]
        inverse_temperatures = [1.0, 2.0, 3.0]
        model = turing_multiple_behavioral_sessions(
            sessions_info,
            (; learning_rate=learning_rates, inverse_temperature=inverse_temperatures),
        )
        @test logjoint(model, (;)) ≈
            sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        @test isempty(keys(rand(Xoshiro(1), model)))
        #It returns the parameters it was given, and what each session saved, in order
        returned_value = returned(model, (;))
        @test returned_value.cognitive_parameters ==
              (; learning_rate=learning_rates, inverse_temperature=inverse_temperatures) &&
            returned_value.environment_parameters == (;)
        saved = returned_value.saved_trajectories
        @test saved isa Vector && length(saved) == 3
        @test [length(session.states) for session in saved] == [4, 3, 5]
        @test saved[2].states[2].expected_reward ≈ 0.5 + 0.3 * (0.0 - 0.5)
    end

    @testset "A missing action in one session is a latent under that session, and leaves the others as they are" begin
        model = turing_multiple_behavioral_sessions(
            sessions_for_test(),
            (; learning_rate=[0.3, 0.3, 0.3], inverse_temperature=[2.0, 2.0, 2.0]),
        )
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test collect(keys(values)) == [@varname(session[3].timestep[3].choice.imputed)]
        imputed = Dict(@varname(session[3].timestep[3].choice.imputed) => false)
        with_false = (;
            FULL_TABLE..., choice=[true, false, true, false, true, true, true, false, true]
        )
        @test (@test_logs match_mode = :any logjoint(model, imputed)) ≈
            sessions_by_hand(with_false, fill(0.3, 3), fill(2.0, 3))
    end

    @testset "An environment runs only in the sessions where an observation is missing, so only they have its variables" begin
        #S1 misses a reward, and S2 and S3 have all of theirs
        table = (; FULL_TABLE..., reward=[1.0, missing, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        sessions_info = sessions_for_test(
            WithDefaultParameters(DistributionsJl()),
            table;
            environment=DriftingRewards(),
            impute_missing_actions=false,
        )
        model = turing_multiple_behavioral_sessions(
            sessions_info, (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        )
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) ==
            Set([@varname(session[1].timestep[t].environment.drift) for t in 1:3])
    end

    @testset "Sessions whose saved trajectories have different types are returned in one vector which holds them all" begin
        #S3's missing choice isn't imputed, so S3 saves UnimputedMissingValue() there, which the other sessions don't
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; outputs=:choice),
        )
        model = turing_multiple_behavioral_sessions(
            sessions_info,
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0]),
        )
        saved = returned(model, (;)).saved_trajectories
        @test saved[1].actions == [
            (; choice=BeforeFirstTimestep()),
            (; choice=true),
            (; choice=false),
            (; choice=true),
        ]
        @test saved[3].actions == [
            (; choice=BeforeFirstTimestep()),
            (; choice=true),
            (; choice=true),
            (; choice=UnimputedMissingValue()),
            (; choice=true),
        ]
        @test logjoint(model, (;)) ≈
            sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
    end

    @testset "The parameters must have one value for each session" begin
        model = turing_multiple_behavioral_sessions(
            sessions_for_test(; impute_missing_actions=false),
            (; learning_rate=[0.3, 0.3], inverse_temperature=[2.0, 2.0]),
        )
        @test_throws BoundsError logjoint(model, (;))
    end

    @testset "Environment parameters are given as one vector for each parameter, like the cognitive parameters, and only when the sessions have an environment" begin
        table = (;
            FULL_TABLE..., reward=[1.0, missing, 1.0, 0.0, 1.0, 1.0, missing, 0.0, 1.0]
        )
        sessions_info = sessions_for_test(
            WithDefaultParameters(DistributionsJl()),
            table;
            environment=RewardsPreviousChoice(),
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; inputs=SaveEverything()),
        )
        parameters = (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        returned_value = returned(
            turing_multiple_behavioral_sessions(
                sessions_info, parameters, (; scale=[2.0, 1.0, 3.0])
            ),
            (;),
        )
        #RewardsPreviousChoice gives the previous choice scaled, so the second reward of S1 is 2.0 after a choice of true, and S3's is 3.0
        saved = returned_value.saved_trajectories
        @test saved[1].observations[3] == (; reward=2.0) &&
            saved[3].observations[3] == (; reward=3.0)
        #The environment parameters are returned as they were given
        @test returned_value.environment_parameters == (; scale=[2.0, 1.0, 3.0])
        #Without environment parameters, its defaults are used, and none are returned
        returned_value = returned(
            turing_multiple_behavioral_sessions(sessions_info, parameters), (;)
        )
        @test returned_value.saved_trajectories[1].observations[3] == (; reward=1.0) &&
            returned_value.environment_parameters == (;)
        @test_throws "Environment parameters were given, (scale = [2.0, 2.0, 2.0],), but the session has no environment." logjoint(
            turing_multiple_behavioral_sessions(
                sessions_for_test(; impute_missing_actions=false),
                parameters,
                (; scale=fill(2.0, 3)),
            ),
            (;),
        )
    end
end

@testset "full_model" begin
    @testset "IndependentParameters samples each parameter for every session from its prior, named after the parameter" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        model = full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info
        )
        values = rand(Xoshiro(1), model)
        @test Set(keys(values)) ==
            Set([@varname(learning_rate.session), @varname(inverse_temperature.session)])
        @test length(values[@varname(learning_rate.session)]) == 3
        #The log density is the priors and the sessions with those parameters
        learning_rates = [0.2, 0.3, 0.4]
        inverse_temperatures = [1.0, 2.0, 3.0]
        given = Dict(
            @varname(learning_rate.session) => learning_rates,
            @varname(inverse_temperature.session) => inverse_temperatures,
        )
        expected_prior =
            sum(logpdf.(Beta(2, 2), learning_rates)) +
            sum(logpdf.(LogNormal(0, 0.5), inverse_temperatures))
        @test logprior(model, given) ≈ expected_prior
        @test loglikelihood(model, given) ≈
            sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        @test logjoint(model, given) ≈
            expected_prior +
              sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
    end

    @testset "Fixed parameters are given to every session, next to those of the parameter model" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        learning_rate_priors = IndependentParameters((; learning_rate=Beta(2, 2)))
        model = full_model(
            TuringBackend(),
            learning_rate_priors,
            sessions_info;
            fixed_cognitive_parameters=(; inverse_temperature=3.0),
        )
        learning_rates = [0.2, 0.3, 0.4]
        given = Dict(@varname(learning_rate.session) => learning_rates)
        @test collect(keys(rand(Xoshiro(1), model))) == [@varname(learning_rate.session)]
        @test logjoint(model, given) ≈
            sum(logpdf.(Beta(2, 2), learning_rates)) +
              sessions_by_hand(FULL_TABLE, learning_rates, fill(3.0, 3))
        #The fixed values are returned with the parameters of every session
        @test returned(model, given).cognitive_parameters.inverse_temperature ==
            fill(3.0, 3)
        #It is the composite of the parameter model and the fixed values
        by_hand = full_model(
            TuringBackend(),
            CompositeParameters(
                learning_rate_priors, FixedParameters((; inverse_temperature=3.0))
            ),
            sessions_info,
        )
        @test logjoint(model, given) ≈ logjoint(by_hand, given)
        #A part of a composite given with a name has its variables under that name, also next to the fixed values
        named_model = full_model(
            TuringBackend(),
            CompositeParameters(; learning=learning_rate_priors),
            sessions_info;
            fixed_cognitive_parameters=(; inverse_temperature=3.0),
        )
        @test collect(keys(rand(Xoshiro(1), named_model))) ==
            [@varname(learning.learning_rate.session)]
        @test logjoint(
            named_model, Dict(@varname(learning.learning_rate.session) => learning_rates)
        ) ≈ logjoint(model, given)
        #A parameter model of one's own takes them too, and a parameter can't be both fixed and given
        with_own_model = full_model(
            TuringBackend(),
            SharedLearningRate(),
            sessions_for_test(StartsFromParameter(); impute_missing_actions=false);
            fixed_cognitive_parameters=(; initial_value=0.3),
        )
        @test returned(with_own_model, rand(Xoshiro(2), with_own_model)).cognitive_parameters.initial_value ==
            fill(0.3, 3)
        @test_throws "fixed_cognitive_parameters fixes the parameters (:learning_rate,), which the parameter model, a IndependentParameters, gives too. A parameter is either fixed or given by the parameter model." full_model(
            TuringBackend(),
            learning_rate_priors,
            sessions_info;
            fixed_cognitive_parameters=(; learning_rate=0.3),
        )
    end

    @testset "It returns the parameters of every session, what each session saved, and what the downstream model returned" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        model = saving_full_model(IndependentParameters(FULL_PRIORS), sessions_info)
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
        )
        returned_value = returned(model, given)
        @test keys(returned_value) == (
            :cognitive_parameters,
            :environment_parameters,
            :saved_trajectories,
            :downstream_values,
        )
        @test returned_value.environment_parameters === (;)
        @test returned_value.cognitive_parameters ==
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        @test length(returned_value.saved_trajectories) == 3
        @test returned_value.saved_trajectories[2].states[2].expected_reward ≈
            0.5 + 0.3 * (0.0 - 0.5)
        @test returned_value.downstream_values === nothing
    end

    @testset "With include_submodel_prefixes, the parameter models and the downstream model have their variables under their names" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        model = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            downstream_model=MeasurementRegression([0.4, 0.6, 0.5]),
            include_submodel_prefixes=true,
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set([
            @varname(cognitive_parameters_model.learning_rate.session),
            @varname(cognitive_parameters_model.inverse_temperature.session),
            @varname(downstream_model.slope),
        ])
        without = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            downstream_model=MeasurementRegression([0.4, 0.6, 0.5]),
        )
        @test Set(keys(rand(Xoshiro(1), without))) == Set([
            @varname(learning_rate.session),
            @varname(inverse_temperature.session),
            @varname(slope),
        ])
        #The log densities are the same with the names changed
        given = Dict(
            @varname(cognitive_parameters_model.learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(cognitive_parameters_model.inverse_temperature.session) =>
                [1.0, 2.0, 3.0],
            @varname(downstream_model.slope) => 0.8,
        )
        given_without = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
            @varname(slope) => 0.8,
        )
        @test logjoint(model, given) ≈ logjoint(without, given_without)
    end

    @testset "A parameter model of one's own gives the parameters in the same form" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        model = full_model(TuringBackend(), SharedLearningRate(), sessions_info)
        @test Set(keys(rand(Xoshiro(1), model))) ==
            Set([@varname(learning_rate), @varname(inverse_temperature)])
        given = Dict(
            @varname(learning_rate) => 0.3, @varname(inverse_temperature) => [1.0, 2.0, 3.0]
        )
        @test loglikelihood(model, given) ≈
            sessions_by_hand(FULL_TABLE, fill(0.3, 3), [1.0, 2.0, 3.0])
        @test returned(model, given).cognitive_parameters.learning_rate == fill(0.3, 3)
    end

    @testset "A downstream model of its own gets what the sessions saved" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        measurements = [0.4, 0.6, 0.5]
        model = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            downstream_model=MeasurementRegression(measurements),
        )
        @test @varname(slope) in keys(rand(Xoshiro(1), model))
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
            @varname(slope) => 0.8,
        )
        returned_value = returned(model, given)
        @test returned_value.downstream_values == 0.8
        last_expected_rewards = [
            session.states[end].expected_reward for
            session in returned_value.saved_trajectories
        ]
        @test loglikelihood(model, given) ≈
            sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0]) +
              logpdf(MvNormal(0.8 .* last_expected_rewards, 1.0), measurements)
    end

    @testset "A tuple of parameter models gives the model of a composite of them, and a NamedTuple names their variables under its names" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        learning = IndependentParameters((; learning_rate=FULL_PRIORS.learning_rate))
        temperature = IdenticalParameters((;
            inverse_temperature=FULL_PRIORS.inverse_temperature
        ))
        from_tuple = full_model(TuringBackend(), (learning, temperature), sessions_info)
        composite = full_model(
            TuringBackend(), CompositeParameters(learning, temperature), sessions_info
        )
        values = Dict(
            @varname(learning_rate.session) => [0.2, 0.5, 0.8],
            @varname(inverse_temperature.shared) => 2.0,
        )
        @test Set(keys(rand(Xoshiro(1), from_tuple))) == Set(keys(values))
        @test logjoint(from_tuple, values) ≈ logjoint(composite, values)
        from_named_tuple = full_model(
            TuringBackend(), (; learning, temperature), sessions_info
        )
        named_values = Dict(
            @varname(learning.learning_rate.session) => [0.2, 0.5, 0.8],
            @varname(temperature.inverse_temperature.shared) => 2.0,
        )
        @test Set(keys(rand(Xoshiro(1), from_named_tuple))) == Set(keys(named_values))
        @test logjoint(from_named_tuple, named_values) ≈ logjoint(composite, values)
    end

    @testset "The environment parameters and the kinds of the parts are checked" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        #The parts are checked when the model is made
        @test_throws "The sessions have no environment model, but the parameter model of the environment, a FixedParameters, gives the parameters (:scale,)." full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=FixedParameters((; scale=2.0)),
        )
        @test_throws "A parameter model must be a subtype of AbstractParametersModel" full_model(
            TuringBackend(), FULL_PRIORS, sessions_info
        )
        @test_throws "The downstream model must be a subtype of AbstractDownstreamModel" full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            downstream_model=:none,
        )
        @test_throws "A parameter model must be a subtype of AbstractParametersModel" full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=(; scale=2.0),
        )
    end

    @testset "With IdenticalParameters for the environment, each of its parameters is sampled once, and every session then scores its observations" begin
        sessions_info = sessions_for_test(;
            environment=RewardsAroundMean(), impute_missing_actions=false
        )
        shared = IdenticalParameters((; mean_reward=Normal(0, 1)))
        model = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=shared,
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set([
            @varname(learning_rate.session),
            @varname(inverse_temperature.session),
            @varname(mean_reward.shared),
        ])
        learning_rates = [0.2, 0.3, 0.4]
        inverse_temperatures = [1.0, 2.0, 3.0]
        given = Dict(
            @varname(learning_rate.session) => learning_rates,
            @varname(inverse_temperature.session) => inverse_temperatures,
            @varname(mean_reward.shared) => 0.6,
        )
        #Every reward in the data counts with the environment's distribution, though no session misses one
        @test loglikelihood(model, given) ≈
            sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures) +
              sum(logpdf.(Normal(0.6, 1.0), FULL_TABLE.reward))
        @test logprior(model, given) ≈
            sum(logpdf.(Beta(2, 2), learning_rates)) +
              sum(logpdf.(LogNormal(0, 0.5), inverse_temperatures)) +
              logpdf(Normal(0, 1), 0.6)
        #With the parameters given, sessions with nothing missing run without the environment
        fixed = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=FixedParameters((; mean_reward=0.6)),
        )
        @test loglikelihood(
            fixed,
            Dict(
                @varname(learning_rate.session) => learning_rates,
                @varname(inverse_temperature.session) => inverse_temperatures,
            ),
        ) ≈ sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        #With the prefixes, its variables are under environment_parameters_model
        prefixed = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=shared,
            include_submodel_prefixes=true,
        )
        @test @varname(environment_parameters_model.mean_reward.shared) in
            keys(rand(Xoshiro(1), prefixed))
        #It can be sampled with NUTS
        chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)
        @test @varname(mean_reward.shared) in Turing.FlexiChains.parameters(chain)
    end

    @testset "The cognitive parameters can be fixed, for every session or by session, with nothing sampled" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        same = saving_full_model(
            FixedParameters((; learning_rate=0.3, inverse_temperature=2.0)), sessions_info
        )
        @test isempty(keys(rand(Xoshiro(1), same)))
        @test logjoint(same, (;)) ≈ sessions_by_hand(FULL_TABLE, fill(0.3, 3), fill(2.0, 3))
        @test returned(same, (;)).cognitive_parameters ==
            (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        #By session, the values are found by the session's data identifier, in any order
        by_session = saving_full_model(
            SessionwiseFixedParameters((;
                S2=(; learning_rate=0.3, inverse_temperature=2.0),
                S3=(; learning_rate=0.4, inverse_temperature=3.0),
                S1=(; learning_rate=0.2, inverse_temperature=1.0),
            )),
            sessions_info,
        )
        @test isempty(keys(rand(Xoshiro(1), by_session)))
        @test logjoint(by_session, (;)) ≈
            sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
        @test returned(by_session, (;)).cognitive_parameters ==
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        #A session without values is an error when the model is made, since the values are looked up then
        @test_logs SAVING_WARNING @test_throws "SessionwiseFixedParameters has no parameters for the session S3." full_model(
            TuringBackend(),
            SessionwiseFixedParameters((;
                S1=(; learning_rate=0.2, inverse_temperature=1.0),
                S2=(; learning_rate=0.3, inverse_temperature=2.0),
            )),
            sessions_info,
        )
    end

    @testset "Sessions which save without a downstream model give a warning when the model is made, since the fit is slower for nothing" begin
        saving = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        @test_logs SAVING_WARNING full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), saving
        )
        #The environment's history counts too, and with a downstream model or without saving there is no warning
        @test_logs SAVING_WARNING full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(;
                impute_missing_actions=false,
                environment=RewardsAroundMean(),
                environment_save_history=SaveHistory(; outputs=:reward),
            );
            environment_parameters_model=FixedParameters((; mean_reward=0.5)),
        )
        @test_logs full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            saving;
            downstream_model=MeasurementRegression([0.4, 0.6, 0.5]),
        )
        @test_logs full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(; impute_missing_actions=false),
        )
    end

    @testset "It returns the environment parameters of every session too, as the parameter model gave them" begin
        sessions_info = sessions_for_test(;
            environment=RewardsAroundMean(), impute_missing_actions=false
        )
        shared = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=IdenticalParameters((; mean_reward=Normal(0, 1))),
        )
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
            @varname(mean_reward.shared) => 0.6,
        )
        @test returned(shared, given).environment_parameters == (; mean_reward=fill(0.6, 3))
        by_session = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=SessionwiseFixedParameters((;
                S1=(; mean_reward=0.1), S2=(; mean_reward=0.2), S3=(; mean_reward=0.3)
            )),
        )
        @test returned(by_session, given).environment_parameters ==
            (; mean_reward=[0.1, 0.2, 0.3])
        #Without environment parameters there are none
        @test returned(
            full_model(TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_info),
            given,
        ).environment_parameters === (;)
    end

    @testset "IdenticalParameters gives every session the same sampled cognitive parameters, as name.shared" begin
        sessions_info = sessions_for_test(; impute_missing_actions=false)
        model = full_model(TuringBackend(), IdenticalParameters(FULL_PRIORS), sessions_info)
        @test Set(keys(rand(Xoshiro(1), model))) ==
            Set([@varname(learning_rate.shared), @varname(inverse_temperature.shared)])
        given = Dict(
            @varname(learning_rate.shared) => 0.3,
            @varname(inverse_temperature.shared) => 2.0,
        )
        @test loglikelihood(model, given) ≈
            sessions_by_hand(FULL_TABLE, fill(0.3, 3), fill(2.0, 3))
        @test logprior(model, given) ≈
            logpdf(Beta(2, 2), 0.3) + logpdf(LogNormal(0, 0.5), 2.0)
        @test returned(model, given).cognitive_parameters ==
            (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)
        @test Set(Turing.FlexiChains.parameters(chain)) ==
            Set([@varname(learning_rate.shared), @varname(inverse_temperature.shared)])
    end

    @testset "The environment can take different given parameters in each session, and only the sessions with something missing run it" begin
        #S1 misses a reward, which its environment gives around its own mean
        table = (; FULL_TABLE..., reward=[1.0, missing, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        sessions_info = sessions_for_test(
            WithDefaultParameters(DistributionsJl()),
            table;
            environment=RewardsAroundMean(),
            impute_missing_actions=false,
        )
        model = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            environment_parameters_model=SessionwiseFixedParameters((;
                S1=(; mean_reward=0.6), S2=(; mean_reward=0.0), S3=(; mean_reward=1.0)
            )),
        )
        values = rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([
            @varname(learning_rate.session),
            @varname(inverse_temperature.session),
            @varname(session[1].timestep[2].environment.reward.imputed),
        ])
        #S1's rewards, the imputed one and the two in the data, are scored against its own mean, since its environment runs, and the rewards of the other sessions aren't
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
            @varname(session[1].timestep[2].environment.reward.imputed) => 0.8,
        )
        with_imputed = (; table..., reward=[1.0, 0.8, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        @test logjoint(model, given) ≈
            sum(logpdf.(Beta(2, 2), [0.2, 0.3, 0.4])) +
              sum(logpdf.(LogNormal(0, 0.5), [1.0, 2.0, 3.0])) +
              sum(logpdf.(Normal(0.6, 1.0), [1.0, 0.8, 1.0])) +
              sessions_by_hand(with_imputed, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
    end

    @testset "Sessions that run their environment and a session that doesn't save their trajectories in one vector, also in parallel" begin
        #S1 and S3 run their environment, whose mean is fixed, to impute a reward each, and S2 doesn't run it
        sessions_info = sessions_for_test(
            WithDefaultParameters(DistributionsJl()),
            TWO_MISSING_REWARDS_TABLE;
            environment=RewardsAroundMean(),
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        @test [session.includes_environment for session in sessions_info.sessions] == [true, false, true]
        in_sequence = saving_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_info;
            fixed_environment_parameters=(; mean_reward=0.6),
        )
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
            @varname(session[1].timestep[2].environment.reward.imputed) => 0.8,
            @varname(session[3].timestep[2].environment.reward.imputed) => 0.1,
        )

        #The rewards of S1 and S3, imputed and recorded, are scored against the environment, and those of S2 aren't
        with_imputed = (;
            TWO_MISSING_REWARDS_TABLE...,
            reward=[1.0, 0.8, 1.0, 0.0, 1.0, 1.0, 0.1, 0.0, 1.0],
        )
        @test logjoint(in_sequence, given) ≈
            sum(logpdf.(Beta(2, 2), [0.2, 0.3, 0.4])) +
              sum(logpdf.(LogNormal(0, 0.5), [1.0, 2.0, 3.0])) +
              sum(logpdf.(Normal(0.6, 1.0), [1.0, 0.8, 1.0, 1.0, 0.1, 0.0, 1.0])) +
              sessions_by_hand(with_imputed, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])

        #One vector of one type holds what each session saved: the states from the imputed rewards, and a record of the environment, which saves nothing, for every saved timestep, empty for S2 too
        saved = returned(in_sequence, given).saved_trajectories
        @test isconcretetype(eltype(saved))
        @test saved[1].states[3].expected_reward ≈ 0.6 + 0.2 * (0.8 - 0.6)
        @test saved[3].states[3].expected_reward ≈ 0.7 + 0.4 * (0.1 - 0.7)
        @test saved[1].environment.states == fill((;), 4)
        @test saved[2].environment.states == fill((;), 3)
        @test saved[2].environment.observations == fill((;), 3)

        #In parallel, the log density and what the sessions saved are the same
        in_parallel = Turing.DynamicPPL.setthreadsafe(
            saving_full_model(
                IndependentParameters(FULL_PRIORS),
                sessions_info;
                fixed_environment_parameters=(; mean_reward=0.6),
                parallelization=ParallelSessions(),
            ),
            true,
        )
        @test logjoint(in_parallel, given) ≈ logjoint(in_sequence, given)
        @test returned(in_parallel, given).saved_trajectories == saved
    end

    @testset "A warning says when both parameter models sample a parameter which the cognitive model and the environment share, unless the prefixes are included" begin
        sessions_info = sessions_for_test(;
            environment=RewardsWithLearningRate(), impute_missing_actions=false
        )
        warning = (
            :warn,
            "The cognitive model and the environment model both have the parameters (:learning_rate,), and both parameter models sample, so their variables can have the same names. A name that both use is one variable for both, which sample rejects. Give full_model include_submodel_prefixes = true to keep them apart.",
        )
        shared = IdenticalParameters((; learning_rate=Beta(2, 2)))
        model = @test_logs warning full_model(
            TuringBackend(), shared, sessions_info; environment_parameters_model=shared
        )
        #The warning is given when the model is made, and not when it runs, since the model is differentiated
        values = @test_logs rand(Xoshiro(1), model)
        #The two parameters are then one variable, which sample rejects, and which the log density gives one value with both priors
        @test collect(keys(values)) == [@varname(learning_rate.shared)]
        @test logprior(model, Dict(@varname(learning_rate.shared) => 0.3)) ≈
            2 * logpdf(Beta(2, 2), 0.3)
        @test_logs match_mode = :any @test_throws ArgumentError sample(
            Xoshiro(1), model, NUTS(), 5; progress=false
        )
        #With the prefixes, no warning, and the parameters are kept apart
        prefixed = @test_logs full_model(
            TuringBackend(),
            shared,
            sessions_info;
            environment_parameters_model=shared,
            include_submodel_prefixes=true,
        )
        @test Set(keys(rand(Xoshiro(1), prefixed))) == Set([
            @varname(cognitive_parameters_model.learning_rate.shared),
            @varname(environment_parameters_model.learning_rate.shared),
        ])
        #No warning when the environment's parameters are fixed, nor when the cognitive parameters are
        @test_logs full_model(
            TuringBackend(),
            shared,
            sessions_info;
            environment_parameters_model=FixedParameters((; learning_rate=0.5)),
        )
        @test_logs full_model(
            TuringBackend(),
            FixedParameters((; learning_rate=0.5, inverse_temperature=1.0)),
            sessions_info;
            environment_parameters_model=shared,
        )
        #No warning when the two models' parameters have different names, nor without an environment
        @test_logs full_model(
            TuringBackend(),
            shared,
            sessions_for_test(;
                environment=RewardsAroundMean(), impute_missing_actions=false
            );
            environment_parameters_model=IdenticalParameters((; mean_reward=Normal())),
        )
        @test_logs full_model(TuringBackend(), shared, sessions_for_test())
    end

    @testset "It can be sampled, and returned gives the parameters and the states of every session for each sample" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        model = saving_full_model(IndependentParameters(FULL_PRIORS), sessions_info)
        chain = sample(Xoshiro(1), model, NUTS(), 50; progress=false)
        @test size(chain, 1) == 50
        @test Set(Turing.FlexiChains.parameters(chain)) ==
            Set([@varname(learning_rate.session), @varname(inverse_temperature.session)])
        returned_values = returned(model, chain)
        @test length(returned_values) == 50
        @test length(returned_values[1].cognitive_parameters.learning_rate) == 3
        @test [
            length(session.states) for session in returned_values[1].saved_trajectories
        ] == [4, 3, 5]
        #With the missing choice imputed, its latent is sampled too, by a sampler for discrete variables
        with_imputed = full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), sessions_for_test()
        )
        chain = @test_logs match_mode = :any sample(
            Xoshiro(1),
            with_imputed,
            Gibbs(
                (@varname(learning_rate), @varname(inverse_temperature)) => NUTS(),
                @varname(session) => PG(5),
            ),
            20;
            progress=false,
        )
        @test @varname(session[3].timestep[3].choice.imputed) in
            Turing.FlexiChains.parameters(chain)
    end

    @testset "turing_catch_invalid_parameters wraps the sessions, so one rejected session rejects the whole sample" begin
        table = (; FULL_TABLE..., choice=[0.2, 0.4, 0.6, 0.8, 1.0, 0.2, 0.4, 0.6, 0.8])
        sessions_info = sessions_for_test(
            RejectsHighLearningRate(),
            table;
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        parameters = (; learning_rate=[0.3, 1.5, 0.3])
        @test_throws "The learning rate must be at most 1." logjoint(
            turing_multiple_behavioral_sessions(sessions_info, parameters), (;)
        )
        caught = turing_catch_invalid_parameters(
            turing_multiple_behavioral_sessions(sessions_info, parameters)
        )
        @test logjoint(caught, (;)) == -Inf && returned(caught, (;)) === nothing
        #Valid parameters are unaffected
        valid = turing_catch_invalid_parameters(
            turing_multiple_behavioral_sessions(
                sessions_info, (; learning_rate=fill(0.3, 3))
            ),
        )
        @test logjoint(valid, (;)) ≈ logjoint(
            turing_multiple_behavioral_sessions(
                sessions_info, (; learning_rate=fill(0.3, 3))
            ),
            (;),
        )
        @test length(returned(valid, (;))) == 3
        #The full model puts it around the sessions with the keyword, and returns nothing for a rejected sample without running the downstream model
        model = full_model(
            TuringBackend(),
            IndependentParameters((; learning_rate=Uniform(0, 2))),
            sessions_info;
            catch_invalid_parameters=true,
            downstream_model=MeasurementRegression([0.4, 0.6, 0.5]),
        )
        rejected = Dict(
            @varname(learning_rate.session) => [0.3, 1.5, 0.3], @varname(slope) => 0.8
        )
        @test logjoint(model, rejected) == -Inf
        @test returned(model, rejected) === nothing
        accepted = Dict(
            @varname(learning_rate.session) => fill(0.3, 3), @varname(slope) => 0.8
        )
        @test isfinite(logjoint(model, accepted)) &&
            returned(model, accepted).downstream_values == 0.8
        #A sampler starts when most first draws are rejected, since the parameters are sampled outside the try block
        without_downstream = saving_full_model(
            IndependentParameters((; learning_rate=Uniform(0, 2))),
            sessions_info;
            catch_invalid_parameters=true,
        )
        for seed in 1:3
            chain = sample(Xoshiro(seed), without_downstream, NUTS(), 10; progress=false)
            @test size(chain, 1) == 10 && all(
                returned_value -> returned_value isa NamedTuple,
                returned(without_downstream, chain),
            )
        end
    end

    @testset "ParallelSessions runs the sessions over the threads, and needs the model to be marked thread-safe" begin
        sessions_info = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        parameters = (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        in_sequence = turing_multiple_behavioral_sessions(sessions_info, parameters)
        in_parallel = turing_multiple_behavioral_sessions(
            sessions_info, parameters, (;), ParallelSessions()
        )
        @test_throws "The sessions run in parallel, so the model that is evaluated must be marked thread-safe with setthreadsafe(model, true) from DynamicPPL." logjoint(
            in_parallel, (;)
        )
        threadsafe = Turing.DynamicPPL.setthreadsafe(in_parallel, true)
        @test logjoint(threadsafe, (;)) ≈ logjoint(in_sequence, (;))
        @test returned(threadsafe, (;)).saved_trajectories[2].states ==
            returned(in_sequence, (;)).saved_trajectories[2].states
        #Sessions whose saved trajectories have different types give what they give in sequence, since S3 saves UnimputedMissingValue() for its missing choice
        saving_actions = sessions_for_test(;
            impute_missing_actions=false,
            cognitive_save_history=SaveHistory(; outputs=:choice),
        )
        in_parallel_saving_actions = Turing.DynamicPPL.setthreadsafe(
            turing_multiple_behavioral_sessions(
                saving_actions, parameters, (;), ParallelSessions()
            ),
            true,
        )
        @test returned(in_parallel_saving_actions, (;)) ==
            returned(turing_multiple_behavioral_sessions(saving_actions, parameters), (;))
        #The full model passes it on
        full = Turing.DynamicPPL.setthreadsafe(
            saving_full_model(
                IndependentParameters(FULL_PRIORS),
                sessions_info;
                parallelization=ParallelSessions(),
            ),
            true,
        )
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
        )
        @test loglikelihood(full, given) ≈
            sessions_by_hand(FULL_TABLE, [0.2, 0.3, 0.4], [1.0, 2.0, 3.0])
    end

    @testset "A chain from a model that saves nothing gives the states through returned with a copy that saves" begin
        unsaved_sessions = sessions_for_test(; impute_missing_actions=false)
        saving_sessions = MultipleSessionsInfo(
            unsaved_sessions; cognitive_save_history=SaveHistory(; states=:expected_reward)
        )
        unsaved = full_model(
            TuringBackend(), IndependentParameters(FULL_PRIORS), unsaved_sessions
        )
        saving = saving_full_model(IndependentParameters(FULL_PRIORS), saving_sessions)
        chain = sample(Xoshiro(2), unsaved, NUTS(), 20; progress=false)
        returned_values = returned(saving, chain)
        @test length(returned_values) == 20
        @test [
            length(session.states) for session in returned_values[1].saved_trajectories
        ] == [4, 3, 5]
        @test returned(unsaved, chain)[1].saved_trajectories[1].states == fill((;), 4)
        #The first state of each session is the initial one, and the parameters are the chain's
        learning_rates = returned_values[1].cognitive_parameters.learning_rate
        @test returned_values[1].saved_trajectories[1].states[2].expected_reward ≈
            0.5 + learning_rates[1] * (1.0 - 0.5)
    end

    @testset "Evaluating the sessions only allocates the distributions they keep, also with a column that can hold a MissingValue" begin
        function table_with(n_timesteps)
            return (;
                id=repeat([1, 2], inner=n_timesteps),
                reward=repeat([1.0, 0.0], n_timesteps),
                choice=Vector{Union{Missing,Bool}}(repeat([true, false], n_timesteps)),
            )
        end
        model_with(n_timesteps) = full_model(
            TuringBackend(),
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(
                WithDefaultParameters(DistributionsJl()), table_with(n_timesteps)
            ),
        )
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3],
            @varname(inverse_temperature.session) => [1.0, 2.0],
        )
        short_model = model_with(5_000)
        long_model = model_with(10_000)
        logjoint(short_model, given)
        logjoint(long_model, given)
        #Doubling the timesteps only adds the vectors of the distributions of the actions to condition on, about 23 bytes per timestep
        @test @allocated(logjoint(long_model, given)) -
              @allocated(logjoint(short_model, given)) <= 40 * 10_000
    end
end

## Parameter domains in fits ##

#The environment with rewards around a mean, with a domain for the mean
struct BoundedRewardsAroundMean <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(::BoundedRewardsAroundMean, parameters)
    return (;)
end
function CognitiveModeling.run_timestep(::BoundedRewardsAroundMean, context::ModelContext)
    return run_timestep(RewardsAroundMean(), context)
end
function CognitiveModeling.attribute_names(::BoundedRewardsAroundMean)
    return CognitiveModeling.attribute_names(RewardsAroundMean())
end
function CognitiveModeling.parameter_domains(::BoundedRewardsAroundMean)
    return (; mean_reward=Domain(; lower=0, upper=1),)
end

@testset "Parameter domains in fits" begin
    sessions_info = sessions_for_test(
        WithDomains(DistributionsJl()); impute_missing_actions=false
    )
    unchecked_sessions = sessions_for_test(
        WithDomains(DistributionsJl());
        impute_missing_actions=false,
        check_parameter_domains=false,
    )
    within_domains = IndependentParameters((;
        learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
    ))
    outside_domains = FixedParameters((; learning_rate=1.5, inverse_temperature=2.0))

    @testset "A value outside its domain stops the evaluation, also when invalid parameters are caught" begin
        model = @test_logs full_model(TuringBackend(), within_domains, sessions_info)
        @test isfinite(
            logjoint(
                model,
                Dict(
                    @varname(learning_rate.session) => fill(0.3, 3),
                    @varname(inverse_temperature.session) => fill(2.0, 3),
                ),
            ),
        )
        #A value that a sampler proposes is checked when the sessions run
        proposed_outside = Dict(
            @varname(learning_rate.session) => [0.3, 0.3, 1.5],
            @varname(inverse_temperature.session) => fill(2.0, 3),
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{DistributionsJl} is 1.5, which is outside its domain, from 0 to 1.",
        ) logjoint(model, proposed_outside)
        #The error isn't the one that rejects a sample, so it stops the evaluation there too
        @test_throws ParameterDomainError logjoint(
            full_model(
                TuringBackend(),
                within_domains,
                sessions_info;
                catch_invalid_parameters=true,
            ),
            proposed_outside,
        )
    end

    @testset "A number of ForwardDiff.jl that equals a bound of its domain is within it" begin
        Dual = Turing.ForwardDiff.Dual
        check(learning_rate) = CognitiveModeling.check_parameters_in_domains(
            WithDomains(DistributionsJl()), (; learning_rate, inverse_temperature=2.0)
        )
        #A learning rate of exactly 1 or 0, as when a transformation reaches the bound while a sampler explores, with a derivative that points beyond it
        @test check(Dual(1.0, 1.0e-30)) === nothing
        @test check(Dual(0.0, -1.0)) === nothing
        @test check(Dual(Dual(1.0, 1.0e-30), Dual(1.0e-30, 0.0))) === nothing
        @test_throws ParameterDomainError check(Dual(1.5, 0.0))
        @test_throws ParameterDomainError check(Dual(nextfloat(1.0), 1.0))
        @test_throws ParameterDomainError check(Dual(NaN, 1.0))
    end

    @testset "A fixed value outside its domain is an error when the sessions run" begin
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{DistributionsJl} is 1.5, which is outside its domain, from 0 to 1.",
        ) logjoint(full_model(TuringBackend(), outside_domains, sessions_info), (;))
        @test_throws ParameterDomainError logjoint(
            full_model(
                TuringBackend(),
                IndependentParameters((; inverse_temperature=LogNormal(0, 0.5))),
                sessions_info;
                fixed_cognitive_parameters=(; learning_rate=1.5),
            ),
            Dict(@varname(inverse_temperature.session) => fill(2.0, 3)),
        )
    end

    @testset "Sessions that don't check the domains run with any value" begin
        unchecked = logjoint(
            full_model(TuringBackend(), outside_domains, unchecked_sessions), (;)
        )
        @test isfinite(unchecked)
        #The log density is that of the same model without domains
        @test unchecked == logjoint(
            full_model(
                TuringBackend(),
                outside_domains,
                sessions_for_test(; impute_missing_actions=false),
            ),
            (;),
        )
    end

    @testset "The parameters of an environment that runs are checked too" begin
        with_environment = sessions_for_test(
            WithDomains(DistributionsJl());
            environment=BoundedRewardsAroundMean(),
            include_environment=true,
            impute_missing_actions=false,
        )
        cognitive_parameters = FixedParameters((;
            learning_rate=0.3, inverse_temperature=2.0
        ))
        @test isfinite(
            logjoint(
                full_model(
                    TuringBackend(),
                    cognitive_parameters,
                    with_environment;
                    environment_parameters_model=FixedParameters((; mean_reward=0.5)),
                ),
                (;),
            ),
        )
        #A fixed value outside its domain is an error when the sessions run
        @test_throws ParameterDomainError(
            "The parameter mean_reward of an environment model of type BoundedRewardsAroundMean is 2.0, which is outside its domain, from 0 to 1.",
        ) logjoint(
            full_model(
                TuringBackend(),
                cognitive_parameters,
                with_environment;
                environment_parameters_model=FixedParameters((; mean_reward=2.0)),
            ),
            (;),
        )
        #A value that a sampler proposes is checked when the sessions run, for both models
        sampled = full_model(
            TuringBackend(),
            within_domains,
            with_environment;
            environment_parameters_model=IdenticalParameters((; mean_reward=Beta(2, 2))),
        )
        @test_throws ParameterDomainError(
            "The parameter mean_reward of an environment model of type BoundedRewardsAroundMean is 2.0, which is outside its domain, from 0 to 1.",
        ) logjoint(
            sampled,
            Dict(
                @varname(learning_rate.session) => fill(0.3, 3),
                @varname(inverse_temperature.session) => fill(2.0, 3),
                @varname(mean_reward.shared) => 2.0,
            ),
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{DistributionsJl} is 1.5, which is outside its domain, from 0 to 1.",
        ) logjoint(
            sampled,
            Dict(
                @varname(learning_rate.session) => [0.3, 0.3, 1.5],
                @varname(inverse_temperature.session) => fill(2.0, 3),
                @varname(mean_reward.shared) => 0.5,
            ),
        )
    end

    @testset "A single session checks its parameters, and turing_catch_invalid_parameters doesn't catch the error" begin
        session_info = first(sessions_info.sessions)
        parameters = (; learning_rate=1.5, inverse_temperature=2.0)
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{DistributionsJl} is 1.5, which is outside its domain, from 0 to 1.",
        ) logjoint(turing_behavioral_session(session_info, parameters), (;))
        @test_throws ParameterDomainError logjoint(
            turing_catch_invalid_parameters(
                turing_behavioral_session(session_info, parameters)
            ),
            (;),
        )
        @test isfinite(
            logjoint(
                turing_behavioral_session(first(unchecked_sessions.sessions), parameters),
                (;),
            ),
        )
    end

    @testset "A prior that reaches outside a domain is left to the check when the sessions run" begin
        outside_priors = IndependentParameters((;
            learning_rate=Normal(0, 1), inverse_temperature=LogNormal(0, 0.5)
        ))
        #The model is made without an error or a warning
        model = @test_logs full_model(TuringBackend(), outside_priors, sessions_info)
        #A value within the domain is evaluated, and a value outside it stops the evaluation
        @test isfinite(
            logjoint(
                model,
                Dict(
                    @varname(learning_rate.session) => fill(0.3, 3),
                    @varname(inverse_temperature.session) => fill(2.0, 3),
                ),
            ),
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{DistributionsJl} is -0.5, which is outside its domain, from 0 to 1.",
        ) logjoint(
            model,
            Dict(
                @varname(learning_rate.session) => [0.3, 0.3, -0.5],
                @varname(inverse_temperature.session) => fill(2.0, 3),
            ),
        )
    end
end

@testset "A copy of a full model with other sessions keeps the environment running when its parameters are estimated" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
    sessions_info = sessions_for_test(;
        environment=RewardsAroundMean(), impute_missing_actions=false
    )
    @test !any(session -> session.includes_environment, sessions_info.sessions)
    model = full_model(
        TuringBackend(),
        IndependentParameters(FULL_PRIORS),
        sessions_info;
        environment_parameters_model=IdenticalParameters((; mean_reward=Normal())),
    )
    @test all(session -> session.includes_environment, model.args.sessions_info.sessions)
    saving = extension.with_session_info(
        model,
        info -> MultipleSessionsInfo(
            info; cognitive_save_history=SaveHistory(; states=:expected_reward)
        ),
        "get_session_trajectories",
    )
    @test all(session -> session.includes_environment, saving.args.sessions_info.sessions)
end

@testset "session_model, multiple_sessions_model and catch_invalid_parameters make the Turing.jl models with TuringBackend()" begin
    sessions_info = sessions_for_test(; impute_missing_actions=false)
    parameters = (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
    session_parameters = (; learning_rate=0.2, inverse_temperature=1.0)
    session_info = sessions_info.sessions[1]

    #The model of one session, without and with environment parameters
    @test session_model(TuringBackend(), session_info, session_parameters) ===
        turing_behavioral_session(session_info, session_parameters)
    with_environment = sessions_for_test(
        WithDefaultParameters(DistributionsJl()),
        TWO_MISSING_REWARDS_TABLE;
        environment=RewardsAroundMean(),
        impute_missing_actions=false,
    )
    @test session_model(
        TuringBackend(),
        with_environment.sessions[1],
        session_parameters,
        (; mean_reward=0.5),
    ) === turing_behavioral_session(
        with_environment.sessions[1], session_parameters, (; mean_reward=0.5)
    )

    #The model of several sessions, with the parallelization as a keyword
    @test multiple_sessions_model(TuringBackend(), sessions_info, parameters) ===
        turing_multiple_behavioral_sessions(
        sessions_info, parameters, (;), SerialSessions()
    )
    @test multiple_sessions_model(
        TuringBackend(), sessions_info, parameters; parallelization=ParallelSessions()
    ) === turing_multiple_behavioral_sessions(
        sessions_info, parameters, (;), ParallelSessions()
    )

    #The model which rejects invalid parameters, around any of them
    session = turing_behavioral_session(session_info, session_parameters)
    @test catch_invalid_parameters(TuringBackend(), session) ===
        turing_catch_invalid_parameters(session)
end
