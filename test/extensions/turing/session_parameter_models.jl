## The session parameter models on their own, with the three sessions of FULL_TABLE, S1, S2 and S3 ##

@testset "turing_session_parameters_model" begin
    sessions_info = sessions_for_test(; impute_missing_actions=false)

    @testset "NoSessionParameters gives no parameters, and samples nothing" begin
        model = CognitiveModeling.turing_session_parameters_model(
            NoSessionParameters(), sessions_info
        )
        @test model() === (;)
        @test isempty(keys(rand(Xoshiro(1), model)))
    end

    @testset "FixedSessionParameters gives every session the same values, and samples nothing" begin
        model = CognitiveModeling.turing_session_parameters_model(
            FixedSessionParameters((; learning_rate=0.3, inverse_temperature=2.0)),
            sessions_info,
        )
        @test model() == (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        @test isempty(keys(rand(Xoshiro(1), model)))
        @test logjoint(model, (;)) == 0
    end

    @testset "FixedSessionParametersBySession gives each session its own values, found by its identifier, and samples nothing" begin
        model = CognitiveModeling.turing_session_parameters_model(
            FixedSessionParametersBySession((;
                S3=(; learning_rate=0.4, inverse_temperature=3.0),
                S1=(; learning_rate=0.2, inverse_temperature=1.0),
                S2=(; learning_rate=0.3, inverse_temperature=2.0),
            )),
            sessions_info,
        )
        @test model() ==
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        @test isempty(keys(rand(Xoshiro(1), model)))
        #The same from a dictionary keyed by the sessions' identifiers
        from_dictionary = CognitiveModeling.turing_session_parameters_model(
            FixedSessionParametersBySession(
                Dict(
                    "S3" => (; learning_rate=0.4, inverse_temperature=3.0),
                    "S1" => (; learning_rate=0.2, inverse_temperature=1.0),
                    "S2" => (; learning_rate=0.3, inverse_temperature=2.0),
                ),
            ),
            sessions_info,
        )
        @test from_dictionary() == model()
        @test_throws "FixedSessionParametersBySession has no parameters for the session S3. It has parameters for (:S1, :S2)." CognitiveModeling.turing_session_parameters_model(
            FixedSessionParametersBySession((;
                S1=(; learning_rate=0.2), S2=(; learning_rate=0.3)
            )),
            sessions_info,
        )()
    end

    @testset "IdenticalSessionParameters samples each parameter once, as name.shared, and gives its value to every session" begin
        model = CognitiveModeling.turing_session_parameters_model(
            IdenticalSessionParameters(FULL_PRIORS), sessions_info
        )
        @test Set(keys(rand(Xoshiro(1), model))) ==
            Set([@varname(learning_rate.shared), @varname(inverse_temperature.shared)])
        given = Dict(
            @varname(learning_rate.shared) => 0.3,
            @varname(inverse_temperature.shared) => 2.0,
        )
        @test returned(model, given) ==
            (; learning_rate=fill(0.3, 3), inverse_temperature=fill(2.0, 3))
        @test logjoint(model, given) ≈
            logpdf(Beta(2, 2), 0.3) + logpdf(LogNormal(0, 0.5), 2.0)
    end

    @testset "IndependentSessionParameters samples each parameter for every session, as the vector name.session" begin
        model = CognitiveModeling.turing_session_parameters_model(
            IndependentSessionParameters(FULL_PRIORS), sessions_info
        )
        values = rand(Xoshiro(1), model)
        @test Set(keys(values)) ==
            Set([@varname(learning_rate.session), @varname(inverse_temperature.session)])
        @test length(values[@varname(learning_rate.session)]) == 3
        given = Dict(
            @varname(learning_rate.session) => [0.2, 0.3, 0.4],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
        )
        @test returned(model, given) ==
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        @test logjoint(model, given) ≈
            sum(logpdf.(Beta(2, 2), [0.2, 0.3, 0.4])) +
              sum(logpdf.(LogNormal(0, 0.5), [1.0, 2.0, 3.0]))
    end

    @testset "Without priors, the kinds that sample give no parameters, and sample nothing" begin
        for parameters_model in
            (IdenticalSessionParameters((;)), IndependentSessionParameters((;)))
            model = CognitiveModeling.turing_session_parameters_model(
                parameters_model, sessions_info
            )
            @test model() === (;)
            @test isempty(keys(rand(Xoshiro(1), model)))
        end
    end
end

## The session parameter models which sample from the distributions of groups, or combine others ##

@testset "GroupedHierarchicalSessionParameters and CompositeSessionParameters" begin
    sessions_info = sessions_for_test(
        SessionRescorlaWagner(), GROUPED_FULL_TABLE; impute_missing_actions=false
    )

    @testset "Each session's value is its group's mean plus a standardized value times its group's standard deviation, transformed" begin
        hierarchical = GroupedHierarchicalSessionParameters((;
            learning_rate=HierarchicalParameter(
                GROUP_PRIORS; transformation=to_unit_interval, group_by=:group
            ),
            inverse_temperature=HierarchicalParameter(GROUP_PRIORS; transformation=exp),
        ))
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(hierarchical, sessions_info),
            sessions_info,
        )
        #The learning rate has two groups, with the values of each parameter of its distribution under their names, the inverse temperature one value of each, and every session has a standardized value
        @test Set(keys(rand(Xoshiro(1), model))) == Set([
            @varname(learning_rate.mean.group),
            @varname(learning_rate.sd.group),
            @varname(learning_rate.standardized),
            @varname(inverse_temperature.mean.group),
            @varname(inverse_temperature.sd.group),
            @varname(inverse_temperature.standardized),
        ])
        @test keys(rand(Xoshiro(1), model)[@varname(learning_rate.mean.group)]) == (:a, :b)
        @test rand(Xoshiro(1), model)[@varname(inverse_temperature.mean.group)] isa Real
        values = Dict(
            @varname(learning_rate.mean.group) => (; a=0.5, b=-1.0),
            @varname(learning_rate.sd.group) => (; a=1.0, b=2.0),
            @varname(learning_rate.standardized) => [0.1, 0.2, -0.3],
            @varname(inverse_temperature.mean.group) => 0.3,
            @varname(inverse_temperature.sd.group) => 0.5,
            @varname(inverse_temperature.standardized) => [1.0, 0.0, -1.0],
        )
        parameters = returned(model, values)
        @test parameters.learning_rate ≈
            to_unit_interval.([0.5 + 1.0 * 0.1, -1.0 + 2.0 * 0.2, 0.5 + 1.0 * -0.3])
        @test parameters.inverse_temperature ≈ exp.(0.3 .+ 0.5 .* [1.0, 0.0, -1.0])
        @test logjoint(model, values) ≈
            sum(logpdf.(Normal(0, 1), [0.5, -1.0, 0.3])) +
              sum(logpdf.(Exponential(1), [1.0, 2.0, 0.5])) +
              sum(logpdf.(Normal(), [0.1, 0.2, -0.3, 1.0, 0.0, -1.0]))
        #A model that wasn't prepared finds the groups when it runs, and gives the same
        unprepared = CognitiveModeling.turing_session_parameters_model(
            hierarchical, sessions_info
        )
        @test returned(unprepared, values) == parameters
        @test logjoint(unprepared, values) ≈ logjoint(model, values)
    end

    @testset "A centered parameter is sampled from the distribution of its group, as name.session" begin
        centered = GroupedHierarchicalSessionParameters((;
            learning_rate=HierarchicalParameter(
                GROUP_PRIORS;
                transformation=to_unit_interval,
                group_by=:group,
                centered=true,
            ),
        ))
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(centered, sessions_info),
            sessions_info,
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set([
            @varname(learning_rate.mean.group),
            @varname(learning_rate.sd.group),
            @varname(learning_rate.session),
        ])
        values = Dict(
            @varname(learning_rate.mean.group) => (; a=0.5, b=-1.0),
            @varname(learning_rate.sd.group) => (; a=1.0, b=2.0),
            @varname(learning_rate.session) => [0.6, -0.6, 0.2],
        )
        @test returned(model, values).learning_rate ≈ to_unit_interval.([0.6, -0.6, 0.2])
        @test logjoint(model, values) ≈
            sum(logpdf.(Normal(0, 1), [0.5, -1.0])) +
              sum(logpdf.(Exponential(1), [1.0, 2.0])) +
              logpdf(Normal(0.5, 1.0), 0.6) +
              logpdf(Normal(-1.0, 2.0), -0.6) +
              logpdf(Normal(0.5, 1.0), 0.2)
    end

    @testset "Another distribution takes the group's values of its parameters, in the order of the priors" begin
        with_laplace = GroupedHierarchicalSessionParameters((;
            learning_rate=HierarchicalParameter(
                (; location=Normal(0, 1), spread=Exponential(1));
                distribution=(location, spread) -> Laplace(location, spread),
                transformation=to_unit_interval,
            ),
        ))
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(with_laplace, sessions_info),
            sessions_info,
        )
        values = Dict(
            @varname(learning_rate.location.group) => 0.5,
            @varname(learning_rate.spread.group) => 2.0,
            @varname(learning_rate.session) => [0.6, -0.6, 0.2],
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(values))
        @test logjoint(model, values) ≈
            logpdf(Normal(0, 1), 0.5) +
              logpdf(Exponential(1), 2.0) +
              sum(logpdf.(Laplace(0.5, 2.0), [0.6, -0.6, 0.2]))
    end

    @testset "A composite runs each part with its own variables, and merges what they give" begin
        composite = CompositeSessionParameters(
            GroupedHierarchicalSessionParameters((;
                learning_rate=HierarchicalParameter(
                    GROUP_PRIORS; transformation=to_unit_interval
                )
            )),
            IndependentSessionParameters((; inverse_temperature=LogNormal(0, 0.5))),
        )
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(composite, sessions_info),
            sessions_info,
        )
        values = Dict(
            @varname(learning_rate.mean.group) => 0.5,
            @varname(learning_rate.sd.group) => 1.0,
            @varname(learning_rate.standardized) => [0.1, 0.2, -0.3],
            @varname(inverse_temperature.session) => [1.0, 2.0, 3.0],
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(values))
        parameters = returned(model, values)
        @test keys(parameters) == (:learning_rate, :inverse_temperature)
        @test parameters.learning_rate ≈ to_unit_interval.(0.5 .+ [0.1, 0.2, -0.3])
        @test parameters.inverse_temperature == [1.0, 2.0, 3.0]
        #Parts that don't tell which parameters they give are checked when the model runs
        twice = CompositeSessionParameters(SharedLearningRate(), SharedLearningRate())
        @test_throws "Two parts of CompositeSessionParameters give the parameters (:learning_rate, :inverse_temperature). Each parameter must come from one part." rand(
            Xoshiro(1),
            CognitiveModeling.turing_session_parameters_model(twice, sessions_info),
        )
    end

    @testset "Values that were looked up for every session are given as they are" begin
        looked_up = CognitiveModeling.prepare_parameters_model(
            FixedSessionParametersBySession(
                Dict(
                    "S3" => (; learning_rate=0.4, inverse_temperature=3.0),
                    "S1" => (; learning_rate=0.2, inverse_temperature=1.0),
                    "S2" => (; learning_rate=0.3, inverse_temperature=2.0),
                ),
            ),
            sessions_info,
        )
        model = CognitiveModeling.turing_session_parameters_model(looked_up, sessions_info)
        @test model() ==
            (; learning_rate=[0.2, 0.3, 0.4], inverse_temperature=[1.0, 2.0, 3.0])
        @test isempty(keys(rand(Xoshiro(1), model)))
    end

    @testset "The full model fits the sessions with the parameters of their groups" begin
        hierarchical = GroupedHierarchicalSessionParameters((;
            learning_rate=HierarchicalParameter(
                GROUP_PRIORS; transformation=to_unit_interval, group_by=:group
            ),
            inverse_temperature=HierarchicalParameter(GROUP_PRIORS; transformation=exp),
        ))
        model = turing_full_model(hierarchical, sessions_info)
        values = Dict(
            @varname(learning_rate.mean.group) => (; a=0.5, b=-1.0),
            @varname(learning_rate.sd.group) => (; a=1.0, b=2.0),
            @varname(learning_rate.standardized) => [0.1, 0.2, -0.3],
            @varname(inverse_temperature.mean.group) => 0.3,
            @varname(inverse_temperature.sd.group) => 0.5,
            @varname(inverse_temperature.standardized) => [1.0, 0.0, -1.0],
        )
        learning_rates = to_unit_interval.([
            0.5 + 1.0 * 0.1, -1.0 + 2.0 * 0.2, 0.5 + 1.0 * -0.3
        ])
        inverse_temperatures = exp.(0.3 .+ 0.5 .* [1.0, 0.0, -1.0])
        @test returned(model, values).cognitive_parameters.learning_rate ≈ learning_rates
        @test logjoint(model, values) ≈
            sum(logpdf.(Normal(0, 1), [0.5, -1.0, 0.3])) +
              sum(logpdf.(Exponential(1), [1.0, 2.0, 0.5])) +
              sum(logpdf.(Normal(), [0.1, 0.2, -0.3, 1.0, 0.0, -1.0])) +
              sessions_by_hand(FULL_TABLE, learning_rates, inverse_temperatures)
        chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)
        @test size(chain, 1) == 20
    end
end

## The session parameter model which gives each parameter by a regression on columns of the table ##

#The three sessions of GROUPED_FULL_TABLE with the age and the participant of each: S1 and S3 are of P1, in group a, and S2 is of P2, in group b
const REGRESSION_FULL_TABLE = merge(
    GROUPED_FULL_TABLE,
    (;
        age=[20, 20, 20, 30, 30, 40, 40, 40, 40],
        participant=["P1", "P1", "P1", "P2", "P2", "P1", "P1", "P1", "P1"],
    ),
)

@testset "HierarchicalRegressionSessionParameters" begin
    sessions_info = sessions_for_test(
        SessionRescorlaWagner(), REGRESSION_FULL_TABLE; impute_missing_actions=false
    )
    regression = HierarchicalRegressionSessionParameters((;
        learning_rate=RegressionParameter(
            Intercept(Normal(0, 1)),
            ContinuousPredictor(:age, Normal(0, 0.5)),
            CategoricalPredictor(:group, Normal(0, 2)),
            Interaction((:age, :group), Normal(0, 0.3)),
            VaryingEffects(
                :participant,
                Intercept(Exponential(1)),
                ContinuousPredictor(:age, Exponential(0.2)),
            ),
            VaryingEffects(:id, Intercept(Exponential(0.5)); name=:session);
            transformation=to_unit_interval,
        ),
        inverse_temperature=RegressionParameter(
            Intercept(Normal(0, 1)); transformation=exp
        ),
    ))
    values = Dict(
        @varname(learning_rate.coef) =>
            (; intercept=0.5, age=0.01, group_b=-1.0, age_by_group_b=0.02),
        @varname(learning_rate.participant.intercept.sd) => 2.0,
        @varname(learning_rate.participant.intercept.standardized) => [0.3, -0.4],
        @varname(learning_rate.participant.age.sd) => 0.1,
        @varname(learning_rate.participant.age.standardized) => [0.5, -1.0],
        @varname(learning_rate.session.intercept.sd) => 0.5,
        @varname(learning_rate.session.intercept.standardized) => [0.1, 0.2, -0.3],
        @varname(inverse_temperature.coef) => (; intercept=0.7),
    )
    #The sum of each session by hand: the intercept, the effects of its age, its group and their product, the deviations of its participant's intercept and of its participant's slope of the age times its age, and the deviation of the session itself
    learning_rates = to_unit_interval.([
        0.5 + 0.01 * 20 + 2.0 * 0.3 + 0.1 * 0.5 * 20 + 0.5 * 0.1,
        0.5 + 0.01 * 30 - 1.0 + 0.02 * 30 + 2.0 * -0.4 + 0.1 * -1.0 * 30 + 0.5 * 0.2,
        0.5 + 0.01 * 40 + 2.0 * 0.3 + 0.1 * 0.5 * 40 + 0.5 * -0.3,
    ])
    log_prior =
        logpdf(Normal(0, 1), 0.5) +
        logpdf(Normal(0, 0.5), 0.01) +
        logpdf(Normal(0, 2), -1.0) +
        logpdf(Normal(0, 0.3), 0.02) +
        logpdf(Exponential(1), 2.0) +
        sum(logpdf.(Normal(), [0.3, -0.4])) +
        logpdf(Exponential(0.2), 0.1) +
        sum(logpdf.(Normal(), [0.5, -1.0])) +
        logpdf(Exponential(0.5), 0.5) +
        sum(logpdf.(Normal(), [0.1, 0.2, -0.3])) +
        logpdf(Normal(0, 1), 0.7)

    @testset "Each session's value is the sum of the effects of the coefficients and of the deviations of its levels of the varying effects, transformed" begin
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(regression, sessions_info),
            sessions_info,
        )
        #The coefficients are one variable under the parameter, and the variables of varying effects are under their name and the name of each coefficient
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(values))
        parameters = returned(model, values)
        @test parameters.learning_rate ≈ learning_rates
        @test parameters.inverse_temperature ≈ fill(exp(0.7), 3)
        @test logjoint(model, values) ≈ log_prior
        #A model that wasn't prepared builds the designs when it runs, and gives the same
        unprepared = CognitiveModeling.turing_session_parameters_model(
            regression, sessions_info
        )
        @test returned(unprepared, values) == parameters
        @test logjoint(unprepared, values) ≈ log_prior
    end

    @testset "Centered varying effects are sampled as deviation, and grouped ones have a standard deviation for each group, under the group's name" begin
        centered_and_grouped = HierarchicalRegressionSessionParameters((;
            learning_rate=RegressionParameter(
                VaryingEffects(:participant, Intercept(Exponential(1)); centered=true),
                VaryingEffects(
                    :id, Intercept(Exponential(0.5)); group_by=:group, name=:session
                ),
            ),
        ))
        model = CognitiveModeling.turing_session_parameters_model(
            centered_and_grouped, sessions_info
        )
        given = Dict(
            @varname(learning_rate.participant.intercept.sd) => 2.0,
            @varname(learning_rate.participant.intercept.deviation) => [0.6, -0.8],
            @varname(learning_rate.session.intercept.sd) => (; a=0.5, b=3.0),
            @varname(learning_rate.session.intercept.standardized) => [0.1, 0.2, -0.3],
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(given))
        #S1 and S3 are in group a and S2 in group b, which scales its standardized value by its own standard deviation
        @test returned(model, given).learning_rate ≈
            [0.6 + 0.5 * 0.1, -0.8 + 3.0 * 0.2, 0.6 + 0.5 * -0.3]
        @test logjoint(model, given) ≈
            logpdf(Exponential(1), 2.0) +
              sum(logpdf.(Normal(0, 2.0), [0.6, -0.8])) +
              sum(logpdf.(Exponential(0.5), [0.5, 3.0])) +
              sum(logpdf.(Normal(), [0.1, 0.2, -0.3]))
        #Centered and grouped, each deviation is sampled with the standard deviation of its group
        both = HierarchicalRegressionSessionParameters((;
            learning_rate=RegressionParameter(
                VaryingEffects(
                    :id,
                    Intercept(Exponential(0.5));
                    group_by=:group,
                    centered=true,
                    name=:session,
                ),
            ),
        ))
        model = CognitiveModeling.turing_session_parameters_model(both, sessions_info)
        given = Dict(
            @varname(learning_rate.session.intercept.sd) => (; a=0.5, b=3.0),
            @varname(learning_rate.session.intercept.deviation) => [0.1, 0.2, -0.3],
        )
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(given))
        @test returned(model, given).learning_rate ≈ [0.1, 0.2, -0.3]
        @test logjoint(model, given) ≈
            sum(logpdf.(Exponential(0.5), [0.5, 3.0])) +
              logpdf(Normal(0, 0.5), 0.1) +
              logpdf(Normal(0, 3.0), 0.2) +
              logpdf(Normal(0, 0.5), -0.3)
    end

    @testset "With an intercept and a varying intercept over the sessions alone, it gives what GroupedHierarchicalSessionParameters gives without groups" begin
        for centered in (false, true)
            as_regression = CognitiveModeling.turing_session_parameters_model(
                HierarchicalRegressionSessionParameters((;
                    learning_rate=RegressionParameter(
                        Intercept(GROUP_PRIORS.mean),
                        VaryingEffects(
                            :id, Intercept(GROUP_PRIORS.sd); centered, name=:session
                        );
                        transformation=to_unit_interval,
                    ),
                )),
                sessions_info,
            )
            as_groups = CognitiveModeling.turing_session_parameters_model(
                GroupedHierarchicalSessionParameters((;
                    learning_rate=HierarchicalParameter(
                        GROUP_PRIORS; transformation=to_unit_interval, centered
                    ),
                )),
                sessions_info,
            )
            #The same values of the mean, the standard deviation and the sessions, as each model names them. A centered regression samples the deviations from the mean, and the groups the values themselves
            regression_values = Dict(
                @varname(learning_rate.coef) => (; intercept=0.5),
                @varname(learning_rate.session.intercept.sd) => 2.0,
                if centered
                    @varname(learning_rate.session.intercept.deviation) => [0.2, 0.4, -0.6]
                else
                    @varname(learning_rate.session.intercept.standardized) =>
                        [0.1, 0.2, -0.3]
                end,
            )
            group_values = Dict(
                @varname(learning_rate.mean.group) => 0.5,
                @varname(learning_rate.sd.group) => 2.0,
                if centered
                    @varname(learning_rate.session) => [0.7, 0.9, -0.1]
                else
                    @varname(learning_rate.standardized) => [0.1, 0.2, -0.3]
                end,
            )
            @test returned(as_regression, regression_values).learning_rate ≈
                returned(as_groups, group_values).learning_rate
            @test logjoint(as_regression, regression_values) ≈
                logjoint(as_groups, group_values)
        end
    end

    @testset "With a coefficient for every level of a categorical predictor and no intercept, the coefficient of each level is its value" begin
        every_level = CognitiveModeling.turing_session_parameters_model(
            HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(
                    CategoricalPredictor(:group, Normal(0, 1); coding=:full_dummy);
                    transformation=to_unit_interval,
                ),
            )),
            sessions_info,
        )
        given = Dict(@varname(learning_rate.coef) => (; group_a=0.2, group_b=-0.3))
        @test Set(keys(rand(Xoshiro(1), every_level))) == Set(keys(given))
        #S1 and S3 are in group a, and S2 in group b
        @test returned(every_level, given).learning_rate ≈
            to_unit_interval.([0.2, -0.3, 0.2])
        @test logjoint(every_level, given) ≈
            logpdf(Normal(0, 1), 0.2) + logpdf(Normal(0, 1), -0.3)
    end

    @testset "As a part of a composite it gives its parameters next to the others, with its variables under the name of the part" begin
        composite = CompositeSessionParameters(
            FixedSessionParameters((; inverse_temperature=2.0));
            learning=HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(
                    Intercept(Normal(0, 1)), ContinuousPredictor(:age, Normal(0, 0.5))
                ),
            )),
        )
        model = CognitiveModeling.turing_session_parameters_model(
            CognitiveModeling.prepare_parameters_model(composite, sessions_info),
            sessions_info,
        )
        given = Dict(@varname(learning.learning_rate.coef) => (; intercept=0.1, age=0.01))
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(given))
        @test returned(model, given) == (;
            inverse_temperature=fill(2.0, 3),
            learning_rate=[0.1 + 0.01 * 20, 0.1 + 0.01 * 30, 0.1 + 0.01 * 40],
        )
    end

    @testset "The full model fits the sessions with the parameters of their regressions, for the cognitive model and for the environment" begin
        model = turing_full_model(regression, sessions_info)
        @test returned(model, values).cognitive_parameters.learning_rate ≈ learning_rates
        @test logjoint(model, values) ≈
            log_prior + sessions_by_hand(FULL_TABLE, learning_rates, fill(exp(0.7), 3))
        chain = sample(Xoshiro(1), model, NUTS(), 20; progress=false)
        @test size(chain, 1) == 20
        #A parameter that is fixed is given next to those of the regression
        with_fixed = turing_full_model(
            HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(
                    Intercept(Normal(0, 1)); transformation=to_unit_interval
                ),
            )),
            sessions_info;
            fixed_cognitive_parameters=(; inverse_temperature=3.0),
        )
        @test returned(
            with_fixed, Dict(@varname(learning_rate.coef) => (; intercept=0.0))
        ).cognitive_parameters ==
            (; learning_rate=fill(0.5, 3), inverse_temperature=fill(3.0, 3))
        #The mean reward of the environment of each session, by its group
        with_environment = sessions_for_test(
            SessionRescorlaWagner(),
            REGRESSION_FULL_TABLE;
            environment=RewardsAroundMean(),
            impute_missing_actions=false,
        )
        model = turing_full_model(
            FixedSessionParameters((; learning_rate=0.3)),
            with_environment;
            environment_parameters_model=HierarchicalRegressionSessionParameters((;
                mean_reward=RegressionParameter(
                    Intercept(Normal(0, 1)), CategoricalPredictor(:group, Normal(0, 2))
                ),
            )),
        )
        given = Dict(@varname(mean_reward.coef) => (; intercept=0.6, group_b=-0.2))
        @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(given))
        @test returned(model, given).environment_parameters.mean_reward ≈ [0.6, 0.4, 0.6]
    end

    @testset "A transformation that reaches outside a domain gives a warning when the model is made, for a regression with varying effects" begin
        bounded_sessions = sessions_for_test(
            BoundedRescorlaWagner(), REGRESSION_FULL_TABLE; impute_missing_actions=false
        )
        within_domain = RegressionParameter(
            Intercept(Normal(0, 1)),
            VaryingEffects(:id, Intercept(Exponential(1)));
            transformation=exp,
        )
        @test_logs (
            :warn,
            r"The parameter learning_rate can be given values from -Inf to Inf, by the transformation identity of its regression, but its domain in a cognitive model of type BoundedRescorlaWagner is from 0 to 1\.",
        ) turing_full_model(
            HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(
                    Intercept(Normal(0, 1)), VaryingEffects(:id, Intercept(Exponential(1)))
                ),
                inverse_temperature=within_domain,
            )),
            bounded_sessions,
        )
        @test_logs turing_full_model(
            HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(
                    Intercept(Normal(0, 1)),
                    VaryingEffects(:id, Intercept(Exponential(1)));
                    transformation=to_unit_interval,
                ),
                inverse_temperature=within_domain,
            )),
            bounded_sessions,
        )
        #Without varying effects, what the regression can give isn't known, so nothing is warned about
        @test_logs turing_full_model(
            HierarchicalRegressionSessionParameters((;
                learning_rate=RegressionParameter(Intercept(Beta(2, 2))),
                inverse_temperature=within_domain,
            )),
            bounded_sessions,
        )
    end
end
