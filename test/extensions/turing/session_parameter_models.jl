## The session parameter models on their own, with the three sessions of FULL_TABLE, S1, S2 and S3 ##

@testset "turing_session_parameters_model" begin
    sessions_info = sessions_for_test(; impute_missing_actions=false)

    @testset "NoParameters gives no parameters, and samples nothing" begin
        model = CognitiveModeling.turing_session_parameters_model(
            NoParameters(), sessions_info
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

    @testset "FixedParametersBySession gives each session its own values, found by its identifier, and samples nothing" begin
        model = CognitiveModeling.turing_session_parameters_model(
            FixedParametersBySession((;
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
            FixedParametersBySession(
                Dict(
                    "S3" => (; learning_rate=0.4, inverse_temperature=3.0),
                    "S1" => (; learning_rate=0.2, inverse_temperature=1.0),
                    "S2" => (; learning_rate=0.3, inverse_temperature=2.0),
                ),
            ),
            sessions_info,
        )
        @test from_dictionary() == model()
        @test_throws "FixedParametersBySession has no values for the session S3. It has values for (:S1, :S2)." CognitiveModeling.turing_session_parameters_model(
            FixedParametersBySession((;
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
        #The learning rate has two groups and the inverse temperature one, each with the parameters of its distribution, and every session has a standardized value
        @test Set(keys(rand(Xoshiro(1), model))) == Set([
            @varname(learning_rate.mean.group),
            @varname(learning_rate.sd.group),
            @varname(learning_rate.standardized),
            @varname(inverse_temperature.mean.group),
            @varname(inverse_temperature.sd.group),
            @varname(inverse_temperature.standardized),
        ])
        values = Dict(
            @varname(learning_rate.mean.group) => [0.5, -1.0],
            @varname(learning_rate.sd.group) => [1.0, 2.0],
            @varname(learning_rate.standardized) => [0.1, 0.2, -0.3],
            @varname(inverse_temperature.mean.group) => [0.3],
            @varname(inverse_temperature.sd.group) => [0.5],
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
            @varname(learning_rate.mean.group) => [0.5, -1.0],
            @varname(learning_rate.sd.group) => [1.0, 2.0],
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
            @varname(learning_rate.location.group) => [0.5],
            @varname(learning_rate.spread.group) => [2.0],
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
            @varname(learning_rate.mean.group) => [0.5],
            @varname(learning_rate.sd.group) => [1.0],
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
            FixedParametersBySession(
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
            @varname(learning_rate.mean.group) => [0.5, -1.0],
            @varname(learning_rate.sd.group) => [1.0, 2.0],
            @varname(learning_rate.standardized) => [0.1, 0.2, -0.3],
            @varname(inverse_temperature.mean.group) => [0.3],
            @varname(inverse_temperature.sd.group) => [0.5],
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
