## The parameters models on their own, with the three sessions of FULL_TABLE, S1, S2 and S3 ##

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
