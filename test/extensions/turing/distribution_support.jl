@testset "The values that the distributions of Distributions.jl can give" begin
    support = CognitiveModeling.distribution_support

    @testset "A univariate distribution gives the values from its minimum to its maximum" begin
        @test support(Normal(0, 1)) == (; lower=(-Inf), upper=Inf, discrete=false)
        @test support(Beta(2, 2)) == (; lower=0.0, upper=1.0, discrete=false)
        @test support(LogNormal(0, 1)) == (; lower=0.0, upper=Inf, discrete=false)
        @test support(truncated(Normal(0, 1), -1, 2)) ==
            (; lower=-1.0, upper=2.0, discrete=false)
    end

    @testset "A discrete distribution gives discrete values" begin
        @test support(Poisson(3)) == (; lower=0, upper=Inf, discrete=true)
        @test support(DiscreteUniform(1, 4)) == (; lower=1, upper=4, discrete=true)
        @test support(Bernoulli(0.5)) == (; lower=false, upper=true, discrete=true)
    end

    @testset "The values of other distributions aren't known" begin
        @test support(MvNormal(zeros(2), 1.0)) === nothing
        @test support(filldist(Normal(0, 1), 2)) === nothing
    end

    @testset "A parameter model reads the values of its priors from it" begin
        ranges = CognitiveModeling.parameter_value_ranges(
            IndependentParameters((;
                learning_rate=Beta(2, 2), noise=MvNormal(zeros(2), 1.0)
            )),
        )
        @test keys(ranges) === (:learning_rate,)
        @test ranges.learning_rate.lower == 0.0 &&
            ranges.learning_rate.upper == 1.0 &&
            !ranges.learning_rate.discrete
        @test ranges.learning_rate.source == "its prior, $(Beta(2, 2))"
    end
end
