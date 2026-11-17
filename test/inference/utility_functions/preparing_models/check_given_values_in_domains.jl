using Random: AbstractRNG

#A stand-in for a prior whose values are known: from lower to upper, discrete or not. A draw from it is its lowest value
struct PriorBetween
    lower::Float64
    upper::Float64
    discrete::Bool
end
PriorBetween(lower, upper) = PriorBetween(lower, upper, false)
function CognitiveModeling.distribution_support(prior::PriorBetween)
    return (; lower=prior.lower, upper=prior.upper, discrete=prior.discrete)
end
Base.rand(rng::AbstractRNG, prior::PriorBetween) = prior.lower

#A model with a domain for a probability, a count, a choice between two and a name, and with a parameter without one. All but the probability have default values
struct CountsWithDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::CountsWithDomains)
    return ModelAttributes(;
        parameters=(:probability, :n_options, :uses_rewards, :strategy, :bias),
        states=:n_timesteps,
        inputs=:reward,
        outputs=:choice,
    )
end
function CognitiveModeling.parameter_domains(::CountsWithDomains)
    return (;
        probability=ParameterDomain(; lower=0, upper=1),
        n_options=ParameterDomain(; type=Integer, lower=1),
        uses_rewards=ParameterDomain(; type=Bool),
        strategy=ParameterDomain(; type=Symbol),
    )
end
function CognitiveModeling.default_parameters(::CountsWithDomains)
    return (; n_options=2, uses_rewards=true, strategy=:greedy, bias=0.0)
end

#An environment whose parameter has a domain
struct GivesRewardWithDomain <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesRewardWithDomain)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end
function CognitiveModeling.parameter_domains(::GivesRewardWithDomain)
    return (; scale=ParameterDomain(; lower=0))
end

#A model whose domains aren't ParameterDomains
struct CountsWithDomainsAsNamedTuples <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::CountsWithDomainsAsNamedTuples)
    return ModelAttributes(; parameters=:n_options, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::CountsWithDomainsAsNamedTuples)
    return (; n_options=(; lower=1, discrete=true),)
end

#A transformation to between 0 and 1
squash(value) = 1 / (1 + exp(-value))

#The sessions of the grouped table with the model with domains
function sessions_with_domains(; keywords...)
    return MultipleSessionsInfo(
        CountsWithDomains(),
        GROUPS_TABLE;
        observation_columns=:outcome,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

@testset "The values that a parameter model can give" begin
    value_ranges = CognitiveModeling.parameter_value_ranges

    @testset "A distribution's values aren't known unless a method says them, and a parameter model of one's own gives none" begin
        @test CognitiveModeling.distribution_support(1.0) === nothing
        @test CognitiveModeling.distribution_support(PriorBetween(0, 1)) ==
            (; lower=0.0, upper=1.0, discrete=false)
        @test value_ranges(ParametersModelOfOwn()) === (;)
        @test value_ranges(NoParameters()) === (;)
        @test value_ranges(FixedParameters((; probability=0.5))) === (;)
    end

    @testset "Priors give their own values, and a prior whose values aren't known is left out" begin
        priors = (;
            probability=PriorBetween(0, 1), n_options=PriorBetween(1, 4, true), bias=2
        )
        expected = (;
            probability=(;
                lower=0.0,
                upper=1.0,
                discrete=false,
                source="its prior, PriorBetween(0.0, 1.0, false)",
            ),
            n_options=(;
                lower=1.0,
                upper=4.0,
                discrete=true,
                source="its prior, PriorBetween(1.0, 4.0, true)",
            ),
        )
        @test value_ranges(IndependentParameters(priors)) == expected
        @test value_ranges(IdenticalParameters(priors)) == expected
    end

    @testset "A hierarchical parameter with the default normal distribution gives what its transformation makes of the real line" begin
        group_priors = (; mean=1, sd=2)
        ranges = value_ranges(
            HierarchicalGroupedParameters((;
                unbounded=HierarchicalParameter(group_priors),
                positive=HierarchicalParameter(group_priors; transformation=exp),
                probability=HierarchicalParameter(group_priors; transformation=squash),
                #The values of another distribution depend on its parameters
                other_distribution=HierarchicalParameter(
                    group_priors; distribution=(mean, sd) -> nothing
                ),
                #A transformation that throws for an end of the real line, and one that doesn't give a number
                throws=HierarchicalParameter(
                    group_priors; transformation=value -> error("not for $value")
                ),
                not_a_number=HierarchicalParameter(
                    group_priors; transformation=value -> :high
                ),
            )),
        )
        @test keys(ranges) === (:unbounded, :positive, :probability)
        @test ranges.unbounded == (;
            lower=(-Inf),
            upper=Inf,
            discrete=false,
            source="the transformation identity of its group-level normal distribution",
        )
        @test (ranges.positive.lower, ranges.positive.upper) == (0.0, Inf)
        @test (ranges.probability.lower, ranges.probability.upper) == (0.0, 1.0)
        @test ranges.positive.source ==
            "the transformation exp of its group-level normal distribution"
    end

    @testset "A regression gives what its transformation makes of its lowest and its highest sum" begin
        sessions_info = grouped_sessions()
        function ranges_of(terms...; keywords...)
            parameters_model = HierarchicalRegressionParameters((;
                probability=RegressionParameter(terms...; keywords...)
            ))
            return value_ranges(
                CognitiveModeling.prepare_parameters_model(parameters_model, sessions_info)
            )
        end
        bounds_of(range) = (range.lower, range.upper)

        #The sum of a session is lowest and highest when each coefficient times its value is: the site is 1 or 2
        with_predictor = ranges_of(
            Intercept(PriorBetween(0, 1)),
            ContinuousPredictor(:site, PriorBetween(-1, 2)),
        ).probability
        @test with_predictor == (;
            lower=-2.0,
            upper=5.0,
            discrete=false,
            source="the transformation identity of its regression",
        )
        #A value below zero turns the two ends of its coefficient around, as that of the reference level with effects coding
        effects_coded = ranges_of(
            Intercept(PriorBetween(0, 1)),
            CategoricalPredictor(:diagnosis, PriorBetween(0, 3); coding=:effects),
        ).probability
        @test bounds_of(effects_coded) == (-3.0, 4.0)
        #The transformation is applied to both ends
        @test bounds_of(
            ranges_of(Intercept(PriorBetween(-1, 1)); transformation=exp).probability
        ) == (exp(-1), exp(1))
        #A coefficient whose prior isn't known can have any value, and adds nothing where it is multiplied by zero
        @test bounds_of(ranges_of(Intercept(1)).probability) == (-Inf, Inf)
        @test bounds_of(
            ranges_of(Intercept(PriorBetween(0, 1)), CategoricalPredictor(:diagnosis, 1)).probability,
        ) == (-Inf, Inf)
        @test bounds_of(
            ranges_of(
                Intercept(PriorBetween(0, 1)),
                CategoricalPredictor(:diagnosis, 1);
                transformation=squash,
            ).probability,
        ) == (0.0, 1.0)
        #Varying effects are normal, so the sum can have any value
        @test bounds_of(
            ranges_of(
                Intercept(PriorBetween(0, 1)),
                VaryingEffects(:id, Intercept(2));
                transformation=exp,
            ).probability,
        ) == (0.0, Inf)
        @test bounds_of(
            ranges_of(VaryingEffects(:id, Intercept(2)); transformation=squash).probability
        ) == (0.0, 1.0)
        #With a family distribution, the values are those of the distribution of a transformed sum, also when the sum has varying effects
        @test ranges_of(
            Intercept(PriorBetween(-1, 1)),
            VaryingEffects(:id, Intercept(2));
            family_distribution=value -> PriorBetween(0, 1, true),
        ).probability == (;
            lower=0.0,
            upper=1.0,
            discrete=true,
            source="its family distribution, such as PriorBetween(0.0, 1.0, true)",
        )
        #The sum is one between the lowest and the highest, transformed, and the other parameters are drawn from their priors
        around_sum = ranges_of(
            Intercept(PriorBetween(2, 5));
            transformation=sum -> 10 * sum,
            family_distribution=PriorBetween,
            family_distribution_priors=(; upper=PriorBetween(30, 40)),
        ).probability
        @test bounds_of(around_sum) == (20.0, 30.0)
        @test around_sum.source ==
            "its family distribution, such as PriorBetween(20.0, 30.0, false)"
        #A family distribution whose values aren't known, or which can't be made, leaves the parameter out
        @test ranges_of(Intercept(1); family_distribution=value -> :unknown) === (;)
        @test ranges_of(
            Intercept(1); family_distribution=value -> error("not for $value")
        ) === (;)
        @test ranges_of(
            Intercept(1);
            family_distribution=PriorBetween,
            family_distribution_priors=(; upper=:not_a_prior),
        ) === (;)
        #A transformation that throws leaves the parameter out, and so does a regression whose design isn't built
        @test ranges_of(Intercept(1); transformation=value -> error("not for $value")) ===
            (;)
        @test value_ranges(
            HierarchicalRegressionParameters((;
                probability=RegressionParameter(Intercept(PriorBetween(0, 1)))
            )),
        ) === (;)
    end
end

@testset "check_given_values_in_domains" begin
    check = CognitiveModeling.check_given_values_in_domains
    model = CountsWithDomains()

    @testset "Values within the domains pass, and so does a parameter without a domain" begin
        @test check(
            IndependentParameters((;
                probability=PriorBetween(0, 1),
                n_options=PriorBetween(1, 4, true),
                uses_rewards=PriorBetween(0, 1, true),
                bias=PriorBetween(-Inf, Inf),
            )),
            model,
        ) === nothing
        #A prior whose values aren't known is left to the check when a session runs
        @test check(IndependentParameters((; probability=2, strategy=3)), model) === nothing
        @test check(ParametersModelOfOwn(), model) === nothing
        @test check(NoParameters(), model) === nothing
    end

    @testset "Values that reach outside the bounds of a domain are an error" begin
        @test_throws ArgumentError(
            "The parameter probability can be given values from -1.0 to 1.0, by its prior, PriorBetween(-1.0, 1.0, false), but its domain in a cognitive model of type CountsWithDomains is from 0 to 1. A value outside the domain is a ParameterDomainError when a session runs, which stops a fit. Give the parameter a prior or a transformation that stays within the domain, or leave the check out with check_parameter_domains = false in the sessions.",
        ) check(IndependentParameters((; probability=PriorBetween(-1, 1))), model)
        @test_throws "The parameter probability can be given values from 0.0 to 1.5" check(
            IdenticalParameters((; probability=PriorBetween(0, 1.5))), model
        )
        @test_throws "The parameter n_options can be given values from 0.0 to 4.0, by its prior, PriorBetween(0.0, 4.0, true), but its domain in a cognitive model of type CountsWithDomains is from 1 to Inf." check(
            IndependentParameters((; n_options=PriorBetween(0, 4, true))), model
        )
        #A hierarchical parameter and a regression are judged by what their transformations give
        group_priors = (; mean=1, sd=2)
        @test_throws "The parameter probability can be given values from 0.0 to Inf, by the transformation exp of its group-level normal distribution, but its domain in a cognitive model of type CountsWithDomains is from 0 to 1." check(
            HierarchicalGroupedParameters((;
                probability=HierarchicalParameter(group_priors; transformation=exp)
            )),
            model,
        )
        @test check(
            HierarchicalGroupedParameters((;
                probability=HierarchicalParameter(group_priors; transformation=squash)
            )),
            model,
        ) === nothing
        sessions_info = grouped_sessions()
        prepared_regression(regression_parameter) =
            CognitiveModeling.prepare_parameters_model(
                HierarchicalRegressionParameters((; probability=regression_parameter)),
                sessions_info,
            )
        @test_throws "The parameter probability can be given values from -Inf to Inf, by the transformation identity of its regression, but its domain in a cognitive model of type CountsWithDomains is from 0 to 1." check(
            prepared_regression(RegressionParameter(Intercept(1))), model
        )
        @test check(
            prepared_regression(RegressionParameter(Intercept(1); transformation=squash)),
            model,
        ) === nothing
        @test check(
            prepared_regression(RegressionParameter(Intercept(PriorBetween(0, 1)))), model
        ) === nothing
        #With a family distribution, by the values of the distribution
        @test_throws "The parameter probability can be given values from 0.0 to 3.0, by its family distribution, such as PriorBetween(0.0, 3.0, false), but its domain in a cognitive model of type CountsWithDomains is from 0 to 1." check(
            prepared_regression(
                RegressionParameter(
                    Intercept(1);
                    transformation=squash,
                    family_distribution=value -> PriorBetween(0, 3),
                ),
            ),
            model,
        )
        @test check(
            prepared_regression(
                RegressionParameter(
                    Intercept(1); family_distribution=value -> PriorBetween(0, 1)
                ),
            ),
            model,
        ) === nothing
    end

    @testset "Continuous values for a discrete parameter are an error, and so are numbers for a parameter whose values aren't numbers" begin
        @test_throws ArgumentError(
            "The parameter n_options is discrete in a cognitive model of type CountsWithDomains, with values of type Integer, but it is given continuous values by its prior, PriorBetween(1.0, 4.0, false). A value of another type is a ParameterDomainError when a session runs, which stops a fit.",
        ) check(IndependentParameters((; n_options=PriorBetween(1, 4))), model)
        #A domain of Bool takes the values 0 and 1
        @test_throws "The parameter uses_rewards can be given values from 0.0 to 2.0, by its prior, PriorBetween(0.0, 2.0, true), but its domain in a cognitive model of type CountsWithDomains is from 0 to 1." check(
            IndependentParameters((; uses_rewards=PriorBetween(0, 2, true))), model
        )
        @test_throws "The parameter uses_rewards is discrete in a cognitive model of type CountsWithDomains, with values of type Bool, but it is given continuous values" check(
            IndependentParameters((; uses_rewards=PriorBetween(0, 1))), model
        )
        @test_throws ArgumentError(
            "The parameter strategy has values of type Symbol in a cognitive model of type CountsWithDomains, but it is given numbers by its prior, PriorBetween(1.0, 3.0, true).",
        ) check(IndependentParameters((; strategy=PriorBetween(1, 3, true))), model)
        #A regression gives a discrete parameter its values through a family distribution
        sessions_info = grouped_sessions()
        uses_rewards_from(regression_parameter) =
            CognitiveModeling.prepare_parameters_model(
                HierarchicalRegressionParameters((; uses_rewards=regression_parameter)),
                sessions_info,
            )
        @test_throws "The parameter uses_rewards is discrete in a cognitive model of type CountsWithDomains, with values of type Bool, but it is given continuous values by the transformation squash of its regression." check(
            uses_rewards_from(RegressionParameter(Intercept(1); transformation=squash)),
            model,
        )
        @test check(
            uses_rewards_from(
                RegressionParameter(
                    Intercept(1);
                    transformation=squash,
                    family_distribution=value -> PriorBetween(0, 1, true),
                ),
            ),
            model,
        ) === nothing
    end

    @testset "Fixed values are checked as those of a session are when it runs" begin
        @test check(
            FixedParameters((; probability=0.5, n_options=3.0, strategy=:greedy, bias=-2)),
            model,
        ) === nothing
        @test_throws ParameterDomainError(
            "The parameter probability of a cognitive model of type CountsWithDomains is 1.5, which is outside its domain, from 0 to 1.",
        ) check(FixedParameters((; probability=1.5)), model)
        @test_throws ParameterDomainError check(FixedParameters((; n_options=2.5)), model)
        @test_throws ParameterDomainError check(FixedParameters((; strategy=1)), model)
        #The values of every session, in one vector for each parameter or by session
        @test check(
            SessionwiseFixedParameters((; probability=[0.1, 0.2, 0.3, 0.4])), model
        ) === nothing
        @test_throws ParameterDomainError(
            "The parameter probability of a cognitive model of type CountsWithDomains is -0.3, which is outside its domain, from 0 to 1.",
        ) check(SessionwiseFixedParameters((; probability=[0.1, 0.2, -0.3, 0.4])), model)
        @test_throws ParameterDomainError check(
            SessionwiseFixedParameters((; S1=(; probability=0.1), S2=(; probability=7.0))),
            model,
        )
        @test_throws ParameterDomainError check(
            SessionwiseFixedParameters(
                Dict("S1" => (; probability=0.1), "S2" => (; probability=7.0))
            ),
            model,
        )
    end

    @testset "Every part of a composite is checked" begin
        @test check(
            CompositeParameters(
                IndependentParameters((; probability=PriorBetween(0, 1))),
                FixedParameters((; n_options=3)),
            ),
            model,
        ) === nothing
        @test_throws ParameterDomainError check(
            CompositeParameters(
                IndependentParameters((; probability=PriorBetween(0, 1))),
                FixedParameters((; n_options=0)),
            ),
            model,
        )
        @test_throws "The parameter probability can be given values from 0.0 to 2.0" check(
            CompositeParameters(
                FixedParameters((; n_options=3));
                noise=IndependentParameters((; probability=PriorBetween(0, 2))),
            ),
            model,
        )
    end

    @testset "A model without domains and sessions without an environment have nothing to check, and domains that aren't ParameterDomains are an error" begin
        @test check(
            IndependentParameters((; probability=PriorBetween(-1, 2))), SessionCounter()
        ) === nothing
        @test check(
            IndependentParameters((; scale=PriorBetween(-1, 2))), NoEnvironment()
        ) === nothing
        @test_throws "parameter_domains of a cognitive model of type CountsWithDomainsAsNamedTuples returns (n_options = (lower = 1, discrete = true),), but it must return a NamedTuple with a ParameterDomain under the name of each parameter that has a domain" check(
            IndependentParameters((; n_options=PriorBetween(1, 4, true))),
            CountsWithDomainsAsNamedTuples(),
        )
        @test_throws "but it must return a NamedTuple with a ParameterDomain" check(
            FixedParameters((; n_options=2)), CountsWithDomainsAsNamedTuples()
        )
    end
end

@testset "prepare_full_model checks what the parameter models can give against the domains" begin
    prepare = CognitiveModeling.prepare_full_model
    within = IndependentParameters((; probability=PriorBetween(0, 1)))
    outside = IndependentParameters((; probability=PriorBetween(-1, 1)))

    @testset "A prior or a fixed value outside a domain is an error, unless the sessions leave the check out" begin
        sessions_info = sessions_with_domains()
        @test prepare(within, sessions_info).cognitive_parameters_model === within
        @test_throws "The parameter probability can be given values from -1.0 to 1.0" prepare(
            outside, sessions_info
        )
        @test_throws ParameterDomainError prepare(
            within, sessions_info; fixed_cognitive_parameters=(; n_options=0)
        )
        @test_throws ParameterDomainError prepare(
            SessionwiseFixedParameters((;
                S1=(; probability=0.1),
                S2=(; probability=0.2),
                S3=(; probability=1.3),
                S4=(; probability=0.4),
            )),
            sessions_info,
        )
        unchecked_sessions = sessions_with_domains(; check_parameter_domains=false)
        @test prepare(outside, unchecked_sessions).cognitive_parameters_model === outside
        @test prepare(
            within, unchecked_sessions; fixed_cognitive_parameters=(; n_options=0)
        ) isa NamedTuple
    end

    @testset "A regression is checked with the design that was built for the sessions" begin
        sessions_info = sessions_with_domains()
        @test_throws "The parameter probability can be given values from -2.0 to 5.0, by the transformation identity of its regression" prepare(
            HierarchicalRegressionParameters((;
                probability=RegressionParameter(
                    Intercept(PriorBetween(0, 1)),
                    ContinuousPredictor(:site, PriorBetween(-1, 2)),
                )
            )),
            sessions_info,
        )
        @test prepare(
            HierarchicalRegressionParameters((;
                probability=RegressionParameter(
                    Intercept(1), VaryingEffects(:id, Intercept(2)); transformation=squash
                )
            )),
            sessions_info,
        ) isa NamedTuple
    end

    @testset "The parameters of an environment are checked when a session runs it" begin
        sessions_info = sessions_with_domains(; environment=GivesRewardWithDomain())
        @test !any(session -> session.includes_environment, sessions_info.sessions)
        #No session runs the environment, so what it is given is never used
        @test prepare(
            within,
            sessions_info;
            environment_parameters_model=FixedParameters((; scale=-1.0)),
        ) isa NamedTuple
        #When its parameters are estimated, every session runs it
        @test_throws "The parameter scale can be given values from -1.0 to 1.0, by its prior, PriorBetween(-1.0, 1.0, false), but its domain in an environment model of type GivesRewardWithDomain is from 0 to Inf." prepare(
            within,
            sessions_info;
            environment_parameters_model=IdenticalParameters((; scale=PriorBetween(-1, 1))),
        )
        @test prepare(
            within,
            sessions_info;
            environment_parameters_model=IdenticalParameters((; scale=PriorBetween(0, 1))),
        ) isa NamedTuple
    end
end
