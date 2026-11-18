#The function that the tutorial on discrete parameters shows for reading the names of the discrete and of the continuous variables of a Turing.jl model, as it is written there. It is all Turing.jl's and DynamicPPL.jl's own, so these tests say when a new version of either changes what the tutorial shows
function discrete_and_continuous_variable_names(model)
    distributions = Turing.DynamicPPL.extract_priors(model)
    variable_names = collect(keys(distributions))
    is_discrete = map(name -> distributions[name] isa DiscreteDistribution, variable_names)
    return (;
        discrete=variable_names[is_discrete], continuous=variable_names[.!is_discrete]
    )
end

#A Turing.jl model that has nothing to do with sessions, with discrete and continuous variables of one value and of several
Turing.@model function model_with_both_kinds_of_variables(observed_value)
    n_components ~ Poisson(3)
    component ~ Categorical([0.2, 0.3, 0.5])
    means ~ filldist(Normal(0, 1), 3)
    sd ~ Exponential(1)
    flags ~ filldist(Bernoulli(0.5), 2)
    #A discrete distribution whose values aren't whole numbers
    grid_value ~ DiscreteNonParametric([0.1, 0.5, 0.9], [0.2, 0.3, 0.5])
    return observed_value ~ Normal(means[component] + grid_value, sd)
end

@testset "The discrete and the continuous variables of a model, as the tutorial reads them" begin
    @testset "A variable is discrete when its distribution is, in any Turing.jl model" begin
        variable_names = discrete_and_continuous_variable_names(
            model_with_both_kinds_of_variables(0.3)
        )
        @test variable_names.discrete == [
            @varname(n_components),
            @varname(component),
            @varname(flags),
            @varname(grid_value),
        ]
        @test variable_names.continuous == [@varname(means), @varname(sd)]
    end

    @testset "A discrete parameter of the sessions and an imputed choice are discrete variables, and each kind takes a sampler" begin
        parameters_model = IndependentParameters((;
            learning_rate=Beta(2, 2), uses_rewards=Bernoulli(0.5)
        ))
        model = turing_full_model(
            parameters_model, sessions_for_test(LearnsOrNot(); impute_missing_actions=true)
        )
        variable_names = discrete_and_continuous_variable_names(model)
        @test variable_names.discrete == [
            @varname(uses_rewards.session), @varname(session[3].timestep[3].choice.imputed)
        ]
        @test variable_names.continuous == [@varname(learning_rate.session)]

        #A sampler which uses gradients can't sample the model
        @test_throws ArgumentError sample(Xoshiro(1), model, NUTS(), 20; progress=false)
        #With a sampler for each kind, and adaptation steps that are left out of the chain, the variables of both kinds change between samples
        sampler = Gibbs(
            variable_names.continuous => NUTS(), variable_names.discrete => MH()
        )
        chain = sample(
            Xoshiro(1), model, sampler, 60; nadapts=30, discard_initial=30, progress=false
        )
        @test size(chain, 1) == 60
        sampled_parameters = map(
            sampled -> sampled.cognitive_parameters, returned(model, chain)
        )
        @test all(
            parameters -> parameters.uses_rewards isa Vector{Bool}, sampled_parameters
        )
        @test length(
            unique(map(parameters -> parameters.learning_rate, sampled_parameters))
        ) > 1
        @test length(
            unique(map(parameters -> parameters.uses_rewards, sampled_parameters))
        ) > 1
    end

    @testset "The variables of a regression with a family are of both kinds" begin
        learning_rate = RegressionParameter(
            Intercept(Normal(0, 1)),
            VaryingEffects(:id, Intercept(Exponential(1)));
            transformation=to_unit_interval,
        )
        uses_rewards = RegressionParameter(Intercept(Normal(0, 1)); family=BernoulliLogit)
        model = turing_full_model(
            HierarchicalRegressionParameters((; learning_rate, uses_rewards)),
            sessions_for_test(
                LearnsOrNot(), REGRESSION_FULL_TABLE; impute_missing_actions=false
            ),
        )
        variable_names = discrete_and_continuous_variable_names(model)
        @test variable_names.discrete == [@varname(uses_rewards.session)]
        @test variable_names.continuous == [
            @varname(learning_rate.coef),
            @varname(learning_rate.id.intercept.sd),
            @varname(learning_rate.id.intercept.deviation_z),
            @varname(uses_rewards.coef),
        ]
    end

    @testset "A model without discrete variables has none" begin
        model = turing_full_model(
            IndependentParameters(FULL_PRIORS),
            sessions_for_test(; impute_missing_actions=false),
        )
        variable_names = discrete_and_continuous_variable_names(model)
        @test isempty(variable_names.discrete)
        @test variable_names.continuous ==
            [@varname(learning_rate.session), @varname(inverse_temperature.session)]
    end
end
