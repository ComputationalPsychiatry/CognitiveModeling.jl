## Models for the tests ##
#Only has the normal functions, so it gets the defaults in Turing
struct OnlyNormalFunction <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyNormalFunction, parameters) = (; total=0.0)
function CognitiveModeling.apply_model(model::OnlyNormalFunction, context::ModelContext)
    return ModelOutput(
        (; choice=Bernoulli(0.5)),
        (; total=context.previous_state.total + context.input.reward),
    )
end
function CognitiveModeling.attribute_names(model::OnlyNormalFunction)
    return ModelAttributes(; parameters=(), states=:total, inputs=:reward, outputs=:choice)
end

#Has its own Turing version, which samples within the timestep
struct SamplesWithinTimestep <: AbstractCognitiveModel end
Turing.@model function CognitiveModeling.turing_apply_model(
    model::SamplesWithinTimestep, context::ModelContext
)
    noise ~ Normal(0, 1)
    return ModelOutput(
        (; choice=Bernoulli(0.5)),
        (; total=context.previous_state.total + context.input.reward + noise),
    )
end

#Runs several timesteps as submodels, each with its own prefix
Turing.@model function run_timesteps_as_submodels(model, rewards)
    state = (; total=0.0)
    for t in eachindex(rewards)
        context = ModelContext((; reward=rewards[t]), state, (; choice=NoAction()), (;))
        output ~ to_submodel(
            prefix(turing_apply_model(model, context), @varname(timestep[t])), false
        )
        state = output.new_state
    end
    return state
end

#Samples when it initializes, in Turing
struct SamplesWhenInitializing <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::SamplesWhenInitializing, parameters)
    return (; total=parameters.start)
end
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::SamplesWhenInitializing, parameters
)
    offset ~ Normal(0, 1)
    return (; total=parameters.start + offset)
end

#An environment with only the normal functions
struct OnlyNormalEnvironment <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::OnlyNormalEnvironment, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_model(
    environment::OnlyNormalEnvironment, context::ModelContext
)
    return ModelOutput(
        (; reward=Deterministic(1.0)),
        (; n_timesteps=context.previous_state.n_timesteps + 1),
    )
end

#A model without methods of its own
struct ModelWithoutMethods <: AbstractCognitiveModel end

#A parameters model and a downstream model without methods of their own
struct ParametersModelWithoutMethod <: AbstractSessionParametersModel end
struct DownstreamModelWithoutMethod <: AbstractDownstreamModel end

@testset "turing_initialize_state" begin
    @testset "The default gives what initialize_state returns, for both kinds of model, and has no variables" begin
        @test turing_initialize_state(OnlyNormalFunction(), (;))() ===
            initialize_state(OnlyNormalFunction(), (;))
        @test turing_initialize_state(OnlyNormalEnvironment(), (;))() === (; n_timesteps=0)
        @test isempty(
            keys(rand(Xoshiro(1), turing_initialize_state(OnlyNormalEnvironment(), (;))))
        )
    end

    @testset "A model's own method is used instead of the default, with its variable" begin
        model = turing_initialize_state(SamplesWhenInitializing(), (; start=0.5))
        @test collect(keys(rand(Xoshiro(1), model))) == [@varname(offset)]
        @test returned(model, (; offset=0.25)) === (; total=0.75)
        @test logjoint(model, (; offset=0.25)) ≈ logpdf(Normal(0, 1), 0.25)
    end

    @testset "A model without an initialize_state method gives the informative error" begin
        @test_throws "Each model must define" turing_initialize_state(
            ModelWithoutMethods(), (;)
        )()
    end
end

@testset "The parts of a full model without methods give informative errors" begin
    sessions_info = MultipleSessionsInfo(
        OnlyNormalFunction(),
        (; id=[1, 1], reward=[1.0, 0.0], choice=[true, false]);
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    )
    @test_throws "There is no turing_session_parameters_model method for a parameters model of type ParametersModelWithoutMethod. Each parameters model must define Turing.@model function CognitiveModeling.turing_session_parameters_model(parameters_model::ParametersModelWithoutMethod, sessions_info::MultipleSessionsInfo), which returns a NamedTuple with one vector for each parameter." CognitiveModeling.turing_session_parameters_model(
        ParametersModelWithoutMethod(), sessions_info
    )
    @test_throws "A parameters model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it has type @NamedTuple{learning_rate::Beta{Float64}}." CognitiveModeling.turing_session_parameters_model(
        (; learning_rate=Beta(2, 2)), sessions_info
    )
    @test_throws "There is no turing_downstream_model method for a downstream model of type DownstreamModelWithoutMethod." CognitiveModeling.turing_downstream_model(
        DownstreamModelWithoutMethod(), [], (;), sessions_info
    )
    @test_throws "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type Nothing." CognitiveModeling.turing_downstream_model(
        nothing, [], (;), sessions_info
    )
    #The default downstream model samples nothing and returns nothing
    @test CognitiveModeling.turing_downstream_model(
        NoDownstreamModel(), [], (;), sessions_info
    )() === nothing
    @test isempty(
        keys(
            rand(
                Xoshiro(1),
                CognitiveModeling.turing_downstream_model(
                    NoDownstreamModel(), [], (;), sessions_info
                ),
            ),
        ),
    )
end

@testset "turing_apply_model" begin
    context = ModelContext((; reward=1.0), (; total=0.5), (; choice=NoAction()), (;))

    @testset "The default gives what apply_model returns" begin
        output = turing_apply_model(OnlyNormalFunction(), context)()
        @test output isa ModelOutput
        @test output.new_state == apply_model(OnlyNormalFunction(), context).new_state
        @test output.output_distribution ==
            apply_model(OnlyNormalFunction(), context).output_distribution
    end

    @testset "The default works as a submodel, and has no variables" begin
        model = run_timesteps_as_submodels(OnlyNormalFunction(), [1.0, 2.0])
        @test isempty(keys(rand(Xoshiro(1), model)))
        @test model() == (; total=3.0)
    end

    @testset "A model without an apply_model method gives the informative error" begin
        @test_throws "Each model must define" turing_apply_model(
            ModelWithoutMethods(), context
        )()
    end

    @testset "The default works for an environment model too" begin
        environment_context = ModelContext(
            (; choice=true), (; n_timesteps=2), (; reward=0.0), (;), 1
        )
        output = turing_apply_model(OnlyNormalEnvironment(), environment_context)()
        @test output isa ModelOutput && output.new_state === (; n_timesteps=3)
    end

    @testset "A model's own method is used instead of the default" begin
        @test @varname(noise) in
            keys(rand(Xoshiro(1), turing_apply_model(SamplesWithinTimestep(), context)))
    end

    @testset "Each timestep has its own variable, and they count in the log density" begin
        model = run_timesteps_as_submodels(SamplesWithinTimestep(), [1.0, 2.0])
        #Turing warns about the array of timesteps when it handles the values, which @test_logs hides
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) ==
            Set([@varname(timestep[1].noise), @varname(timestep[2].noise)])
        noises = Dict(
            @varname(timestep[1].noise) => 0.25, @varname(timestep[2].noise) => -0.5
        )
        log_density = @test_logs match_mode = :any logjoint(model, noises)
        @test log_density ≈ logpdf(Normal(0, 1), 0.25) + logpdf(Normal(0, 1), -0.5)
        @test (@test_logs match_mode = :any returned(model, noises)) == (; total=2.75)
    end
end
