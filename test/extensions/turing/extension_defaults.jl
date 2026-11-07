## Models for the tests ##
#Only has the normal functions, so it gets the defaults in Turing
struct OnlyNormalFunction <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyNormalFunction, parameters) = (; total = 0.0)
function CognitiveModeling.apply_model(model::OnlyNormalFunction, input::ModelInput)
    return ModelOutput((; choice = Bernoulli(0.5)), (; total = input.previous_state.total + input.observation.reward))
end
CognitiveModeling.attribute_names(model::OnlyNormalFunction) = ModelAttributes(; parameters = (), states = :total, observations = :reward, actions = :choice)

#Has its own Turing version, which samples within the timestep
struct SamplesWithinTimestep <: AbstractCognitiveModel end
Turing.@model function CognitiveModeling.turing_apply_model(model::SamplesWithinTimestep, input::ModelInput)
    noise ~ Normal(0, 1)
    return ModelOutput((; choice = Bernoulli(0.5)), (; total = input.previous_state.total + input.observation.reward + noise))
end

#Runs several timesteps as submodels, each with its own prefix
Turing.@model function run_timesteps_as_submodels(model, rewards)
    state = (; total = 0.0)
    for t in eachindex(rewards)
        input = ModelInput((; reward = rewards[t]), state, (; choice = NoAction()), (;))
        output ~ to_submodel(prefix(turing_apply_model(model, input), @varname(timestep[t])), false)
        state = output.new_state
    end
    return state
end

#Samples when it initializes, in Turing
struct SamplesWhenInitializing <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::SamplesWhenInitializing, parameters) = (; total = parameters.start)
Turing.@model function CognitiveModeling.turing_initialize_state(model::SamplesWhenInitializing, parameters)
    offset ~ Normal(0, 1)
    return (; total = parameters.start + offset)
end

#An environment with only the normal functions
struct OnlyNormalEnvironment <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::OnlyNormalEnvironment, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::OnlyNormalEnvironment, input::EnvironmentInput)
    return EnvironmentOutput((; reward = Deterministic(1.0)), (; n_timesteps = input.previous_state.n_timesteps + 1))
end

#A population model and a downstream model without methods of their own
struct PopulationModelWithoutMethod <: AbstractPopulationModel end
struct DownstreamModelWithoutMethod <: AbstractDownstreamModel end

@testset "turing_initialize_state" begin
    @testset "The default gives what initialize_state returns, for both kinds of model, and has no variables" begin
        @test turing_initialize_state(OnlyNormalFunction(), (;))() === initialize_state(OnlyNormalFunction(), (;))
        @test turing_initialize_state(OnlyNormalEnvironment(), (;))() === (; n_timesteps = 0)
        @test isempty(keys(rand(Xoshiro(1), turing_initialize_state(OnlyNormalEnvironment(), (;)))))
    end

    @testset "A model's own method is used instead of the default, with its variable" begin
        model = turing_initialize_state(SamplesWhenInitializing(), (; start = 0.5))
        @test collect(keys(rand(Xoshiro(1), model))) == [@varname(offset)]
        @test returned(model, (; offset = 0.25)) === (; total = 0.75)
        @test logjoint(model, (; offset = 0.25)) ≈ logpdf(Normal(0, 1), 0.25)
    end

    @testset "A model without an initialize_state method gives the informative error" begin
        @test_throws "Each model must define" turing_initialize_state(ModelWithoutMethods(), (;))()
    end
end

@testset "The parts of a full model without methods give informative errors" begin
    sessions_info = MultipleSessionsInfo(OnlyNormalFunction(), (; id = [1, 1], reward = [1.0, 0.0], choice = [true, false]); observation_columns = :reward, action_columns = :choice, session_columns = :id, who_starts = EnvironmentStarts())
    @test_throws "There is no turing_population_model method for a population model of type PopulationModelWithoutMethod. Each population model must define Turing.@model function CognitiveModeling.turing_population_model(population_model::PopulationModelWithoutMethod, sessions_info::MultipleSessionsInfo)" CognitiveModeling.turing_population_model(PopulationModelWithoutMethod(), sessions_info)
    @test_throws "The population model must be a subtype of AbstractPopulationModel, such as IndependentSessions(priors) with the priors in a NamedTuple, but it has type @NamedTuple{learning_rate::Beta{Float64}}." CognitiveModeling.turing_population_model((; learning_rate = Beta(2, 2)), sessions_info)
    @test_throws "There is no turing_downstream_model method for a downstream model of type DownstreamModelWithoutMethod." CognitiveModeling.turing_downstream_model(DownstreamModelWithoutMethod(), [], (;), sessions_info)
    @test_throws "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type Nothing." CognitiveModeling.turing_downstream_model(nothing, [], (;), sessions_info)
    #The default downstream model samples nothing and returns nothing
    @test CognitiveModeling.turing_downstream_model(NoDownstreamModel(), [], (;), sessions_info)() === nothing
    @test isempty(keys(rand(Xoshiro(1), CognitiveModeling.turing_downstream_model(NoDownstreamModel(), [], (;), sessions_info))))
end

@testset "turing_apply_model" begin
    input = ModelInput((; reward = 1.0), (; total = 0.5), (; choice = NoAction()), (;))

    @testset "The default gives what apply_model returns" begin
        output = turing_apply_model(OnlyNormalFunction(), input)()
        @test output isa ModelOutput
        @test output.new_state == apply_model(OnlyNormalFunction(), input).new_state
        @test output.action_distribution == apply_model(OnlyNormalFunction(), input).action_distribution
    end

    @testset "The default works as a submodel, and has no variables" begin
        model = run_timesteps_as_submodels(OnlyNormalFunction(), [1.0, 2.0])
        @test isempty(keys(rand(Xoshiro(1), model)))
        @test model() == (; total = 3.0)
    end

    @testset "A model without an apply_model method gives the informative error" begin
        @test_throws "Each model must define" turing_apply_model(ModelWithoutMethods(), input)()
    end

    @testset "The default works for an environment model too" begin
        environment_input = EnvironmentInput((; choice = true), (; n_timesteps = 2), (; reward = 0.0), (;), 1)
        output = turing_apply_model(OnlyNormalEnvironment(), environment_input)()
        @test output isa EnvironmentOutput && output.new_state === (; n_timesteps = 3)
    end

    @testset "A model's own method is used instead of the default" begin
        @test @varname(noise) in keys(rand(Xoshiro(1), turing_apply_model(SamplesWithinTimestep(), input)))
    end

    @testset "Each timestep has its own variable, and they count in the log density" begin
        model = run_timesteps_as_submodels(SamplesWithinTimestep(), [1.0, 2.0])
        #Turing warns about the array of timesteps when it handles the values, which @test_logs hides
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([@varname(timestep[1].noise), @varname(timestep[2].noise)])
        noises = Dict(@varname(timestep[1].noise) => 0.25, @varname(timestep[2].noise) => -0.5)
        log_density = @test_logs match_mode = :any logjoint(model, noises)
        @test log_density ≈ logpdf(Normal(0, 1), 0.25) + logpdf(Normal(0, 1), -0.5)
        @test (@test_logs match_mode = :any returned(model, noises)) == (; total = 2.75)
    end
end
