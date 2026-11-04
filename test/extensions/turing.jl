using CognitiveModeling: apply_model, turing_apply_model
using Random: Xoshiro

## Models for the tests ##
#Only has the normal function, so it gets the default in Turing
struct OnlyNormalFunction <: AbstractCognitiveModel end
function CognitiveModeling.apply_model(model::OnlyNormalFunction, input::ModelInput)
    return ModelOutput((; choice = Bernoulli(0.5)), (; total = input.previous_state.total + input.observation.reward))
end

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
        input = ModelInput((; reward = rewards[t]), state, NoAction(), (;))
        output ~ to_submodel(prefix(turing_apply_model(model, input), @varname(timestep[t])), false)
        state = output.new_state
    end
    return state
end

@testset "Turing extension" begin
    @test Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt) isa Module

    input = ModelInput((; reward = 1.0), (; total = 0.5), NoAction(), (;))

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
