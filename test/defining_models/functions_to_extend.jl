struct ModelWithoutMethods <: AbstractCognitiveModel end
struct EnvironmentWithoutMethods <: AbstractEnvironmentModel end

#Leaves the input without a type annotation, which must not be ambiguous with the fallback
struct ModelWithUnannotatedInput <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::ModelWithUnannotatedInput, input) = ModelOutput(0.7)

@testset "Functions to extend" begin
    model_input = ModelInput(NoObservation(), (;), NoAction(), (;))
    environment_input = EnvironmentInput(NoAction(), (;), NoObservation(), (;))

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "CognitiveModeling.apply_model(model::ModelWithoutMethods, input::CognitiveModeling.ModelInput)" CognitiveModeling.apply_model(ModelWithoutMethods(), model_input)
        @test_throws "CognitiveModeling.apply_model(environment::EnvironmentWithoutMethods, input::CognitiveModeling.EnvironmentInput)" CognitiveModeling.apply_model(EnvironmentWithoutMethods(), environment_input)
    end

    @testset "A timestep method can leave its input without a type" begin
        @test CognitiveModeling.apply_model(ModelWithUnannotatedInput(), model_input).action_distribution.action == 0.7
    end

    @testset "The defaults of the other functions" begin
        model = ModelWithoutMethods()
        @test CognitiveModeling.initialize_state(model, (;)) === (;)
        @test CognitiveModeling.initialize_state(EnvironmentWithoutMethods(), (;)) === (;)
        @test isempty(CognitiveModeling.default_parameters(model))
    end

    @testset "A missing method for listing names gives an informative error" begin
        model = ModelWithoutMethods()
        @test_throws "CognitiveModeling.get_parameters(model::ModelWithoutMethods)" CognitiveModeling.get_parameters(model)
        @test_throws "CognitiveModeling.get_states(model::ModelWithoutMethods)" CognitiveModeling.get_states(model)
        @test_throws "CognitiveModeling.get_observations(model::ModelWithoutMethods)" CognitiveModeling.get_observations(model)
        @test_throws "CognitiveModeling.get_actions(model::ModelWithoutMethods)" CognitiveModeling.get_actions(model)
    end
end
