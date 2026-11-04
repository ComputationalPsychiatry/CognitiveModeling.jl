struct ModelWithoutMethods <: AbstractCognitiveModel end
struct EnvironmentWithoutMethods <: AbstractEnvironmentModel end

#Leaves the input without a type annotation, which must not be ambiguous with the fallback
struct ModelWithUnannotatedInput <: AbstractCognitiveModel end
CognitiveModeling.apply_model(model::ModelWithUnannotatedInput, input) = ModelOutput((; choice = 0.7))

@testset "Functions to extend" begin
    model_input = ModelInput(NoObservation(), (;), NoAction(), (;))
    environment_input = EnvironmentInput(NoAction(), (;), NoObservation(), (;))

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "CognitiveModeling.apply_model(model::ModelWithoutMethods, input::CognitiveModeling.ModelInput)" CognitiveModeling.apply_model(ModelWithoutMethods(), model_input)
        @test_throws "CognitiveModeling.apply_model(environment::EnvironmentWithoutMethods, input::CognitiveModeling.EnvironmentInput)" CognitiveModeling.apply_model(EnvironmentWithoutMethods(), environment_input)
    end

    @testset "A timestep method can leave its input without a type" begin
        @test CognitiveModeling.apply_model(ModelWithUnannotatedInput(), model_input).action_distribution.choice == 0.7
    end

    @testset "A missing method for the initial state gives an informative error" begin
        @test_throws "CognitiveModeling.initialize_state(model::ModelWithoutMethods, parameters)" CognitiveModeling.initialize_state(ModelWithoutMethods(), (;))
        @test_throws "CognitiveModeling.initialize_state(environment::EnvironmentWithoutMethods, parameters)" CognitiveModeling.initialize_state(EnvironmentWithoutMethods(), (;))
    end

    @testset "The defaults of the other functions" begin
        model = ModelWithoutMethods()
        @test isempty(CognitiveModeling.default_parameters(model))
        #Preparing a state for saving does nothing to it
        state = (; values = [0.5])
        @test CognitiveModeling.prepare_state_for_saving(model, state) === state
        @test CognitiveModeling.prepare_state_for_saving(EnvironmentWithoutMethods(), state) === state
    end

    @testset "A missing method for the names gives an informative error" begin
        @test_throws "CognitiveModeling.attribute_names(model::ModelWithoutMethods)" CognitiveModeling.attribute_names(ModelWithoutMethods())
    end
end
