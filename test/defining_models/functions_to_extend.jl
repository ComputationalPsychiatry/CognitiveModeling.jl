struct ModelWithoutMethods <: AbstractCognitiveModel end
struct EnvironmentWithoutMethods <: AbstractEnvironmentModel end

#Leaves the context without a type annotation, which must not be ambiguous with the fallback
struct ModelWithUnannotatedContext <: AbstractCognitiveModel end
function CognitiveModeling.apply_timestep_model(model::ModelWithUnannotatedContext, context)
    return ModelOutput((; choice=0.7))
end

@testset "Functions to extend" begin
    model_context = ModelContext((; reward=NoValue()), (;), (; choice=NoValue()), (;))
    environment_context = ModelContext((; choice=NoValue()), (;), (; reward=NoValue()), (;))

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "There is no apply_timestep_model method for a cognitive model of type ModelWithoutMethods with a context of type" CognitiveModeling.apply_timestep_model(
            ModelWithoutMethods(), model_context
        )
        @test_throws "CognitiveModeling.apply_timestep_model(model::ModelWithoutMethods, context::CognitiveModeling.ModelContext)" CognitiveModeling.apply_timestep_model(
            ModelWithoutMethods(), model_context
        )
        #An environment model gets the same message, named as one
        @test_throws "There is no apply_timestep_model method for an environment model of type EnvironmentWithoutMethods with a context of type" CognitiveModeling.apply_timestep_model(
            EnvironmentWithoutMethods(), environment_context
        )
        @test_throws "CognitiveModeling.apply_timestep_model(model::EnvironmentWithoutMethods, context::CognitiveModeling.ModelContext)" CognitiveModeling.apply_timestep_model(
            EnvironmentWithoutMethods(), environment_context
        )
    end

    @testset "A timestep method can leave its context without a type" begin
        @test CognitiveModeling.apply_timestep_model(
            ModelWithUnannotatedContext(), model_context
        ).output_distribution.choice == 0.7
    end

    @testset "A missing method for the initial state gives an informative error" begin
        @test_throws "There is no initialize_state method for a cognitive model of type ModelWithoutMethods. Each model must define CognitiveModeling.initialize_state(model::ModelWithoutMethods, parameters)" CognitiveModeling.initialize_state(
            ModelWithoutMethods(), (;)
        )
        @test_throws "There is no initialize_state method for an environment model of type EnvironmentWithoutMethods. Each model must define CognitiveModeling.initialize_state(model::EnvironmentWithoutMethods, parameters)" CognitiveModeling.initialize_state(
            EnvironmentWithoutMethods(), (;)
        )
    end

    @testset "The defaults of the other functions" begin
        model = ModelWithoutMethods()
        @test isempty(CognitiveModeling.default_parameters(model))
        @test isempty(CognitiveModeling.default_parameters(EnvironmentWithoutMethods()))
        #Preparing a state for saving does nothing to it
        state = (; values=[0.5])
        @test CognitiveModeling.prepare_state_for_saving(model, state) === state
        @test CognitiveModeling.prepare_state_for_saving(
            EnvironmentWithoutMethods(), state
        ) === state
        #Neither kind of model ends by default, whatever its state and parameters
        @test CognitiveModeling.stopping_criterion(model, state, (;)) === false
        @test CognitiveModeling.stopping_criterion(
            EnvironmentWithoutMethods(), state, (; last_timestep=0)
        ) === false
        @test CognitiveModeling.stopping_criterion(model, nothing, (;)) === false
    end

    @testset "A missing method for the names gives an informative error" begin
        @test_throws "There is no attribute_names method for a cognitive model of type ModelWithoutMethods. Each model must define CognitiveModeling.attribute_names(model::ModelWithoutMethods), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs." CognitiveModeling.attribute_names(
            ModelWithoutMethods()
        )
        @test_throws "There is no attribute_names method for an environment model of type EnvironmentWithoutMethods. Each model must define CognitiveModeling.attribute_names(model::EnvironmentWithoutMethods), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs." CognitiveModeling.attribute_names(
            EnvironmentWithoutMethods()
        )
    end

    @testset "The functions to extend are public, but not exported" begin
        for name in (
            :apply_timestep_model,
            :initialize_state,
            :attribute_names,
            :prepare_state_for_saving,
            :default_parameters,
            :stopping_criterion,
            :turing_apply_timestep_model,
        )
            @test Base.ispublic(CognitiveModeling, name)
            @test !Base.isexported(CognitiveModeling, name)
        end
    end
end
