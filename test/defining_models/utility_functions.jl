struct ModelWithAttributes <: AbstractCognitiveModel end
CognitiveModeling.get_parameters(model::ModelWithAttributes) = (:learning_rate, :action_noise)
CognitiveModeling.get_states(model::ModelWithAttributes) = (:expected_reward,)
CognitiveModeling.get_observations(model::ModelWithAttributes) = (:reward,)
CognitiveModeling.get_actions(model::ModelWithAttributes) = (:choice,)

struct ModelWithFields{T} <: AbstractCognitiveModel
    n_options::Int
    prior::T
end

#Says that it has no observations, which is different from not saying anything
struct ModelWithoutObservations <: AbstractCognitiveModel end
CognitiveModeling.get_observations(model::ModelWithoutObservations) = ()

#Gets its states from its supertype
abstract type FamilyOfModels <: AbstractCognitiveModel end
CognitiveModeling.get_states(model::FamilyOfModels) = (:shared_state,)
struct ModelInFamily <: FamilyOfModels end

@testset "Utility functions" begin
    @testset "get_attributes collects the attributes, and is type stable" begin
        attributes = @inferred CognitiveModeling.get_attributes(ModelWithAttributes())
        @test attributes == (parameters = (:learning_rate, :action_noise), states = (:expected_reward,), observations = (:reward,), actions = (:choice,))
    end

    @testset "get_attributes throws an error if a method is missing" begin
        @test_throws "get_parameters" CognitiveModeling.get_attributes(ModelWithFields(4, 0.5))
    end

    @testset "The REPL shows a summary of the model" begin
        @test repr("text/plain", ModelWithAttributes()) == """
            -- Cognitive model --
            Type: ModelWithAttributes
            Parameters: learning_rate, action_noise
            States: expected_reward
            Observations: reward
            Actions: choice"""
        @test occursin("States: not specified", repr("text/plain", ModelWithFields(4, 0.5)))
        @test occursin("Observations: none", repr("text/plain", ModelWithoutObservations()))
        @test occursin("States: shared_state", repr("text/plain", ModelInFamily()))
    end

    @testset "Everywhere else a model is printed on one line, with its fields" begin
        @test string(ModelWithFields(4, 0.5)) == "ModelWithFields{Float64}(4, 0.5)"
        @test sprint(show, ModelWithAttributes()) == "ModelWithAttributes()"
    end
end
