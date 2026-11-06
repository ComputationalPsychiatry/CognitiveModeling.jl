struct ModelWithAttributes <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithAttributes)
    return ModelAttributes(; parameters = (:learning_rate, :action_noise), states = :expected_reward, observations = :reward, actions = :choice)
end

struct ModelWithFields{T} <: AbstractCognitiveModel
    n_options::Int
    prior::T
end

#Says that it has no observations, which is different from not saying anything
struct ModelWithoutObservations <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithoutObservations)
    return ModelAttributes(; parameters = :noise, states = (), observations = (), actions = :choice)
end

#Gets its names from its supertype
abstract type FamilyOfModels <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::FamilyOfModels)
    return ModelAttributes(; parameters = (), states = :shared_state, observations = (), actions = ())
end
struct ModelInFamily <: FamilyOfModels end

#Returns a NamedTuple instead of a ModelAttributes
struct ReturnsNamedTupleOfNames <: AbstractCognitiveModel end
CognitiveModeling.attribute_names(model::ReturnsNamedTupleOfNames) = (; parameters = (:noise,), states = (), observations = (), actions = ())

#Environment models with names, without them, and with names that aren't a ModelAttributes
struct EnvironmentWithAttributes <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(environment::EnvironmentWithAttributes)
    return ModelAttributes(; parameters = :reward_probability, states = (), observations = :reward, actions = :choice)
end
struct EnvironmentWithFields <: AbstractEnvironmentModel
    n_arms::Int
end
struct EnvironmentReturnsNamedTupleOfNames <: AbstractEnvironmentModel end
CognitiveModeling.attribute_names(environment::EnvironmentReturnsNamedTupleOfNames) = (; parameters = (), states = (), observations = (:reward,), actions = ())

@testset "Utility functions" begin
    @testset "The REPL shows a summary of the model with its names" begin
        @test repr("text/plain", ModelWithAttributes()) == """
            -- Cognitive model --
            Type: ModelWithAttributes
            Parameters: learning_rate, action_noise
            States: expected_reward
            Observations: reward
            Actions: choice"""
        @test occursin("Attributes: not specified", repr("text/plain", ModelWithFields(4, 0.5)))
        @test occursin("Observations: none", repr("text/plain", ModelWithoutObservations()))
        @test occursin("States: shared_state", repr("text/plain", ModelInFamily()))
    end

    @testset "The REPL shows a summary of an environment model with its names" begin
        @test repr("text/plain", EnvironmentWithAttributes()) == """
            -- Environment model --
            Type: EnvironmentWithAttributes
            Parameters: reward_probability
            States: none
            Observations: reward
            Actions: choice"""
        @test occursin("Attributes: not specified", repr("text/plain", EnvironmentWithFields(2)))
        @test string(EnvironmentWithFields(2)) == "EnvironmentWithFields(2)"
    end

    @testset "A vector of models without names is shown without an error" begin
        @test occursin("ModelWithFields{Float64}(4, 0.5)", repr("text/plain", [ModelWithFields(4, 0.5)]))
        @test occursin("EnvironmentWithFields(2)", repr("text/plain", [EnvironmentWithFields(2)]))
    end

    @testset "Names that aren't a ModelAttributes give an informative error" begin
        @test_throws "The attribute_names method of a cognitive model of type ReturnsNamedTupleOfNames must return a ModelAttributes" repr("text/plain", ReturnsNamedTupleOfNames())
        @test_throws "must return a ModelAttributes" CognitiveModeling.check_type_attribute_names(CognitiveModeling.attribute_names(ReturnsNamedTupleOfNames()), ReturnsNamedTupleOfNames())
        @test_throws "The attribute_names method of an environment model of type EnvironmentReturnsNamedTupleOfNames must return a ModelAttributes" repr("text/plain", EnvironmentReturnsNamedTupleOfNames())
        @test_throws "must return a ModelAttributes" CognitiveModeling.check_type_attribute_names(CognitiveModeling.attribute_names(EnvironmentReturnsNamedTupleOfNames()), EnvironmentReturnsNamedTupleOfNames())
    end

    @testset "The REPL shows the names in a ModelAttributes on one line each" begin
        attributes = ModelAttributes(; parameters = (:learning_rate, :noise), states = :expected_reward, observations = (), actions = :choice)
        @test repr("text/plain", attributes) == """
            ModelAttributes
            Parameters: learning_rate, noise
            States: expected_reward
            Observations: none
            Actions: choice"""
    end

    @testset "Everywhere else a model is printed on one line, with its fields" begin
        @test string(ModelWithFields(4, 0.5)) == "ModelWithFields{Float64}(4, 0.5)"
        @test sprint(show, ModelWithAttributes()) == "ModelWithAttributes()"
    end
end
