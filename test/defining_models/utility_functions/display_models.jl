struct ModelWithFields{T} <: AbstractCognitiveModel
    n_options::Int
    prior::T
end

#Says that it has no observations, which is different from not saying anything
struct ModelWithoutObservations <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithoutObservations)
    return ModelAttributes(; parameters=:noise, states=(), inputs=(), outputs=:choice)
end

#Gets its names from its supertype
abstract type FamilyOfModels <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::FamilyOfModels)
    return ModelAttributes(; parameters=(), states=:shared_state, inputs=(), outputs=())
end
struct ModelInFamily <: FamilyOfModels end

#An environment model without names
struct EnvironmentWithFields <: AbstractEnvironmentModel
    n_arms::Int
end

#Has more parameters than a display lists
struct ModelWithManyParameters <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithManyParameters)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f), states=:value, inputs=:reward, outputs=()
    )
end

@testset "Displaying models" begin
    @testset "The REPL shows a summary of the model with its names" begin
        @test repr("text/plain", ModelWithAttributes()) == """
            -- Cognitive model --
            Type: ModelWithAttributes
            Parameters: learning_rate, action_noise
            States: expected_reward
            Inputs: reward
            Outputs: choice"""
        @test occursin(
            "Attributes: not specified", repr("text/plain", ModelWithFields(4, 0.5))
        )
        @test occursin("Inputs: none", repr("text/plain", ModelWithoutObservations()))
        @test occursin("States: shared_state", repr("text/plain", ModelInFamily()))
    end

    @testset "The REPL shows the values of a model's fields, if it has any" begin
        @test repr("text/plain", ModelWithFields(4, 0.5)) == """
            -- Cognitive model --
            Type: ModelWithFields
            Fields: n_options = 4, prior = 0.5
            Attributes: not specified"""
        @test occursin(
            "\nFields: n_arms = 2\n", repr("text/plain", EnvironmentWithFields(2))
        )
        @test !occursin("Fields", repr("text/plain", ModelWithAttributes()))
    end

    @testset "The REPL shows a summary of an environment model with its names" begin
        @test repr("text/plain", EnvironmentWithAttributes()) == """
            -- Environment model --
            Type: EnvironmentWithAttributes
            Parameters: reward_probability
            States: none
            Inputs: choice
            Outputs: reward"""
        @test occursin(
            "Attributes: not specified", repr("text/plain", EnvironmentWithFields(2))
        )
        @test string(EnvironmentWithFields(2)) ==
            "EnvironmentWithFields (environment model)"
    end

    @testset "The REPL shows only how many names of a kind there are when there are more than five" begin
        shown = repr("text/plain", ModelWithManyParameters(); context=:limit => true)
        @test occursin("Parameters: 6 (see attribute_names)", shown)
        @test occursin("States: value", shown)
        #Without a limit, as when printing to a file, every name is listed
        @test occursin(
            "Parameters: a, b, c, d, e, f", repr("text/plain", ModelWithManyParameters())
        )
        #The ModelAttributes lists every name, since that is where the full list is read
        @test occursin(
            "Parameters: a, b, c, d, e, f",
            repr(
                "text/plain",
                CognitiveModeling.attribute_names(ModelWithManyParameters());
                context=:limit => true,
            ),
        )
    end

    @testset "A vector of models without names is shown without an error" begin
        @test occursin(
            "ModelWithFields (cognitive model)",
            repr("text/plain", [ModelWithFields(4, 0.5)]),
        )
        @test occursin(
            "EnvironmentWithFields (environment model)",
            repr("text/plain", [EnvironmentWithFields(2)]),
        )
    end

    @testset "The REPL shows the names in a ModelAttributes on one line each" begin
        attributes = ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=:expected_reward,
            inputs=(),
            outputs=:choice,
        )
        @test repr("text/plain", attributes) == """
            ModelAttributes
            Parameters: learning_rate, noise
            States: expected_reward
            Inputs: none
            Outputs: choice"""
    end

    @testset "Everywhere else a model is printed on one line, with its type and its kind" begin
        @test string(ModelWithFields(4, 0.5)) == "ModelWithFields (cognitive model)"
        @test sprint(show, ModelWithAttributes()) == "ModelWithAttributes (cognitive model)"
        @test sprint(show, EnvironmentWithAttributes()) ==
            "EnvironmentWithAttributes (environment model)"
    end
end
