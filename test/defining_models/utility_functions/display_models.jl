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

#A model which is neither a cognitive model nor an environment model
struct NeitherKindOfModel <: CognitiveModeling.AbstractTimestepModel end

#A model with domains of every kind for its parameters, and one parameter without a domain
struct ModelWithDisplayedDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDisplayedDomains)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :bias, :n_options, :strategy),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDisplayedDomains)
    return (;
        learning_rate=ParameterDomain(; lower=0, upper=1),
        inverse_temperature=ParameterDomain(; lower=0),
        n_options=ParameterDomain(; type=Integer, lower=1),
        strategy=ParameterDomain(; type=Symbol),
    )
end

#A model with more parameters than a display lists, of which one has a domain
struct ModelWithManyParametersAndADomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithManyParametersAndADomain)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithManyParametersAndADomain)
    return (; b=ParameterDomain(; lower=0),)
end

#Models whose domains aren't right: not in a NamedTuple, and not ParameterDomains
struct ModelWithDomainsInAVector <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainsInAVector)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(model::ModelWithDomainsInAVector)
    return [ParameterDomain(; lower=0)]
end
struct ModelWithDomainsAsNamedTuples <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainsAsNamedTuples)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(model::ModelWithDomainsAsNamedTuples)
    return (; learning_rate=(; lower=0, upper=1),)
end

@testset "Displaying models, and describing them in messages" begin
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

    @testset "The REPL shows the domain of a parameter after its name" begin
        @test repr("text/plain", ModelWithDisplayedDomains()) == """
            -- Cognitive model --
            Type: ModelWithDisplayedDomains
            Parameters: learning_rate ∈ [0, 1], inverse_temperature ∈ [0, ∞), bias, n_options ∈ {1, 2, …}, strategy ∈ Symbol
            States: expected_reward
            Inputs: reward
            Outputs: choice"""
        #A ModelAttributes has only the names, and so has the model on one line
        @test occursin(
            "Parameters: learning_rate, inverse_temperature, bias, n_options, strategy",
            repr(
                "text/plain", CognitiveModeling.attribute_names(ModelWithDisplayedDomains())
            ),
        )
        @test string(ModelWithDisplayedDomains()) ==
            "ModelWithDisplayedDomains (cognitive model)"
        #With more parameters than are listed, there are no domains either, and without a limit there are
        @test occursin(
            "Parameters: 6 (see attribute_names)",
            repr("text/plain", ModelWithManyParametersAndADomain(); context=:limit => true),
        )
        @test occursin(
            "Parameters: a, b ∈ [0, ∞), c, d, e, f",
            repr("text/plain", ModelWithManyParametersAndADomain()),
        )
    end

    @testset "A domain is written as an interval, as whole numbers, or as a type" begin
        describe = CognitiveModeling.describe_domain
        @test describe(ParameterDomain()) == "(-∞, ∞)"
        @test describe(ParameterDomain(; lower=0, upper=1)) == "[0, 1]"
        @test describe(ParameterDomain(; lower=0.0, upper=0.5)) == "[0.0, 0.5]"
        @test describe(ParameterDomain(; lower=0)) == "[0, ∞)"
        @test describe(ParameterDomain(; upper=0.5)) == "(-∞, 0.5]"
        @test describe(ParameterDomain(; type=Integer)) == "{…, -1, 0, 1, …}"
        @test describe(ParameterDomain(; type=Integer, lower=1)) == "{1, 2, …}"
        @test describe(ParameterDomain(; type=Integer, upper=4)) == "{…, 3, 4}"
        @test describe(ParameterDomain(; type=Integer, lower=0, upper=5)) == "{0, 1, …, 5}"
        @test describe(ParameterDomain(; type=Int, lower=1, upper=3)) == "{1, 2, 3}"
        @test describe(ParameterDomain(; type=Integer, lower=2, upper=2)) == "{2}"
        #Bounds that aren't whole numbers are rounded to the whole numbers within them
        @test describe(ParameterDomain(; type=Integer, lower=0.5, upper=9.5)) ==
            "{1, 2, …, 9}"
        @test describe(ParameterDomain(; type=Bool)) == "{false, true}"
        @test describe(ParameterDomain(; type=Symbol)) == "Symbol"
        @test describe(ParameterDomain(; type=AbstractString)) == "AbstractString"
    end

    @testset "A model whose domains aren't right is shown without them, and without an error" begin
        @test occursin(
            "Parameters: learning_rate\n", repr("text/plain", ModelWithDomainsInAVector())
        )
        @test occursin(
            "Parameters: learning_rate\n",
            repr("text/plain", ModelWithDomainsAsNamedTuples()),
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

    @testset "In a message a model is named by its kind and its type" begin
        @test CognitiveModeling.describe_model(ModelWithAttributes()) ==
            "a cognitive model of type ModelWithAttributes"
        @test CognitiveModeling.describe_model(EnvironmentWithAttributes()) ==
            "an environment model of type EnvironmentWithAttributes"
        @test CognitiveModeling.describe_model(NeitherKindOfModel()) ==
            "a model of type NeitherKindOfModel"
    end
end
