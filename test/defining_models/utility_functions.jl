struct ModelWithAttributes <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithAttributes)
    return ModelAttributes(;
        parameters=(:learning_rate, :action_noise),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

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

#Returns a NamedTuple instead of a ModelAttributes
struct ReturnsNamedTupleOfNames <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ReturnsNamedTupleOfNames)
    return (; parameters=(:noise,), states=(), inputs=(), outputs=())
end

#Environment models with names, without them, and with names that aren't a ModelAttributes. The inputs of an
#environment are the actions it receives, and its outputs the observations it gives
struct EnvironmentWithAttributes <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(environment::EnvironmentWithAttributes)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end
struct EnvironmentWithFields <: AbstractEnvironmentModel
    n_arms::Int
end
struct EnvironmentReturnsNamedTupleOfNames <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(environment::EnvironmentReturnsNamedTupleOfNames)
    return (; parameters=(), states=(), inputs=(), outputs=(:reward,))
end

#Has more parameters than a display lists
struct ModelWithManyParameters <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithManyParameters)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f), states=:value, inputs=:reward, outputs=()
    )
end

#A model whose parameters have domains: one within bounds, one discrete with a lower bound, and one without a domain
struct ModelWithDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomains)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options, :bias), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDomains)
    return (; learning_rate=(; lower=0, upper=1), n_options=(; lower=1, discrete=true))
end

#Models whose domains don't fit them: a parameter that isn't in the model's attributes, and a domain with a field there isn't
struct ModelWithUnknownDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithUnknownDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
CognitiveModeling.parameter_domains(::ModelWithUnknownDomain) = (; noise=(; lower=0),)
struct ModelWithMisspelledDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithMisspelledDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithMisspelledDomain)
    return (; learning_rate=(; lowest=0),)
end

@testset "Utility functions" begin
    @testset "Parameters are checked against their domains, and a model has none by default" begin
        @test CognitiveModeling.parameter_domains(ModelWithAttributes()) === (;)
        @test CognitiveModeling.check_parameter_domains(
            ModelWithAttributes(), (; learning_rate=-5.0, noise=2.0)
        ) === nothing
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=0.5, n_options=3, bias=-10.0)
        ) === nothing
        #The bounds are part of the domain, and a whole number can be given as a float
        @test CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=1, n_options=1.0, bias=0.0)
        ) === nothing
        @test_throws InvalidParametersError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is 1.5, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=1.5, n_options=3, bias=0.0)
        )
        @test_throws InvalidParametersError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 0, which is outside its domain, from 1 to Inf.",
        ) CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=0.5, n_options=0, bias=0.0)
        )
        @test_throws InvalidParametersError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 2.5, but it is discrete, so it must be a whole number.",
        ) CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=0.5, n_options=2.5, bias=0.0)
        )
        #A value that isn't a number is outside every domain
        @test_throws InvalidParametersError CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=NaN, n_options=3, bias=0.0)
        )
    end

    @testset "Every element of an array is checked, parameters that aren't given aren't, and values that aren't real numbers are left alone" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=[0.2, 0.8], n_options=3)
        ) === nothing
        @test_throws "The parameter learning_rate of a cognitive model of type ModelWithDomains is 1.2" CognitiveModeling.check_parameter_domains(
            model, (; learning_rate=[0.2, 1.2], n_options=3)
        )
        @test CognitiveModeling.check_parameter_domains(model, (; bias=0.0)) === nothing
        @test CognitiveModeling.check_parameter_domains(
            model, (; learning_rate="high", n_options=:many)
        ) === nothing
    end

    @testset "The domains must be of parameters in the model's attributes, with only lower, upper and discrete" begin
        @test_throws "parameter_domains of a cognitive model of type ModelWithUnknownDomain has domains for (:noise,), but the parameters in its attributes are (:learning_rate,)." CognitiveModeling.check_parameter_domains(
            ModelWithUnknownDomain(), (; learning_rate=0.5)
        )
        @test_throws "The domain of learning_rate in parameter_domains of a cognitive model of type ModelWithMisspelledDomain is (lowest = 0,), but it must be a NamedTuple with any of lower, upper and discrete, such as (; lower = 0, upper = 1)." CognitiveModeling.check_parameter_domains(
            ModelWithMisspelledDomain(), (; learning_rate=0.5)
        )
    end

    @testset "The discrete parameters of a model are those whose domains say so" begin
        @test CognitiveModeling.discrete_parameter_names(ModelWithDomains()) ===
            (:n_options,)
        @test CognitiveModeling.discrete_parameter_names(ModelWithAttributes()) === ()
    end

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

    @testset "Names that aren't a ModelAttributes give an informative error" begin
        @test_throws "The attribute_names method of a cognitive model of type ReturnsNamedTupleOfNames must return a ModelAttributes" repr(
            "text/plain", ReturnsNamedTupleOfNames()
        )
        @test_throws "must return a ModelAttributes" CognitiveModeling.check_type_attribute_names(
            CognitiveModeling.attribute_names(ReturnsNamedTupleOfNames()),
            ReturnsNamedTupleOfNames(),
        )
        @test_throws "The attribute_names method of an environment model of type EnvironmentReturnsNamedTupleOfNames must return a ModelAttributes" repr(
            "text/plain", EnvironmentReturnsNamedTupleOfNames()
        )
        @test_throws "must return a ModelAttributes" CognitiveModeling.check_type_attribute_names(
            CognitiveModeling.attribute_names(EnvironmentReturnsNamedTupleOfNames()),
            EnvironmentReturnsNamedTupleOfNames(),
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
