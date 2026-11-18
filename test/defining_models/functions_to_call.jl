#A model with a continuous parameter and a discrete one, and a model without domains
struct ModelWithDiscreteParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDiscreteParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDiscreteParameter)
    return (;
        learning_rate=ParameterDomain(; lower=0, upper=1),
        n_options=ParameterDomain(; type=Integer, lower=1),
    )
end
#A model with parameters that aren't numbers, and with a parameter that is true or false
struct ModelWithNamedStrategy <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithNamedStrategy)
    return ModelAttributes(;
        parameters=(:strategy, :learning_rate, :label, :is_greedy),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithNamedStrategy)
    return (;
        strategy=ParameterDomain(; type=Symbol),
        learning_rate=ParameterDomain(),
        label=ParameterDomain(; type=AbstractString),
        is_greedy=ParameterDomain(; type=Bool),
    )
end
#A model whose parameters are arrays and tuples, of real numbers, of whole numbers and of true or false
struct ModelWithArrayParameters <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithArrayParameters)
    return ModelAttributes(;
        parameters=(:choice_probabilities, :counts, :flags),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithArrayParameters)
    return (;
        choice_probabilities=ParameterDomain(; type=AbstractVector{Real}, lower=0, upper=1),
        counts=ParameterDomain(; type=AbstractVector{Integer}, lower=0),
        flags=ParameterDomain(; type=NTuple{2,Bool}),
    )
end
#Models whose domains don't fit them: one for a parameter that isn't in the attributes, and domains in the form of NamedTuples
struct ModelWithDomainForUnknownParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainForUnknownParameter)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithDomainForUnknownParameter)
    return (; n_options=ParameterDomain(; type=Integer),)
end
struct ModelWithOldFormOfDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithOldFormOfDomains)
    return ModelAttributes(; parameters=:n_options, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithOldFormOfDomains)
    return (; n_options=(; lower=1, discrete=true),)
end
#A model whose only domain is that of its discrete parameter, between two parameters without one
struct ModelWithDomainForOneParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainForOneParameter)
    return ModelAttributes(;
        parameters=(:bias, :n_options, :learning_rate), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDomainForOneParameter)
    return (; n_options=ParameterDomain(; type=Integer, lower=1),)
end
struct ModelWithoutDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithoutDomains)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end

@testset "Functions that users call to read a model" begin
    @testset "The discrete parameters of a model are those whose domains have a type of whole numbers, or one that isn't of real numbers" begin
        @test CognitiveModeling.discrete_parameter_names(ModelWithDiscreteParameter()) ===
            (:n_options,)
        @test CognitiveModeling.discrete_parameter_names(ModelWithoutDomains()) === ()
        @test CognitiveModeling.discrete_parameter_names(ModelWithNamedStrategy()) ===
            (:strategy, :label, :is_greedy)
        @test CognitiveModeling.continuous_parameter_names(ModelWithNamedStrategy()) ===
            (:learning_rate,)
        #An array or a tuple is discrete when its elements are
        @test CognitiveModeling.discrete_parameter_names(ModelWithArrayParameters()) ===
            (:counts, :flags)
        @test CognitiveModeling.continuous_parameter_names(ModelWithArrayParameters()) ===
            (:choice_probabilities,)
    end

    @testset "Domains that don't fit the model are an error when the names are read" begin
        @test_throws "parameter_domains of a cognitive model of type ModelWithDomainForUnknownParameter has domains for (:n_options,), but the parameters in its attributes are (:learning_rate,)." CognitiveModeling.discrete_parameter_names(
            ModelWithDomainForUnknownParameter()
        )
        @test_throws "it must return a NamedTuple with a ParameterDomain under the name of each parameter that has a domain" CognitiveModeling.discrete_parameter_names(
            ModelWithOldFormOfDomains()
        )
        @test_throws ArgumentError CognitiveModeling.continuous_parameter_names(
            ModelWithOldFormOfDomains()
        )
    end

    @testset "The continuous parameters of a model are all the others, also those without a domain" begin
        @test CognitiveModeling.continuous_parameter_names(ModelWithDiscreteParameter()) ===
            (:learning_rate,)
        @test CognitiveModeling.continuous_parameter_names(ModelWithoutDomains()) ===
            (:learning_rate,)
        #A parameter without a domain is continuous, and they are in the order of the model's attributes
        @test CognitiveModeling.continuous_parameter_names(
            ModelWithDomainForOneParameter()
        ) === (:bias, :learning_rate)
        @test CognitiveModeling.discrete_parameter_names(
            ModelWithDomainForOneParameter()
        ) === (:n_options,)
    end
end
