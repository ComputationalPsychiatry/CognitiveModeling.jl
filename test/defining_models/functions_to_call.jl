#A model with a continuous parameter and a discrete one, and a model without domains
struct ModelWithDiscreteParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDiscreteParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDiscreteParameter)
    return (; learning_rate=(; lower=0, upper=1), n_options=(; lower=1, discrete=true))
end
#A model whose only domain is that of its discrete parameter, between two parameters without one
struct ModelWithDomainForOneParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainForOneParameter)
    return ModelAttributes(;
        parameters=(:bias, :n_options, :learning_rate), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDomainForOneParameter)
    return (; n_options=(; lower=1, discrete=true))
end
struct ModelWithoutDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithoutDomains)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end

@testset "Functions that users call to read a model" begin
    @testset "The discrete parameters of a model are those whose domains say so" begin
        @test CognitiveModeling.discrete_parameter_names(ModelWithDiscreteParameter()) ===
            (:n_options,)
        @test CognitiveModeling.discrete_parameter_names(ModelWithoutDomains()) === ()
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
