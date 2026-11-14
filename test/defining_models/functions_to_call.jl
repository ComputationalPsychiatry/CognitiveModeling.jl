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
end
