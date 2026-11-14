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

@testset "Checking parameters against their domains" begin
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
end
