#A model whose parameters have domains: one within bounds, one of whole numbers with a lower bound, one that is a name, one that is true or false, and one without a domain
struct ModelWithDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomains)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options, :strategy, :is_greedy, :bias),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDomains)
    return (;
        learning_rate=ParameterDomain(; lower=0, upper=1),
        n_options=ParameterDomain(; type=Integer, lower=1),
        strategy=ParameterDomain(; type=Symbol),
        is_greedy=ParameterDomain(; type=Bool),
    )
end

#Models whose domains don't fit them: a parameter that isn't in the model's attributes, a domain that isn't a ParameterDomain, and domains that aren't in a NamedTuple
struct ModelWithUnknownDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithUnknownDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithUnknownDomain)
    return (; noise=ParameterDomain(; lower=0),)
end
struct ModelWithNamedTupleDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithNamedTupleDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithNamedTupleDomain)
    return (; learning_rate=(; lower=0, upper=1),)
end
struct ModelWithUnnamedDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithUnnamedDomains)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithUnnamedDomains)
    return [ParameterDomain(; lower=0, upper=1)]
end

#A number that isn't a real number, as the numbers that Reactant.jl traces aren't, which can't be compared with anything
struct NumberWithoutValue <: Number end

#Checks the same parameters many times, to measure what a check costs when a model runs
function check_many_times(model, parameters, n_times)
    for _ in 1:n_times
        CognitiveModeling.check_parameters_in_domains(model, parameters)
    end
    return nothing
end

@testset "Checking parameters against their domains" begin
    #Parameters of ModelWithDomains which are all within their domains
    valid_parameters = (;
        learning_rate=0.5, n_options=3, strategy=:greedy, is_greedy=true, bias=-10.0
    )

    @testset "A model has no domains by default, and its parameters can then take any value" begin
        @test CognitiveModeling.parameter_domains(ModelWithAttributes()) === (;)
        @test CognitiveModeling.check_parameters_in_domains(
            ModelWithAttributes(), (; learning_rate=-5.0, action_noise="loud")
        ) === nothing
    end

    @testset "Parameters within their domains pass, with the bounds included" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(model, valid_parameters) ===
            nothing
        @test CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=0))
        ) === nothing
        @test CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=1, n_options=1))
        ) === nothing
    end

    @testset "A domain of real numbers takes every type of them" begin
        model = ModelWithDomains()
        for learning_rate in (0.5f0, 0.5, big"0.5", 1//2, true, Float16(0.5))
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; learning_rate))
            ) === nothing
        end
    end

    @testset "A domain of whole numbers takes them in any number type, and one of Bool takes 0 and 1" begin
        model = ModelWithDomains()
        for n_options in (3, 3.0, 3.0f0, Int8(3), big(3), 6//2, true)
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; n_options))
            ) === nothing
        end
        for is_greedy in (true, false, 0, 1, 0.0, 1.0)
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; is_greedy))
            ) === nothing
        end
    end

    @testset "A value outside the bounds is a ParameterDomainError, and so is a NaN" begin
        model = ModelWithDomains()
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is 1.5, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=1.5))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is -0.1, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=-0.1))
        )
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 0, which is outside its domain, from 1 to Inf.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options=0))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is NaN, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=NaN))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=Inf))
        )
        #The error isn't the one a model throws for parameters to be sampled again
        @test !(ParameterDomainError("") isa InvalidParametersError)
    end

    @testset "A value of another type than the domain's is a ParameterDomainError" begin
        model = ModelWithDomains()
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 2.5, of type Float64, but its domain takes whole numbers, as its type is Integer.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options=2.5))
        )
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is \"three\", of type String, but its domain takes values of type Integer.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options="three"))
        )
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithDomains is \"greedy\", of type String, but its domain takes values of type Symbol.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy="greedy"))
        )
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithDomains is 1, of type Int64, but its domain takes values of type Symbol.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy=1))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is :high, of type Symbol, but its domain takes values of type Real.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=:high))
        )
        @test_throws ParameterDomainError(
            "The parameter is_greedy of a cognitive model of type ModelWithDomains is 2, of type Int64, but its domain takes true and false, or 0 and 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; is_greedy=2))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; is_greedy=0.5))
        )
    end

    @testset "Every element of an array is checked" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            merge(
                valid_parameters,
                (;
                    learning_rate=[0.2, 0.8],
                    n_options=[2 3; 4 5],
                    strategy=[:greedy, :random],
                ),
            ),
        ) === nothing
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is 1.2, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=[0.2, 1.2]))
        )
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithDomains is \"random\", of type String, but its domain takes values of type Symbol.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy=Any[:greedy, "random"]))
        )
        #The elements of an array of arrays are checked too
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=[[0.2], [0.8, 1.2]]))
        )
    end

    @testset "Numbers that aren't real numbers are left alone, also in arrays" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            merge(
                valid_parameters,
                (; learning_rate=NumberWithoutValue(), n_options=2.5 + 0.0im),
            ),
        ) === nothing
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            merge(
                valid_parameters,
                (; learning_rate=[NumberWithoutValue()], n_options=[2.5 + 0.0im]),
            ),
        ) === nothing
    end

    @testset "The order of the parameters doesn't matter, and those without a domain can be anything" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            (;
                bias="any value",
                is_greedy=false,
                strategy=:random,
                n_options=2,
                learning_rate=0.1,
            ),
        ) === nothing
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model,
            (;
                bias="any value",
                is_greedy=false,
                strategy=:random,
                n_options=0,
                learning_rate=0.1,
            ),
        )
    end

    @testset "The domains must be ParameterDomains under the names of parameters in the model's attributes" begin
        @test_throws ArgumentError(
            "parameter_domains of a cognitive model of type ModelWithUnknownDomain has domains for (:noise,), but the parameters in its attributes are (:learning_rate,).",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithUnknownDomain(), (; learning_rate=0.5)
        )
        @test_throws ArgumentError(
            "parameter_domains of a cognitive model of type ModelWithNamedTupleDomain returns (learning_rate = (lower = 0, upper = 1),), but it must return a NamedTuple with a ParameterDomain under the name of each parameter that has a domain, such as (; learning_rate = ParameterDomain(; lower = 0, upper = 1)).",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithNamedTupleDomain(), (; learning_rate=0.5)
        )
        @test_throws "parameter_domains of a cognitive model of type ModelWithUnnamedDomains returns" CognitiveModeling.check_parameters_in_domains(
            ModelWithUnnamedDomains(), (; learning_rate=0.5)
        )
    end

    @testset "The check is inferred and doesn't allocate, with and without domains" begin
        model = ModelWithDomains()
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(model, valid_parameters)
        ) === nothing
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(
                ModelWithAttributes(), (; learning_rate=0.5, action_noise=1.0)
            )
        ) === nothing
        #Run once first, so that compiling isn't measured
        check_many_times(model, valid_parameters, 1)
        @test @allocated(check_many_times(model, valid_parameters, 100)) == 0
        vector_parameters = merge(valid_parameters, (; learning_rate=[0.2, 0.8]))
        check_many_times(model, vector_parameters, 1)
        @test @allocated(check_many_times(model, vector_parameters, 100)) == 0
        without_domains = (; learning_rate=0.5, action_noise=1.0)
        check_many_times(ModelWithAttributes(), without_domains, 1)
        @test @allocated(check_many_times(ModelWithAttributes(), without_domains, 100)) == 0
    end
end
