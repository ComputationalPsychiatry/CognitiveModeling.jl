#A model with a continuous parameter and a discrete one, and a model without domains
struct ModelWithDiscreteParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDiscreteParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDiscreteParameter)
    return (;
        learning_rate=Domain(; lower=0, upper=1), n_options=Domain(; type=Integer, lower=1)
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
        strategy=Domain(; type=Symbol),
        learning_rate=Domain(),
        label=Domain(; type=AbstractString),
        is_greedy=Domain(; type=Bool),
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
        choice_probabilities=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        counts=Domain(; type=AbstractVector{Integer}, lower=0),
        flags=Domain(; type=NTuple{2,Bool}),
    )
end
#A model whose parameters have possible values, names and real numbers, or a constraint
struct ModelWithPossibleValuesAndConstraint <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithPossibleValuesAndConstraint)
    return ModelAttributes(;
        parameters=(:strategy, :step_size, :thresholds), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithPossibleValuesAndConstraint)
    return (;
        strategy=Domain(; possible_values=(:greedy, :random)),
        step_size=Domain(; possible_values=(0.5, 1.0)),
        thresholds=Domain(; type=AbstractVector{Real}, constraint=issorted),
    )
end
#Models whose domains don't fit them: one for a parameter that isn't in the attributes, and domains in the form of NamedTuples
struct ModelWithDomainForUnknownParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainForUnknownParameter)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithDomainForUnknownParameter)
    return (; n_options=Domain(; type=Integer),)
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
    return (; n_options=Domain(; type=Integer, lower=1),)
end
struct ModelWithoutDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithoutDomains)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end

#A model with a default for one of its two parameters
struct ModelWithDefaultParameter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDefaultParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(model::ModelWithDefaultParameter)
    return (; inverse_temperature=1.0)
end

#A perception with a default for one of its parameters and a domain for the other, and a response with a discrete
#parameter that has a default, which are only ever submodels
struct PerceptionWithDefault <: AbstractSubmodel end
function CognitiveModeling.attribute_names(perception::PerceptionWithDefault)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(perception::PerceptionWithDefault)
    return (; initial_value=0.5)
end
function CognitiveModeling.parameter_domains(perception::PerceptionWithDefault)
    return (; learning_rate=Domain(; lower=0, upper=1))
end
struct ResponseWithDiscreteDefault <: AbstractSubmodel end
function CognitiveModeling.attribute_names(response::ResponseWithDiscreteDefault)
    return ModelAttributes(;
        parameters=(:inverse_temperature, :n_options), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(response::ResponseWithDiscreteDefault)
    return (; n_options=2)
end
function CognitiveModeling.parameter_domains(response::ResponseWithDiscreteDefault)
    return (; n_options=Domain(; type=Integer, lower=1))
end

#A cognitive model with the two as submodels, and two parameters of its own: one with a default, and a discrete one.
#Whether the flat names of the submodels' parameters have prefixes is chosen when it is constructed
struct ModelWithSubmodelsToRead{prefixes} <: AbstractCognitiveModel
    perception::PerceptionWithDefault
    response::ResponseWithDiscreteDefault
end
function ModelWithSubmodelsToRead(; prefixes=false)
    return ModelWithSubmodelsToRead{prefixes}(
        PerceptionWithDefault(), ResponseWithDiscreteDefault()
    )
end
function CognitiveModeling.submodel_fields(model::ModelWithSubmodelsToRead)
    return (:perception, :response)
end
function CognitiveModeling.uses_submodel_prefixes(
    model::ModelWithSubmodelsToRead{prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(model::ModelWithSubmodelsToRead)
    return ModelAttributes(;
        parameters=(:lapse_rate, :is_greedy), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(model::ModelWithSubmodelsToRead)
    return (; lapse_rate=0.0)
end
function CognitiveModeling.parameter_domains(model::ModelWithSubmodelsToRead)
    return (; is_greedy=Domain(; type=Bool))
end

#A model with the same perception twice and no prefixes, so that two parameters have the same flat name
struct ModelWithTwoPerceptions <: AbstractCognitiveModel
    first_perception::PerceptionWithDefault
    second_perception::PerceptionWithDefault
end
function CognitiveModeling.submodel_fields(model::ModelWithTwoPerceptions)
    return (:first_perception, :second_perception)
end
function CognitiveModeling.attribute_names(model::ModelWithTwoPerceptions)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=())
end

@testset "Functions that users call to read a model" begin
    @testset "The parameters of a model, and those without a default, which a fit needs priors for" begin
        @test CognitiveModeling.get_parameter_names(ModelWithDefaultParameter()) ===
            (:learning_rate, :inverse_temperature)
        @test CognitiveModeling.get_parameter_names(
            ModelWithDefaultParameter(); with_default_parameters=true
        ) === (:learning_rate, :inverse_temperature)
        @test CognitiveModeling.get_parameter_names(
            ModelWithDefaultParameter(); with_default_parameters=false
        ) === (:learning_rate,)
        #A model without defaults has every parameter both ways, in the order of its attributes
        @test CognitiveModeling.get_parameter_names(ModelWithDiscreteParameter()) ===
            (:learning_rate, :n_options)
        @test CognitiveModeling.get_parameter_names(
            ModelWithDiscreteParameter(); with_default_parameters=false
        ) === (:learning_rate, :n_options)
    end

    @testset "The names of the attributes of a model leave out the parameters with a default when asked to, and keep the other names" begin
        model = ModelWithSubmodelsToRead(; prefixes=true)
        every_name = CognitiveModeling.get_attribute_names(model)
        @test every_name.parameters === (
            :lapse_rate,
            :is_greedy,
            :perception__learning_rate,
            :perception__initial_value,
            :response__inverse_temperature,
            :response__n_options,
        )
        @test CognitiveModeling.get_attribute_names(model; with_default_parameters=true) ===
            every_name
        #The model's own default and those of its submodels are left out, under their flat names
        without_defaults = CognitiveModeling.get_attribute_names(
            model; with_default_parameters=false
        )
        @test without_defaults.parameters ===
            (:is_greedy, :perception__learning_rate, :response__inverse_temperature)
        @test without_defaults.states === every_name.states &&
            without_defaults.inputs === every_name.inputs &&
            without_defaults.outputs === every_name.outputs
        #Without prefixes too, and a model without defaults has the same names both ways
        @test CognitiveModeling.get_attribute_names(
            ModelWithSubmodelsToRead(); with_default_parameters=false
        ).parameters === (:is_greedy, :learning_rate, :inverse_temperature)
        @test CognitiveModeling.get_attribute_names(
            ModelWithDiscreteParameter(); with_default_parameters=false
        ) === CognitiveModeling.get_attribute_names(ModelWithDiscreteParameter())
        #The names are in the types, so both are inferred
        @test @inferred(CognitiveModeling.get_attribute_names(model)) === every_name
        @test @inferred(
            CognitiveModeling.without_default_parameters(
                every_name, CognitiveModeling.get_default_parameters(model)
            )
        ) === without_defaults
    end

    @testset "The default values and the domains of the parameters of a model are read together with those of its submodels, under the flat names" begin
        model = ModelWithSubmodelsToRead()
        @test CognitiveModeling.get_default_parameters(model) ===
            (; lapse_rate=0.0, initial_value=0.5, n_options=2)
        @test CognitiveModeling.get_parameter_domains(model) === (;
            is_greedy=Domain(; type=Bool),
            learning_rate=Domain(; lower=0, upper=1),
            n_options=Domain(; type=Integer, lower=1),
        )
        #The model's own are what it implements, without those of its submodels
        @test CognitiveModeling.default_parameters(model) === (; lapse_rate=0.0)
        @test keys(CognitiveModeling.parameter_domains(model)) === (:is_greedy,)
        #With prefixes, those of the submodels are under their flat names
        prefixed = ModelWithSubmodelsToRead(; prefixes=true)
        @test CognitiveModeling.get_default_parameters(prefixed) ===
            (; lapse_rate=0.0, perception__initial_value=0.5, response__n_options=2)
        @test keys(CognitiveModeling.get_parameter_domains(prefixed)) ===
            (:is_greedy, :perception__learning_rate, :response__n_options)
        #A submodel is read on its own, and a model without defaults or domains has none
        @test CognitiveModeling.get_default_parameters(PerceptionWithDefault()) ===
            (; initial_value=0.5)
        @test keys(CognitiveModeling.get_parameter_domains(PerceptionWithDefault())) ===
            (:learning_rate,)
        @test CognitiveModeling.get_default_parameters(ModelWithoutDomains()) === (;)
        @test CognitiveModeling.get_parameter_domains(ModelWithoutDomains()) === (;)
        #Both are inferred
        @test @inferred(CognitiveModeling.get_default_parameters(prefixed)) isa NamedTuple
        @test @inferred(CognitiveModeling.get_parameter_domains(prefixed)) isa NamedTuple
    end

    @testset "The default values and the domains are read after the submodels are checked, so that two parameters with the same flat name are an error" begin
        twice = ModelWithTwoPerceptions(PerceptionWithDefault(), PerceptionWithDefault())
        @test_throws "has two parameters with the flat name :learning_rate: first_perception.learning_rate and second_perception.learning_rate." CognitiveModeling.get_default_parameters(
            twice
        )
        @test_throws "has two parameters with the flat name :learning_rate" CognitiveModeling.get_parameter_domains(
            twice
        )
        @test_throws "has two parameters with the flat name :learning_rate" CognitiveModeling.discrete_parameter_names(
            twice
        )
    end

    @testset "The discrete and the continuous parameters of a model are read together with those of its submodels, with or without those that have a default" begin
        model = ModelWithSubmodelsToRead(; prefixes=true)
        @test CognitiveModeling.discrete_parameter_names(model) ===
            (:is_greedy, :response__n_options)
        @test CognitiveModeling.continuous_parameter_names(model) === (
            :lapse_rate,
            :perception__learning_rate,
            :perception__initial_value,
            :response__inverse_temperature,
        )
        @test CognitiveModeling.discrete_parameter_names(
            model; with_default_parameters=false
        ) === (:is_greedy,)
        @test CognitiveModeling.continuous_parameter_names(
            model; with_default_parameters=false
        ) === (:perception__learning_rate, :response__inverse_temperature)
        #Without prefixes
        @test CognitiveModeling.discrete_parameter_names(ModelWithSubmodelsToRead()) ===
            (:is_greedy, :n_options)
        #A submodel is read on its own
        @test CognitiveModeling.discrete_parameter_names(ResponseWithDiscreteDefault()) ===
            (:n_options,)
        @test CognitiveModeling.continuous_parameter_names(
            ResponseWithDiscreteDefault()
        ) === (:inverse_temperature,)
        @test CognitiveModeling.continuous_parameter_names(
            PerceptionWithDefault(); with_default_parameters=false
        ) === (:learning_rate,)
    end

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
        #A parameter with possible values is discrete, also when they are real numbers, and a constraint doesn't change it
        @test CognitiveModeling.discrete_parameter_names(
            ModelWithPossibleValuesAndConstraint()
        ) === (:strategy, :step_size)
        @test CognitiveModeling.continuous_parameter_names(
            ModelWithPossibleValuesAndConstraint()
        ) === (:thresholds,)
    end

    @testset "Domains that don't fit the model are an error when the names are read" begin
        @test_throws "parameter_domains of a cognitive model of type ModelWithDomainForUnknownParameter has domains for (:n_options,), but the parameters in its attributes are (:learning_rate,)." CognitiveModeling.discrete_parameter_names(
            ModelWithDomainForUnknownParameter()
        )
        @test_throws "it must return a NamedTuple with a Domain under the name of each parameter that has a domain" CognitiveModeling.discrete_parameter_names(
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
