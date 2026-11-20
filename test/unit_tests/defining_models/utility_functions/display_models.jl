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
        learning_rate=Domain(; lower=0, upper=1),
        inverse_temperature=Domain(; lower=0),
        n_options=Domain(; type=Integer, lower=1),
        strategy=Domain(; type=Symbol),
    )
end
function CognitiveModeling.default_parameters(model::ModelWithDisplayedDomains)
    return (; inverse_temperature=2.0, bias=0.5, strategy=:greedy)
end

#A model with default values for some of its parameters, and no domains
struct ModelWithDisplayedDefaults <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDisplayedDefaults)
    return ModelAttributes(;
        parameters=(:learning_rate, :weights), states=(), inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(model::ModelWithDisplayedDefaults)
    return (; weights=collect(1.0:100.0))
end

#A model with more parameters than a display lists one on a line, of which one has a domain
struct ModelWithManyParametersAndADomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithManyParametersAndADomain)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithManyParametersAndADomain)
    return (; b=Domain(; lower=0),)
end

#A model with domains for an input and an output, beside an input without one, and one whose input domains aren't in a NamedTuple
struct ModelWithInputAndOutputDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithInputAndOutputDomains)
    return ModelAttributes(;
        parameters=:learning_rate, states=(), inputs=(:reward, :cue), outputs=:choice
    )
end
function CognitiveModeling.input_domains(model::ModelWithInputAndOutputDomains)
    return (; reward=Domain(; lower=0, upper=1))
end
function CognitiveModeling.output_domains(model::ModelWithInputAndOutputDomains)
    return (; choice=Domain(; possible_values=(1, 2)))
end
struct ModelWithInputDomainsInAVector <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithInputDomainsInAVector)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=())
end
function CognitiveModeling.input_domains(model::ModelWithInputDomainsInAVector)
    return [Domain(; lower=0)]
end

#Models whose domains aren't right: not in a NamedTuple, and not Domains
struct ModelWithDomainsInAVector <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomainsInAVector)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(model::ModelWithDomainsInAVector)
    return [Domain(; lower=0)]
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

    @testset "The REPL lists the parameters one on a line when any has a domain or a default value" begin
        @test repr("text/plain", ModelWithDisplayedDomains()) == """
            -- Cognitive model --
            Type: ModelWithDisplayedDomains
            Parameters:
              learning_rate ∈ [0, 1]
              inverse_temperature ∈ [0, ∞) (default 2.0)
              bias (default 0.5)
              n_options ∈ {1, 2, …}
              strategy ∈ Symbol (default :greedy)
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
        #Default values alone give the same list, with a large array shortened in the REPL
        @test repr("text/plain", ModelWithDisplayedDefaults(); context=:limit => true) ==
            """
-- Cognitive model --
Type: ModelWithDisplayedDefaults
Parameters:
  learning_rate
  weights (default [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  …  91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
States: none
Inputs: none
Outputs: none"""
    end

    @testset "The REPL lists the inputs or the outputs one on a line when any has a domain" begin
        @test repr("text/plain", ModelWithInputAndOutputDomains()) == """
            -- Cognitive model --
            Type: ModelWithInputAndOutputDomains
            Parameters: learning_rate
            States: none
            Inputs:
              reward ∈ [0, 1]
              cue
            Outputs:
              choice ∈ {1, 2}"""
        #Domains that aren't in a NamedTuple are left out
        @test occursin(
            "Inputs: reward", repr("text/plain", ModelWithInputDomainsInAVector())
        )
        @test CognitiveModeling.input_domains(ModelWithFields(3, 1.0)) === (;) &&
            CognitiveModeling.output_domains(ModelWithFields(3, 1.0)) === (;)
    end

    @testset "The REPL lists at most ten parameters one on a line, and says how many more there are" begin
        shown = repr(
            "text/plain", ModelWithManyParametersAndADomain(); context=:limit => true
        )
        @test occursin("\nParameters:\n  a\n  b ∈ [0, ∞)\n  c\n", shown)
        @test occursin("\n  j\n  … and 2 more (see attribute_names)\nStates: none", shown)
        @test !occursin("\n  k\n", shown)
        twelve_names =
            CognitiveModeling.attribute_names(ModelWithManyParametersAndADomain()).parameters
        limited_io = IOContext(IOBuffer(), :limit => true)
        @test CognitiveModeling.n_attribute_lines(limited_io, twelve_names) == 10
        @test CognitiveModeling.n_attribute_lines(limited_io, (:a, :b, :c)) == 3
        #Without a limit, as when printing to a file, every parameter is listed
        unlimited = repr("text/plain", ModelWithManyParametersAndADomain())
        @test occursin("\n  k\n  l\nStates: none", unlimited)
        @test !occursin("more", unlimited)
        @test CognitiveModeling.n_attribute_lines(IOBuffer(), twelve_names) == 12
    end

    @testset "A parameter has more to show than its name when it has a domain or a default value" begin
        domains = (; learning_rate=Domain(; lower=0), noise=(; lower=0))
        defaults = (; bias=0.5)
        @test CognitiveModeling.has_domain_or_default(:learning_rate, domains, defaults)
        @test CognitiveModeling.has_domain_or_default(:bias, domains, defaults)
        @test !CognitiveModeling.has_domain_or_default(:temperature, domains, defaults)
        #A domain that isn't a Domain isn't shown
        @test !CognitiveModeling.has_domain_or_default(:noise, domains, defaults)
    end

    @testset "A domain is written as an interval, as whole numbers, or as a type" begin
        describe = CognitiveModeling.describe_domain
        @test describe(Domain()) == "(-∞, ∞)"
        @test describe(Domain(; lower=0, upper=1)) == "[0, 1]"
        @test describe(Domain(; lower=0.0, upper=0.5)) == "[0.0, 0.5]"
        @test describe(Domain(; lower=0)) == "[0, ∞)"
        @test describe(Domain(; upper=0.5)) == "(-∞, 0.5]"
        @test describe(Domain(; type=Integer)) == "{…, -1, 0, 1, …}"
        @test describe(Domain(; type=Integer, lower=1)) == "{1, 2, …}"
        @test describe(Domain(; type=Integer, upper=4)) == "{…, 3, 4}"
        @test describe(Domain(; type=Integer, lower=0, upper=5)) == "{0, 1, …, 5}"
        @test describe(Domain(; type=Int, lower=1, upper=3)) == "{1, 2, 3}"
        @test describe(Domain(; type=Integer, lower=2, upper=2)) == "{2}"
        #Bounds that aren't whole numbers are rounded to the whole numbers within them
        @test describe(Domain(; type=Integer, lower=0.5, upper=9.5)) == "{1, 2, …, 9}"
        @test describe(Domain(; type=Bool)) == "{false, true}"
        @test describe(Domain(; type=Symbol)) == "Symbol"
        @test describe(Domain(; type=AbstractString)) == "AbstractString"
    end

    @testset "A domain of arrays or tuples is written as the values of each element" begin
        describe = CognitiveModeling.describe_domain
        @test describe(Domain(; type=AbstractVector{Real}, lower=0, upper=1)) ==
            "[0, 1] for each element of a vector"
        @test describe(Domain(; type=AbstractMatrix{Integer}, lower=1)) ==
            "{1, 2, …} for each element of a matrix"
        @test describe(Domain(; type=AbstractArray{Real})) ==
            "(-∞, ∞) for each element of an array"
        @test describe(Domain(; type=NTuple{3,Real}, lower=0)) ==
            "[0, ∞) for each element of a tuple of 3"
        @test describe(Domain(; type=Tuple{Vararg{Bool}})) ==
            "{false, true} for each element of a tuple"
        @test describe(Domain(; type=AbstractVector{Symbol})) ==
            "Symbol for each element of a vector"
    end

    @testset "A domain with possible values is written as them, and a constraint is named after the values" begin
        describe = CognitiveModeling.describe_domain
        @test describe(Domain(; possible_values=(:greedy, :random))) == "{:greedy, :random}"
        @test describe(Domain(; possible_values=(2, 4, 8))) == "{2, 4, 8}"
        @test describe(Domain(; possible_values=("low", "high"))) == "{\"low\", \"high\"}"
        #A range of more than three values is written with its first two and its last
        @test describe(Domain(; possible_values=1:3)) == "{1, 2, 3}"
        @test describe(Domain(; possible_values=1:10)) == "{1, 2, …, 10}"
        @test describe(Domain(; possible_values=0.0:0.25:1.0)) == "{0.0, 0.25, …, 1.0}"
        @test describe(Domain(; type=NTuple{2,Symbol}, possible_values=(:a, :b))) ==
            "{:a, :b} for each element of a tuple of 2"
        @test describe(
            Domain(; type=AbstractVector{Real}, lower=0, upper=1, constraint=sums_to_one)
        ) == "[0, 1] for each element of a vector, with the constraint sums_to_one"
        @test describe(Domain(; lower=0, constraint=isinteger)) ==
            "[0, ∞), with the constraint isinteger"
        @test describe(Domain(; constraint=value -> value != 0)) ==
            "(-∞, ∞), with a constraint"
        @test occursin(
            "  strategy ∈ {:greedy, :random}\n  n_options ∈ {2, 4, 8}\n",
            repr("text/plain", ModelWithPossibleValues()),
        )
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
