#Uses the submodels of the tests of bundle_submodel_parameters.jl

#A model which names a field it doesn't have, one whose field holds something that isn't a submodel, and one whose
#submodel has the name of one of its own parameters
struct NamesMissingField <: AbstractCognitiveModel
    perception::PerceptionForTest
end
CognitiveModeling.submodel_fields(model::NamesMissingField) = (:perception, :response)
function CognitiveModeling.attribute_names(model::NamesMissingField)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
struct HoldsNumberAsSubmodel <: AbstractCognitiveModel
    perception::Float64
end
CognitiveModeling.submodel_fields(model::HoldsNumberAsSubmodel) = (:perception,)
function CognitiveModeling.attribute_names(model::HoldsNumberAsSubmodel)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
struct SubmodelNamedAsParameter <: AbstractCognitiveModel
    perception::PerceptionForTest
end
CognitiveModeling.submodel_fields(model::SubmodelNamedAsParameter) = (:perception,)
function CognitiveModeling.attribute_names(model::SubmodelNamedAsParameter)
    return ModelAttributes(;
        parameters=:perception, states=(), inputs=:reward, outputs=:choice
    )
end

@testset "The names of a model with submodels" begin
    @testset "get_attribute_names collects the flat parameters, the states with a bundle for each submodel with a state, and the model's own inputs and outputs" begin
        model = PerceptionAndResponse(PerceptionForTest(), ResponseForTest())
        attributes = CognitiveModeling.get_attribute_names(model)
        @test attributes.parameters ==
            (:lapse_rate, :learning_rate, :initial_value, :inverse_temperature)
        #The response has no state, so only the perception's bundle is a state
        @test attributes.states == (:perception,)
        @test attributes.inputs == (:reward,) && attributes.outputs == (:choice,)
        #With prefixes, the flat names of the parameters have them
        prefixed = PerceptionAndResponse(
            PerceptionForTest(), ResponseForTest(); prefixes=true
        )
        @test CognitiveModeling.get_attribute_names(prefixed).parameters == (
            :lapse_rate,
            :perception__learning_rate,
            :perception__initial_value,
            :response__inverse_temperature,
        )
        #A submodel has a state when one of its own submodels has one
        nested = PerceptionAndResponse(
            ResponseForTest(),
            PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes=true);
            prefixes=true,
        )
        @test CognitiveModeling.get_attribute_names(nested).states == (:response,)
        #A model without submodels has the names of attribute_names
        @test CognitiveModeling.get_attribute_names(PerceptionForTest()) ===
            CognitiveModeling.attribute_names(PerceptionForTest())
        #The names are in the type, so they are inferred
        @test @inferred(CognitiveModeling.get_attribute_names(prefixed)) isa ModelAttributes
    end

    @testset "submodel_state_names gives the submodels with a state, with theirs, down through the submodels" begin
        #The response has no state, so only the perception has one
        @test CognitiveModeling.submodel_state_names(
            PerceptionAndResponse(PerceptionForTest(), ResponseForTest())
        ) == (; perception=(;))
        #A submodel has a state when one of its own submodels has one
        nested = PerceptionAndResponse(
            ResponseForTest(),
            PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes=true);
            prefixes=true,
        )
        @test CognitiveModeling.submodel_state_names(nested) ==
            (; response=(; perception=(;)))
        @test CognitiveModeling.submodel_state_names(PerceptionForTest()) == (;)
        @test @inferred(CognitiveModeling.submodel_state_names(nested)) isa NamedTuple
    end

    @testset "Submodels that aren't declared or named as they must be give informative errors" begin
        @test_throws "CognitiveModeling.submodel_fields of a cognitive model of type NamesMissingField names :response, but it has no field :response. Its fields are (:perception,)." CognitiveModeling.get_attribute_names(
            NamesMissingField(PerceptionForTest())
        )
        @test_throws "The submodel :perception of a cognitive model of type HoldsNumberAsSubmodel is a Float64, which isn't an AbstractSubmodel." CognitiveModeling.get_attribute_names(
            HoldsNumberAsSubmodel(0.5)
        )
        @test_throws "A cognitive model of type SubmodelNamedAsParameter has a submodel named :perception, which is also one of its own names in attribute_names." CognitiveModeling.get_attribute_names(
            SubmodelNamedAsParameter(PerceptionForTest())
        )
    end

    @testset "Two parameters with the same flat name are an error that says to use prefixes, also within a submodel" begin
        #Two responses without prefixes both have an inverse temperature
        twice = PerceptionAndResponse(ResponseForTest(), ResponseForTest())
        @test_throws "has two parameters with the flat name :inverse_temperature: perception.inverse_temperature and response.inverse_temperature. Give it a method of CognitiveModeling.uses_submodel_prefixes which returns true" CognitiveModeling.get_attribute_names(
            twice
        )
        #The same pair within a submodel is reported for that submodel, even when the model around it uses prefixes
        around = PerceptionAndResponse(twice, ResponseForTest(); prefixes=true)
        @test_throws "A cognitive model of type PerceptionAndResponse{ResponseForTest, ResponseForTest, false} has two parameters" CognitiveModeling.get_attribute_names(
            around
        )
        #With prefixes there is nothing to report
        @test CognitiveModeling.get_attribute_names(
            PerceptionAndResponse(ResponseForTest(), ResponseForTest(); prefixes=true)
        ).parameters == (
            :lapse_rate, :perception__inverse_temperature, :response__inverse_temperature
        )
    end

    @testset "A name used for two kinds among the collected names is an error, as among a model's own" begin
        #A submodel named lapse_rate, whose state bundle has the name of the model's own parameter, is caught as a
        #submodel with a name of the model's own. Parts in a NamedTuple whose bundle name is a parameter of a part
        clashing = PartsInNamedTuple((; learning_rate=PerceptionForTest()))
        @test_throws "learning_rate" CognitiveModeling.get_attribute_names(clashing)
    end

    @testset "A model with submodels shows them, and the collected names, and a submodel shows as a model does" begin
        model = PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes=true)
        shown = sprint(show, MIME"text/plain"(), model)
        @test occursin(
            "\nSubmodels: perception (PerceptionForTest), response (ResponseForTest)", shown
        )
        @test occursin(
            "\nParameters: lapse_rate, perception__learning_rate, perception__initial_value, response__inverse_temperature",
            shown,
        ) || occursin("\nParameters: 4 (see attribute_names)", shown)
        @test occursin("\nStates: perception", shown)
        @test startswith(
            sprint(show, MIME"text/plain"(), ResponseForTest()),
            "-- Submodel --\nType: ResponseForTest",
        )
        @test sprint(show, ResponseForTest()) == "ResponseForTest (submodel)"
        @test occursin(
            "Attributes: not specified",
            sprint(show, MIME"text/plain"(), SubmodelWithoutMethods()),
        )
    end
end
