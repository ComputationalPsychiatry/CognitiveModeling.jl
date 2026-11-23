## Submodels for the tests ##
#A perception with two parameters and a state, and a response with one parameter, which are only ever submodels
struct PerceptionForTest <: AbstractSubmodel end
function CognitiveModeling.attribute_names(perception::PerceptionForTest)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value),
        states=:expected_reward,
        inputs=(),
        outputs=(),
    )
end
struct ResponseForTest <: AbstractSubmodel end
function CognitiveModeling.attribute_names(response::ResponseForTest)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=(), outputs=()
    )
end

#A cognitive model with a perception and a response in fields of their own, which it names, and a parameter of its own.
#Whether the flat names of the submodels' parameters have prefixes is chosen when it is constructed, and kept in its type
struct PerceptionAndResponse{Tperception,Tresponse,prefixes} <: AbstractCognitiveModel
    perception::Tperception
    response::Tresponse
end
function PerceptionAndResponse(perception, response; prefixes=false)
    return PerceptionAndResponse{typeof(perception),typeof(response),prefixes}(
        perception, response
    )
end
CognitiveModeling.submodel_fields(model::PerceptionAndResponse) = (:perception, :response)
function CognitiveModeling.uses_submodel_prefixes(
    model::PerceptionAndResponse{<:Any,<:Any,prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(model::PerceptionAndResponse)
    return ModelAttributes(;
        parameters=:lapse_rate, states=(), inputs=:reward, outputs=:choice
    )
end

#A submodel without methods of its own
struct SubmodelWithoutMethods <: AbstractSubmodel end

#A submodel which keeps its parts in a NamedTuple, and gives them with a method of submodels of its own
struct PartsInNamedTuple{Tparts<:NamedTuple} <: AbstractSubmodel
    parts::Tparts
end
CognitiveModeling.submodels(model::PartsInNamedTuple) = model.parts
function CognitiveModeling.attribute_names(model::PartsInNamedTuple)
    return ModelAttributes(; parameters=:weight, states=(), inputs=(), outputs=())
end

@testset "Submodels and their parameters" begin
    @testset "Every timestep model is a submodel" begin
        @test CognitiveModeling.AbstractTimestepModel <: AbstractSubmodel
        @test AbstractCognitiveModel <: AbstractSubmodel &&
            AbstractEnvironmentModel <: AbstractSubmodel
    end

    @testset "A submodel has the defaults of a model, and the same errors for what it must define" begin
        @test CognitiveModeling.default_parameters(ResponseForTest()) == (;)
        @test CognitiveModeling.parameter_domains(ResponseForTest()) == (;)
        @test CognitiveModeling.prepare_state_for_saving(
            PerceptionForTest(), (; expected_reward=0.5)
        ) == (; expected_reward=0.5)
        @test_throws "There is no initialize_state method for a submodel of type ResponseForTest." CognitiveModeling.initialize_state(
            ResponseForTest(), (;)
        )
        @test_throws "There is no attribute_names method for a submodel of type SubmodelWithoutMethods." CognitiveModeling.attribute_names(
            SubmodelWithoutMethods()
        )
    end

    @testset "A model names the fields with its submodels, or gives the submodels itself, and uses no prefixes by default" begin
        model = PerceptionAndResponse(PerceptionForTest(), ResponseForTest())
        @test @inferred(CognitiveModeling.submodels(model)) ==
            (; perception=PerceptionForTest(), response=ResponseForTest())
        @test CognitiveModeling.submodel_fields(PerceptionForTest()) == ()
        @test CognitiveModeling.submodels(PerceptionForTest()) == (;)
        @test CognitiveModeling.submodels(PartsInNamedTuple((; first=ResponseForTest()))) ==
            (; first=ResponseForTest())
        @test CognitiveModeling.uses_submodel_prefixes(PerceptionForTest()) === false
        @test CognitiveModeling.uses_submodel_prefixes(
            PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes=true)
        ) === true
    end

    @testset "Flat parameters are bundled under the names of the submodels, and flattened back" begin
        #Without prefixes, the flat names are the submodels' own names, after the model's own
        model = PerceptionAndResponse(PerceptionForTest(), ResponseForTest())
        flat = (;
            lapse_rate=0.1, learning_rate=0.3, initial_value=0.5, inverse_temperature=2.0
        )
        bundled = (;
            lapse_rate=0.1,
            perception=(; learning_rate=0.3, initial_value=0.5),
            response=(; inverse_temperature=2.0),
        )
        @test CognitiveModeling.flat_parameter_names(model) ==
            (:lapse_rate, :learning_rate, :initial_value, :inverse_temperature)
        @test CognitiveModeling.bundle_parameters(model, flat) === bundled
        @test CognitiveModeling.flatten_parameters(model, bundled) === flat
        #Flat parameters in another order give the same, since they are read by name
        @test CognitiveModeling.bundle_parameters(model, reverse(flat)) === bundled
        #With prefixes, the flat names of the submodels' parameters start with the submodel's name
        prefixed = PerceptionAndResponse(
            PerceptionForTest(), ResponseForTest(); prefixes=true
        )
        flat_prefixed = (;
            lapse_rate=0.1,
            perception__learning_rate=0.3,
            perception__initial_value=0.5,
            response__inverse_temperature=2.0,
        )
        @test CognitiveModeling.flat_parameter_names(prefixed) == keys(flat_prefixed)
        @test CognitiveModeling.bundle_parameters(prefixed, flat_prefixed) === bundled
        @test CognitiveModeling.flatten_parameters(prefixed, bundled) === flat_prefixed
        #A model without submodels gets its own parameters, in the order of its attributes
        @test CognitiveModeling.bundle_parameters(
            PerceptionForTest(), (; initial_value=0.5, learning_rate=0.3)
        ) === (; learning_rate=0.3, initial_value=0.5)
    end

    @testset "Submodels can have submodels, of any kind, also in a NamedTuple and of the same type twice" begin
        #A cognitive model as the perception of another, with prefixes on both levels, and as its response parts in a
        #NamedTuple, which use no prefixes of their own
        inner = PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes=true)
        parts = PartsInNamedTuple((; first=PerceptionForTest(), second=ResponseForTest()))
        outer = PerceptionAndResponse(inner, parts; prefixes=true)
        @test CognitiveModeling.flat_parameter_names(outer) == (
            :lapse_rate,
            :perception__lapse_rate,
            :perception__perception__learning_rate,
            :perception__perception__initial_value,
            :perception__response__inverse_temperature,
            :response__weight,
            :response__learning_rate,
            :response__initial_value,
            :response__inverse_temperature,
        )
        flat_names = CognitiveModeling.flat_parameter_names(outer)
        flat = NamedTuple{flat_names}(Tuple(float.(1:length(flat_names))))
        bundled = CognitiveModeling.bundle_parameters(outer, flat)
        @test bundled.perception.perception == (; learning_rate=3.0, initial_value=4.0)
        @test bundled.response == (;
            weight=6.0,
            first=(; learning_rate=7.0, initial_value=8.0),
            second=(; inverse_temperature=9.0),
        )
        @test CognitiveModeling.flatten_parameters(outer, bundled) === flat
        #Two submodels of the same type, kept apart by prefixes
        twice = PerceptionAndResponse(ResponseForTest(), ResponseForTest(); prefixes=true)
        @test CognitiveModeling.bundle_parameters(
            twice,
            (;
                lapse_rate=0.1,
                perception__inverse_temperature=1.0,
                response__inverse_temperature=2.0,
            ),
        ) === (;
            lapse_rate=0.1,
            perception=(; inverse_temperature=1.0),
            response=(; inverse_temperature=2.0),
        )
        #Three levels deep
        three_levels = PerceptionAndResponse(
            PerceptionAndResponse(
                PerceptionAndResponse(
                    PerceptionForTest(), ResponseForTest(); prefixes=true
                ),
                ResponseForTest();
                prefixes=true,
            ),
            ResponseForTest();
            prefixes=true,
        )
        flat_names = CognitiveModeling.flat_parameter_names(three_levels)
        flat = NamedTuple{flat_names}(Tuple(float.(1:length(flat_names))))
        bundled = CognitiveModeling.bundle_parameters(three_levels, flat)
        @test bundled.perception.perception.perception.learning_rate ==
            flat.perception__perception__perception__learning_rate
        @test CognitiveModeling.flatten_parameters(three_levels, bundled) === flat
    end

    @testset "Bundling is decided when the code is compiled, so its type is inferred and it allocates nothing" begin
        for prefixes in (false, true)
            model = PerceptionAndResponse(
                PerceptionAndResponse(PerceptionForTest(), ResponseForTest(); prefixes),
                ResponseForTest();
                prefixes=true,
            )
            flat_names = CognitiveModeling.flat_parameter_names(model)
            flat = NamedTuple{flat_names}(Tuple(float.(1:length(flat_names))))
            bundled = @inferred CognitiveModeling.bundle_parameters(model, flat)
            @test @inferred(CognitiveModeling.flatten_parameters(model, bundled)) === flat
            @test @inferred(CognitiveModeling.flat_parameter_names(model)) == flat_names
            @test (@allocated CognitiveModeling.bundle_parameters(model, flat)) == 0
        end
    end
end
