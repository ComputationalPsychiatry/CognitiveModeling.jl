#Returns a NamedTuple instead of a ModelAttributes
struct ReturnsNamedTupleOfNames <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ReturnsNamedTupleOfNames)
    return (; parameters=(:noise,), states=(), inputs=(), outputs=())
end

#An environment model which does the same
struct EnvironmentReturnsNamedTupleOfNames <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(environment::EnvironmentReturnsNamedTupleOfNames)
    return (; parameters=(), states=(), inputs=(), outputs=(:reward,))
end

@testset "Checking a model's own methods" begin
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
end
