#A model which is neither a cognitive model nor an environment model
struct NeitherKindOfModel <: CognitiveModeling.AbstractTimestepModel end

@testset "Describing a model in messages" begin
    @testset "A model is named by its kind and its type" begin
        @test CognitiveModeling.describe_model(ModelWithAttributes()) ==
            "a cognitive model of type ModelWithAttributes"
        @test CognitiveModeling.describe_model(EnvironmentWithAttributes()) ==
            "an environment model of type EnvironmentWithAttributes"
        @test CognitiveModeling.describe_model(NeitherKindOfModel()) ==
            "a model of type NeitherKindOfModel"
    end
end
