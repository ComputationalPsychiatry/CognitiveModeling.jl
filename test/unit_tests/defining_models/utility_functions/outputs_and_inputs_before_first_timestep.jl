@testset "Outputs and inputs before the first timestep" begin
    @testset "Every output and every input in a model's attributes is BeforeFirstTimestep" begin
        @test CognitiveModeling.output_before_first_timestep(ModelWithAttributes()) ===
            (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(ModelWithAttributes()) ===
            (; reward=BeforeFirstTimestep())
        #The outputs of an environment model are the observations it gives, and its inputs the actions it receives
        @test CognitiveModeling.output_before_first_timestep(
            EnvironmentWithAttributes()
        ) === (; reward=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(EnvironmentWithAttributes()) ===
            (; choice=BeforeFirstTimestep())
    end

    @testset "The same from a ModelAttributes, with several names or none" begin
        attributes = ModelAttributes(;
            parameters=(), states=(), inputs=(:reward, :cue), outputs=()
        )
        @test CognitiveModeling.input_before_first_timestep(attributes) ===
            (; reward=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
        @test CognitiveModeling.output_before_first_timestep(attributes) === (;)
    end
end
