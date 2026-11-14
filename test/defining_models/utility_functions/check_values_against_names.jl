@testset "Checking values against a model's names" begin
    model = ModelWithAttributes()
    parameters = (; learning_rate=0.1, action_noise=1.0)
    state = (; expected_reward=0.5)
    previous_output = (; choice=BeforeFirstTimestep())

    @testset "The parameters, the state and the previous output must have the names in the model's attributes" begin
        @test CognitiveModeling.check_match_attribute_names(
            model, parameters, state, previous_output
        ) === nothing
        #The names can be in any order
        @test CognitiveModeling.check_match_attribute_names(
            model, (; action_noise=1.0, learning_rate=0.1), state, previous_output
        ) === nothing
        @test_throws "The parameters are named (:learning_rate,), but the parameters in the attributes of a cognitive model of type ModelWithAttributes are (:learning_rate, :action_noise)." CognitiveModeling.check_match_attribute_names(
            model, (; learning_rate=0.1), state, previous_output
        )
        @test_throws "The previous output is (bet = 1,), but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type ModelWithAttributes, (:choice,)." CognitiveModeling.check_match_attribute_names(
            model, parameters, state, (; bet=1)
        )
        @test_throws "The previous output is true, but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type ModelWithAttributes, (:choice,)." CognitiveModeling.check_match_attribute_names(
            model, parameters, state, true
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type ModelWithAttributes are (:expected_reward,)." CognitiveModeling.check_match_attribute_names(
            model, parameters, (; value=0.5), previous_output
        )
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of a cognitive model of type ModelWithAttributes, (:expected_reward,), also when it is a single object." CognitiveModeling.check_match_attribute_names(
            model, parameters, [0.5], previous_output
        )
        #The same for an environment model, whose previous output is the observation it gave
        @test CognitiveModeling.check_match_attribute_names(
            EnvironmentWithAttributes(),
            (; reward_probability=0.7),
            (;),
            (; reward=BeforeFirstTimestep()),
        ) === nothing
        @test_throws "The parameters are named (), but the parameters in the attributes of an environment model of type EnvironmentWithAttributes are (:reward_probability,)." CognitiveModeling.check_match_attribute_names(
            EnvironmentWithAttributes(), (;), (;), (; reward=BeforeFirstTimestep())
        )
    end

    @testset "A single input must have every input in the model's attributes, and can have more" begin
        @test CognitiveModeling.check_input_names(model, (; reward=1.0)) === nothing
        @test CognitiveModeling.check_input_names(model, (; reward=1.0, cue=2)) === nothing
        @test_throws "The input is (cue = 2,), but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type ModelWithAttributes, (:reward,)." CognitiveModeling.check_input_names(
            model, (; cue=2)
        )
        @test_throws "The input is 1.0, but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type ModelWithAttributes, (:reward,)." CognitiveModeling.check_input_names(
            model, 1.0
        )
    end
end
