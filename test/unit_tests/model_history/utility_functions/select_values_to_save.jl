#A model whose state is prepared for saving by doubling it, and one whose state can't be prepared at all
struct DoublesStateForSaving <: AbstractCognitiveModel end
function CognitiveModeling.prepare_state_for_saving(model::DoublesStateForSaving, state)
    return map(value -> 2 * value, state)
end
struct StateCanNotBePrepared <: AbstractCognitiveModel end
function CognitiveModeling.prepare_state_for_saving(model::StateCanNotBePrepared, state)
    return error("The state was prepared for saving")
end

@testset "Selecting the values to save" begin
    @testset "select_names keeps what is selected" begin
        state = (; expected_reward=0.5, surprise=0.1, n_timesteps=3)
        @test CognitiveModeling.select_names(state, Val(SaveEverything())) === state
        @test CognitiveModeling.select_names(state, Val(())) === (;)
        @test CognitiveModeling.select_names(state, Val((:surprise, :expected_reward))) ===
            (; surprise=0.1, expected_reward=0.5)
        @test CognitiveModeling.select_names(
            (; reward=NoValue(), cue=1), Val((:reward,))
        ) === (; reward=NoValue())
        @test CognitiveModeling.select_names(NoValue(), Val(())) === (;)
        @test CognitiveModeling.select_names([0.5], Val(())) === (;)
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(
            NoValue(), Val((:reward,))
        )
        @test_throws FieldError CognitiveModeling.select_names(state, Val((:nope,)))
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(
            [0.5], Val((:expected_reward,))
        )
    end

    @testset "The state to save is what is selected of the state as the model prepares it, which isn't prepared when nothing is selected" begin
        state = (; expected_reward=0.5, surprise=0.1)
        @test CognitiveModeling.select_state_to_save(
            DoublesStateForSaving(), state, Val((:surprise,))
        ) === (; surprise=0.2)
        @test CognitiveModeling.select_state_to_save(
            DoublesStateForSaving(), state, Val(SaveEverything())
        ) === (; expected_reward=1.0, surprise=0.2)
        @test CognitiveModeling.select_state_to_save(
            StateCanNotBePrepared(), state, Val(())
        ) === (;)
        @test_throws "The state was prepared for saving" CognitiveModeling.select_state_to_save(
            StateCanNotBePrepared(), state, Val((:surprise,))
        )
    end
end
