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

    @testset "select_names keeps the states that are selected of a submodel under the name of the submodel" begin
        state = (;
            surprise=0.1,
            perception=(;
                expected_reward=0.5,
                prediction_error=0.2,
                learning=(; volatility=1.0, mean=2.0),
            ),
            n_timesteps=3,
        )
        selection_of(states) = CognitiveModeling.state_selection(SaveTrajectories(; states))
        @test CognitiveModeling.select_names(
            state, selection_of(:perception => :expected_reward)
        ) === (; perception=(; expected_reward=0.5))
        #With states of the model itself, in the order of the selection
        @test CognitiveModeling.select_names(
            state,
            selection_of((:surprise, :perception => (:prediction_error, :expected_reward))),
        ) === (; surprise=0.1, perception=(; prediction_error=0.2, expected_reward=0.5))
        #Down through the submodels of the submodel, where a name alone keeps all of its state
        @test CognitiveModeling.select_names(
            state,
            selection_of((
                :perception => (:learning => :mean, :expected_reward), :surprise
            )),
        ) === (; perception=(; learning=(; mean=2.0), expected_reward=0.5), surprise=0.1)
        @test CognitiveModeling.select_names(
            state, selection_of(:perception => :learning)
        ) === (; perception=(; learning=(; volatility=1.0, mean=2.0)))
        #The names are in the type, so what is selected is inferred, and selecting allocates nothing
        deep = selection_of((
            :surprise, :perception => (:learning => :mean, :expected_reward)
        ))
        @test @inferred(CognitiveModeling.select_names(state, deep)) ===
            (; surprise=0.1, perception=(; learning=(; mean=2.0), expected_reward=0.5))
        allocations(state, selection) =
            @allocated CognitiveModeling.select_names(state, selection)
        allocations(state, deep)
        @test allocations(state, deep) == 0
        #A state that the submodel doesn't have, a value that isn't a NamedTuple, and a name that is selected twice
        @test_throws FieldError CognitiveModeling.select_names(
            state, selection_of(:perception => :nope)
        )
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(
            [0.5], selection_of(:perception => :expected_reward)
        )
        @test_throws "A SaveTrajectories can select each state once, but it selects (:perception, :perception)" CognitiveModeling.select_names(
            state,
            selection_of((
                :perception => :expected_reward, :perception => :prediction_error
            )),
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
