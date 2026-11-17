#A model with a name of each kind, for the names a save history can select
struct ModelForSelections <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelForSelections)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=(:expected_reward, :surprise),
        inputs=:reward,
        outputs=:choice,
    )
end

@testset "Checking the names a save history selects" begin
    @testset "A save history can select the names in the model's attributes, or everything" begin
        model = ModelForSelections()
        @test CognitiveModeling.check_savehistory_selection_names(SaveHistory(), model) ===
            nothing
        @test CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(SaveEverything()), model
        ) === nothing
        @test CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(;
                states=:surprise,
                inputs=:reward,
                outputs=:choice,
                output_distributions=:choice,
            ),
            model,
        ) === nothing
    end

    @testset "A name that isn't in the model's attributes is an error which names the kind" begin
        model = ModelForSelections()
        @test_throws "The SaveHistory selects the states (:value,), but the states in the attributes of a cognitive model of type ModelForSelections are (:expected_reward, :surprise). A history can only select names in the model's attributes." CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(; states=:value), model
        )
        @test_throws "The SaveHistory selects the inputs (:cue,)" CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(; inputs=:cue), model
        )
        #The output distributions are named like the outputs
        @test_throws "The SaveHistory selects the output distributions (:bet,), but the outputs in the attributes of a cognitive model of type ModelForSelections are (:choice,)." CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(; output_distributions=:bet), model
        )
        @test CognitiveModeling.check_selection_names(
            SaveEverything(), (:choice,), "outputs", model
        ) === nothing
    end
end
