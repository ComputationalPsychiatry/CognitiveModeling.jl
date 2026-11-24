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

#A model with a state of its own and two submodels: a perception with two states and a submodel of its own, a learning
#with two states, and a response without a state
struct LearningForSelections <: AbstractSubmodel end
function CognitiveModeling.attribute_names(learning::LearningForSelections)
    return ModelAttributes(;
        parameters=(), states=(:volatility, :mean), inputs=(), outputs=()
    )
end
struct PerceptionForSelections <: AbstractSubmodel
    learning::LearningForSelections
end
CognitiveModeling.submodel_fields(perception::PerceptionForSelections) = (:learning,)
function CognitiveModeling.attribute_names(perception::PerceptionForSelections)
    return ModelAttributes(;
        parameters=(), states=(:expected_reward, :prediction_error), inputs=(), outputs=()
    )
end
struct ResponseForSelections <: AbstractSubmodel end
function CognitiveModeling.attribute_names(response::ResponseForSelections)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=(), outputs=()
    )
end
struct ModelWithSubmodelsForSelections <: AbstractCognitiveModel
    perception::PerceptionForSelections
    response::ResponseForSelections
end
function CognitiveModeling.submodel_fields(model::ModelWithSubmodelsForSelections)
    return (:perception, :response)
end
function CognitiveModeling.attribute_names(model::ModelWithSubmodelsForSelections)
    return ModelAttributes(;
        parameters=(), states=:surprise, inputs=:reward, outputs=:choice
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
        #A state that is selected twice
        @test_throws "The SaveHistory selects the states (:surprise, :surprise), some of them twice." CognitiveModeling.check_savehistory_selection_names(
            SaveHistory(; states=(:surprise, :surprise)), model
        )
    end

    model_with_submodels = ModelWithSubmodelsForSelections(
        PerceptionForSelections(LearningForSelections()), ResponseForSelections()
    )
    check_states(states) = CognitiveModeling.check_savehistory_selection_names(
        SaveHistory(; states), model_with_submodels
    )

    @testset "A save history can select the state of a submodel, or some of its states, down through its submodels" begin
        @test check_states(:perception) === nothing
        @test check_states((:surprise, :perception => :expected_reward)) === nothing
        @test check_states(:perception => (:prediction_error, :learning)) === nothing
        @test check_states(:perception => :learning => (:mean, :volatility)) === nothing
        @test check_states((
            :perception => (:learning => :mean, :expected_reward), :surprise
        )) === nothing
        @test check_states(SaveEverything()) === nothing
    end

    @testset "A state that isn't in the attributes of the submodel is an error which names the submodel" begin
        @test_throws "The SaveHistory selects the states (:value,) of the submodel perception, but the states in the attributes of a submodel of type PerceptionForSelections are (:expected_reward, :prediction_error, :learning). A history can only select names in the model's attributes." check_states(
            :perception => :value
        )
        @test_throws "The SaveHistory selects the states (:value,) of the submodel perception.learning, but the states in the attributes of a submodel of type LearningForSelections are (:volatility, :mean)." check_states(
            :perception => :learning => :value
        )
        #A submodel without a state has no name among the states
        @test_throws "The SaveHistory selects the states (:response,), but the states in the attributes of a cognitive model of type ModelWithSubmodelsForSelections are (:surprise, :perception)." check_states(
            :response => :value
        )
    end

    @testset "A state of a submodel that is selected without the name of the submodel is an error which says how to select it" begin
        @test_throws "The SaveHistory selects the states (:expected_reward,), but the states in the attributes of a cognitive model of type ModelWithSubmodelsForSelections are (:surprise, :perception). A history can only select names in the model's attributes. :expected_reward is a state of the submodel :perception, which is selected with :perception => :expected_reward." check_states(
            :expected_reward
        )
        @test_throws ":mean is a state of the submodel :learning, which is selected with :learning => :mean." check_states(
            :perception => :mean
        )
        #A name that no submodel has either gets no such sentence
        message = try
            check_states(:value)
        catch error
            error.msg
        end
        @test endswith(
            message, "A history can only select names in the model's attributes."
        )
    end

    @testset "A submodel that is named twice, and states of something that isn't a submodel, are errors" begin
        @test_throws "The SaveHistory selects the states (:perception, :perception), some of them twice. The states to save of a submodel are given together, as in :perception => (:expected_reward, :prediction_error)." check_states((
            :perception => :expected_reward, :perception => :prediction_error
        ))
        @test_throws "The SaveHistory selects the states (:mean, :mean) of the submodel perception.learning, some of them twice." check_states(
            :perception => :learning => (:mean, :mean)
        )
        @test_throws "The SaveHistory selects the states (:value,) of :surprise, but :surprise isn't a submodel of a cognitive model of type ModelWithSubmodelsForSelections. It is a state of its own, which is selected by its name alone." check_states(
            :surprise => :value
        )
        @test_throws "The SaveHistory selects the states (:value,) of :expected_reward of the submodel perception, but :expected_reward isn't a submodel of a submodel of type PerceptionForSelections." check_states(
            :perception => :expected_reward => :value
        )
    end
end
