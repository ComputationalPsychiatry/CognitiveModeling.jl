#Uses ChooseWithProbability, ZeroOrOne and simulate_for_test from simulate.jl

## A Rescorla-Wagner model in one, and as a perception and a response ##
#The model in one, whose state is the expected reward
struct LearnerInOne <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnerInOne, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::LearnerInOne, context::ModelContext)
    (; learning_rate, inverse_temperature) = context.parameters
    expected_reward = context.previous_state.expected_reward
    expected_reward += learning_rate * (context.input.reward - expected_reward)
    probability = 1 / (1 + exp(-inverse_temperature * (expected_reward - 0.5)))
    return ModelOutput((; choice=ChooseWithProbability(probability)), (; expected_reward))
end
function CognitiveModeling.attribute_names(model::LearnerInOne)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

#The perception, which learns the expected reward, with a default and a domain for its learning rate
struct LearnsExpectedReward <: AbstractSubmodel end
function CognitiveModeling.initialize_state(perception::LearnsExpectedReward, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.attribute_names(perception::LearnsExpectedReward)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=(), outputs=()
    )
end
function CognitiveModeling.default_parameters(perception::LearnsExpectedReward)
    return (; learning_rate=0.5)
end
function CognitiveModeling.parameter_domains(perception::LearnsExpectedReward)
    return (; learning_rate=Domain(; lower=0, upper=1))
end
function learn_reward(perception::LearnsExpectedReward, parameters, state, reward)
    expected_reward = state.expected_reward
    return (;
        expected_reward=expected_reward +
                        parameters.learning_rate * (reward - expected_reward)
    )
end

#The response, which chooses by the expected reward
struct ChoosesByExpectedReward <: AbstractSubmodel end
function CognitiveModeling.attribute_names(response::ChoosesByExpectedReward)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=(), outputs=()
    )
end
function choice_probability(response::ChoosesByExpectedReward, parameters, expected_reward)
    return 1 / (1 + exp(-parameters.inverse_temperature * (expected_reward - 0.5)))
end

#The model with the two as submodels, which keeps the perception's state under its name, and uses prefixes when it is
#constructed with them
struct LearnerWithSubmodels{Tperception,Tresponse,prefixes} <: AbstractCognitiveModel
    perception::Tperception
    response::Tresponse
end
function LearnerWithSubmodels(; prefixes=false)
    perception, response = LearnsExpectedReward(), ChoosesByExpectedReward()
    return LearnerWithSubmodels{typeof(perception),typeof(response),prefixes}(
        perception, response
    )
end
CognitiveModeling.submodel_fields(model::LearnerWithSubmodels) = (:perception, :response)
function CognitiveModeling.uses_submodel_prefixes(
    model::LearnerWithSubmodels{<:Any,<:Any,prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(model::LearnerWithSubmodels)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
function CognitiveModeling.initialize_state(model::LearnerWithSubmodels, parameters)
    return (;
        perception=CognitiveModeling.initialize_state(
            model.perception, parameters.perception
        ),
    )
end
function CognitiveModeling.timestep_model(
    model::LearnerWithSubmodels, context::ModelContext
)
    perception = learn_reward(
        model.perception,
        context.parameters.perception,
        context.previous_state.perception,
        context.input.reward,
    )
    probability = choice_probability(
        model.response,
        context.parameters.response,
        current_expected_reward(perception.expected_reward),
    )
    return ModelOutput((; choice=ChooseWithProbability(probability)), (; perception))
end

#The expected reward as a number, from a perception that keeps it as one or in a vector
current_expected_reward(expected_reward::Real) = expected_reward
current_expected_reward(expected_reward::AbstractVector) = expected_reward[1]

#A perception which keeps its expected reward in a vector that it changes in place, and copies it when it is saved
struct LearnsInPlace <: AbstractSubmodel end
function CognitiveModeling.initialize_state(perception::LearnsInPlace, parameters)
    return (; expected_reward=[0.5])
end
function CognitiveModeling.attribute_names(perception::LearnsInPlace)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=(), outputs=()
    )
end
function CognitiveModeling.prepare_state_for_saving(perception::LearnsInPlace, state)
    return (; expected_reward=copy(state.expected_reward))
end
function learn_reward(perception::LearnsInPlace, parameters, state, reward)
    state.expected_reward[1] +=
        parameters.learning_rate * (reward - state.expected_reward[1])
    return state
end

@testset "Models with submodels" begin
    flat_parameters = (; learning_rate=0.3, inverse_temperature=2.0)
    environment = FixedObservationsEnvironment((; reward=[1.0, 0.0, 1.0, 1.0, 0.0]))
    simulate_with(model, cognitive_parameters; keywords...) = simulate_for_test(
        model,
        environment;
        n_steps=5,
        cognitive_parameters,
        environment_parameters=(;),
        action_map=NameMap(),
        keywords...,
    )

    @testset "A model with submodels gives what the same model in one gives, from flat parameters, with the states of its submodels under their names" begin
        in_one = simulate_with(LearnerInOne(), flat_parameters)
        with_submodels = simulate_with(LearnerWithSubmodels(), flat_parameters)
        @test with_submodels.actions == in_one.actions
        @test map(state -> state.perception, with_submodels.states) == in_one.states
        #With prefixes, the flat names start with the submodels' names
        prefixed = simulate_with(
            LearnerWithSubmodels(; prefixes=true),
            (; perception__learning_rate=0.3, response__inverse_temperature=2.0),
        )
        @test prefixed.actions == in_one.actions && prefixed.states == with_submodels.states
    end

    @testset "A submodel's defaults fill in, and its domains are checked with its own name" begin
        defaulted = simulate_with(LearnerWithSubmodels(), (; inverse_temperature=2.0))
        @test defaulted.actions ==
            simulate_with(LearnerInOne(), (; learning_rate=0.5, inverse_temperature=2.0)).actions
        @test_throws ParameterDomainError simulate_with(
            LearnerWithSubmodels(), (; learning_rate=2.0, inverse_temperature=2.0)
        )
        @test_throws "LearnsExpectedReward" simulate_with(
            LearnerWithSubmodels(), (; learning_rate=2.0, inverse_temperature=2.0)
        )
    end

    @testset "The parameters and the state are checked against the collected names, also within the states of the submodels" begin
        #A missing flat parameter is found when the parameters are bundled for the initial state
        @test_throws "The parameters of a model with submodels are given flat, under the names that CognitiveModeling.get_attribute_names gives, but :response__inverse_temperature isn't among them" simulate_with(
            LearnerWithSubmodels(; prefixes=true), flat_parameters
        )
        #One too many, when a state is given
        @test_throws "but the parameters in the attributes of a cognitive model of type LearnerWithSubmodels" simulate_with(
            LearnerWithSubmodels(),
            (; flat_parameters..., lapse_rate=0.1);
            previous_cognitive_state=(; perception=(; expected_reward=0.5)),
        )
        @test_throws "The state is named (:expected_reward,), but the states in the attributes of a cognitive model of type LearnerWithSubmodels" simulate_with(
            LearnerWithSubmodels(),
            flat_parameters;
            previous_cognitive_state=(; expected_reward=0.5),
        )
        @test_throws "The state of a cognitive model of type LearnerWithSubmodels" simulate_with(
            LearnerWithSubmodels(),
            flat_parameters;
            previous_cognitive_state=(; perception=(; value=0.5)),
        )
    end

    @testset "What is saved of a submodel's state is prepared by its own method, and a history selects the state of a submodel by its name" begin
        model = LearnerWithSubmodels{LearnsInPlace,ChoosesByExpectedReward,false}(
            LearnsInPlace(), ChoosesByExpectedReward()
        )
        simulation = simulate_with(
            model,
            flat_parameters;
            cognitive_save_history=SaveHistory(;
                states=:perception, save_initial_timestep_values=false
            ),
        )
        @test map(state -> state.perception.expected_reward[1], simulation.states) == map(
            state -> state.expected_reward,
            simulate_with(LearnerInOne(), flat_parameters).states,
        )
        @test_throws "The SaveHistory selects the states (:expected_reward,)" simulate_with(
            model,
            flat_parameters;
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
    end

    @testset "An instantiated model, a single timestep and the names of the parameters work with submodels too" begin
        agent = InstantiatedModel(
            LearnerWithSubmodels(),
            flat_parameters;
            save_history=SaveHistory(SaveEverything()),
        )
        @test agent.parameters == (; learning_rate=0.3, inverse_temperature=2.0)
        @test agent.state == (; perception=(; expected_reward=0.5))
        simulate_timestep!(agent, (; reward=1.0))
        @test agent.state.perception.expected_reward ≈ 0.65
        reset!(agent)
        @test agent.state == (; perception=(; expected_reward=0.5))
        output, state = simulate_timestep(
            LearnerWithSubmodels(), (; reward=1.0); parameters=flat_parameters
        )
        @test state.perception.expected_reward ≈ 0.65
        @test CognitiveModeling.continuous_parameter_names(
            LearnerWithSubmodels(; prefixes=true)
        ) == (:perception__learning_rate, :response__inverse_temperature)
        @test CognitiveModeling.discrete_parameter_names(LearnerWithSubmodels()) == ()
    end

    @testset "The return type of a simulation with submodels is known, and bundling allocates nothing" begin
        model = LearnerWithSubmodels(; prefixes=true)
        parameters = (; perception__learning_rate=0.3, response__inverse_temperature=2.0)
        @inferred simulate_with(model, parameters)
        @test (@allocated CognitiveModeling.bundle_parameters(model, parameters)) == 0
    end
end
