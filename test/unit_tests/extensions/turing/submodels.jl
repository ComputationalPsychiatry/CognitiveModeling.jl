## The baseline as a model with submodels: a perception, which learns the expected reward, and a response ##
#The perception keeps the expected reward as its state
struct PerceivesRewards <: AbstractSubmodel end
function CognitiveModeling.attribute_names(perception::PerceivesRewards)
    return ModelAttributes(;
        parameters=:learning_rate, states=:expected_reward, inputs=(), outputs=()
    )
end
function CognitiveModeling.initialize_state(perception::PerceivesRewards, parameters)
    return (; expected_reward=0.5)
end

#The response chooses by the expected reward
struct RespondsToRewards <: AbstractSubmodel end
function CognitiveModeling.attribute_names(response::RespondsToRewards)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=(), outputs=()
    )
end

#The baseline with the two as submodels, which uses prefixes when it is constructed with them
struct BaselineOfSubmodels{prefixes} <: AbstractCognitiveModel
    perception::PerceivesRewards
    response::RespondsToRewards
end
function BaselineOfSubmodels(; prefixes=false)
    return BaselineOfSubmodels{prefixes}(PerceivesRewards(), RespondsToRewards())
end
CognitiveModeling.submodel_fields(model::BaselineOfSubmodels) = (:perception, :response)
function CognitiveModeling.uses_submodel_prefixes(
    model::BaselineOfSubmodels{prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(model::BaselineOfSubmodels)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
function CognitiveModeling.initialize_state(model::BaselineOfSubmodels, parameters)
    return (;
        perception=CognitiveModeling.initialize_state(
            model.perception, parameters.perception
        ),
    )
end
function CognitiveModeling.timestep_model(model::BaselineOfSubmodels, context::ModelContext)
    expected_reward = ExampleModels.learn(
        context.previous_state.perception.expected_reward,
        context.parameters.perception.learning_rate,
        context.input.reward,
    )
    logit = ExampleModels.choice_logit(
        expected_reward, context.parameters.response.inverse_temperature
    )
    return ModelOutput(
        (; choice=bernoulli_logit(DistributionsJl(), logit)),
        (; perception=(; expected_reward)),
    )
end

@testset "Models with submodels are fitted with their parameters flat" begin
    #The sessions of FULL_TABLE, whose choice that wasn't recorded isn't imputed, so that the only variables are parameters
    sessions_of(model) = sessions_for_test(model; impute_missing_actions=false)
    priors = (; learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5))
    baseline = full_model(
        TuringBackend(),
        IndependentParameters(priors),
        sessions_of(Baseline(DistributionsJl())),
    )
    with_submodels = full_model(
        TuringBackend(), IndependentParameters(priors), sessions_of(BaselineOfSubmodels())
    )
    values = Dict(
        @varname(learning_rate.sessions) => [0.3, 0.5, 0.7],
        @varname(inverse_temperature.sessions) => [1.0, 2.0, 3.0],
    )

    @testset "The model has the log density of the same model in one, and its priors are given under the flat names" begin
        @test logjoint(with_submodels, values) ≈ logjoint(baseline, values)
        #With prefixes, the flat names start with the submodels' names
        prefixed = full_model(
            TuringBackend(),
            IndependentParameters((;
                perception__learning_rate=Beta(2, 2),
                response__inverse_temperature=LogNormal(0, 0.5),
            )),
            sessions_of(BaselineOfSubmodels(; prefixes=true)),
        )
        @test logjoint(
            prefixed,
            Dict(
                @varname(perception__learning_rate.sessions) => [0.3, 0.5, 0.7],
                @varname(response__inverse_temperature.sessions) => [1.0, 2.0, 3.0],
            ),
        ) ≈ logjoint(baseline, values)
        #A parameter model must give the flat names
        @test_throws "gives the parameters (:learning_rate, :inverse_temperature), which aren't in the attributes" full_model(
            TuringBackend(),
            IndependentParameters(priors),
            sessions_of(BaselineOfSubmodels(; prefixes=true)),
        )
    end

    @testset "The trajectories of a fit have an entry for each state of a submodel, named after the submodel" begin
        saving_perception = SaveHistory(; states=:perception)
        chain = sample(Xoshiro(1), with_submodels, Prior(), 5; progress=false)
        trajectories = get_session_trajectories(
            with_submodels, chain; cognitive_save_history=saving_perception
        )
        names = string.(Turing.FlexiChains.get_name.(keys(trajectories)))
        @test "S1.perception.expected_reward" in names && !("S1.perception" in names)
        #The same samples of the baseline give the same expected rewards
        baseline_chain = sample(Xoshiro(1), baseline, Prior(), 5; progress=false)
        baseline_trajectories = get_session_trajectories(
            baseline,
            baseline_chain;
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        )
        @test trajectories[@varname(S1.perception.expected_reward)] ==
            baseline_trajectories[@varname(S1.expected_reward)]
    end
end
