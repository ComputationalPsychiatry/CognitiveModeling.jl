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
    expected_reward = CommonTestFunctionality.learn(
        context.previous_state.perception.expected_reward,
        context.parameters.perception.learning_rate,
        context.input.reward,
    )
    logit = CommonTestFunctionality.choice_logit(
        expected_reward, context.parameters.response.inverse_temperature
    )
    return ModelOutput(
        (; choice=bernoulli_logit(DistributionsJl(), logit)),
        (; perception=(; expected_reward)),
    )
end

#A perception which also keeps how much its expected reward changed, so that it has two states, and the baseline with it
struct PerceivesRewardsAndChanges <: AbstractSubmodel end
function CognitiveModeling.attribute_names(perception::PerceivesRewardsAndChanges)
    return ModelAttributes(;
        parameters=:learning_rate, states=(:expected_reward, :change), inputs=(), outputs=()
    )
end
function CognitiveModeling.initialize_state(
    perception::PerceivesRewardsAndChanges, parameters
)
    return (; expected_reward=0.5, change=0.0)
end
struct BaselineKeepingChanges <: AbstractCognitiveModel
    perception::PerceivesRewardsAndChanges
    response::RespondsToRewards
end
function BaselineKeepingChanges()
    return BaselineKeepingChanges(PerceivesRewardsAndChanges(), RespondsToRewards())
end
function CognitiveModeling.submodel_fields(model::BaselineKeepingChanges)
    return (:perception, :response)
end
function CognitiveModeling.attribute_names(model::BaselineKeepingChanges)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
function CognitiveModeling.initialize_state(model::BaselineKeepingChanges, parameters)
    return (;
        perception=CognitiveModeling.initialize_state(
            model.perception, parameters.perception
        ),
    )
end
function CognitiveModeling.timestep_model(
    model::BaselineKeepingChanges, context::ModelContext
)
    previous_expected_reward = context.previous_state.perception.expected_reward
    expected_reward = CommonTestFunctionality.learn(
        previous_expected_reward,
        context.parameters.perception.learning_rate,
        context.input.reward,
    )
    logit = CommonTestFunctionality.choice_logit(
        expected_reward, context.parameters.response.inverse_temperature
    )
    return ModelOutput(
        (; choice=bernoulli_logit(DistributionsJl(), logit)),
        (;
            perception=(;
                expected_reward, change=expected_reward - previous_expected_reward
            ),
        ),
    )
end

#A perception whose learning is a submodel of its own, so that the model has a submodel within a submodel, which uses
#prefixes when it is constructed with them
struct PerceivesRewardsThroughLearning{prefixes} <: AbstractSubmodel
    learning::PerceivesRewards
end
function CognitiveModeling.submodel_fields(perception::PerceivesRewardsThroughLearning)
    return (:learning,)
end
function CognitiveModeling.uses_submodel_prefixes(
    perception::PerceivesRewardsThroughLearning{prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(perception::PerceivesRewardsThroughLearning)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=())
end
function CognitiveModeling.initialize_state(
    perception::PerceivesRewardsThroughLearning, parameters
)
    return (;
        learning=CognitiveModeling.initialize_state(
            perception.learning, parameters.learning
        ),
    )
end

#The baseline with the nested perception, with prefixes on both levels or on neither
struct BaselineOfNestedSubmodels{prefixes} <: AbstractCognitiveModel
    perception::PerceivesRewardsThroughLearning{prefixes}
    response::RespondsToRewards
end
function BaselineOfNestedSubmodels(; prefixes=false)
    return BaselineOfNestedSubmodels{prefixes}(
        PerceivesRewardsThroughLearning{prefixes}(PerceivesRewards()), RespondsToRewards()
    )
end
function CognitiveModeling.submodel_fields(model::BaselineOfNestedSubmodels)
    return (:perception, :response)
end
function CognitiveModeling.uses_submodel_prefixes(
    model::BaselineOfNestedSubmodels{prefixes}
) where {prefixes}
    return prefixes
end
function CognitiveModeling.attribute_names(model::BaselineOfNestedSubmodels)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
function CognitiveModeling.initialize_state(model::BaselineOfNestedSubmodels, parameters)
    return (;
        perception=CognitiveModeling.initialize_state(
            model.perception, parameters.perception
        ),
    )
end
function CognitiveModeling.timestep_model(
    model::BaselineOfNestedSubmodels, context::ModelContext
)
    expected_reward = CommonTestFunctionality.learn(
        context.previous_state.perception.learning.expected_reward,
        context.parameters.perception.learning.learning_rate,
        context.input.reward,
    )
    logit = CommonTestFunctionality.choice_logit(
        expected_reward, context.parameters.response.inverse_temperature
    )
    return ModelOutput(
        (; choice=bernoulli_logit(DistributionsJl(), logit)),
        (; perception=(; learning=(; expected_reward))),
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
        saving_perception = SaveTrajectories(; states=:perception)
        chain = sample(Xoshiro(1), with_submodels, Prior(), 5; progress=false)
        trajectories = get_session_trajectories(
            with_submodels, chain; save_cognitive_trajectories=saving_perception
        )
        names = string.(Turing.FlexiChains.get_name.(keys(trajectories)))
        @test "S1.perception.expected_reward" in names && !("S1.perception" in names)
        #The same samples of the baseline give the same expected rewards
        baseline_chain = sample(Xoshiro(1), baseline, Prior(), 5; progress=false)
        baseline_trajectories = get_session_trajectories(
            baseline,
            baseline_chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:expected_reward),
        )
        @test trajectories[@varname(S1.perception.expected_reward)] ==
            baseline_trajectories[@varname(S1.expected_reward)]
    end

    @testset "A SaveTrajectories selects some of the states of a submodel, and the trajectories of a fit have entries for those only" begin
        keeping_changes = full_model(
            TuringBackend(),
            IndependentParameters(priors),
            sessions_of(BaselineKeepingChanges()),
        )
        @test logjoint(keeping_changes, values) ≈ logjoint(baseline, values)
        chain = sample(Xoshiro(1), keeping_changes, Prior(), 5; progress=false)
        names_saving(states) = string.(
            Turing.FlexiChains.get_name.(
                keys(
                    get_session_trajectories(
                        keeping_changes,
                        chain;
                        save_cognitive_trajectories=SaveTrajectories(; states),
                    ),
                ),
            ),
        )
        @test "S1.perception.expected_reward" in names_saving(:perception) &&
            "S1.perception.change" in names_saving(:perception)
        only_expected_reward = names_saving(:perception => :expected_reward)
        @test "S1.perception.expected_reward" in only_expected_reward &&
            !("S1.perception.change" in only_expected_reward)
        #The entries are those of the whole state of the submodel
        whole = get_session_trajectories(
            keeping_changes,
            chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:perception),
        )
        in_part = get_session_trajectories(
            keeping_changes,
            chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:perception => :change),
        )
        @test in_part[@varname(S2.perception.change)] ==
            whole[@varname(S2.perception.change)]
        #A state that the submodel doesn't have is an error when the sessions are made
        @test_throws "The SaveTrajectories selects the states (:value,) of the submodel perception" get_session_trajectories(
            keeping_changes,
            chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:perception => :value),
        )
    end

    @testset "A submodel can have submodels of its own, which are fitted with their parameters flat, and whose states have entries named after both" begin
        nested = full_model(
            TuringBackend(),
            IndependentParameters(priors),
            sessions_of(BaselineOfNestedSubmodels()),
        )
        @test logjoint(nested, values) ≈ logjoint(baseline, values)
        #With prefixes on both levels, the flat names start with the names of both submodels
        nested_prefixed = full_model(
            TuringBackend(),
            IndependentParameters((;
                perception__learning__learning_rate=Beta(2, 2),
                response__inverse_temperature=LogNormal(0, 0.5),
            )),
            sessions_of(BaselineOfNestedSubmodels(; prefixes=true)),
        )
        @test logjoint(
            nested_prefixed,
            Dict(
                @varname(perception__learning__learning_rate.sessions) => [0.3, 0.5, 0.7],
                @varname(response__inverse_temperature.sessions) => [1.0, 2.0, 3.0],
            ),
        ) ≈ logjoint(baseline, values)
        #The state of the innermost submodel has an entry named after both submodels, with the baseline's expected rewards
        chain = sample(Xoshiro(1), nested, Prior(), 5; progress=false)
        trajectories = get_session_trajectories(
            nested,
            chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:perception),
        )
        names = string.(Turing.FlexiChains.get_name.(keys(trajectories)))
        @test "S1.perception.learning.expected_reward" in names &&
            !("S1.perception.learning" in names)
        baseline_chain = sample(Xoshiro(1), baseline, Prior(), 5; progress=false)
        baseline_trajectories = get_session_trajectories(
            baseline,
            baseline_chain;
            save_cognitive_trajectories=SaveTrajectories(; states=:expected_reward),
        )
        @test trajectories[@varname(S1.perception.learning.expected_reward)] ==
            baseline_trajectories[@varname(S1.expected_reward)]
        #The state of the innermost submodel can be selected alone, with the names of both submodels
        selected = get_session_trajectories(
            nested,
            chain;
            save_cognitive_trajectories=SaveTrajectories(;
                states=:perception => :learning => :expected_reward
            ),
        )
        @test selected[@varname(S1.perception.learning.expected_reward)] ==
            trajectories[@varname(S1.perception.learning.expected_reward)]
    end
end
