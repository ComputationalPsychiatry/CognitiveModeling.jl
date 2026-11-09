using FlexiChains: FlexiChains
using JET: JET

## A model and its sessions ##
#A model which predicts the next outcome with its expected value, for the session info
struct PredictsNextOutcome <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::PredictsNextOutcome, parameters)
    return (; expected_value=0.0)
end
function CognitiveModeling.apply_model(model::PredictsNextOutcome, context::ModelContext)
    r = context.input.outcome
    V₋₁ = context.previous_state.expected_value
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + context.parameters.learning_rate * (r - V₋₁)
    return ModelOutput((; prediction=Deterministic(Vₜ)), (; expected_value=Vₜ))
end
function CognitiveModeling.attribute_names(model::PredictsNextOutcome)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=:expected_value,
        inputs=:outcome,
        outputs=:prediction,
    )
end

#Participant P1 has two blocks of one trial each, and P2 one block of one trial
const CHAIN_TABLE = (;
    participant=["P1", "P1", "P2"],
    block=[1, 2, 1],
    outcome=[1.0, 0.0, 1.0],
    prediction=[0.0, 0.3, 0.0],
)

#The sessions of the table, one for each participant, or one for each block of each participant
function chain_sessions(session_columns=:participant)
    return MultipleSessionsInfo(
        PredictsNextOutcome(),
        CHAIN_TABLE;
        observation_columns=:outcome,
        action_columns=:prediction,
        session_columns,
        who_starts=AgentStarts(),
    )
end

## Results made by hand, so that no PPL is needed ##
#What a session returns, in the shape turing_behavioural_session gives it: for each kind, a vector over the timesteps from
#timestep 0, with an empty NamedTuple on every timestep when nothing of that kind is saved, and the environment's record
#under environment when the session has one
function session_result(
    n_timesteps;
    observations=fill((;), n_timesteps + 1),
    actions=fill((;), n_timesteps + 1),
    action_distributions=fill((;), n_timesteps + 1),
    states=fill((;), n_timesteps + 1),
    environment=nothing,
)
    timestep_sizes = Union{BeforeFirstTimestep,Int}[
        BeforeFirstTimestep();
        fill(1, n_timesteps)
    ]
    saved = (; states, observations, actions, action_distributions)
    environment === nothing || (saved = merge(saved, (; environment)))
    return merge(saved, (; timestep_sizes))
end

#What an environment's record looks like when it saves its level and the rewards it gave, from timestep 0
function environment_record(levels, rewards)
    n_saved = length(levels)
    return (;
        states=[(; level) for level in levels],
        observations=[(; reward=BeforeFirstTimestep()); [(; reward) for reward in rewards]],
        actions=fill((;), n_saved),
        observation_distributions=[
            (; reward=BeforeFirstTimestep());
            [(; reward=Deterministic(reward)) for reward in rewards]
        ],
    )
end

#What a session returns when it saves everything, with the outcomes it saw, the predictions it made and its expected values from timestep 0
function saves_everything(outcomes, predictions, expected_values)
    return session_result(
        length(outcomes);
        observations=[
            (; outcome=BeforeFirstTimestep());
            [(; outcome) for outcome in outcomes]
        ],
        actions=[
            (; prediction=BeforeFirstTimestep());
            [(; prediction) for prediction in predictions]
        ],
        action_distributions=[
            (; prediction=BeforeFirstTimestep());
            [(; prediction=Deterministic(prediction)) for prediction in predictions]
        ],
        states=[(; expected_value) for expected_value in expected_values],
    )
end

#What a session returns when it saves its expected values only, from timestep 0
function saves_expected_values(expected_values)
    return session_result(
        length(expected_values) - 1;
        states=[(; expected_value) for expected_value in expected_values],
    )
end

#What a full model returns, with the parameters of every session for both models
function full_model_result(
    cognitive_parameters, saved_behavioural_values; environment_parameters=(;)
)
    return (;
        cognitive_parameters,
        environment_parameters,
        saved_behavioural_values,
        downstream_values=nothing,
    )
end

#The results of two samples, as two iterations of one chain
two_samples(first_result, second_result) = reshape([first_result, second_result], 2, 1)

#Two samples of a full model of the sessions by participant, each with a learning rate for P1 and P2 and what they saved
const FULL_MODEL_RESULTS = two_samples(
    full_model_result(
        (; learning_rate=[0.5, 0.2]),
        [
            saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.5]),
            saves_everything([1.0], [0.0], [0.0, 0.0]),
        ],
    ),
    full_model_result(
        (; learning_rate=[0.4, 0.1]),
        [
            saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.4]),
            saves_everything([1.0], [0.0], [0.0, 0.0]),
        ],
    ),
)

## Reading the chain ##
#The names of the entries of a chain as text, in their order, which reads the same with the VarName of FlexiChains 0.6 and 0.7
entry_names(chain) = [string(FlexiChains.get_name(key)) for key in keys(chain)]

#The values of the entry whose name is the given text, one for each sample
function entry(chain, name::String)
    return chain[only(
        filter(key -> string(FlexiChains.get_name(key)) == name, collect(keys(chain)))
    )]
end

@testset "FlexiChains extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingFlexiChainsExt)
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own
    @testset "JET" begin
        JET.test_package(extension; target_modules=(extension,))
    end

    @testset "The results of a full model give what each session saved, named after the session and the attribute, in the order of the sessions" begin
        chain = session_values_chain(FULL_MODEL_RESULTS, chain_sessions())
        @test chain isa FlexiChains.FlexiChain
        @test entry_names(chain) == [
            "P1.outcome",
            "P1.prediction",
            "P1.action_distributions.prediction",
            "P1.expected_value",
            "P2.outcome",
            "P2.prediction",
            "P2.action_distributions.prediction",
            "P2.expected_value",
        ]
        #What a session saved has a vector over its timesteps from timestep 0, for each sample
        @test size(entry(chain, "P1.expected_value")) == (2, 1)
        @test entry(chain, "P1.expected_value")[2] == [0.0, 0.0, 0.4]
        @test entry(chain, "P1.outcome")[1] == [BeforeFirstTimestep(), 1.0, 0.0]
        @test entry(chain, "P2.prediction")[1] == [BeforeFirstTimestep(), 0.0]
        @test entry(chain, "P1.action_distributions.prediction")[1] ==
            [BeforeFirstTimestep(), Deterministic(0.0), Deterministic(0.3)]
    end

    @testset "The parameters of the sessions are a chain of their own, with the environment's under environment" begin
        chain = session_parameters_chain(FULL_MODEL_RESULTS, chain_sessions())
        @test entry_names(chain) == ["P1.learning_rate", "P2.learning_rate"]
        @test size(entry(chain, "P1.learning_rate")) == (2, 1)
        @test vec(entry(chain, "P1.learning_rate")) == [0.5, 0.4] &&
            vec(entry(chain, "P2.learning_rate")) == [0.2, 0.1]
        with_environment = two_samples(
            full_model_result(
                (; learning_rate=[0.5, 0.2]),
                [saves_expected_values([0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])];
                environment_parameters=(; mean_reward=[0.6, 0.6]),
            ),
            full_model_result(
                (; learning_rate=[0.4, 0.1]),
                [saves_expected_values([0.0, 0.0, 0.4]), saves_expected_values([0.0, 0.0])];
                environment_parameters=(; mean_reward=[0.7, 0.7]),
            ),
        )
        chain = session_parameters_chain(with_environment, chain_sessions())
        @test entry_names(chain) == [
            "P1.learning_rate",
            "P1.environment.mean_reward",
            "P2.learning_rate",
            "P2.environment.mean_reward",
        ]
        @test vec(entry(chain, "P2.environment.mean_reward")) == [0.6, 0.7]
        #It merges with the chain of the saved values, and takes only the results of a full model
        merged = merge(session_values_chain(with_environment, chain_sessions()), chain)
        @test entry_names(merged) == [
            "P1.expected_value",
            "P2.expected_value",
            "P1.learning_rate",
            "P1.environment.mean_reward",
            "P2.learning_rate",
            "P2.environment.mean_reward",
        ]
        @test_throws "The results must be what turing_full_model returns for each sample" session_parameters_chain(
            reshape([[saves_expected_values([0.0, 0.0, 0.5])]], 1, 1), chain_sessions()
        )
    end

    @testset "What isn't saved has no entry" begin
        results = two_samples(
            full_model_result(
                (; learning_rate=[0.5, 0.2]),
                [saves_expected_values([0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])],
            ),
            full_model_result(
                (; learning_rate=[0.4, 0.1]),
                [saves_expected_values([0.0, 0.0, 0.4]), saves_expected_values([0.0, 0.0])],
            ),
        )
        @test entry_names(session_values_chain(results, chain_sessions())) ==
            ["P1.expected_value", "P2.expected_value"]
    end

    @testset "The environment's record is under environment, named the same way" begin
        sessions = [
            session_result(
                2;
                states=[(; expected_value) for expected_value in [0.0, 0.0, 0.5]],
                environment=environment_record([0.0, 0.1, 0.2], [1.0, 0.0]),
            ),
            session_result(1; environment=environment_record([0.0, 0.1], [1.0])),
        ]
        chain = session_values_chain(reshape([sessions], 1, 1), chain_sessions())
        @test entry_names(chain) == [
            "P1.expected_value",
            "P1.environment.reward",
            "P1.environment.observation_distributions.reward",
            "P1.environment.level",
            "P2.environment.reward",
            "P2.environment.observation_distributions.reward",
            "P2.environment.level",
        ]
        @test entry(chain, "P1.environment.level")[1] == [(0.0), 0.1, 0.2]
        @test entry(chain, "P2.environment.reward")[1] == [BeforeFirstTimestep(), 1.0]
        #An environment that saved nothing, as when it didn't run, has no entries
        empty_environment = session_result(
            1;
            states=[(; expected_value=0.0), (; expected_value=0.5)],
            environment=(;
                states=fill((;), 2),
                observations=fill((;), 2),
                actions=fill((;), 2),
                observation_distributions=fill((;), 2),
            ),
        )
        @test entry_names(session_values_chain(reshape([empty_environment], 1, 1))) ==
            ["expected_value"]
        #A state of the environment that isn't a NamedTuple is one entry
        plain_states = session_result(
            1;
            environment=(;
                states=[0.0, 0.5],
                observations=fill((;), 2),
                actions=fill((;), 2),
                observation_distributions=fill((;), 2),
            ),
        )
        @test entry_names(session_values_chain(reshape([plain_states], 1, 1))) ==
            ["environment.states"]
    end

    @testset "The results of several sessions have the same names, and those of a single session no identifier" begin
        sessions = [
            saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.5]),
            saves_expected_values([0.0, 0.0]),
        ]
        chain = session_values_chain(reshape([sessions], 1, 1), chain_sessions())
        @test entry_names(chain) == [
            "P1.outcome",
            "P1.prediction",
            "P1.action_distributions.prediction",
            "P1.expected_value",
            "P2.expected_value",
        ]
        chain = session_values_chain(reshape([first(sessions)], 1, 1))
        @test entry_names(chain) ==
            ["outcome", "prediction", "action_distributions.prediction", "expected_value"]
        @test entry(chain, "expected_value")[1] == [0.0, 0.0, 0.5]
    end

    @testset "Identifiers from several session columns are joined with underscores, and a state that isn't a NamedTuple is one entry" begin
        sessions = [
            session_result(1; states=[0.0, 0.5]),
            session_result(1; states=[0.0, 0.0]),
            session_result(1; states=[0.0, 0.2]),
        ]
        chain = session_values_chain(
            reshape([sessions], 1, 1), chain_sessions((:participant, :block))
        )
        @test entry_names(chain) == ["P1_1.states", "P1_2.states", "P2_1.states"]
        @test entry(chain, "P1_1.states")[1] == [0.0, 0.5]
    end

    @testset "The samples are numbered as given" begin
        results = fill(
            [saves_expected_values([0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])],
            3,
            2,
        )
        chain = session_values_chain(
            results, chain_sessions(); iter_indices=11:13, chain_indices=[4, 7]
        )
        @test FlexiChains.iter_indices(chain) == 11:13 &&
            FlexiChains.chain_indices(chain) == [4, 7]
    end

    @testset "Results of the wrong kind, rejected samples and clashing names are errors" begin
        one_session = saves_expected_values([0.0, 0.0, 0.5])
        @test_throws "There are no results to make a chain of." session_values_chain(
            Matrix{Any}(undef, 0, 1)
        )
        @test_throws "With a MultipleSessionsInfo, the results must be what turing_full_model or turing_multiple_behavioural_sessions returns" session_values_chain(
            reshape([one_session], 1, 1), chain_sessions()
        )
        @test_throws "Without a session info, the results must be what turing_behavioural_session returns" session_values_chain(
            reshape([[one_session]], 1, 1)
        )
        @test_throws "A sample has nothing as its result" session_values_chain(
            reshape([nothing, one_session], 2, 1)
        )
        @test_throws "The results have 1 sessions, but the session info has 2." session_values_chain(
            reshape([[one_session]], 1, 1), chain_sessions()
        )
        #A model can't name a state like an observation, but results made by hand can, which the chain refuses
        clashing = [
            session_result(
                1;
                states=[(; outcome=0.1), (; outcome=0.2)],
                observations=[(; outcome=BeforeFirstTimestep()), (; outcome=1.0)],
            ),
            saves_expected_values([0.0, 0.0]),
        ]
        @test_throws "Two values would have the same name in the chain, P1.outcome" session_values_chain(
            reshape([clashing], 1, 1), chain_sessions()
        )
    end
end
