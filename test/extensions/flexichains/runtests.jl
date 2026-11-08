using FlexiChains: FlexiChains
using JET: JET

## A model and its sessions ##
#A model which predicts the next outcome with its expected value, for the session info
struct PredictsNextOutcome <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PredictsNextOutcome, parameters) = (; expected_value = 0.0)
function CognitiveModeling.apply_model(model::PredictsNextOutcome, input::ModelInput)
    r = input.observation.outcome
    V₋₁ = input.previous_state.expected_value
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + input.parameters.learning_rate * (r - V₋₁)
    return ModelOutput((; prediction = Deterministic(Vₜ)), (; expected_value = Vₜ))
end
CognitiveModeling.attribute_names(model::PredictsNextOutcome) = ModelAttributes(; parameters = :learning_rate, states = :expected_value, observations = :outcome, actions = :prediction)

#Participant P1 has two blocks of one trial each, and P2 one block of one trial
const CHAIN_TABLE = (; participant = ["P1", "P1", "P2"], block = [1, 2, 1], outcome = [1.0, 0.0, 1.0], prediction = [0.0, 0.3, 0.0])

#The sessions of the table, one for each participant, or one for each block of each participant
function chain_sessions(session_columns = :participant)
    return MultipleSessionsInfo(PredictsNextOutcome(), CHAIN_TABLE; observation_columns = :outcome, action_columns = :prediction, session_columns, who_starts = AgentStarts())
end

## Results made by hand, so that no PPL is needed ##
#What a session returns, in the shape turing_behavioural_session gives it: for each kind, a vector over the timesteps from
#timestep 0, with an empty NamedTuple on every timestep when nothing of that kind is saved
function session_result(n_timesteps; observations = fill((;), n_timesteps + 1), actions = fill((;), n_timesteps + 1), action_distributions = fill((;), n_timesteps + 1), states = fill((;), n_timesteps + 1))
    timestep_sizes = Union{BeforeFirstTimestep, Int}[BeforeFirstTimestep(); fill(1, n_timesteps)]
    return (; observations, actions, action_distributions, states, timestep_sizes)
end

#What a session returns when it saves everything, with the outcomes it saw, the predictions it made and its expected values from timestep 0
function saves_everything(outcomes, predictions, expected_values)
    return session_result(
        length(outcomes);
        observations = [(; outcome = BeforeFirstTimestep()); [(; outcome) for outcome in outcomes]],
        actions = [(; prediction = BeforeFirstTimestep()); [(; prediction) for prediction in predictions]],
        action_distributions = [(; prediction = BeforeFirstTimestep()); [(; prediction = Deterministic(prediction)) for prediction in predictions]],
        states = [(; expected_value) for expected_value in expected_values],
    )
end

#What a session returns when it saves its expected values only, from timestep 0
saves_expected_values(expected_values) = session_result(length(expected_values) - 1; states = [(; expected_value) for expected_value in expected_values])

#What a full model returns, with the parameters of every session
full_model_result(cognitive_parameters, saved_behavioural_values) = (; cognitive_parameters, saved_behavioural_values, downstream_values = nothing)

#The results of two samples, as two iterations of one chain
two_samples(first_result, second_result) = reshape([first_result, second_result], 2, 1)

#Two samples of a full model of the sessions by participant, each with a learning rate for P1 and P2 and what they saved
const FULL_MODEL_RESULTS = two_samples(
    full_model_result((; learning_rate = [0.5, 0.2]), [saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.5]), saves_everything([1.0], [0.0], [0.0, 0.0])]),
    full_model_result((; learning_rate = [0.4, 0.1]), [saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.4]), saves_everything([1.0], [0.0], [0.0, 0.0])]),
)

## Reading the chain ##
#The names of the entries of a chain as text, in their order, which reads the same with the VarName of FlexiChains 0.6 and 0.7
entry_names(chain) = [string(FlexiChains.get_name(key)) for key in keys(chain)]

#The values of the entry whose name is the given text, one for each sample
entry(chain, name::String) = chain[only(filter(key -> string(FlexiChains.get_name(key)) == name, collect(keys(chain))))]

@testset "FlexiChains extension" begin
    extension = Base.get_extension(CognitiveModeling, :CognitiveModelingFlexiChainsExt)
    @test extension isa Module

    #JET's test_package doesn't look inside extensions, so the extension is checked on its own
    @testset "JET" begin
        JET.test_package(extension; target_modules = (extension,))
    end

    @testset "The results of a full model give the parameters and what each session saved, named after it, in the order of the sessions" begin
        chain = session_values_chain(FULL_MODEL_RESULTS, chain_sessions())
        @test chain isa FlexiChains.FlexiChain
        @test entry_names(chain) == [
            "P1.learning_rate", "P1.observations.outcome", "P1.actions.prediction", "P1.action_distributions.prediction", "P1.expected_value",
            "P2.learning_rate", "P2.observations.outcome", "P2.actions.prediction", "P2.action_distributions.prediction", "P2.expected_value",
        ]
        #A parameter has a value for each sample, and what a session saved has a vector over its timesteps from timestep 0
        @test size(entry(chain, "P1.learning_rate")) == (2, 1)
        @test vec(entry(chain, "P1.learning_rate")) == [0.5, 0.4] && vec(entry(chain, "P2.learning_rate")) == [0.2, 0.1]
        @test entry(chain, "P1.expected_value")[2] == [0.0, 0.0, 0.4]
        @test entry(chain, "P1.observations.outcome")[1] == [BeforeFirstTimestep(), 1.0, 0.0]
        @test entry(chain, "P2.actions.prediction")[1] == [BeforeFirstTimestep(), 0.0]
        @test entry(chain, "P1.action_distributions.prediction")[1] == [BeforeFirstTimestep(), Deterministic(0.0), Deterministic(0.3)]
    end

    @testset "What isn't saved has no entry" begin
        results = two_samples(
            full_model_result((; learning_rate = [0.5, 0.2]), [saves_expected_values([0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])]),
            full_model_result((; learning_rate = [0.4, 0.1]), [saves_expected_values([0.0, 0.0, 0.4]), saves_expected_values([0.0, 0.0])]),
        )
        @test entry_names(session_values_chain(results, chain_sessions())) == ["P1.learning_rate", "P1.expected_value", "P2.learning_rate", "P2.expected_value"]
    end

    @testset "The results of several sessions have no parameters, and those of a single session no identifier" begin
        sessions = [saves_everything([1.0, 0.0], [0.0, 0.3], [0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])]
        chain = session_values_chain(reshape([sessions], 1, 1), chain_sessions())
        @test entry_names(chain) == ["P1.observations.outcome", "P1.actions.prediction", "P1.action_distributions.prediction", "P1.expected_value", "P2.expected_value"]
        chain = session_values_chain(reshape([first(sessions)], 1, 1))
        @test entry_names(chain) == ["observations.outcome", "actions.prediction", "action_distributions.prediction", "expected_value"]
        @test entry(chain, "expected_value")[1] == [0.0, 0.0, 0.5]
    end

    @testset "Identifiers from several session columns are joined with underscores, and a state that isn't a NamedTuple is one entry" begin
        sessions = [session_result(1; states = [0.0, 0.5]), session_result(1; states = [0.0, 0.0]), session_result(1; states = [0.0, 0.2])]
        chain = session_values_chain(reshape([sessions], 1, 1), chain_sessions((:participant, :block)))
        @test entry_names(chain) == ["P1_1.states", "P1_2.states", "P2_1.states"]
        @test entry(chain, "P1_1.states")[1] == [0.0, 0.5]
    end

    @testset "The samples are numbered as given" begin
        results = fill([saves_expected_values([0.0, 0.0, 0.5]), saves_expected_values([0.0, 0.0])], 3, 2)
        chain = session_values_chain(results, chain_sessions(); iter_indices = 11:13, chain_indices = [4, 7])
        @test FlexiChains.iter_indices(chain) == 11:13 && FlexiChains.chain_indices(chain) == [4, 7]
    end

    @testset "Results of the wrong kind, rejected samples and clashing names are errors" begin
        one_session = saves_expected_values([0.0, 0.0, 0.5])
        @test_throws "There are no results to make a chain of." session_values_chain(Matrix{Any}(undef, 0, 1))
        @test_throws "With a MultipleSessionsInfo, the results must be what turing_full_model or turing_multiple_behavioural_sessions returns" session_values_chain(reshape([one_session], 1, 1), chain_sessions())
        @test_throws "Without a session info, the results must be what turing_behavioural_session returns" session_values_chain(reshape([[one_session]], 1, 1))
        @test_throws "A sample has nothing as its result" session_values_chain(reshape([nothing, one_session], 2, 1))
        @test_throws "The results have 1 sessions, but the session info has 2." session_values_chain(reshape([[one_session]], 1, 1), chain_sessions())
        #A parameter named like a saved state
        clashing = full_model_result((; expected_value = [0.1, 0.2]), [one_session, saves_expected_values([0.0, 0.0])])
        @test_throws "Two values would have the same name in the chain, P1.expected_value" session_values_chain(reshape([clashing], 1, 1), chain_sessions())
    end
end
