"""
Computational modeling of cognition and behaviour.
"""
module CognitiveModeling

using Random: AbstractRNG, default_rng

include("defining_models/types.jl")
include("defining_models/functions_to_extend.jl")
include("defining_models/utility_functions.jl")
include("environments/common_rl_environment.jl")
include("simulation/save_history.jl")
include("simulation/types.jl")
include("simulation/simulate.jl")
include("inference/types.jl")
include("inference/functions_to_extend.jl")
include("inference/utility_functions.jl")
include("inference/turing_models.jl")
include("inference/chains.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel
export AbstractNoValue, NoObservation, NoAction, BeforeFirstTimestep
export MissingValue
export ModelContext, ModelOutput, ModelAttributes, Deterministic, InvalidParametersError
export CommonRLEnvironment
export InstantiatedModel, AgentStarts, EnvironmentStarts, UnknownWhoStarts
export SaveHistory, SaveEverything, NameMap
export simulate, simulate!, reset!
export SessionInfo, MultipleSessionsInfo, NoEnvironment, predictive_sessions
export AbstractSessionParametersModel,
    NoParameters,
    FixedSessionParameters,
    FixedParametersBySession,
    IdenticalSessionParameters,
    IndependentSessionParameters
export AbstractDownstreamModel, NoDownstreamModel
export NoParallelisation, ParallelSessions
export turing_behavioural_session,
    turing_multiple_behavioural_sessions, turing_full_model, turing_catch_invalid_parameters
export session_values_chain, session_parameters_chain, predict_session_values

#What users call with the package's name, as in CognitiveModeling.apply_model
public apply_model, initialize_state, attribute_names
public prepare_state_for_saving, default_parameters, stopping_criterion
public turing_apply_model,
    turing_initialize_state, turing_session_parameters_model, turing_downstream_model
public ModelHistory, n_sessions, session_index

end
