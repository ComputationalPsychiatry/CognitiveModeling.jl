"""
Computational modeling of cognition and behavior.
"""
module CognitiveModeling

using Random: AbstractRNG, default_rng

include("defining_models/types.jl")
include("defining_models/functions_to_extend.jl")
include("defining_models/utility_functions.jl")
include("environments/common_rl_environment.jl")
include("saving_histories/save_history.jl")
include("saving_histories/storage.jl")
include("simulation/types.jl")
include("simulation/simulate.jl")
include("inference/types.jl")
include("inference/session_parameter_models/supertype.jl")
include("inference/session_parameter_models/no_parameters.jl")
include("inference/session_parameter_models/fixed_session_parameters.jl")
include("inference/session_parameter_models/fixed_parameters_by_session.jl")
include("inference/session_parameter_models/identical_session_parameters.jl")
include("inference/session_parameter_models/independent_session_parameters.jl")
include("inference/session_parameter_models/grouped_hierarchical_session_parameters.jl")
include("inference/session_parameter_models/composite_session_parameters.jl")
include("inference/prepare_full_model.jl")
include("inference/functions_to_extend.jl")
include("inference/utility_functions.jl")
include("inference/turing_models.jl")
include("inference/postprocessing.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel
export AbstractNoValue, NoValue, BeforeFirstTimestep
export MissingValue, UnimputedMissingValue
export ModelContext, ModelOutput, ModelAttributes, Deterministic, InvalidParametersError
export CommonRLEnvironment
export InstantiatedModel, AgentStarts, EnvironmentStarts, UnknownWhoStarts
export SaveHistory, SaveEverything, NameMap
export simulate, simulate!, reset!
export SessionInfo, MultipleSessionsInfo, NoEnvironment, prepare_for_prediction
export AbstractSessionParametersModel,
    NoParameters,
    FixedSessionParameters,
    FixedParametersBySession,
    IdenticalSessionParameters,
    IndependentSessionParameters,
    GroupedHierarchicalSessionParameters,
    HierarchicalParameter,
    CompositeSessionParameters
export AbstractDownstreamModel, NoDownstreamModel
export NoParallelization, ParallelSessions
export turing_behavioral_session,
    turing_multiple_behavioral_sessions, turing_full_model, turing_catch_invalid_parameters
export session_trajectories_chain, session_parameters_chain, predict_session_trajectories

#What users call with the package's name, as in CognitiveModeling.apply_timestep_model
public apply_timestep_model, initialize_state, attribute_names
public prepare_state_for_saving, default_parameters, stopping_criterion
public parameter_domains, discrete_parameter_names
public turing_apply_timestep_model,
    turing_initialize_state, turing_session_parameters_model, turing_downstream_model
public ModelHistory, n_sessions, session_index, session_name, session_column_values
public parameter_names, estimates_parameters, prepare_parameters_model, prepare_full_model

end
