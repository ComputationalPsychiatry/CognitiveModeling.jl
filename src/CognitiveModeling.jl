"""
Computational modeling of cognition and behavior.
"""
module CognitiveModeling

using Random: AbstractRNG, default_rng

include("defining_models/types.jl")
include("defining_models/functions_to_extend.jl")
include("defining_models/functions_to_call.jl")
include("defining_models/utility_functions/check_model_methods.jl")
include("defining_models/utility_functions/read_parameters_valid.jl")
include("defining_models/utility_functions/bundle_submodel_parameters.jl")
include("defining_models/utility_functions/collect_submodel_names.jl")
include("defining_models/utility_functions/collect_submodel_parameters.jl")
include("defining_models/utility_functions/compare_names.jl")
include("defining_models/utility_functions/check_attributes_against_names.jl")
include("defining_models/utility_functions/check_parameters_in_domains.jl")
include("defining_models/utility_functions/outputs_and_inputs_before_first_timestep.jl")
include("defining_models/utility_functions/display_models.jl")
include("model_history/types.jl")
include("model_history/utility_functions/read_selection.jl")
include("model_history/utility_functions/select_values_to_save.jl")
include("model_history/utility_functions/check_selection_names.jl")
include("model_history/utility_functions/timestep_sizes.jl")
include("model_history/utility_functions/storage.jl")
include("model_history/utility_functions/display_save_histories.jl")
include("model_history/utility_functions/display_session_trajectories.jl")
include("simulation/types.jl")
include("simulation/simulate.jl")
include("simulation/mask_timesteps.jl")
include("simulation/fixed_observations_environment.jl")
include("simulation/display_name_maps_and_instantiated_models.jl")
include("inference/types.jl")
include("inference/parameter_models/no_parameters.jl")
include("inference/parameter_models/fixed_parameters.jl")
include("inference/parameter_models/sessionwise_fixed_parameters.jl")
include("inference/parameter_models/identical_parameters.jl")
include("inference/parameter_models/independent_parameters.jl")
include("inference/parameter_models/hierarchical_grouped_parameters.jl")
include("inference/parameter_models/hierarchical_regression_parameters.jl")
include("inference/parameter_models/composite_parameters.jl")
include("inference/parameter_models/functions_to_extend.jl")
include("inference/functions_to_call.jl")
include("inference/full_model.jl")
include("inference/utility_functions/preparing_models/prepare_full_model.jl")
include("inference/utility_functions/preparing_models/prepare_session_data.jl")
include("inference/utility_functions/preparing_models/split_table_into_sessions.jl")
include("inference/utility_functions/preparing_models/include_environment_in_session.jl")
include("inference/utility_functions/preparing_models/check_session_identifiers.jl")
include("inference/utility_functions/preparing_models/read_session_columns.jl")
include("inference/utility_functions/preparing_models/display_sessions.jl")
include("inference/utility_functions/within_models/read_session_setup.jl")
include("inference/utility_functions/within_models/read_timestep_data.jl")
include("inference/utility_functions/within_models/to_condition_on.jl")
include("inference/utility_functions/within_models/save_fitted_session_trajectories.jl")
include("inference/utility_functions/within_models/mask_fitted_timesteps.jl")
include("inference/workflow_and_postprocessing/get_session_trajectories.jl")
include("inference/workflow_and_postprocessing/predict_session_trajectories.jl")
include("inference/workflow_and_postprocessing/conditioning_sessions.jl")

#What the package has for each of its extensions, which add the methods: the types and functions users reach through
#the package, with their docstrings, and the helpers that call an extension
include("extensions/common_rl_interface/common_rl_environment.jl")
include("extensions/tables/column_table.jl")
include("extensions/turing/functions_to_extend.jl")
include("extensions/turing/turing_models.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel, AbstractSubmodel
export AbstractNoValue, NoValue, BeforeFirstTimestep
export ImputedMissingValue, UnimputedMissingValue, SimulatedMissingValue
export ModelContext, ModelOutput, ModelAttributes, Deterministic, ValidByDefault
export Domain, ParameterDomainError
export FixedObservationsEnvironment, CommonRLEnvironment
export InstantiatedModel, AgentStarts, EnvironmentStarts, UnknownWhoStarts
export SaveHistory, SaveEverything, SessionTrajectories, NameMap
export simulate_session, simulate_session!, simulate_timestep, simulate_timestep!, reset!
export SessionInfo, MultipleSessionsInfo, NoEnvironment
export decondition_sessions, condition_sessions
export AbstractParametersModel,
    NoParameters,
    FixedParameters,
    SessionwiseFixedParameters,
    IdenticalParameters,
    IndependentParameters,
    HierarchicalGroupedParameters,
    HierarchicalParameter,
    HierarchicalRegressionParameters,
    RegressionParameter,
    RegressionFamily,
    RegressionOutcome,
    Intercept,
    ContinuousPredictor,
    CategoricalPredictor,
    Interaction,
    VaryingEffects,
    DefaultRegressionPriors,
    DefaultRegressionSettings,
    CompositeParameters
export AbstractDownstreamModel, NoDownstreamModel
export SerialSessions, ParallelSessions
export AbstractInferenceBackend, TuringBackend
export full_model, session_model, multiple_sessions_model
export get_session_trajectories, get_session_parameters, predict_session_trajectories

#What users call with the package's name, as in CognitiveModeling.timestep_model
public timestep_model, initialize_state, attribute_names
public prepare_state_for_saving, default_parameters, stopping_criterion
public parameter_domains, input_domains, output_domains
public submodel_fields, submodels, uses_submodel_prefixes, get_attribute_names
public discrete_parameter_names, continuous_parameter_names
public turing_timestep_model,
    turing_initialize_state, turing_parameters_model, turing_downstream_model
public turing_behavioral_session, turing_multiple_behavioral_sessions, turing_full_model
public ModelHistory, n_sessions, session_index, session_id_to_symbol, session_column_values
public parameter_names, estimates_parameters, prepare_parameters_model, prepare_full_model

end
