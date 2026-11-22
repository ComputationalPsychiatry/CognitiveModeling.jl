module CognitiveModelingTuringExt

using CognitiveModeling
using CognitiveModeling:
    AbstractTimestepModel,
    timestep_model,
    initialize_state,
    attribute_names,
    NoEnvironment,
    get_session_trajectories,
    get_session_parameters,
    with_save_history,
    with_data,
    with_common_element_types,
    cut_after_last_timestep_with_data,
    simulates_values,
    value_passed_on,
    LatentValueMarker
using CognitiveModeling:
    turing_timestep_model,
    turing_initialize_state,
    turing_parameters_model,
    turing_downstream_model,
    turing_behavioral_session,
    turing_multiple_behavioral_sessions,
    turing_full_model
using CognitiveModeling:
    merge_default_parameters,
    check_match_attribute_names,
    check_parameters_in_domains,
    check_model_output,
    map_names
using CognitiveModeling:
    check_environment_parameters,
    action_names_to_condition_on,
    observation_names_to_condition_on,
    find_outputs_to_condition_on,
    n_sessions,
    has_environment,
    remake_session_without_environment,
    include_environments_if_estimated,
    estimates_parameters,
    collect_parameters_in_session_order,
    prepare_full_model,
    as_parameters_model,
    parameter_names,
    merge_parameters_of_parts,
    part_names,
    build_family_distribution
using CognitiveModeling:
    initialize_output_distributions_for_conditioning,
    store_output_distributions_for_conditioning,
    select_row,
    build_named_entry,
    has_missing_value
using CognitiveModeling:
    select_timestep_values_to_save,
    select_initial_timestep_values_to_save,
    map_saved,
    initialize_all_storage,
    store_timestep_values!,
    saved_trajectories,
    with_empty_environment_trajectories,
    timestep_sizes_until
using CognitiveModeling:
    stopping_criterion, check_stopping_criterion_not_met, last_timestep_with_data
using CognitiveModeling:
    parameters_are_valid, combine_parameters_valid, parameters_valid_in_every_session
using CognitiveModeling:
    select_names,
    saves_initial_timestep_values,
    get_timestep_size,
    initialize_storage,
    store_value!
using Random: AbstractRNG, default_rng
using Turing
using Turing: DynamicPPL
using Turing.DynamicPPL: VarName

include("extension_defaults.jl")
#The full model, made from its parts, the Turing.jl models of the full model, the sessions, a session and a timestep, each in a file named after it, and the helpers they use, in files named after the models they serve
include("full_model.jl")
include("behavioral_models/full_model.jl")
include("behavioral_models/multiple_behavioral_sessions.jl")
include("behavioral_models/behavioral_session.jl")
include("behavioral_models/behavioral_timestep.jl")
include("behavioral_models/impute_missing_values.jl")
include("utility_functions/multiple_behavioral_sessions.jl")
include("utility_functions/behavioral_session.jl")

#The Turing.jl models of the parameter models, which give or sample the parameters of every session, each
#in a file named as the file of its type in the package. The same kinds serve the parameters of the cognitive model and
#those of the environment model, and each returns one vector for each parameter, with a value for every session, which
#the helpers in their own file split from an array when the parameter is an array
include("utility_functions/parameter_models.jl")
include("parameter_models/no_parameters.jl")
include("parameter_models/fixed_parameters.jl")
include("parameter_models/sessionwise_fixed_parameters.jl")
include("parameter_models/identical_parameters.jl")
include("parameter_models/independent_parameters.jl")
include("parameter_models/hierarchical_grouped_parameters.jl")
include("parameter_models/hierarchical_regression_parameters.jl")
include("parameter_models/composite_parameters.jl")

#Reading the trajectories and the parameters of fitted sessions from a chain, simulating the sessions again, and taking their data out of them and putting it back
include("workflow_and_postprocessing/get_session_trajectories.jl")
include("workflow_and_postprocessing/predict_session_trajectories.jl")
include("workflow_and_postprocessing/conditioning_sessions.jl")

end
