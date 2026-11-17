module CognitiveModelingTuringExt

using CognitiveModeling
using CognitiveModeling:
    AbstractTimestepModel,
    run_timestep,
    initialize_state,
    attribute_names,
    NoEnvironment,
    session_trajectories_chain,
    session_parameters_chain
using CognitiveModeling:
    turing_run_timestep,
    turing_initialize_state,
    turing_parameters_model,
    turing_downstream_model,
    turing_catch_invalid_parameters
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
    sessions_save_anything,
    collect_parameters_in_session_order,
    prepare_full_model,
    as_parameters_model,
    prepare_parameters_model,
    parameter_names,
    merge_parameters_of_parts,
    part_names
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
    with_empty_environment_entry,
    timestep_sizes_until
using CognitiveModeling:
    stopping_criterion, check_stopping_criterion_not_met, last_timestep_with_data
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
include("turing_behavioral_session.jl")

#The Turing.jl models of the parameter models, which give or sample the parameters of every session, each
#in a file named as the file of its type in the package. The same kinds serve the parameters of the cognitive model and
#those of the environment model, and each returns one vector for each parameter, with a value for every session
include("parameter_models/no_parameters.jl")
include("parameter_models/fixed_parameters.jl")
include("parameter_models/sessionwise_fixed_parameters.jl")
include("parameter_models/identical_parameters.jl")
include("parameter_models/independent_parameters.jl")
include("parameter_models/hierarchical_grouped_parameters.jl")
include("parameter_models/hierarchical_regression_parameters.jl")
include("parameter_models/composite_parameters.jl")

include("session_trajectories_chain.jl")

end
