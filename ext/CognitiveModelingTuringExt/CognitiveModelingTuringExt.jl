module CognitiveModelingTuringExt

using CognitiveModeling
using CognitiveModeling:
    AbstractTimestepModel,
    apply_timestep_model,
    initialize_state,
    attribute_names,
    describe_model,
    NoEnvironment,
    session_trajectories_chain,
    session_parameters_chain
using CognitiveModeling:
    turing_apply_timestep_model,
    turing_initialize_state,
    turing_session_parameters_model,
    turing_downstream_model,
    turing_catch_invalid_parameters
using CognitiveModeling:
    merge_default_parameters,
    check_match_attribute_names,
    check_parameter_domains,
    parameter_domains,
    check_model_output,
    map_names
using CognitiveModeling:
    check_environment_parameters,
    action_names_to_condition_on,
    observation_names_to_condition_on,
    find_outputs_to_condition_on,
    n_sessions,
    has_environment,
    without_environment,
    sessions_for_environment_parameters,
    estimates_parameters,
    sessions_save_anything,
    parameters_by_session,
    prepare_full_model,
    prepare_parameters_model,
    parameter_names,
    merge_parameters_of_parts
using CognitiveModeling:
    initialize_distributions,
    store_distributions,
    select_row,
    build_named_entry,
    has_missing_value
using CognitiveModeling:
    saved_session_values,
    initial_session_values,
    map_saved,
    initialize_all_storage,
    store_all_values!,
    saved_trajectories,
    with_empty_environment_entry,
    timestep_sizes_until
using CognitiveModeling:
    stopping_criterion, check_stopping_criterion_not_met, last_timestep_with_data
using CognitiveModeling:
    select_names, saves_initial_values, get_timestep_size, initialize_storage, store_value!
using Random: AbstractRNG, default_rng
using Turing
using Turing: DynamicPPL
using Turing.DynamicPPL: VarName

include("extension_defaults.jl")
include("turing_behavioral_session.jl")

#The Turing.jl models of the session parameter models, which give or sample the parameters of every session, each
#in a file named as the file of its type in the package. The same kinds serve the parameters of the cognitive model and
#those of the environment model, and each returns one vector for each parameter, with a value for every session
include("session_parameter_models/no_parameters.jl")
include("session_parameter_models/fixed_session_parameters.jl")
include("session_parameter_models/fixed_parameters_by_session.jl")
include("session_parameter_models/identical_session_parameters.jl")
include("session_parameter_models/independent_session_parameters.jl")
include("session_parameter_models/grouped_hierarchical_session_parameters.jl")
include("session_parameter_models/composite_session_parameters.jl")

include("session_trajectories_chain.jl")

end
