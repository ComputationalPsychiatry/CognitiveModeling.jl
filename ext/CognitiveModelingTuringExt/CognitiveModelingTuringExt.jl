module CognitiveModelingTuringExt

using CognitiveModeling
using CognitiveModeling:
    AbstractModel,
    apply_model,
    initialize_state,
    attribute_names,
    session_values_chain,
    session_parameters_chain
using CognitiveModeling:
    turing_apply_model,
    turing_initialize_state,
    turing_session_parameters_model,
    turing_downstream_model,
    turing_catch_invalid_parameters
using CognitiveModeling:
    merge_default_parameters, check_match_attribute_names, check_model_output, map_names
using CognitiveModeling:
    check_environment_parameters,
    recorded_action_names,
    recorded_observation_names,
    recorded_on_timestep,
    n_sessions,
    has_environment,
    without_environment,
    sessions_for_environment_parameters,
    samples_parameters,
    sessions_save_anything,
    parameters_by_session
using CognitiveModeling:
    initialize_distributions, store_distributions, row, entry, has_missing_value
using CognitiveModeling:
    saved_session_outputs,
    initial_session_outputs,
    initialize_session_storage,
    store_session_outputs!,
    session_saved_values,
    with_empty_environment_entry,
    saved_values_until,
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
include("turing_behavioural_session.jl")
include("session_parameters_models.jl")
include("session_values_chain.jl")

end
