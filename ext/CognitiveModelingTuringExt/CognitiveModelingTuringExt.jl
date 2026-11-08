module CognitiveModelingTuringExt

using CognitiveModeling
using CognitiveModeling: AbstractModel, apply_model, initialize_state, session_values_chain
using CognitiveModeling: turing_apply_model, turing_initialize_state, turing_population_model, turing_downstream_model, turing_catch_invalid_parameters
using CognitiveModeling: merge_default_parameters, check_match_attribute_names, check_model_output, check_environment_output
using CognitiveModeling: check_environment_parameters, recorded_action_names, recorded_on_timestep, n_sessions
using CognitiveModeling: initialize_distributions, store_distributions, row, entry, has_missing_value
using CognitiveModeling: saved_session_outputs, initial_session_outputs, initialize_session_storage, store_session_outputs!
using CognitiveModeling: select_names, saves_initial_values, get_timestep_size, initialize_storage, store_output!
using Random: default_rng
using Turing
using Turing: DynamicPPL
using Turing.DynamicPPL: VarName

include("extension_defaults.jl")
include("turing_behavioural_session.jl")
include("population_models.jl")
include("session_values_chain.jl")

end
