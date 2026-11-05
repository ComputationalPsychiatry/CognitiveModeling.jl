"""
Computational modeling of cognition and behaviour.
"""
module CognitiveModeling

using Random: AbstractRNG, default_rng

include("defining_models/types.jl")
include("defining_models/functions_to_extend.jl")
include("defining_models/utility_functions.jl")
include("environments/types.jl")
include("environments/functions_to_extend.jl")
include("environments/utility_functions.jl")
include("environments/common_rl_environment.jl")
include("simulation/types.jl")
include("simulation/simulate.jl")
include("inference/types.jl")
include("inference/functions_to_extend.jl")
include("inference/utility_functions.jl")
include("inference/turing_models.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel
export NoObservation, NoAction, MissingObservation, MissingAction
export ModelInput, ModelOutput, ModelAttributes, InvalidParametersError
export EnvironmentInput, EnvironmentOutput, CommonRLEnvironment
export CognitiveAgent, Environment, AgentStarts, EnvironmentStarts
export SaveHistory, SaveEverything
export simulate, simulate!, turing_single_session

#What users call with the package's name, as in CognitiveModeling.apply_model
public apply_model, initialize_state, attribute_names
public prepare_state_for_saving, default_parameters, turing_apply_model
public AgentHistory, EnvironmentHistory

end
