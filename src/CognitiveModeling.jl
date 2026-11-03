"""
Computational modeling of cognition and behaviour.
"""
module CognitiveModeling

using Random: AbstractRNG, default_rng

include("defining_models/types.jl")
include("defining_models/functions_to_extend.jl")
include("defining_models/utility_functions.jl")
include("environments/types.jl")
include("simulation/types.jl")
include("simulation/simulate.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel
export NoObservation, NoAction
export CognitiveAgent, Environment, AgentStarts, EnvironmentStarts
export simulate, simulate!

#What users call with the package's name, as in CognitiveModeling.get_attributes
public get_attributes

#The functions that users extend, and ModelInput, ModelOutput, EnvironmentInput and EnvironmentOutput, aren't public
#for now. They are used with the package's name, as in CognitiveModeling.apply_model

end
