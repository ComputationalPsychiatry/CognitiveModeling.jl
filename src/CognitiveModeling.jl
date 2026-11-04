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
include("simulation/types.jl")
include("simulation/simulate.jl")
include("inference/functions_to_extend.jl")

#What users work with directly
export AbstractCognitiveModel, AbstractEnvironmentModel
export NoObservation, NoAction
export ModelInput, ModelOutput, ModelAttributes
export CognitiveAgent, Environment, AgentStarts, EnvironmentStarts
export simulate, simulate!

#What users call with the package's name, as in CognitiveModeling.AgentHistory
public AgentHistory

end
