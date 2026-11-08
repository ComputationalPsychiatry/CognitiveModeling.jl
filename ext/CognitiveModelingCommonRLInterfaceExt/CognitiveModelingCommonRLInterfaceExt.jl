module CognitiveModelingCommonRLInterfaceExt

using CognitiveModeling
using CommonRLInterface:
    CommonRLInterface, reset!, observe, act!, terminated, clone, provided

include("common_rl_environment.jl")

end
