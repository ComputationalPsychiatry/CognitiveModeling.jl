module CognitiveModelingFlexiChainsExt

using CognitiveModeling
using CognitiveModeling: MultipleSessionsInfo, session_id_to_symbol
using FlexiChains: FlexiChains, FlexiChain, Parameter, VarName

#The names of the entries are the VarName that FlexiChains.jl uses, made with the macro it exports. Version 0.7 uses the
#VarName of VarNames.jl, and 0.6, which Turing.jl needs until it moves to VarNames.jl, the one of AbstractPPL.jl
@static if pkgversion(FlexiChains) >= v"0.7"
    include("entry_names_varnames.jl")
else
    include("entry_names_abstractppl.jl")
end

include("session_trajectories_chain.jl")

end
