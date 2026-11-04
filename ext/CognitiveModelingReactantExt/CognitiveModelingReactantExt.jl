module CognitiveModelingReactantExt

using CognitiveModeling
using Reactant

#Placeholder for a version of simulate with a traced loop. The normal simulate compiles as it is, but Reactant unrolls
#its loop, so compile times grow quickly with the number of timesteps. A traced loop can't save to a vector of
#NamedTuples, so this version would save to one traced array per number, and return a different shape

end
