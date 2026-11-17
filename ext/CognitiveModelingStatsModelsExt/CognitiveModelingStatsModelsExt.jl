module CognitiveModelingStatsModelsExt

using CognitiveModeling
using CognitiveModeling: with_prior, as_tuple
using StatsModels: FormulaTerm, Term, ConstantTerm, InteractionTerm, FunctionTerm

#The regression of a parameter of a HierarchicalRegressionParameters written as a formula, in a file named as the
#file of the type in the package
include("hierarchical_regression_parameters.jl")

end
