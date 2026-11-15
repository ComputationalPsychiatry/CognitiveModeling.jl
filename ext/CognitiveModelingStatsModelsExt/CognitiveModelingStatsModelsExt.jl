module CognitiveModelingStatsModelsExt

using CognitiveModeling
using CognitiveModeling: with_prior
using StatsModels:
    StatsModels,
    FormulaTerm,
    Term,
    ConstantTerm,
    InteractionTerm,
    FunctionTerm,
    DummyCoding,
    EffectsCoding,
    CategoricalTerm,
    ContinuousTerm

#The regression of a parameter of a HierarchicalRegressionSessionParameters written as a formula, in a file named as the
#file of the type in the package
include("hierarchical_regression_session_parameters.jl")

end
