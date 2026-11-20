#Example models that the test suites share: cognitive models and environments which each have one feature beside the
#baseline they vary, the distributions they give in each suite, and the log likelihoods of their outputs by hand. A runner
#includes this file and loads the module with `using .ExampleModels`, and the Turing.jl suites include turing_methods.jl
#after it. The distributions written by hand and log_density aren't exported, since tests of their own define the same
#names, so they are reached as ExampleModels.BernoulliByHand
module ExampleModels

using CognitiveModeling
using CognitiveModeling: run_timestep, initialize_state, attribute_names
using Random: AbstractRNG

include("distributions.jl")
include("cognitive_models.jl")
include("environments.jl")

#The kinds of distributions, and the functions that give the distributions of a kind
export ByHand, Means
export bernoulli, normal, categorical, symbol_choice, independent_normals
export choice_and_time, reward_and_cue

#The cognitive models
export Baseline, WithoutParameters, WithoutStates, WithoutInputs, WithoutOutputs
export ManyAttributes,
    VectorAttributes, NamedTupleAttributes, ContinuousOutput, SymbolOutput
export WithSymbolParameter, WithDiscreteParameter, WithDefaultParameters, WithDomains
export WithStoppingCriterion, DependsOnPreviousOutput, UsesTimestepSize
export InitialStateFromParameter, SampledInitialState, SamplingWithinTimestep
export OutputNotEveryTimestep, ThrowsInvalidParameters
export StateChangedInPlace, StateChangedInPlaceSavedWithDeepcopy, StateChangesType
export BaselineFamily, MethodsFromSupertype
export EveryFeatureFamily, EveryFeature, EveryFeatureWithoutSampling

#The environments
export BaselineEnvironment, WithoutStatesEnvironment, WithoutParametersEnvironment
export WithoutInputsEnvironment, WithoutOutputsEnvironment
export OutputNotEveryTimestepEnvironment, WithStoppingCriterionEnvironment
export NoOutputBeforeFirstInputEnvironment, SamplingWithinTimestepEnvironment
export OtherNamesEnvironment, ManyAttributesEnvironment, VectorAttributesEnvironment
export NamedTupleAttributesEnvironment, ContinuousInputEnvironment, SymbolInputEnvironment
export EveryFeatureEnvironment

#The log likelihoods by hand, and what the models sample of their own
export log_likelihood_by_hand, latents_from_states, log_density_of_latents_by_hand
export run_timestep_with_latents

end
