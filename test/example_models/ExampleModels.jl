#Example models that the test suites share: cognitive models and environments which each have one feature beside the
#baseline they vary, the distributions they give in each suite, the log likelihoods of their outputs by hand, and each
#cognitive model with the environment it runs with. A runner includes this file and loads the module with
#`using .ExampleModels`, and the Turing.jl suites include turing_methods.jl after it. The distributions written by hand and log_density aren't exported, since tests of their own define the same
#names, so they are reached as ExampleModels.BernoulliByHand
module ExampleModels

using CognitiveModeling
using CognitiveModeling: timestep_model, initialize_state, attribute_names
using Random: AbstractRNG, Xoshiro

include("distributions.jl")
include("cognitive_models.jl")
include("environments.jl")
include("example_pairs.jl")

#The kinds of distributions, and the functions that give the distributions of a kind
export ByHand, Means
export bernoulli, bernoulli_logit, normal, categorical, symbol_choice, independent_normals
export choice_and_time, reward_and_cue

#The cognitive models
export Baseline, WithoutParameters, WithoutStates, WithoutInputs, WithoutOutputs
export ManyAttributes,
    VectorAttributes, NamedTupleAttributes, ContinuousOutput, SymbolOutput
export WithSymbolParameter, WithDiscreteParameter, WithDefaultParameters, WithDomains
export WithStoppingCriterion, DependsOnPreviousOutput, UsesTimestepSize
export InitialStateFromParameter, SampledInitialState, SamplingWithinTimestep
export OutputNotEveryTimestep, WithInvalidParameters
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
export timestep_model_with_latents

#Each cognitive model with the environment it runs with, and helpers to simulate them and to read their data
export EXAMPLE_PAIRS, PAIRS_WITH_MEANS, pair_named, simulate_pair
export inputs_received, timesteps_of, sizes_of_timesteps, columns, WHO_STARTS

end
