#The code that the test suites share, as a package which each suite's environment has: cognitive models and environments
#which each have one feature beside the baseline they vary, the distributions they give in each suite, the log
#likelihoods of their outputs by hand, each cognitive model with the environment it runs with, and the simulation and
#inference cases made from them. Its extension for Turing.jl, which loads with Turing.jl, adds the distributions of
#Distributions.jl, the priors of the inference cases and the Turing.jl methods of the models. The distributions written
#by hand and log_density aren't exported, since tests of their own define the same names, so they are reached as
#CommonTestFunctionality.BernoulliByHand
module CommonTestFunctionality

using CognitiveModeling
using CognitiveModeling: timestep_model, initialize_state, attribute_names
using Random: AbstractRNG, Xoshiro

include("distributions.jl")
include("cognitive_models.jl")
include("environments.jl")
include("model_pairs.jl")
include("helpers.jl")
include("simulation_cases.jl")
include("inference_cases.jl")

#The kinds of distributions, and the functions that give the distributions of a kind. The functions that give the priors
#of the inference cases, beta, log_normal, exponential and product_of, aren't exported, since only the cases use them
export ByHand, Means, DistributionsJl
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
export MODEL_PAIRS, PAIRS_WITH_MEANS, pair_named, simulate_pair
export inputs_received, timesteps_of, sizes_of_timesteps, columns, WHO_STARTS

#What the models sampled in a session of an inference backend, read by the names of its variables
export latents_from_variables, variables_of

#The simulation cases, and what the suites read of them
export simulation_cases, sum_of_reals, real_numbers, with_real_numbers

#The inference cases, their log densities by hand, and what the suites take from them
export session_cases, sessions_cases, full_model_cases
export session_log_density_by_hand, sessions_log_density_by_hand
export simulated_data, with_missing, renamed, environment_name, to_unit_interval

end
