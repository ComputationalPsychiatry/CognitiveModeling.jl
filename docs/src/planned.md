# Planned tutorials and interfaces

This page lists what is planned and not written yet. The list will grow.

## Cognitive models for tutorials

- Models of metacognition, such as HMeta-d.
- The models of hBayesDM, probably every one of them.
- The main models for the Iowa Gambling Task, such as PVL-Delta and ORL.
- Different variants of the Rescorla-Wagner model, and models like it.
- Evidence accumulation models, such as attention-based models.
- Bayesian models made with RxInfer.jl.
- Active inference models, which a separate package is being made for.
- An example of a cognitive model which uses Turing.jl to select its actions.
- Models with discrete parameters, to show how they are fitted. The page
  [Discrete parameters](generated/discrete_parameters.md) is a sketch of it.
- Examples with models from control theory as cognitive models.
- Models from game theory, such as win-stay-lose-shift, and models that take a payoff
  matrix, as an input or as a parameter.
- The k-ToM model of recursive theory of mind.
- An example of inverse reinforcement learning, if it can be done: the reward function
  that an agent acts on, fitted from its actions.

## Interfaces

- BayesianRegressionModels.jl, which could also give regressions on a parameter with
  unordered categories, with a regression for each category but one.
- Simulation-based inference, where a neural network is trained on simulations to estimate
  the parameters: NeuralEstimators.jl, to test or to make an interface with, and perhaps the
  Python package sbi too. A simulation of a cognitive model is what both need.
- StanBlocks.jl, potentially, as another way to write the probabilistic model around a
  session.
- GlobalSensitivity.jl, for Sobol sensitivity analyses of cognitive models: how much each
  parameter, alone and together with the others, changes what a model does.
- Cognitive models in continuous time, probably with DifferentialEquations.jl, whose states
  change between the timesteps as well as on them.
- An MCMCChains.jl extension, so that [`get_session_trajectories`](@ref),
  [`get_session_parameters`](@ref) and [`predict_session_trajectories`](@ref) also take and
  return its chains, beside the `FlexiChain`s of FlexiChains.jl that they take now.

## Defining models

- A macro that defines a cognitive model or an environment in one command, and writes the
  methods of the functions to extend where it can deduce them, such as `attribute_names`
  from the names the model uses.
- Whether a trial needs to be distinguished from a timestep, and whether a model can have
  nested time scales, such as evidence that accumulates over several timesteps within each
  trial and learning between the trials.
- Relations between parameters, checked beside the domain of each parameter: a function of
  all the parameters of a model, such as one that the lower of two thresholds is below the
  upper one, or that the sum of two rates is at most 1, or that a vector has one element for
  each option that another parameter gives.

## Fitting

- Dyadic interaction, with another agent as the environment of the agent that is fitted.
  When the other agent's actions are recorded, each agent is a session of its own whose
  observations are the other's actions. When they aren't, the other agent is the environment
  that fills them in.
- Session identifiers in the names of the variables of a fitted session, instead of
  `sessions[i]`, if a way is found that doesn't compile the session once per identifier.
- A convenient way to fit a single session, with the full model or a variant of it.
- Choosing a sampler for each parameter from whether it is discrete or continuous.
- Summing out a discrete parameter with few values instead of sampling it, so that a sampler
  which uses gradients can fit the whole model, perhaps with one of the Julia packages for
  marginalizing discrete variables.
- Mixture models and model selection within the fitted model, in the style of
  Michael D. Lee: a latent variable for each session, or each group, chooses which of
  several cognitive models generated its behavior, and its posterior compares the models.
- Leave-one-session-out cross-validation, which estimates how well a model predicts the
  behavior of a new participant or session. Summing the log-likelihoods of each session and
  smoothing them with PSIS, as for single data points, gives unreliable estimates when each
  session has parameters of its own, since leaving out a session's data changes its
  parameters the most. They must instead be integrated over their population distribution,
  by fitting again without each session, or by importance sampling over the session's
  parameters. The parameter models know which parameters are a session's, so this would be
  the package's, with PosteriorStats.jl for the smoothing.
- Several sets of experimental data in one hierarchical model, such as different experiments
  or tasks done by the same participants, which share the parameters they have in common.
- Imputing a missing part of a value, such as the time of a response whose choice was
  recorded, or one element of a vector, as a latent variable of its own, for when a later
  timestep uses the value. Leaving a missing part out already works: a distribution of one's
  own, which splits the distribution of the value into its parts, scores the parts that are
  known, as the tutorial on SequentialSamplingModels.jl shows.
- One format for the trajectories and the parameters of fitted sessions, which
  [`get_session_trajectories`](@ref) and [`get_session_parameters`](@ref) return whatever
  inference backend fitted them, so that the package's own plots take what any backend
  gives: the `FlexiChain` of FlexiChains.jl that they return now, or a type of the package's
  own.

## Performance

- Benchmarks which compare the speed of simulating and fitting with other inference
  backends and packages, such as Stan.
- Extended benchmarks with full data sets, beside the small ones in the `benchmark` folder,
  which run on every push.
- Compatibility with KernelAbstractions.jl, and with GPUs in general, through the packages
  of JuliaGPU, such as CUDA.jl, AMDGPU.jl, Metal.jl and oneAPI.jl.
- A version of the simulation for GPUs, which stores what it saves without vectors that
  grow, and samples with a random number generator of the device.

## Plots

- A plot of the parameters of every fitted session, with the averages of their groups.
- A plot of the trajectories of different fitted sessions over the timesteps.
- Plots of the trajectories of a simulation, of an instantiated model or of a fitted session
  over time. The entry before the first timestep is timestep 0, at time 0, and its size is
  `BeforeFirstTimestep()`. The time of a later timestep is the sum of the sizes of the
  timesteps up to it.
