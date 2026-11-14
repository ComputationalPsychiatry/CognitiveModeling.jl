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
- Models with discrete parameters, to show how they are fitted.

## Interfaces

- BayesianRegressionModels.jl.
- Simulation-based inference, where a neural network is trained on simulations to estimate
  the parameters: NeuralEstimators.jl, to test or to make an interface with, and perhaps the
  Python package sbi too. A simulation of a cognitive model is what both need.
- StanBlocks.jl, potentially, as another way to write the probabilistic model around a
  session.
- GlobalSensitivity.jl, for Sobol sensitivity analyses of cognitive models: how much each
  parameter, alone and together with the others, changes what a model does.
- Cognitive models in continuous time, probably with DifferentialEquations.jl, whose states
  change between the timesteps as well as on them.

## Defining models

- A macro that defines a cognitive model or an environment in one command, and writes the
  methods of the functions to extend where it can deduce them, such as `attribute_names`
  from the names the model uses.

## Fitting

- Dyadic interaction, with another agent as the environment of the agent that is fitted.
  When the other agent's actions are recorded, each agent is a session of its own whose
  observations are the other's actions. When they aren't, the other agent is the environment
  that fills them in.
- Session identifiers in the names of the variables of a fitted session, instead of
  `session[i]`, if a way is found that doesn't compile the session once per identifier.
- A convenient way to fit a single session, with the full model or a variant of it.
- Choosing a sampler for each parameter from whether it is discrete or continuous.
- Mixture models and model selection within the fitted model, in the style of
  Michael D. Lee: a latent variable for each session, or each group, chooses which of
  several cognitive models generated its behavior, and its posterior compares the models.
- Several sets of experimental data in one hierarchical model, such as different experiments
  or tasks done by the same participants, which share the parameters they have in common.
- Formulas for a `HierarchicalRegression`, as an extension for StatsModels.jl, so that the
  regression of a parameter can be written as
  `@formula(learning_rate ~ 1 + age + diagnosis + (1 | participant))`.

## Plots

- A plot of the parameters of every fitted session, with the averages of their groups.
- A plot of the trajectories of different fitted sessions over the timesteps.
- Plots of a simulation, a history or a fitted session over time. The entry before the first
  timestep is timestep 0, at time 0, and its size is `BeforeFirstTimestep()`. The time of a
  later timestep is the sum of the sizes of the timesteps up to it.
