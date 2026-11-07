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

## Interfaces

- BayesianRegressionModels.jl.
- Simulation-based inference, where a neural network is trained on simulations to estimate
  the parameters: NeuralEstimators.jl, to test or to make an interface with, and perhaps the
  Python package sbi too. A simulation of a cognitive model is what both need.
- StanBlocks.jl, potentially, as another way to write the probabilistic model around a
  session.

## Fitting

- Dyadic interaction, with another agent as the environment of the agent that is fitted.
  When the other agent's actions are recorded, each agent is a session of its own whose
  observations are the other's actions. When they aren't, the other agent is the environment
  that fills them in.
- Session identifiers in the names of the variables of a fitted session, instead of
  `session[i]`, if a way is found that doesn't compile the session once per identifier.

## Plots

- Plots of a simulation, a history or a fitted session over time. The entry before the first
  timestep is timestep 0, at time 0, and its size is `BeforeFirstTimestep()`. The time of a
  later timestep is the sum of the sizes of the timesteps up to it.
