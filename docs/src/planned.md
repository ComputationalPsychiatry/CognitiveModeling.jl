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

## Plots

- Plots of a simulation, a history or a fitted session over time. The entry before the first
  timestep is timestep 0, at time 0, and its size is `BeforeFirstTimestep()`. The time of a
  later timestep is the sum of the sizes of the timesteps up to it.
