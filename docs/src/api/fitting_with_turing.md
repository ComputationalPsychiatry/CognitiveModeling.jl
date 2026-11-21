# Fitting with Turing.jl

The Turing.jl models of the sessions on the page [Fitting](fitting.md), what a model can add
for them, and reading the trajectories they saved from a chain.

```@index
Pages = ["fitting_with_turing.md"]
```

## Turing.jl models

[`full_model`](@ref), [`session_model`](@ref), [`multiple_sessions_model`](@ref) and
[`catch_invalid_parameters`](@ref) make these models when they are given `TuringBackend()`.
The models are public, but not exported.

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["extensions/turing/turing_models.jl"]
```

## Postprocessing

With Turing.jl loaded, which loads FlexiChains.jl, [`get_session_trajectories`](@ref) and
[`get_session_parameters`](@ref) take a model made by [`full_model`](@ref),
[`multiple_sessions_model`](@ref) or [`session_model`](@ref) with `TuringBackend()`, or
such a model inside [`catch_invalid_parameters`](@ref), and the `FlexiChain` that
`sample` gives for it. A model of one's own around the sessions gives them the values that
`returned(model, chain)` gives instead, with the `MultipleSessionsInfo` of several sessions.

```julia
chain = sample(model, NUTS(), 1000)
session_trajectories = get_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; states = :expected_value))
parameters = get_session_parameters(model, chain)

#The same from a model of one's own around the sessions, which returns what they return
parameters = get_session_parameters(returned(own_model, chain), sessions_info)
```

### Predictive checks

With Turing.jl loaded, [`predict_session_trajectories`](@ref) takes a model made by
[`full_model`](@ref), [`multiple_sessions_model`](@ref) or [`session_model`](@ref) with
`TuringBackend()`, or such a model inside [`catch_invalid_parameters`](@ref), and a chain
sampled from it.
It returns the simulated trajectories as a `FlexiChain`, as
[`get_session_trajectories`](@ref) makes it. For a prior predictive check, give a chain
sampled from the prior, such as `sample(model, Prior(), 1000)`.

```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```

Turing.jl's own `decondition` has a method for these models, which also deconditions their
sessions with [`decondition_sessions`](@ref), since the models take their data as arguments,
which Turing.jl's `decondition` otherwise leaves as they are. Turing.jl's own functions then
simulate the data: `sample` with `Prior()` from the prior, and `predict` with the parameters
of each sample of a chain. `predict` has a method for these models too, which takes only the
parameters, and any start that a model samples, from the chain, and simulates every variable
within the timesteps again, also an action that was imputed in the fit, where Turing.jl's
`predict` otherwise keeps the values that the chain has. It returns the simulated actions
and observations as variables, such as `session[1].timestep[3].choice.imputed`, where
[`predict_session_trajectories`](@ref) returns the trajectories that the sessions save.

```julia
prior_predictions = sample(decondition(model), Prior(), 1000)
predictions = predict(decondition(model), chain)
```

## Functions that a model can implement for Turing.jl

A model can add methods of these to change how it runs when it is fitted with Turing.jl,
and must when it samples anything of its own, within a timestep or when it starts. They are
public, but not exported.

```@docs
CognitiveModeling.turing_run_timestep
CognitiveModeling.turing_initialize_state
```

## Fitting several sessions

These are what the parts of [`CognitiveModeling.turing_full_model`](@ref) are written with,
beside the types and functions on the pages [Fitting](fitting.md) and
[Parameter models](parameter_models.md). They are public, but not exported.
In a Turing.jl model of several sessions, session `i` of
[`CognitiveModeling.session_index`](@ref) has its variables under `session[i]`.

```@docs
CognitiveModeling.turing_parameters_model
CognitiveModeling.turing_downstream_model
```
