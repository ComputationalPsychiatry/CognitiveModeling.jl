# Fitting with Turing.jl

The Turing.jl models of the sessions on the page [Fitting](fitting.md), what a model can add
for them, and reading the trajectories they saved from a chain.

```@index
Pages = ["fitting_with_turing.md"]
```

## Turing.jl models

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["extensions/turing/turing_models.jl"]
```

## Postprocessing

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["extensions/flexichains/session_trajectories_chain.jl"]
```

### Predictive checks

With Turing.jl loaded, [`predict_session_trajectories`](@ref) takes a model made by
[`turing_full_model`](@ref), [`turing_multiple_behavioral_sessions`](@ref) or
[`turing_behavioral_session`](@ref), or such a model inside
[`turing_catch_invalid_parameters`](@ref), and a chain sampled from it. It returns the
simulated trajectories as a `FlexiChain`, as [`session_trajectories_chain`](@ref) makes it.
For a prior predictive check, give a chain sampled from the prior, such as
`sample(model, Prior(), 1000)`.

```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```

## Functions that a model can implement for Turing.jl

A model can add methods of these to change how it runs when it is fitted with Turing.jl,
and must when it samples anything of its own, within a timestep or when it starts. They are
public, but not exported.

```@docs
CognitiveModeling.turing_apply_timestep_model
CognitiveModeling.turing_initialize_state
```

## Fitting several sessions

These are what the parts of [`turing_full_model`](@ref) are written with, beside the types
and functions on the page [Fitting](fitting.md). They are public, but not exported. In a
Turing.jl model of several sessions, session `i` of
[`CognitiveModeling.session_index`](@ref) has its variables under `session[i]`.

```@docs
CognitiveModeling.turing_session_parameters_model
CognitiveModeling.turing_downstream_model
```
