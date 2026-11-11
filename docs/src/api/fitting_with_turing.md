# Fitting with Turing.jl

The Turing.jl models of the sessions on the page [Fitting](fitting.md), what a model can add
for them, and reading what they saved from a chain.

```@index
Pages = ["fitting_with_turing.md"]
```

## Turing.jl models

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/turing_models.jl"]
```

## Postprocessing

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/postprocessing.jl"]
```

## Functions that a model can implement for Turing.jl

A model can add methods of these to change how it runs when it is fitted with Turing.jl,
and must when it samples anything of its own, within a timestep or when it starts. They are
public, but not exported.

```@docs
CognitiveModeling.turing_apply_model
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
