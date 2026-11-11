# Fitting

The sessions to fit, with their data, models and settings, how the parameters of several
sessions are given, and what runs after them, which are the same for every inference
backend. A backend fits several sessions with a full model in three parts: a parameters
model gives or samples the parameters of every session, for the cognitive model and for the
environment model, the sessions run with them, and a downstream model can use what the
sessions saved. Each backend has a page of its own, with the models it makes and what a
model can add for it: [Fitting with Turing.jl](fitting_with_turing.md).

```@index
Pages = ["fitting.md"]
```

## Sessions, parameters and settings

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/types.jl"]
```

## Reading the sessions

These are public, but not exported.

```@docs
CognitiveModeling.n_sessions
CognitiveModeling.session_index
CognitiveModeling.session_name
```
