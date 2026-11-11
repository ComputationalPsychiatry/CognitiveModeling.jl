# API

What users work with, by topic:

- [Defining models](defining_models.md): the types that a cognitive model or an environment
  model is written with, and the functions it implements.
- [Simulation](simulation.md): running models, alone or together.
- [Saving histories](saving_histories.md): what a simulation or a fitted session saves of
  each model.
- [Fitting](fitting.md): the sessions to fit, how the parameters of several sessions are
  given, and what runs after them.
- [Fitting with Turing.jl](fitting_with_turing.md): the Turing.jl models of the sessions, and
  reading what they saved from a chain.

The functions and types that are only used inside the package are under [Internals](@ref).

```@docs
CognitiveModeling
```

```@index
Pages = ["defining_models.md", "simulation.md", "saving_histories.md", "fitting.md", "fitting_with_turing.md"]
```
