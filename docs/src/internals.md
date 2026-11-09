# Internals

These are only used inside the package, and can change without notice. What users work with
is on the page [API](@ref).

```@index
Pages = ["internals.md"]
```

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["defining_models/types.jl", "defining_models/utility_functions.jl", "simulation/save_history.jl", "simulation/types.jl", "simulation/simulate.jl", "inference/types.jl", "inference/utility_functions.jl"]
Filter = object -> object !== Base.show
```
