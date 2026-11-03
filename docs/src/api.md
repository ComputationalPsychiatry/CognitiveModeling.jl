# API

```@index
```

## Public

```@autodocs
Modules = [CognitiveModeling]
Private = false
```

## Defining models and environments

These are what a model or an environment is written with. They aren't public for now, so
use them with the package's name, as in `CognitiveModeling.apply_model`.

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["defining_models/types.jl", "defining_models/functions_to_extend.jl", "defining_models/utility_functions.jl", "environments/types.jl"]
```

## Internals

These are only used inside the package, and can change without notice.

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["simulation/simulate.jl"]
```
