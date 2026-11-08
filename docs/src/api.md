# API

The functions and types that are only used inside the package are on the page
[Internals](@ref).

```@index
Pages = ["api.md"]
```

## Public

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["CognitiveModeling.jl", "defining_models/types.jl", "environments/types.jl", "environments/common_rl_environment.jl", "simulation/save_history.jl", "simulation/types.jl", "simulation/simulate.jl", "inference/types.jl", "inference/turing_models.jl", "inference/chains.jl"]
```

## Defining models and environments

These are what a model or an environment is written with. They are public, but not exported,
so use them with the package's name, as in `CognitiveModeling.apply_model`.

### Functions that a model must implement

```@docs
CognitiveModeling.apply_model(::CognitiveModeling.AbstractCognitiveModel, ::Any)
CognitiveModeling.initialize_state(::CognitiveModeling.AbstractCognitiveModel, ::Any)
CognitiveModeling.attribute_names(::CognitiveModeling.AbstractCognitiveModel)
```

### Functions that an environment model must implement

```@docs
CognitiveModeling.apply_model(::CognitiveModeling.AbstractEnvironmentModel, ::Any)
CognitiveModeling.initialize_state(::CognitiveModeling.AbstractEnvironmentModel, ::Any)
CognitiveModeling.attribute_names(::CognitiveModeling.AbstractEnvironmentModel)
```

### Functions that a model or an environment model can implement

```@docs
CognitiveModeling.prepare_state_for_saving
CognitiveModeling.default_parameters
CognitiveModeling.turing_apply_model
CognitiveModeling.turing_initialize_state
```

### Functions that an environment model can implement

```@docs
CognitiveModeling.stopping_criterion
```

## Fitting several sessions

These are what the parts of [`turing_full_model`](@ref) are written with, beside the types
on the public list. They are public, but not exported.

```@docs
CognitiveModeling.turing_population_model
CognitiveModeling.turing_downstream_model
CognitiveModeling.n_sessions
CognitiveModeling.session_index
```

### Display

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["defining_models/utility_functions.jl", "environments/utility_functions.jl", "inference/utility_functions.jl"]
Filter = object -> object === Base.show
```
