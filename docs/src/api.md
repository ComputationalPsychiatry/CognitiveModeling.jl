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
Pages = ["CognitiveModeling.jl", "defining_models/types.jl", "environments/common_rl_environment.jl", "simulation/save_history.jl", "simulation/types.jl", "simulation/simulate.jl", "inference/types.jl", "inference/turing_models.jl", "inference/chains.jl"]
```

## Defining models

These are what a cognitive model or an environment model is written with, since the two are
defined in the same way. They are public, but not exported, so use them with the package's
name, as in `CognitiveModeling.apply_model`.

### Functions that a model must implement

```@docs
CognitiveModeling.apply_model
CognitiveModeling.initialize_state
CognitiveModeling.attribute_names
```

### Functions that a model can implement

```@docs
CognitiveModeling.prepare_state_for_saving
CognitiveModeling.default_parameters
CognitiveModeling.stopping_criterion
CognitiveModeling.turing_apply_model
CognitiveModeling.turing_initialize_state
```

## Fitting several sessions

These are what the parts of [`turing_full_model`](@ref) are written with, beside the types
on the public list. They are public, but not exported.

```@docs
CognitiveModeling.turing_session_parameters_model
CognitiveModeling.turing_downstream_model
CognitiveModeling.n_sessions
CognitiveModeling.session_index
```

### Display

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["defining_models/utility_functions.jl", "inference/utility_functions.jl"]
Filter = object -> object === Base.show
```
