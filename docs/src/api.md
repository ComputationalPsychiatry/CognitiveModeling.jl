# API

```@index
```

## Public

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["CognitiveModeling.jl", "defining_models/types.jl", "environments/types.jl", "environments/common_rl_environment.jl", "simulation/types.jl", "simulation/simulate.jl"]
```

## Defining models and environments

These are what a model or an environment is written with. They are public, but not exported,
so use them with the package's name, as in `CognitiveModeling.apply_model`.

### Functions that a model must implement

```@docs
CognitiveModeling.apply_model(::CognitiveModeling.AbstractCognitiveModel, ::Any)
CognitiveModeling.initialize_state(::CognitiveModeling.AbstractCognitiveModel, ::Any)
CognitiveModeling.attribute_names
```

### Functions that a model can implement

```@docs
CognitiveModeling.prepare_state_for_saving(::CognitiveModeling.AbstractCognitiveModel, ::Any)
CognitiveModeling.default_parameters
CognitiveModeling.turing_apply_model
```

### Functions that an environment model must implement

```@docs
CognitiveModeling.apply_model(::CognitiveModeling.AbstractEnvironmentModel, ::Any)
CognitiveModeling.initialize_state(::CognitiveModeling.AbstractEnvironmentModel, ::Any)
```

### Functions that an environment model can implement

```@docs
CognitiveModeling.prepare_state_for_saving(::CognitiveModeling.AbstractEnvironmentModel, ::Any)
```

### Display

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["defining_models/utility_functions.jl"]
```

## Internals

These are only used inside the package, and can change without notice.

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["simulation/simulate.jl"]
```
