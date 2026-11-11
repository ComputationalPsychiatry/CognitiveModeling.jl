# Defining models

These are what a cognitive model or an environment model is written with, since the two are
defined in the same way.

```@index
Pages = ["defining_models.md"]
```

## Types

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["defining_models/types.jl"]
```

## Functions that a model must implement

These and the functions that a model can implement are public, but not exported, so use them
with the package's name, as in `CognitiveModeling.apply_model`.

```@docs
CognitiveModeling.apply_model
CognitiveModeling.initialize_state
CognitiveModeling.attribute_names
```

## Functions that a model can implement

```@docs
CognitiveModeling.prepare_state_for_saving
CognitiveModeling.default_parameters
CognitiveModeling.parameter_domains
CognitiveModeling.stopping_criterion
```

## Reading a model's parameters

```@docs
CognitiveModeling.discrete_parameter_names
```

## Environments from other packages

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["environments/common_rl_environment.jl"]
```
