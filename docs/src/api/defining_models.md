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
with the package's name, as in `CognitiveModeling.timestep_model`.

```@docs
CognitiveModeling.timestep_model
CognitiveModeling.initialize_state
CognitiveModeling.attribute_names
```

## Functions that a model can implement

```@docs
CognitiveModeling.prepare_state_for_saving
CognitiveModeling.default_parameters
CognitiveModeling.parameter_domains
CognitiveModeling.input_domains
CognitiveModeling.output_domains
CognitiveModeling.stopping_criterion
```

## Functions that a model with submodels implements

A model can hold submodels, each an [`AbstractSubmodel`](@ref), in its fields. These say
which fields hold them, and how the names of their parameters are given.

```@docs
CognitiveModeling.submodel_fields
CognitiveModeling.submodels
CognitiveModeling.uses_submodel_prefixes
```

## Reading a model's names and parameters

These read a model together with its submodels, where the functions that a model implements
give what is the model's own.

```@docs
CognitiveModeling.get_attribute_names
CognitiveModeling.get_parameter_names
CognitiveModeling.get_default_parameters
CognitiveModeling.get_parameter_domains
CognitiveModeling.discrete_parameter_names
CognitiveModeling.continuous_parameter_names
```

## Bundling the parameters of submodels

A model is given its parameters and those of its submodels in one flat `NamedTuple`, and
reads those of each submodel under the submodel's name. These go from one form to the other.

```@docs
CognitiveModeling.bundle_parameters
CognitiveModeling.flatten_parameters
```

## Environments from other packages

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["extensions/common_rl_interface/common_rl_environment.jl"]
```
