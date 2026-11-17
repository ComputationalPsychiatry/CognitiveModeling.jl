# Parameter models

How a full model gives the parameters of every session, for the cognitive model and for the
environment model, which is the same for every inference backend. The sessions themselves
are on the page [Fitting](fitting.md), and each backend's page says how its models sample
the parameters: [Fitting with Turing.jl](fitting_with_turing.md).

```@index
Pages = ["parameter_models.md"]
```

## Models

```@docs
AbstractParametersModel
```

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = [
    "inference/parameter_models/no_parameters.jl",
    "inference/parameter_models/fixed_parameters.jl",
    "inference/parameter_models/sessionwise_fixed_parameters.jl",
    "inference/parameter_models/identical_parameters.jl",
    "inference/parameter_models/independent_parameters.jl",
    "inference/parameter_models/hierarchical_grouped_parameters.jl",
    "inference/parameter_models/composite_parameters.jl",
]
```

## Regressions

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/parameter_models/hierarchical_regression_parameters.jl"]
```

## Functions that a parameter model can implement

A parameter model of one's own can add methods of these. They are public, but not
exported.

```@docs
CognitiveModeling.parameter_names
CognitiveModeling.estimates_parameters
CognitiveModeling.prepare_parameters_model
CognitiveModeling.parameter_value_ranges
```

## Functions that an inference backend can implement

An inference backend can add methods of this for the distributions it takes, so that the
priors of a full model are checked against the domains of the parameters before a fit. It
is public, but not exported.

```@docs
CognitiveModeling.distribution_support
```
