# Session parameter models

How a full model gives the parameters of every session, for the cognitive model and for the
environment model, which is the same for every inference backend. The sessions themselves
are on the page [Fitting](fitting.md), and each backend's page says how its models sample
the parameters: [Fitting with Turing.jl](fitting_with_turing.md).

```@index
Pages = ["session_parameter_models.md"]
```

## Models

```@docs
AbstractSessionParametersModel
```

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = [
    "inference/session_parameter_models/no_session_parameters.jl",
    "inference/session_parameter_models/fixed_session_parameters.jl",
    "inference/session_parameter_models/fixed_session_parameters_by_session.jl",
    "inference/session_parameter_models/identical_session_parameters.jl",
    "inference/session_parameter_models/independent_session_parameters.jl",
    "inference/session_parameter_models/grouped_hierarchical_session_parameters.jl",
    "inference/session_parameter_models/composite_session_parameters.jl",
]
```

## Regressions

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/session_parameter_models/hierarchical_regression_session_parameters.jl"]
```

## Functions that a session parameter model can implement

A session parameter model of one's own can add methods of these. They are public, but not
exported.

```@docs
CognitiveModeling.parameter_names
CognitiveModeling.estimates_parameters
CognitiveModeling.prepare_parameters_model
```
