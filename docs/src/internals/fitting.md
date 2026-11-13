# Internals of fitting

```@index
Pages = ["fitting.md"]
```

## Sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/types.jl"]
```

## Session parameter models

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = [
    "inference/session_parameter_models/supertype.jl",
    "inference/session_parameter_models/no_parameters.jl",
    "inference/session_parameter_models/fixed_session_parameters.jl",
    "inference/session_parameter_models/fixed_parameters_by_session.jl",
    "inference/session_parameter_models/identical_session_parameters.jl",
    "inference/session_parameter_models/independent_session_parameters.jl",
    "inference/session_parameter_models/grouped_hierarchical_session_parameters.jl",
    "inference/session_parameter_models/composite_session_parameters.jl",
]
```

## Preparing a full model

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/prepare_full_model.jl"]
```

## Utility functions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions.jl"]
```

## Reading tables with Tables.jl

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["extensions/tables/column_table.jl"]
```
