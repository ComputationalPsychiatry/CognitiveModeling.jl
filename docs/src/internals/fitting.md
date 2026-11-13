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

### Preparing the data of a session

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/prepare_session_data.jl"]
```

### Splitting a table into sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/split_table_into_sessions.jl"]
```

### Reading what a session is set up with

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/read_session_setup.jl"]
```

### Values to condition on

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/to_condition_on.jl"]
```

### Saving what a fitted session returns

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/save_fitted_session_trajectories.jl"]
```

### Displaying sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/display_sessions.jl"]
```

## Reading tables with Tables.jl

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["extensions/tables/column_table.jl"]
```
