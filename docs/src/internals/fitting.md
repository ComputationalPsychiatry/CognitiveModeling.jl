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

## Preparing sessions

What runs once, when the sessions and the parts of a full model are prepared.

### Preparing the data of a session

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/preparing_sessions/prepare_session_data.jl"]
```

### Splitting a table into sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/preparing_sessions/split_table_into_sessions.jl"]
```

### Including the environment in a session, or leaving it out

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/preparing_sessions/include_environment.jl"]
```

### Checking the identifiers of the sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/preparing_sessions/check_session_identifiers.jl"]
```

## Running sessions

What an inference backend's model of the sessions calls.

### Reading what a session is set up with

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/running_sessions/read_session_setup.jl"]
```

### Reading the data of one timestep

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/running_sessions/read_timestep_data.jl"]
```

### Values to condition on

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/running_sessions/to_condition_on.jl"]
```

### Saving what a fitted session returns

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/running_sessions/save_fitted_session_trajectories.jl"]
```

## Displaying sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/display_sessions.jl"]
```

## Reading tables with Tables.jl

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["extensions/tables/column_table.jl"]
```
