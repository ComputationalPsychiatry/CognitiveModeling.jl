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
    "inference/session_parameter_models/no_session_parameters.jl",
    "inference/session_parameter_models/fixed_session_parameters.jl",
    "inference/session_parameter_models/fixed_session_parameters_by_session.jl",
    "inference/session_parameter_models/identical_session_parameters.jl",
    "inference/session_parameter_models/independent_session_parameters.jl",
    "inference/session_parameter_models/grouped_hierarchical_session_parameters.jl",
    "inference/session_parameter_models/hierarchical_regression_session_parameters.jl",
    "inference/session_parameter_models/composite_session_parameters.jl",
]
```

## Utility functions

### Preparing models

What runs outside an inference backend's model: preparing the sessions and the parts of a
full model once, before the model is made, and showing the sessions in the REPL.

#### Preparing a full model

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/prepare_full_model.jl"]
```

#### Preparing the data of a session

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/prepare_session_data.jl"]
```

#### Splitting a table into sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/split_table_into_sessions.jl"]
```

#### Including the environment in a session, or leaving it out

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/include_environment.jl"]
```

#### Checking the identifiers of the sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/check_session_identifiers.jl"]
```

#### Reading the columns that a session parameter model uses

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/read_session_columns.jl"]
```

#### Displaying sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/display_sessions.jl"]
```

### Within models

What an inference backend's model of the sessions calls.

#### Reading what a session is set up with

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/within_models/read_session_setup.jl"]
```

#### Reading the data of one timestep

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/within_models/read_timestep_data.jl"]
```

#### Values to condition on

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/within_models/to_condition_on.jl"]
```

#### Saving what a fitted session returns

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/within_models/save_fitted_session_trajectories.jl"]
```

## Reading tables with Tables.jl

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["extensions/tables/column_table.jl"]
```
