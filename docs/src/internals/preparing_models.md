# Utility functions for preparing models

What runs outside an inference backend's model: preparing the sessions and the parts of a
full model once, before the model is made, and showing the sessions in the REPL.

```@index
Pages = ["preparing_models.md"]
```

## Preparing a full model

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/prepare_full_model.jl"]
```

## Preparing the data of a session

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/prepare_session_data.jl"]
```

## Splitting a table into sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/split_table_into_sessions.jl"]
```

## Including the environment in a session, or leaving it out

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/include_environment.jl"]
```

## Checking the identifiers of the sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/check_session_identifiers.jl"]
```

## Reading the columns that a session parameter model uses

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/read_session_columns.jl"]
```

## Displaying sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/display_sessions.jl"]
```
