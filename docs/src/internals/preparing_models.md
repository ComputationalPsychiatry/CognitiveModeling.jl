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

## Checking what a parameter model can give against the domains of the parameters

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/check_given_values_in_domains.jl"]
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
Pages = ["inference/utility_functions/preparing_models/include_environment_in_session.jl"]
```

## Checking the data identifiers of the sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/utility_functions/preparing_models/check_session_identifiers.jl"]
```

## Reading the columns that a parameter model uses

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
