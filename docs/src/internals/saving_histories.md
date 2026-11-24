# Internals of saving histories

```@index
Pages = ["saving_histories.md"]
```

## Types

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/types.jl"]
```

## Utility functions

### Reading the selection of a save history

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/utility_functions/read_selection.jl"]
```

### Selecting the values to save

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/utility_functions/select_values_to_save.jl"]
```

### Checking the names a save history selects

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/utility_functions/check_selection_names.jl"]
```

### The sizes of the timesteps that are saved

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/utility_functions/timestep_sizes.jl"]
```

### Storing what is saved

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["model_history/utility_functions/storage.jl"]
```

### Displaying what is selected and what was saved

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = [
    "model_history/utility_functions/display_save_histories.jl",
    "model_history/utility_functions/display_session_trajectories.jl",
]
```
