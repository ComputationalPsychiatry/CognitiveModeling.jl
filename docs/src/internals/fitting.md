# Internals of fitting

The internals of the sessions and of reading them from a table. Those of parameter
models and of the utility functions are on pages of their own:
[Parameter models](parameter_models.md),
[Preparing models](preparing_models.md), [Within models](within_models.md) and
[Workflow and postprocessing](workflow_and_postprocessing.md).

```@index
Pages = ["fitting.md"]
```

## Sessions

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["inference/types.jl"]
```

## Reading tables with Tables.jl

```@autodocs
Modules = [CognitiveModeling]
Public = false
Pages = ["extensions/tables/column_table.jl"]
```
