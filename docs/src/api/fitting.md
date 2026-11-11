# Fitting

The sessions to fit, with their data, models and settings, how the parameters of several
sessions are given, and what runs after them, which are the same for every inference
backend. A backend fits several sessions with a full model in three parts: a session parameter
model gives or samples the parameters of every session, for the cognitive model and for the
environment model, the sessions run with them, and a downstream model can use the
trajectories the sessions saved. Each backend has a page of its own, with the models it makes and what a
model can add for it: [Fitting with Turing.jl](fitting_with_turing.md).

```@index
Pages = ["fitting.md"]
```

## Sessions and settings

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/types.jl"]
```

## Session parameter models

```@autodocs
Modules = [CognitiveModeling]
Private = false
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

## Functions that a session parameter model can implement

A session parameter model of one's own can add methods of these. They are public, but not
exported.

```@docs
CognitiveModeling.parameter_names
CognitiveModeling.estimates_parameters
CognitiveModeling.prepare_parameters_model
```

## Preparing a full model

This is what an inference backend calls before it makes its full model. It is public, but
not exported.

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/prepare_full_model.jl"]
```

## Reading the sessions

These are public, but not exported.

```@docs
CognitiveModeling.n_sessions
CognitiveModeling.session_index
CognitiveModeling.session_name
CognitiveModeling.session_column_values
```
