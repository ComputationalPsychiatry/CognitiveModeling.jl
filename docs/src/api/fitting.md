# Fitting

The sessions to fit, with their data, models and settings, and what runs after them, which
are the same for every inference backend. A backend fits several sessions with a full model
in three parts: a session parameter model gives or samples the parameters of every session,
for the cognitive model and for the environment model, as on the page
[Session parameter models](session_parameter_models.md), the sessions run with them, and a
downstream model can use the trajectories the sessions saved. Each backend has a page of its
own, with the models it makes and what a model can add for it:
[Fitting with Turing.jl](fitting_with_turing.md).

```@index
Pages = ["fitting.md"]
```

## Sessions and settings

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/types.jl"]
Filter = t -> t !== AbstractSessionParametersModel
```

## Preparing a full model

This is what an inference backend calls before it makes its full model. It is public, but
not exported.

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/utility_functions/preparing_models/prepare_full_model.jl"]
```

## Reading the sessions

A session is identified in three ways: by its order among the sessions, by its identifier,
which is the value of its session column, or a `NamedTuple` of the values of several, and
by its symbol name, a `Symbol` made from the identifier, which the chains of a fit name its
entries after. These functions are public, but not exported.

```@docs
CognitiveModeling.n_sessions
CognitiveModeling.session_index
CognitiveModeling.session_id_to_symbol
CognitiveModeling.session_column_values
```

## Workflow and postprocessing

What is done with an inference backend's model after it is made. Each backend adds the
methods for its own models, and its page describes which models they take.

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["inference/workflow_and_postprocessing/predict_session_trajectories.jl"]
```
