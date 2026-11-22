# Fitting with Turing.jl

The Turing.jl models of the sessions on the page [Fitting](fitting.md), what a model can add
for them, and reading the trajectories they saved from a chain.

```@index
Pages = ["fitting_with_turing.md"]
```

## Turing.jl models

[`full_model`](@ref), [`session_model`](@ref) and [`multiple_sessions_model`](@ref) make
these models when they are given `TuringBackend()`. The models are public, but not exported.

```@autodocs
Modules = [CognitiveModeling]
Private = false
Pages = ["extensions/turing/turing_models.jl"]
```

## Postprocessing

With Turing.jl loaded, which loads FlexiChains.jl, [`get_session_trajectories`](@ref) and
[`get_session_parameters`](@ref) take a model made by [`full_model`](@ref),
[`multiple_sessions_model`](@ref) or [`session_model`](@ref) with `TuringBackend()`, and the
`FlexiChain` that `sample` gives for it. A model of one's own around the sessions gives them the values that
`returned(model, chain)` gives instead, with the `MultipleSessionsInfo` of several sessions.

```julia
chain = sample(model, NUTS(), 1000)
session_trajectories = get_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; states = :expected_value))
parameters = get_session_parameters(model, chain)

#The same from a model of one's own around the sessions, which returns what they return
parameters = get_session_parameters(returned(own_model, chain), sessions_info)
```

### Predictive checks

With Turing.jl loaded, [`predict_session_trajectories`](@ref) takes a model made by
[`full_model`](@ref), [`multiple_sessions_model`](@ref) or [`session_model`](@ref) with
`TuringBackend()`, and a chain sampled from it.
It returns the simulated trajectories as a `FlexiChain`, as
[`get_session_trajectories`](@ref) makes it. For a prior predictive check, give a chain
sampled from the prior, such as `sample(model, Prior(), 1000)`.

```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```

[`decondition_sessions`](@ref) takes these models too, and returns a copy whose sessions are
deconditioned, since the models take the data of their sessions as arguments, which
Turing.jl's `decondition` leaves as they are. Turing.jl's own functions then simulate the
data: `sample` with `Prior()` from the prior, and `predict` with the parameters of each
sample of a chain. They return the simulated actions and observations as variables, such as
`session[1].timestep[3].choice.simulated`, where [`predict_session_trajectories`](@ref)
returns the trajectories that the sessions save. These variables are named apart from those
that a fit imputes, such as `session[1].timestep[3].choice.imputed`, so `predict` simulates
every action and observation again, also those that the fit imputed. It fails when the fit
imputed values at different timesteps in different sessions, where
[`predict_session_trajectories`](@ref) works. Turing.jl's `condition` and `decondition` work
on these models as on any other, also for a variable within the sessions, such as a
simulated action.

```julia
prior_predictions = sample(decondition_sessions(model), Prior(), 1000)
predictions = predict(decondition_sessions(model), chain)
```

### Parameter recovery

[`condition_sessions`](@ref) takes these models too, with values of their variables in the
collections that Turing.jl's `condition` takes them in: a `Dict` from `VarName`s to values,
or the `VarNamedTuple` that `rand(model)` gives. It returns a copy whose sessions have the
values of their actions and observations as data, such as that of
`session[1].timestep[3].choice.simulated`, and a value of any other variable is an error.
`DynamicPPL.subset` leaves out the parameters of a draw from the prior of the deconditioned
model, under `@varname(session)` for several sessions and `@varname(timestep)` for a single
one, so that the copy is fitted to the simulated data, as in a parameter recovery.
Turing.jl's `condition` on the same values gives the same posterior, as the tutorial
[Predictive checks and parameter
recovery](../generated/predictive_checks_and_parameter_recovery.md)
shows, which also compares the two ways.

```julia
deconditioned_model = decondition_sessions(model)
simulated = rand(deconditioned_model)
recovery_model = condition_sessions(deconditioned_model, DynamicPPL.subset(simulated, [@varname(session)]))
recovered = sample(recovery_model, NUTS(), 1000)
```

## Functions that a model can implement for Turing.jl

A model can add methods of these to change how it runs when it is fitted with Turing.jl,
and must when it samples anything of its own, within a timestep or when it starts. They are
public, but not exported.

```@docs
CognitiveModeling.turing_timestep_model
CognitiveModeling.turing_initialize_state
```

## Fitting several sessions

These are what the parts of [`CognitiveModeling.turing_full_model`](@ref) are written with,
beside the types and functions on the pages [Fitting](fitting.md) and
[Parameter models](parameter_models.md). They are public, but not exported.
In a Turing.jl model of several sessions, session `i` of
[`CognitiveModeling.session_index`](@ref) has its variables under `session[i]`.

```@docs
CognitiveModeling.turing_parameters_model
CognitiveModeling.turing_downstream_model
```
