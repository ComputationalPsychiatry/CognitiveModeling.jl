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
session_trajectories = get_session_trajectories(model, chain; save_cognitive_trajectories = SaveTrajectories(; states = :expected_value))
parameters = get_session_parameters(model, chain)

#The same from a model of one's own around the sessions, which returns what they return
parameters = get_session_parameters(returned(own_model, chain), sessions_info)
```

### Why the package has its own decondition and condition

Turing.jl's `decondition` and `condition` work on the variables of a model: `condition`
gives a variable a value, so that the model scores the value instead of sampling it, and
`decondition` takes the value away again. The models of the package don't have the data of
their sessions as variables. They take it as arguments, in a [`SessionInfo`](@ref) or a
[`MultipleSessionsInfo`](@ref), because the data does more than give values to score:

- The models receive it on each timestep, as observations and previous actions.
- It decides how each session runs: its number of timesteps, which values are imputed,
  whether the environment runs, and where the stopping criterion is checked.
- It is checked before a fit, against the names and the domains of the models.
- The observations of a session without an environment have no distribution, so they can
  only be data.
- Data in arguments is scored all at once, which makes each evaluation several times faster
  than values that are looked up by name.

Turing.jl's `decondition` therefore leaves the data as it is: it can only free variables,
and the data aren't variables. [`decondition_sessions`](@ref) turns the data into variables:
it puts [`SimulatedMissingValue`](@ref) in place of every action, and of every observation
when the sessions have an environment, so that the model samples each of them. Turing.jl's
`sample` with `Prior()` and its `predict` then simulate the data, as for any model. The
deconditioned sessions are named apart from the fitted ones, so that `predict` draws every
variable within them again.

[`condition_sessions`](@ref) goes the other way: it puts values of those variables back in
the data. Turing.jl's `condition` works on the same variables too, and gives the same
posterior for most models, but the values then stay variables. Each is looked up by name on
every evaluation, which is slower; a value with a misspelled name is left out without an
error; and with a stopping criterion, a session ends where the criterion is met at the
parameters of each evaluation, before its last value or after it. With `condition_sessions`
the model is the one that the same data in a table gives: as fast, checked in the same way,
with an error for a value that has no place in the data, and with each session ending after
its last value.

Together they allow prior and posterior predictive checks and parameter recoveries with
Turing.jl's own functions, on the models that the package makes, without writing the
simulated data to a table first.

### Predictive checks

With Turing.jl loaded, [`predict_session_trajectories`](@ref) takes a model made by
[`full_model`](@ref), [`multiple_sessions_model`](@ref) or [`session_model`](@ref) with
`TuringBackend()`, and a chain sampled from it.
It returns the simulated trajectories as a `FlexiChain`, as
[`get_session_trajectories`](@ref) makes it. For a prior predictive check, give a chain
sampled from the prior, such as `sample(model, Prior(), 1000)`.

```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_trajectories(model, chain; save_cognitive_trajectories = SaveTrajectories(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```

[`decondition_sessions`](@ref) takes these models too, and returns a copy whose sessions are
deconditioned, since the models take the data of their sessions as arguments, which
Turing.jl's `decondition` leaves as they are. Turing.jl's own functions then simulate the
data: `sample` with `Prior()` from the prior, and `predict` with the parameters of each
sample of a chain. They return the simulated actions and observations as variables, such as
`simulated_sessions[1].timesteps[3].choice.simulated`, where
[`predict_session_trajectories`](@ref) returns the trajectories that the sessions save. The
deconditioned sessions are named `simulated_sessions[i]`, apart from the fitted sessions,
`sessions[i]`, and a single session has the prefix `simulated_session`, so that `predict`
draws every variable within them again: every action and observation, also those that the
fit imputed, and what a model samples of its own, at its start or within a timestep.
Turing.jl's `condition` and `decondition` work on these models as on any other, also for a
variable within the sessions, such as a simulated action.

```julia
prior_predictions = sample(decondition_sessions(model), Prior(), 1000)
predictions = predict(decondition_sessions(model), chain)
```

### Parameter recovery

[`condition_sessions`](@ref) takes these models too, with values of their variables in the
collections that Turing.jl's `condition` takes them in: a `Dict` from `VarName`s to values,
or the `VarNamedTuple` that `rand(model)` gives. It returns a copy whose sessions have the
values of their actions and observations as data, such as that of
`simulated_sessions[1].timesteps[3].choice.simulated`, and a value of any other variable is
an error. `DynamicPPL.subset` with `@varname(simulated_sessions)`, or with
`@varname(simulated_session)` for a single session, leaves out the parameters of a draw from
the prior of the deconditioned model, so that the copy is fitted to the simulated data, as
in a parameter recovery. The sessions of the copy are named `sessions[i]` again once no
value of theirs is simulated, and a single session keeps its prefix. Each session of the
copy ends after its last timestep with a value, so a simulation that a
[`CognitiveModeling.stopping_criterion`](@ref) ended is fitted with the timesteps it had.
Turing.jl's `condition` on the same values gives the same posterior for models without a
stopping criterion of their own, as the tutorial
[Predictive checks and parameter recovery](../generated/predictive_checks_and_parameter_recovery.md)
shows, which also compares the two ways. With one, it keeps the timesteps after the last
value, and the session ends where the criterion is met at the parameters of each evaluation:
before its last value, which leaves out the values after it with no error, or after it, with
a variable for every action and observation that follows.

```julia
deconditioned_model = decondition_sessions(model)
simulated = rand(deconditioned_model)
recovery_model = condition_sessions(deconditioned_model, DynamicPPL.subset(simulated, [@varname(simulated_sessions)]))
recovered = sample(recovery_model, NUTS(), 1000)
```

[`condition_sessions`](@ref) also takes the values as pairs, as Turing.jl's `condition`
does, and the data in the forms the sessions are made from: a table of several sessions,
which it reads with the column settings that the sessions were read with, such as one
written from simulations, or the observations and the actions of a single session, as
[`SessionInfo`](@ref) takes them. A table must have the same sessions, in the same order,
since what the model was prepared with for each session must still fit.

```julia
recovery_model = condition_sessions(deconditioned_model, @varname(simulated_sessions[1].timesteps[1].choice.simulated) => true)
recovery_model = condition_sessions(model, simulated_table)
single_recovery_model = condition_sessions(single_session_model, (; reward = rewards), (; choice = choices))
```

## Functions that a model can implement for Turing.jl

A model can add methods of these to change how it runs when it is fitted with Turing.jl,
and must when it samples anything of its own, within a timestep or when it starts. They are
public, but not exported.

```@docs
CognitiveModeling.turing_timestep_model
CognitiveModeling.turing_initialize_state
```

## Variables that only some samples have

Turing.jl's samplers, and its `maximum_a_posteriori` and `maximum_likelihood`, take the
variables of a model from one draw from the prior. They make it with Julia's default random
number generator, whatever `initial_params` they are given. A sample of a model of the
package can have fewer variables than another in two ways.

A session can end early, when its parameters turn invalid, or when a stopping criterion is
met while its data continues. The sample then has a log density of `-Inf`, and it doesn't
have what would have been sampled after the timestep where the session ended: the actions
and observations that later timesteps impute, what a model samples within a later timestep,
and the variables of the downstream model, which doesn't run. A sampler rejects such
samples, so every sample in the chain has every variable. But when the first draw ends
early, the fit throws a `KeyError` that names a variable the draw doesn't have, such as
`KeyError: key slope not found`, even when `initial_params` are valid. When the first draw
doesn't end early but the parameters that the fit starts from do, a sampler throws an
`ArgumentError` which says that some indices in the output vector were not set, and
`maximum_a_posteriori` and `maximum_likelihood` throw an error which says that some values
are missing. Valid `initial_params` prevent those. These errors happen less often the more
seldom the priors give parameters that end a session early, and `Random.seed!` sets the
generator that makes the first draw. A model in which nothing is sampled after a session
ends has none of them.

A stopping criterion that is met already before the first timestep of a session, with its
initial states and the parameters of a draw, is an error instead, which stops the fit: when
the first draw gives such parameters, whatever `initial_params` are given, and when a sampler
proposes them later. Priors under which the criterion isn't met before the first timestep
prevent it.

`maximum_a_posteriori` and `maximum_likelihood` don't look for parameters to start from
whose log density is finite, as the samplers do. Without valid `initial_params`, they can
return a mode with a log density of `-Inf`, without an error, for any model whose priors
give parameters that end a session early.

A model can also sample a variable within a timestep on some timesteps only. This is no
problem when the data or the number of the timestep chooses those timesteps, as when an
action that is missing in the data is imputed: every sample then has the variable on the
same timesteps. When the parameters choose them, directly or through the state, samples with
valid parameters have different variables, which Turing.jl's samplers don't allow for, and
don't check. Sampling then either fails to start, with the same `KeyError`, or gives a wrong
posterior without an error. In a sample that doesn't use the variable, nothing holds its
value, so the sampler moves it freely, far from where its prior puts it. A sample that uses
it again then has a low density, so the sampler stays among the samples that don't use it.
In a test where a parameter should be above 0.5 in half of the samples, and a variable was
sampled only then, the parameter was above 0.5 in at most 1 of 400. Instead, sample such a
variable on every timestep, and use it only where it is needed, as for noise that is added
to the expected reward `Vₜ` only when it is above 0.6:

```julia
noise ~ Normal(0, 0.1)
if Vₜ > 0.6
    Vₜ += noise
end
```

A chain sampled with `Prior()`, and the predictions of `predict`, have `missing` for a
variable in the samples that don't have it.

## Fitting several sessions

These are what the parts of [`CognitiveModeling.turing_full_model`](@ref) are written with,
beside the types and functions on the pages [Fitting](fitting.md) and
[Parameter models](parameter_models.md). They are public, but not exported.
In a Turing.jl model of several sessions, session `i` of
[`CognitiveModeling.session_index`](@ref) has its variables under `sessions[i]`.

```@docs
CognitiveModeling.turing_parameters_model
CognitiveModeling.turing_downstream_model
```
