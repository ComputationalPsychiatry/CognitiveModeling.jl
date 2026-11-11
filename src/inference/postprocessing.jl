## What fitted sessions saved, as a chain, made in the FlexiChains extension ##

"""
    session_values_chain(results, sessions_info::MultipleSessionsInfo;
                         iter_indices=1:size(results, 1), chain_indices=1:size(results, 2))
    session_values_chain(results; iter_indices=1:size(results, 1),
                         chain_indices=1:size(results, 2))
    session_values_chain(model, chain; cognitive_save_history=SaveHistory(),
                         environment_save_history=SaveHistory())

Return what fitted sessions saved, for every sample, as a `FlexiChain` from FlexiChains.jl
with the same samples as the fit, so that it can be summarised and plotted like the chain
itself. This needs FlexiChains.jl to be loaded, which Turing.jl loads. The names of the
entries are the `VarName`s that FlexiChains.jl uses, written as
`@varname(P1.expected_value)` with FlexiChains.jl 0.6, which Turing.jl uses, and as
`@vn(P1.expected_value)` with 0.7.

`results` is a matrix with one element for each sample, iterations down and chains across,
as `returned(model, chain)` gives it in Turing.jl. Its elements are what one of three models
returns:
- [`turing_full_model`](@ref), given with its `MultipleSessionsInfo`. The entries of each
  session are named after its identifier, such as `P1.expected_value`. The parameters of the
  sessions are a chain of their own, see [`session_parameters_chain`](@ref), and what a
  downstream model returned isn't included.
- [`turing_multiple_behavioural_sessions`](@ref), given with its `MultipleSessionsInfo`,
  named in the same way.
- [`turing_behavioural_session`](@ref), given without a session info, with no session
  identifier in the names, such as `expected_value`.
In a model of one's own around these, that model must return what they return, for
`returned` to give it.

What each session saved, as its [`SaveHistory`](@ref)s selected, is one entry for each name,
with the session's identifier first. The cognitive model's values are named after the
attribute alone, since a model uses a name for one attribute only: a state as
`P1.expected_value`, an observation as `P1.reward` and an action as `P1.choice`. An action
distribution is named like its action, so it is under its kind, as
`P1.action_distributions.choice`. The environment's values are under `environment`, in the
same way: `P1.environment.level` for a state, `P1.environment.reward` for an observation it
gave, `P1.environment.choice` for an action it received and
`P1.environment.observation_distributions.reward`. A saved state that isn't a `NamedTuple`
is one entry, `P1.states` or `P1.environment.states`. Each entry holds a vector over the
timesteps for every sample, which starts with timestep 0 when the initial values are saved,
so that element `t + 1` is timestep `t`, as everywhere in the package. The observations, the
actions and the distributions of timestep 0 are [`BeforeFirstTimestep`](@ref), so a summary
of those entries needs timestep 0 left out. An identifier from several session columns is
named by the name and the value of each column, as in `id_S1__block_2`, which
[`CognitiveModeling.session_name`](@ref) gives. `iter_indices` and `chain_indices` number
the samples as the chain that was fitted does.

The third form, in the Turing extension, takes a model made by one of the three functions,
or such a model inside [`turing_catch_invalid_parameters`](@ref), and a chain sampled from
it. It makes a copy of the model whose sessions save what the two save histories select,
gives it to `returned`, and makes the chain of the results, numbered as `chain` is. A fit is
fastest when its sessions save nothing, and what they saved is then read afterwards in this
way. Selecting anything in `environment_save_history` makes every session run its
environment, see [`SessionInfo`](@ref).

# Examples
```julia
chain = sample(model, NUTS(), 1000)
session_values = session_values_chain(model, chain; cognitive_save_history = SaveHistory(; states = :expected_value))

#The expected values of participant P2 over the timesteps, for every sample
session_values[@varname(P2.expected_value)]

#The parameters of the sessions, in the same chain
everything = merge(session_values, session_parameters_chain(model, chain))
```
"""
function session_values_chain end

"""
    session_parameters_chain(results, sessions_info::MultipleSessionsInfo;
                             iter_indices=1:size(results, 1),
                             chain_indices=1:size(results, 2))
    session_parameters_chain(model, chain)

Return the parameters of every fitted session, for every sample, as a `FlexiChain` with the
same samples as the fit, named after the session: `P1.learning_rate` for a parameter of the
cognitive model, and `P1.environment.mean_reward` for one of the environment model. This
needs FlexiChains.jl to be loaded, which Turing.jl loads.

`results` is a matrix with what [`turing_full_model`](@ref) returns for each sample, as
`returned(model, chain)` gives it, with the `MultipleSessionsInfo` of the model. The
parameters are those the parameters models gave each session, so a parameters model other
than [`IndependentSessionParameters`](@ref) gives parameters that aren't in the chain
itself, such as one value for every session from [`IdenticalSessionParameters`](@ref), and
values from [`FixedSessionParameters`](@ref) that were never sampled. The second form, in
the Turing extension, takes a model made by `turing_full_model` and a chain sampled from it,
and numbers the samples as the chain does. The result merges with the chain of
[`session_values_chain`](@ref), which has the same samples, with `merge`.

# Examples
```julia
chain = sample(model, NUTS(), 1000)
parameters = session_parameters_chain(model, chain)

#The learning rate of participant P1, for every sample
parameters[@varname(P1.learning_rate)]
```
"""
function session_parameters_chain end

"""
    predict_session_values([rng,] model, chain; cognitive_save_history=SaveHistory(),
                           environment_save_history=SaveHistory())

Simulate the sessions of `model` again with the parameters of every sample in `chain`, and
return what they saved as a `FlexiChain`, as [`session_values_chain`](@ref) makes it. This
is for prior and posterior predictive checks, and needs Turing.jl to be loaded.

`model` is made by [`turing_full_model`](@ref),
[`turing_multiple_behavioural_sessions`](@ref) or [`turing_behavioural_session`](@ref), or
is such a model inside [`turing_catch_invalid_parameters`](@ref). Its sessions are prepared
with [`prepare_for_prediction`](@ref), so that every action is simulated, and every
observation too when the sessions have an environment, and they save what the two save
histories select. The
model of those sessions then runs once for each sample, with the sample's parameters, and
any start that a model samples, and with every variable within the timesteps drawn again, as
data is in a predictive check: an action imputed in the fit is simulated like the others.
What the sessions return is kept. The sessions have at most the number of timesteps of their
data, and fewer when a [`stopping_criterion`](@ref) of a model is met, since no timestep of
a predictive session has data. For a prior predictive check, give a chain sampled from the
prior, such as `sample(model, Prior(), 1000)`.

# Examples
```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_values(model, chain; cognitive_save_history = SaveHistory(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```
"""
function predict_session_values end
