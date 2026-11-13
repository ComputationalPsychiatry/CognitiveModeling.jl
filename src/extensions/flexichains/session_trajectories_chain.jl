## The trajectories of fitted sessions, as a chain, made in the FlexiChains extension ##

"""
    session_trajectories_chain(results, sessions_info::MultipleSessionsInfo;
                               iter_indices=1:size(results, 1),
                               chain_indices=1:size(results, 2))
    session_trajectories_chain(results; iter_indices=1:size(results, 1),
                               chain_indices=1:size(results, 2))
    session_trajectories_chain(model, chain; cognitive_save_history=SaveHistory(),
                               environment_save_history=SaveHistory())

Return the trajectories that fitted sessions saved, for every sample, as a `FlexiChain` from
FlexiChains.jl with the same samples as the fit, so that they can be summarized and plotted
like the chain itself. The trajectories of a session are what changes over its timesteps:
the states, the observations and the actions of its models, and the distributions they
gave. This needs FlexiChains.jl to be loaded, which Turing.jl loads. The names of the
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
- [`turing_multiple_behavioral_sessions`](@ref), given with its `MultipleSessionsInfo`,
  named in the same way.
- [`turing_behavioral_session`](@ref), given without a session info, with no session
  identifier in the names, such as `expected_value`.
In a model of one's own around these, that model must return what they return, for
`returned` to give it.

The trajectories each session saved, as its [`SaveHistory`](@ref)s selected, are one entry
for each name, with the session's identifier first. The cognitive model's values are named
after the attribute alone, since a model uses a name for one attribute only: a state as
`P1.expected_value`, an observation as `P1.reward` and an action as `P1.choice`. An action
distribution is named like its action, so it is under its kind, as
`P1.action_distributions.choice`. The environment's values are under `environment`, in the
same way: `P1.environment.level` for a state, `P1.environment.reward` for an observation it
gave, `P1.environment.choice` for an action it received and
`P1.environment.observation_distributions.reward`. Each entry holds a vector over the
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
fastest when its sessions save nothing, and their trajectories are then read afterwards in
this way. Selecting anything in `environment_save_history` makes every session run its
environment, see [`SessionInfo`](@ref).

# Examples
```julia
chain = sample(model, NUTS(), 1000)
session_trajectories = session_trajectories_chain(model, chain; cognitive_save_history = SaveHistory(; states = :expected_value))

#The expected values of participant P2 over the timesteps, for every sample
session_trajectories[@varname(P2.expected_value)]

#The parameters of the sessions, in the same chain
everything = merge(session_trajectories, session_parameters_chain(model, chain))
```
"""
function session_trajectories_chain end

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
parameters are those the session parameter models gave each session, so a session parameter
model other than [`IndependentSessionParameters`](@ref) gives parameters that aren't in the
chain itself, such as one value for every session from [`IdenticalSessionParameters`](@ref),
and values from [`FixedSessionParameters`](@ref) that were never sampled. The second form,
in the Turing extension, takes a model made by `turing_full_model` and a chain sampled from
it, and numbers the samples as the chain does. The result merges with the chain of
[`session_trajectories_chain`](@ref), which has the same samples, with `merge`.

# Examples
```julia
chain = sample(model, NUTS(), 1000)
parameters = session_parameters_chain(model, chain)

#The learning rate of participant P1, for every sample
parameters[@varname(P1.learning_rate)]
```
"""
function session_parameters_chain end
