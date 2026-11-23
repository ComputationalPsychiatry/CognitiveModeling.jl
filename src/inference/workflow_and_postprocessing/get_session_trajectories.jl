## The trajectories and the parameters of fitted sessions, read from a fit ##

"""
    get_session_trajectories(returned_values, sessions_info::MultipleSessionsInfo;
                             iter_indices=1:size(returned_values, 1),
                             chain_indices=1:size(returned_values, 2))
    get_session_trajectories(returned_values; iter_indices=1:size(returned_values, 1),
                             chain_indices=1:size(returned_values, 2))
    get_session_trajectories(model, chain; cognitive_save_history=SaveHistory(),
                             environment_save_history=SaveHistory())

Return the trajectories that fitted sessions saved, for every sample, as a `FlexiChain` from
FlexiChains.jl with the same samples as the fit, so that they can be summarized and plotted
like the chain itself. The trajectories of a session are what changes over its timesteps:
the states, the observations and the actions of its models, and the distributions they
gave. This needs FlexiChains.jl to be loaded. The names of the entries are the `VarName`s
that FlexiChains.jl uses, written as `@varname(P1.expected_value)` with FlexiChains.jl 0.6,
and as `@vn(P1.expected_value)` with 0.7.

`returned_values` is a matrix with one element for each sample, iterations down and chains
across, with what an inference backend's model of the fitted sessions returned for that
sample. Its elements are what one of three models returns:
- The full model, given with its `MultipleSessionsInfo`. The entries of each session are
  named after its symbol name, such as `P1.expected_value`. The parameters of the sessions
  are a chain of their own, see [`get_session_parameters`](@ref), and what a downstream
  model returned isn't included.
- The model of several sessions, given with its `MultipleSessionsInfo`, named in the same
  way.
- The model of a single session, given without a session info, with no symbol name of a
  session in the names, such as `expected_value`.
A model of one's own around the model of several sessions or of a single session can return
what they return, and its returned values are then read in the same way.

The trajectories each session saved, as its [`SaveHistory`](@ref)s selected, are one entry
for each name, with the session's symbol name first. The cognitive model's values are named
after the attribute alone, since a model uses a name for one attribute only: a state as
`P1.expected_value`, an observation as `P1.reward` and an action as `P1.choice`. An action
distribution is named like its action, so it is under its kind, as
`P1.action_distributions.choice`. The environment's values are under `environment`, in the
same way: `P1.environment.level` for a state, `P1.environment.reward` for an observation it
gave, `P1.environment.choice` for an action it received and
`P1.environment.observation_distributions.reward`. The state of a submodel, which the
trajectories name under `submodel_states`, has an entry for each of its own states, named
after the submodel too, as `P1.perception.expected_value`. Each entry holds a vector over the
timesteps for every sample, which starts with timestep 0 when the initial timestep values
are saved, so that element `t + 1` is timestep `t`, as everywhere in the package. The
observations, the actions and the distributions of timestep 0 are
[`BeforeFirstTimestep`](@ref), so a summary of those entries needs timestep 0 left out. The
symbol name of a session with several session columns joins the name and the value of each
column, as in `id_S1__block_2`, as [`CognitiveModeling.session_id_to_symbol`](@ref) gives
it. When the models can set `parameters_valid` in their [`ModelOutput`](@ref), each session
also has an entry `parameters_valid`, as in `P1.parameters_valid`, which says whether its
parameters stayed valid in each sample; where they didn't, its trajectories end before the
timestep on which they turned invalid. `iter_indices` and `chain_indices` number the
samples as the chain that was fitted does.

The third form takes one of the three models and a chain sampled from it, a `FlexiChain`. It
runs a copy of the model whose sessions save what the two save histories select on every
sample, and makes the chain of the returned values, numbered as `chain` is. A fit is fastest
when its sessions save nothing, and their trajectories are then read afterwards in this way.
Selecting anything in `environment_save_history` makes every session run its environment,
see [`SessionInfo`](@ref). Each inference backend has a method of this form for its own
models, which needs its package to be loaded, and its documentation says which models it
takes.

# Examples
```julia
#The expected values of participant P2 over the timesteps, for every sample of a fit
session_trajectories = get_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; states = :expected_value))
session_trajectories[@varname(P2.expected_value)]

#The parameters of the sessions, in the same chain
everything = merge(session_trajectories, get_session_parameters(model, chain))
```
"""
function get_session_trajectories end

"""
    get_session_parameters(returned_values, sessions_info::MultipleSessionsInfo;
                           iter_indices=1:size(returned_values, 1),
                           chain_indices=1:size(returned_values, 2))
    get_session_parameters(returned_values; iter_indices=1:size(returned_values, 1),
                           chain_indices=1:size(returned_values, 2))
    get_session_parameters(model, chain)

Return the parameters of every fitted session, for every sample, as a `FlexiChain` with the
same samples as the fit, named after the session: `P1.learning_rate` for a parameter of the
cognitive model, and `P1.environment.mean_reward` for one of the environment model. This
needs FlexiChains.jl to be loaded.

`returned_values` is a matrix with one element for each sample, with what one of three
models returned for it, as for [`get_session_trajectories`](@ref): the full model or the
model of several sessions, given with their `MultipleSessionsInfo`, or the model of a single
session, given without a session info, whose parameters are named without a session, as
`learning_rate` and `environment.mean_reward`. The parameters are those each session was
given, without the default values of its models. With a full model, they are those the
parameter models gave each session, so a parameter model other than
[`IndependentParameters`](@ref) gives parameters that aren't in the chain itself, such as
one value for every session from [`IdenticalParameters`](@ref), and values from
[`FixedParameters`](@ref) that were never sampled. A model of one's own around the model of
several sessions or of a single session gives the parameters it passed to the sessions, such
as transformations of its variables, when it returns what they return. The third form takes
one of the three models and a chain sampled from it, a `FlexiChain`, and numbers the samples
as the chain does, with a method of each inference backend for its own models. It merges
with the chain of [`get_session_trajectories`](@ref), which has the same samples, with
`merge`.

# Examples
```julia
#The learning rate of participant P1, for every sample of a fit
parameters = get_session_parameters(model, chain)
parameters[@varname(P1.learning_rate)]
```
"""
function get_session_parameters end

"""
    with_save_history(info, cognitive_save_history, environment_save_history)

Return a copy of `info`, a [`SessionInfo`](@ref) or a [`MultipleSessionsInfo`](@ref), whose
sessions save what the two save histories select. Used by the methods of
[`get_session_trajectories`](@ref) and [`predict_session_trajectories`](@ref) which take a
model, to read or simulate the sessions again with what is to be saved.
"""
function with_save_history(
    info::SessionInfo,
    cognitive_save_history::SaveHistory,
    environment_save_history::SaveHistory,
)
    return SessionInfo(info; cognitive_save_history, environment_save_history)
end
function with_save_history(
    info::MultipleSessionsInfo,
    cognitive_save_history::SaveHistory,
    environment_save_history::SaveHistory,
)
    return MultipleSessionsInfo(info; cognitive_save_history, environment_save_history)
end
