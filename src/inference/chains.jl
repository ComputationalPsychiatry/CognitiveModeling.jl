## What fitted sessions saved, as a chain, made in the FlexiChains extension ##

"""
    session_values_chain(results, sessions_info::MultipleSessionsInfo;
                         iter_indices=1:size(results, 1), chain_indices=1:size(results, 2))
    session_values_chain(results; iter_indices=1:size(results, 1),
                         chain_indices=1:size(results, 2))
    session_values_chain(model, chain; save_history::SaveHistory)

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
  session are named after its identifier, such as `P1.expected_value`, and the session's
  parameters are included as the population model returned them, such as
  `P1.learning_rate`, since a population model other than [`IndependentSessions`](@ref)
  can give parameters that aren't in the chain itself. What a downstream model returned
  isn't included.
- [`turing_multiple_behavioural_sessions`](@ref), given with its `MultipleSessionsInfo`,
  named in the same way, and without the parameters.
- [`turing_behavioural_session`](@ref), given without a session info, with no session
  identifier in the names, such as `expected_value`.
In a model of one's own around these, that model must return what they return, for
`returned` to give it.

What each session saved, as its [`SaveHistory`](@ref) selected, is one entry for each name:
a state as `P1.expected_value`, and an observation, an action or an action distribution under
its kind, as `P1.observations.reward`, `P1.actions.choice` and
`P1.action_distributions.choice`. A saved state that isn't a `NamedTuple` is one entry,
`P1.states`. Each entry holds a vector over the timesteps for every sample, which starts with
timestep 0 when the initial values are saved, so that element `t + 1` is timestep `t`, as
everywhere in the package. The observation, the action and the action distribution of
timestep 0 are [`BeforeFirstTimestep`](@ref), so a summary of those entries needs timestep 0
left out. An identifier from several session columns is joined with underscores, as in
`S1_2`. `iter_indices` and `chain_indices` number the samples as the chain that was fitted
does.

The third form, in the Turing extension, takes a model made by one of the three functions,
or such a model inside [`turing_catch_invalid_parameters`](@ref), and a chain sampled from
it. It makes a copy of the model whose sessions save what `save_history` selects, gives it to
`returned`, and makes the chain of the results, numbered as `chain` is. A fit is fastest when
its sessions save nothing, and what they saved is then read afterwards in this way.

# Examples
```julia
chain = sample(model, NUTS(), 1000)
session_values = session_values_chain(model, chain; save_history = SaveHistory(; states = :expected_value))

#The expected values of participant P2 over the timesteps, for every sample
session_values[@varname(P2.expected_value)]

#The mean learning rate of each participant, with the identifier interpolated as a Symbol
for participant in Symbol.(sessions_info.session_ids)
    println(participant, ": ", mean(session_values[@varname(\$participant.learning_rate)]))
end
```
"""
function session_values_chain end
