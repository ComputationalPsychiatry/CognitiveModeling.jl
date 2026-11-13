## The trajectories of fitted sessions simulated again, as a chain ##

"""
    predict_session_trajectories([rng,] model, chain;
                                 cognitive_save_history=SaveHistory(),
                                 environment_save_history=SaveHistory())

Simulate the sessions of `model` again with the parameters of every sample in `chain`, and
return the trajectories they saved as a `FlexiChain`, as
[`session_trajectories_chain`](@ref) makes it. This is for prior and posterior predictive
checks, and needs Turing.jl to be loaded.

`model` is made by [`turing_full_model`](@ref),
[`turing_multiple_behavioral_sessions`](@ref) or [`turing_behavioral_session`](@ref), or
is such a model inside [`turing_catch_invalid_parameters`](@ref). Its sessions are prepared
with [`prepare_for_prediction`](@ref), so that every action is simulated, and every
observation too when the sessions have an environment, and they save what the two save
histories select. The model of those sessions then runs once for each sample, with the
sample's parameters, and any start that a model samples, and with every variable within the
timesteps drawn again, as data is in a predictive check: an action imputed in the fit is
simulated like the others. The trajectories the sessions return are kept. The sessions have
at most the number of timesteps of their data, and fewer when a [`stopping_criterion`](@ref)
of a model is met, since no timestep of a predictive session has data. For a prior
predictive check, give a chain sampled from the prior, such as
`sample(model, Prior(), 1000)`.

# Examples
```julia
chain = sample(model, NUTS(), 1000)
simulated = predict_session_trajectories(model, chain; cognitive_save_history = SaveHistory(; outputs = :choice))

#The simulated choices of participant P1, for every sample
simulated[@varname(P1.choice)]
```
"""
function predict_session_trajectories end
