## The trajectories of fitted sessions simulated again ##

"""
    predict_session_trajectories([rng,] model, chain;
                                 cognitive_save_history=SaveHistory(),
                                 environment_save_history=SaveHistory())

Simulate the sessions of `model`, an inference backend's model of fitted sessions, again
with the parameters of every sample in `chain`, and return the trajectories they saved for
every sample. This can be used for prior and posterior predictive checks.

The sessions are prepared with [`prepare_for_prediction`](@ref), so that every action is
simulated, and every observation too when the sessions have an environment, and they save
what the two save histories select. The model of those sessions then runs once for each
sample, with the sample's parameters, and any start that a model samples, and with every
variable within the timesteps drawn again, as data is in a predictive check: an action
imputed in the fit is simulated like the others. The trajectories the sessions return are
kept. The sessions have at most the number of timesteps of their data, and fewer when a
[`stopping_criterion`](@ref) of a model is met, since no timestep of a predictive session
has data. For a prior predictive check, give a chain sampled from the prior.

Each inference backend has methods for its own models and chains, which need its package to
be loaded. Which models they take, and the form of the chain they return, are described in
the documentation of the inference backend.
"""
function predict_session_trajectories end
