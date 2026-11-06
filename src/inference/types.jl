#Types for data where an action or an observation wasn't recorded

"""
    MissingAction()

Stand-in for an action that was taken on a timestep but isn't in the data, as one entry of
the actions `NamedTuple`, such as `(; choice = MissingAction())`. Fitting imputes it by
sampling it from the model's action distribution on that timestep, or passes it on as
[`NoAction`](@ref) when it is told not to impute. `NoAction` itself is different: it says
that the model took no action on that timestep.
"""
struct MissingAction end

"""
    MissingObservation()

Stand-in for an observation that was made on a timestep but isn't in the data, as one entry
of the observations `NamedTuple`, such as `(; reward = MissingObservation())`. Fitting fills
it in from an environment model. [`NoObservation`](@ref) is different: it says that nothing
was observed on that timestep.
"""
struct MissingObservation end
