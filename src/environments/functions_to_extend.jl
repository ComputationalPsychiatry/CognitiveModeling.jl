## Functions that an environment model must implement ##

"""
    apply_model(environment, input::EnvironmentInput)::EnvironmentOutput

Run a single timestep of `environment`: update its state and give the next observation.

# Implementation
Every environment model must add a method of this function, which throws an error otherwise.
Anything random should be sampled from `input.rng`, so that a simulation can be reproduced.

See also [`simulate`](@ref), which has an example.
"""
function apply_model(environment::AbstractEnvironmentModel, input)
    error("There is no apply_model method for an environment model of type $(typeof(environment)) with an input of type $(typeof(input)). Each environment model must define CognitiveModeling.apply_model(environment::$(typeof(environment)), input::CognitiveModeling.EnvironmentInput).")
end

"""
    initialize_state(environment::AbstractEnvironmentModel, parameters)

Create the state that `environment` has before its first timestep, which can depend on
`parameters`.

# Implementation
Every environment model must add a method of this function, which throws an error otherwise.
An environment model without a state returns an empty `NamedTuple`, `(;)`.
"""
function initialize_state(environment::AbstractEnvironmentModel, parameters)
    error("There is no initialize_state method for an environment model of type $(typeof(environment)). Each environment model must define CognitiveModeling.initialize_state(environment::$(typeof(environment)), parameters), which returns (;) if the environment model has no state.")
end

## Functions that an environment model can implement ##

"""
    prepare_state_for_saving(environment::AbstractEnvironmentModel, state)

Return what a simulation saves of `state`, as it is about to save it. The state that the
next timestep starts from isn't affected.

# Implementation
An environment model can add a method of this function. The default returns `state` as it
is. An environment model which changes an object in its state in place must add a method
which copies it, as for a cognitive model.
"""
function prepare_state_for_saving(environment::AbstractEnvironmentModel, state)
    return state
end
