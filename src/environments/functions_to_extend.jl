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

"""
    attribute_names(environment::AbstractEnvironmentModel)::ModelAttributes

Return the names of the parameters, states, observations and actions of `environment`, in a
[`ModelAttributes`](@ref).

The observations are the names in the observation that [`apply_model`](@ref) gives, which
the cognitive model receives, and the actions are the names it reads from `input.action`,
which the cognitive model gives. The parameters and the states are as for a cognitive model.

# Implementation
Every environment model must add a method of this function, which throws an error
otherwise. An environment model without one kind of attribute gives `()` for it, and so does
one whose state isn't a `NamedTuple`. [`simulate`](@ref) and [`Environment`](@ref) use the
observation names for the observation before the first timestep, and check the parameters,
the previous observation and a `NamedTuple` state against the names. A simulation also
checks that the environment model and the cognitive model fit together: the environment must
give every observation the model lists, and the model must give every action the environment
lists.

# Examples
```jldoctest
julia> struct BernoulliRewards <: AbstractEnvironmentModel end

julia> function CognitiveModeling.attribute_names(environment::BernoulliRewards)
           return ModelAttributes(;
               parameters = :reward_probability,
               states = (),
               observations = :reward,
               actions = :choice,
           )
       end;

julia> CognitiveModeling.attribute_names(BernoulliRewards()).observations
(:reward,)
```
"""
function attribute_names(environment::AbstractEnvironmentModel)
    error("There is no attribute_names method for an environment model of type $(typeof(environment)). Each environment model must define CognitiveModeling.attribute_names(environment::$(typeof(environment))), which returns a ModelAttributes with the names of its parameters, states, observations and actions.")
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

"""
    default_parameters(environment::AbstractEnvironmentModel)

Return the default parameter values of `environment`, such as `(; reward_probability = 0.8)`,
which [`Environment`](@ref) and [`simulate`](@ref) use for the parameters they aren't given.

# Implementation
An environment model can add a method of this function. The default is an empty
`NamedTuple`, `(;)`, so there are no default values.
"""
function default_parameters(environment::AbstractEnvironmentModel)
    return (;)
end
