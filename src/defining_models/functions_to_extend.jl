## Functions that a model must implement ##

"""
    apply_model(model::AbstractCognitiveModel, input::ModelInput)::ModelOutput

Run a single timestep of `model`. Usually this involves updating cognitive states and returning an object from which the action is sampled.

# Implementation
Every model must add a method of this function, which throws an error otherwise. Anything
random should be sampled from `input.rng`, so that a simulation can be reproduced. 
If states are mutated in place, a prepare_state_for_saving method must be added.
For use within PPLs, there may be special constraints - see the documentation for the specific PPL for details.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: BernoulliLogit

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)

    ## Unpack input ##
    #Reward observed on this timestep
    r = input.observation.reward
    #Expected reward from the previous state
    V₋₁ = input.previous_state.expected_reward
    #Learning rate parameter
    α = input.parameters.learning_rate
    #Inverse temperature parameter
    β = input.parameters.inverse_temperature

    ## Update state ##
    #Update expected reward using the Rescorla-Wagner rule
    Vₜ = V₋₁ + α * (r - V₋₁)

    ## Compute action probabilities ##
    #Softmax function, instantiated as a Bernoulli logit distribution
    Pₜ = BernoulliLogit(β * (Vₜ - 0.5))

    return ModelOutput((; choice = Pₜ), (; expected_reward = Vₜ))
end

parameters = (; learning_rate = 0.1, inverse_temperature = 2.0)
input = ModelInput((; reward = 1.0), (; expected_reward = 0.5), (; choice = NoAction()), parameters)

CognitiveModeling.apply_model(RescorlaWagner(), input).new_state

# output

(expected_reward = 0.55,)
```
"""
function apply_model(model::AbstractCognitiveModel, input)
    error("There is no apply_model method for a model of type $(typeof(model)) with an input of type $(typeof(input)). Each model must define CognitiveModeling.apply_model(model::$(typeof(model)), input::CognitiveModeling.ModelInput).")
end

"""
    initialize_state(model::AbstractCognitiveModel, parameters)

Create the initial state that `model` has before its first timestep, which can depend on
`parameters`.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A model
without a state returns an empty `NamedTuple`, `(;)`.

# Examples
```jldoctest
julia> struct RescorlaWagner <: AbstractCognitiveModel end

julia> function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
           return (; expected_reward = parameters.initial_value)
       end;

julia> CognitiveModeling.initialize_state(RescorlaWagner(), (; initial_value = 0.5))
(expected_reward = 0.5,)
```
"""
function initialize_state(model::AbstractCognitiveModel, parameters)
    error("There is no initialize_state method for a model of type $(typeof(model)). Each model must define CognitiveModeling.initialize_state(model::$(typeof(model)), parameters), which returns (;) if the model has no state.")
end

"""
    attribute_names(model::AbstractCognitiveModel)::ModelAttributes

Return the names of the parameters, states, observations and actions of `model`, in a
[`ModelAttributes`](@ref).

The names define the model, so they must match what the model uses. The parameters are the
names that [`apply_model`](@ref) and [`initialize_state`](@ref) read from the parameters.
The states are the names in the state that `initialize_state` returns and in `new_state`,
which `apply_model` reads from `input.previous_state`. The observations are the names that
`apply_model` reads from `input.observation`, and the actions are the names in
`action_distribution`, which are also those in `input.previous_action`. Nothing checks that
they match while the model runs.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A model
without one kind of attribute gives `()` for it. [`simulate`](@ref) and
[`CognitiveAgent`](@ref) use the action names for the action before the first timestep, and
check the parameters, the previous action and a `NamedTuple` state against the names. A
simulation with an environment also checks that the two fit together: the environment must
give every observation the model lists, and the model must give every action the environment
lists.

# Examples
```jldoctest
julia> struct RescorlaWagner <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::RescorlaWagner)
           return ModelAttributes(;
               parameters = (:learning_rate, :inverse_temperature),
               states = :expected_reward,
               observations = :reward,
               actions = :choice,
           )
       end;

julia> CognitiveModeling.attribute_names(RescorlaWagner()).parameters
(:learning_rate, :inverse_temperature)
```
"""
function attribute_names(model::AbstractCognitiveModel)
    error("There is no attribute_names method for a model of type $(typeof(model)). Each model must define CognitiveModeling.attribute_names(model::$(typeof(model))), which returns a ModelAttributes with the names of its parameters, states, observations and actions.")
end

## Functions that a model can implement ##

"""
    prepare_state_for_saving(model::AbstractCognitiveModel, state)

Return what a simulation saves of `state`, as it is about to save it. The state that the
next timestep starts from isn't affected.

# Implementation
A model can add a method of this function. The default returns `state` as it is, so nothing
is copied. A model which changes an object in its state in place must add a method which
copies it, or all the saved states show the final values, with no error.

# Examples
```jldoctest
julia> struct LearnsInPlace <: AbstractCognitiveModel end

julia> function CognitiveModeling.prepare_state_for_saving(model::LearnsInPlace, state)
           return (; values = copy(state.values))
       end;

julia> state = (; values = [0.5, 0.5]);

julia> CognitiveModeling.prepare_state_for_saving(LearnsInPlace(), state).values === state.values
false
```
"""
function prepare_state_for_saving(model::AbstractCognitiveModel, state)
    return state
end

"""
    default_parameters(model::AbstractCognitiveModel)

Return the default parameter values of `model`, such as `(; learning_rate = 0.1)`, which
[`CognitiveAgent`](@ref) and [`simulate`](@ref) use for the parameters they aren't given.

# Implementation
A model can add a method of this function. The default is an empty `NamedTuple`, `(;)`, so
there are no default values.
"""
function default_parameters(model::AbstractCognitiveModel)
    return (;)
end
