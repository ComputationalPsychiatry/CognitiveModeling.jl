## Functions that a user must implement for their model ##

"""
    apply_model(model::AbstractCognitiveModel, input::ModelInput)::ModelOutput

Run a single timestep of `model`. Usually this involves updating cognitive states and returning an object from which the action is sampled.

# Implementation
Every model must add a method of this function, which throws an error otherwise. Anything
random should be sampled from `input.rng`, so that a simulation can be reproduced.

# Examples
```jldoctest
using CognitiveModeling
using CognitiveModeling: ModelInput, ModelOutput
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

    return ModelOutput(Pₜ, (; expected_reward = Vₜ))
end

parameters = (; learning_rate = 0.1, inverse_temperature = 2.0)
input = ModelInput((; reward = 1.0), (; expected_reward = 0.5), NoAction(), parameters)

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

Create the state that `model` has before its first timestep, which can depend on
`parameters`.

# Implementation
The default is an empty `NamedTuple`, so only models with a state need to add a method.

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
    return (;)
end

"""
    default_parameters(model::AbstractCognitiveModel)

Return the default parameter values of `model`, such as `(; learning_rate = 0.1)`.

# Implementation
By default there are no default values.
"""
function default_parameters(model::AbstractCognitiveModel)
    return (;)
end

"""
    get_parameters(model::AbstractCognitiveModel)

Return the names of the parameters of `model`, such as `(:learning_rate, :action_noise)`.

# Implementation
The default is an empty list.
"""
function get_parameters(model::AbstractCognitiveModel)
    return ()
end

"""
    get_states(model::AbstractCognitiveModel)

Return the names of the states of `model`, such as `(:expected_reward,)`.

# Implementation
The default is an empty list.
"""
function get_states(model::AbstractCognitiveModel)
    return ()
end

"""
    get_observations(model::AbstractCognitiveModel)

Return the names of the observations that `model` receives, such as `(:reward,)`.

# Implementation
The default is an empty list.
"""
function get_observations(model::AbstractCognitiveModel)
    return ()
end

"""
    get_actions(model::AbstractCognitiveModel)

Return the names of the actions that `model` produces, such as `(:choice,)`.

# Implementation
The default is an empty list.
"""
function get_actions(model::AbstractCognitiveModel)
    return ()
end
