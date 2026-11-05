"""
    AbstractCognitiveModel

Supertype for cognitive models.

# Implementation
Create a subtype and add methods of [`apply_model`](@ref), [`initialize_state`](@ref) and
[`attribute_names`](@ref) for it. A model can also add methods of
[`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
[`turing_apply_model`](@ref).
"""
abstract type AbstractCognitiveModel end

#Types for cases where there is no observation or action

"""
    NoObservation()

Stand-in for the observation on a timestep where a model observes nothing, such as the first
timestep when the agent starts.
"""
struct NoObservation end

"""
    NoAction()

Stand-in for an action when there is none, such as the previous action on the first
timestep. A model without actions gives this as its action distribution.
"""
struct NoAction end

"""
    ModelInput(observation, previous_state, previous_action, parameters, timestep_size=1,
               rng=default_rng())

Everything a model needs to run a single timestep, which [`apply_model`](@ref) receives.

Each argument is kept in a field of the same name, such as `input.observation`.

# Arguments
- `observation`: what the model observes on this timestep, or [`NoObservation`](@ref).
- `previous_state`: the state of the model after the previous timestep. On the first
  timestep, this is the output of [`initialize_state`](@ref).
- `previous_action`: the actions from the previous timestep, in a `NamedTuple` with the
  names the model gave them, as in `input.previous_action.choice`, or [`NoAction`](@ref).
- `parameters`: the cognitive parameters of the model. Will generally be a NamedTuple. 
- `timestep_size=1`: how much time this timestep covers.
- `rng=default_rng()`: the random number generator to sample from if anything in the model
  is random, so that a simulation can be reproduced.

# Examples
```jldoctest
julia> observation = (; reward = 1.0);

julia> state = (; expected_reward = 0.5);

julia> parameters = (; learning_rate = 0.1);

julia> ModelInput(observation, state, NoAction(), parameters).timestep_size
1

julia> ModelInput(observation, state, NoAction(), parameters, 0.5).timestep_size
0.5
```
"""
struct ModelInput{Tobservation, Tstate, Taction, Tparameters, Ttimestep_size, Trng}
    observation::Tobservation
    previous_state::Tstate
    previous_action::Taction
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
end

#Sets the timestep size to 1 and the generator to the default one if they aren't specified.
function ModelInput(observation, previous_state, previous_action, parameters, timestep_size = 1)
    return ModelInput(observation, previous_state, previous_action, parameters, timestep_size, default_rng())
end

"""
    ModelOutput(action_distribution, new_state=(;))

The outputs of a model for a single timestep, which [`apply_model`](@ref) returns.

Each argument is kept in a field of the same name, such as `output.new_state`.

# Arguments
- `action_distribution::Union{NamedTuple, NoAction}`: what the actions on this timestep are
  sampled from, in a `NamedTuple` with one entry for each action, such as
  `(; choice = Bernoulli(0.7))`. Each entry can be anything that works with `rand`, such as
  a distribution from Distributions.jl, and the sampled actions keep the names. A model
  without actions gives [`NoAction`](@ref).
- `new_state=(;)`: the cognitive state of the model after this timestep, which is the
  `previous_state` of the next one.

# Examples
```jldoctest
julia> using Distributions: Bernoulli

julia> ModelOutput((; choice = Bernoulli(0.7)), (; expected_reward = 0.5)).new_state
(expected_reward = 0.5,)

julia> ModelOutput((; choice = Bernoulli(0.7))).new_state
NamedTuple()

julia> ModelOutput(NoAction()).action_distribution
NoAction()
```
"""
struct ModelOutput{Taction, Tstate}
    action_distribution::Taction
    new_state::Tstate

    #Only takes named action distributions, or no action, so that the model names every action
    function ModelOutput(action_distribution::Union{NamedTuple, NoAction}, new_state = (;))
        return new{typeof(action_distribution), typeof(new_state)}(action_distribution, new_state)
    end
end

"""
    ModelAttributes(; parameters, states, observations, actions)

The names of the attributes of a model, which [`attribute_names`](@ref) returns.

Each argument is a `Tuple` of `Symbol`s, and is kept in a field of the same name, such as
`attributes.parameters`. A single `Symbol` is put in a tuple, and a `Vector` of them is made
into one. All four must be given, and `()` means that the model has none of that kind.

# Examples
```jldoctest
julia> attributes = ModelAttributes(;
           parameters = (:learning_rate, :inverse_temperature),
           states = :expected_reward,
           observations = [:reward],
           actions = (),
       )
ModelAttributes
Parameters: learning_rate, inverse_temperature
States: expected_reward
Observations: reward
Actions: none

julia> attributes.states
(:expected_reward,)
```
"""
struct ModelAttributes{Tparameters <: Tuple{Vararg{Symbol}}, Tstates <: Tuple{Vararg{Symbol}}, Tobservations <: Tuple{Vararg{Symbol}}, Tactions <: Tuple{Vararg{Symbol}}}
    parameters::Tparameters
    states::Tstates
    observations::Tobservations
    actions::Tactions
end

#Takes the names by keyword, and puts a single name or a vector of names in a tuple
function ModelAttributes(; parameters, states, observations, actions)
    return ModelAttributes(as_tuple_of_names(parameters), as_tuple_of_names(states), as_tuple_of_names(observations), as_tuple_of_names(actions))
end

#Names are kept in a tuple of symbols
as_tuple_of_names(names::Tuple{Vararg{Symbol}}) = names
as_tuple_of_names(name::Symbol) = (name,)
as_tuple_of_names(names::AbstractVector{Symbol}) = Tuple(names)

"""
    InvalidParametersError(message)
    InvalidParametersError()

An error for a model to throw from [`apply_model`](@ref) or [`initialize_state`](@ref) when
its parameters aren't valid, as in `throw(InvalidParametersError())`.

The default message says that the parameters are not valid, possibly dependent on the
observations, actions and states, and a model can give a more specific one. A simulation
stops with it as with any other error, and fitting can catch it to give the parameters a
probability of zero, so a model should throw this rather than an ordinary error when the
parameters are the problem.

# Examples
```jldoctest
julia> try
           throw(InvalidParametersError())
       catch error
           sprint(showerror, error)
       end
"InvalidParametersError: Model parameters are not valid, possibly dependent on observations, actions and states."

julia> sprint(showerror, InvalidParametersError("The learning rate must be between 0 and 1."))
"InvalidParametersError: The learning rate must be between 0 and 1."
```
"""
struct InvalidParametersError <: Exception
    message::String
end

#The default message says what the error means
function InvalidParametersError()
    return InvalidParametersError("Model parameters are not valid, possibly dependent on observations, actions and states.")
end

#Prints the message when the error isn't caught
function Base.showerror(io::IO, error::InvalidParametersError)
    print(io, "InvalidParametersError: ", error.message)
    return nothing
end
