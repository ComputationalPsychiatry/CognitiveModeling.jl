"""
    AbstractModel

Supertype for cognitive models and environment models, which the functions that treat the
two in the same way take. A model is a subtype of [`AbstractCognitiveModel`](@ref) or of
[`AbstractEnvironmentModel`](@ref), and not of this type itself.
"""
abstract type AbstractModel end

"""
    AbstractCognitiveModel

Supertype for cognitive models.

# Implementation
Create a subtype and add methods of [`apply_model`](@ref), [`initialize_state`](@ref) and
[`attribute_names`](@ref) for it. A model can also add methods of
[`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
[`turing_apply_model`](@ref).
"""
abstract type AbstractCognitiveModel <: AbstractModel end

#Types for cases where there is no observation or action

"""
    AbstractNoValue

Supertype for what stands in for an observation or an action that isn't there:
[`NoObservation`](@ref), [`NoAction`](@ref) and [`BeforeFirstTimestep`](@ref). A model which
treats them all the same can check for it, as in
`input.observation.reward isa AbstractNoValue`.
"""
abstract type AbstractNoValue end

"""
    NoObservation()

Stand-in for an observation that isn't made on a timestep, as one entry of the observations
`NamedTuple`, such as `(; reward = NoObservation())`. Before the first timestep, an
observation is [`BeforeFirstTimestep`](@ref) instead.
"""
struct NoObservation <: AbstractNoValue end

"""
    NoAction()

Stand-in for an action that isn't taken on a timestep, as one entry of the actions
`NamedTuple`, such as `(; choice = NoAction())`. Before the first timestep, an action is
[`BeforeFirstTimestep`](@ref) instead.
"""
struct NoAction <: AbstractNoValue end

"""
    BeforeFirstTimestep()

Stand-in for an observation or an action from before the first timestep, when there is none
yet, as one entry of a `NamedTuple`, such as `(; choice = BeforeFirstTimestep())`.

A cognitive model receives it for every previous action on its first timestep, and for every
observation too when the agent starts. An environment model receives it for every previous
observation on its first timestep, and for every action too when the environment starts.
[`NoObservation`](@ref) and [`NoAction`](@ref) are different: they say that nothing was
observed or done on a timestep that did run.

Before the first timestep is timestep 0, which a simulation, a history and a fitted session
start with when their [`SaveHistory`](@ref) saves the initial values, as it does by
default. Its observation and action are this marker unless others are given, and so is its
size.
"""
struct BeforeFirstTimestep <: AbstractNoValue end

#Type for an action or an observation that isn't random

"""
    Deterministic(value)

An action or an observation that is always `value`, for a cognitive model or an environment
model which gives it without any randomness, as in `(; reward = Deterministic(1.0))`.

A simulation samples every action and every observation with `rand`, and sampling from a
`Deterministic` gives `value` itself. The value is kept in the field `value`.

A simulation throws an error for a number, a string or a symbol that isn't wrapped, since
none of them is something to sample from. A vector, a tuple or a range that isn't wrapped is
sampled from as it is, which is rarely what is meant: `rand` picks one of its elements, with
no error.

# Examples
```jldoctest
julia> using Random: Xoshiro

julia> rand(Xoshiro(1), Deterministic(1.0))
1.0

julia> ModelOutput((; choice = Deterministic(true))).action_distribution.choice.value
true
```
"""
struct Deterministic{T}
    value::T
end

#Sampling from it gives the value itself
function Base.rand(::AbstractRNG, deterministic::Deterministic)
    return deterministic.value
end

#Types for who moves first on a timestep

"""
    AgentStarts()

Make the agent move first on each timestep of a simulation or a fitted session.

On a timestep, the cognitive model updates and acts on the previous observation, and the
environment then responds to its action. On the first timestep, the cognitive model receives
an observation with every entry [`BeforeFirstTimestep`](@ref). The model receives it as the
`who_starts` of its [`ModelInput`](@ref), so that it can act before it updates when that is
what it means.

See also [`EnvironmentStarts`](@ref), [`UnknownWhoStarts`](@ref), [`simulate`](@ref) and
[`SessionInfo`](@ref).
"""
struct AgentStarts end

"""
    EnvironmentStarts()

Make the environment move first on each timestep of a simulation or a fitted session.

On a timestep, the environment gives an observation in response to the previous action, and
the cognitive model then updates and acts on it. On the first timestep, the environment
receives an action with every entry [`BeforeFirstTimestep`](@ref). The model receives it as
the `who_starts` of its [`ModelInput`](@ref).

See also [`AgentStarts`](@ref), [`UnknownWhoStarts`](@ref), [`simulate`](@ref) and
[`SessionInfo`](@ref).
"""
struct EnvironmentStarts end

"""
    UnknownWhoStarts()

The `who_starts` of a [`ModelInput`](@ref) when neither side is known to move first on a
timestep, as when a model runs on a single observation with [`simulate`](@ref), or is used
outside a simulation. [`AgentStarts`](@ref) and [`EnvironmentStarts`](@ref) are the other
two, which a simulation or a fitted session passes on from its own `who_starts`.
"""
struct UnknownWhoStarts end

"""
    ModelInput(observation, previous_state, previous_action, parameters, timestep_size=1,
               rng=default_rng(), who_starts=UnknownWhoStarts())

Everything a model needs to run a single timestep, which [`apply_model`](@ref) receives.

Each argument is kept in a field of the same name, such as `input.observation`.

# Arguments
- `observation`: what the model observes on this timestep, in a `NamedTuple` with the names
  the environment gave them, as in `input.observation.reward`. An observation that isn't
  made on this timestep is [`NoObservation`](@ref). On the first timestep when the agent
  starts, every observation is [`BeforeFirstTimestep`](@ref).
- `previous_state`: the state of the model after the previous timestep. On the first
  timestep, this is the output of [`initialize_state`](@ref).
- `previous_action`: the actions from the previous timestep, in a `NamedTuple` with the
  names the model gave them, as in `input.previous_action.choice`. On the first timestep,
  each of them is [`BeforeFirstTimestep`](@ref).
- `parameters`: the cognitive parameters of the model. Will generally be a NamedTuple. 
- `timestep_size=1`: how much time this timestep covers.
- `rng=default_rng()`: the random number generator to sample from if anything in the model
  is random, so that a simulation can be reproduced.
- `who_starts=UnknownWhoStarts()`: who moves first on the timestep, [`AgentStarts`](@ref) or
  [`EnvironmentStarts`](@ref) as the simulation or the fitted session was told, or
  [`UnknownWhoStarts`](@ref) when neither is known. A model which can update before it acts
  or act before it updates can choose by it: with `EnvironmentStarts()` the observation is
  this timestep's, and with `AgentStarts()` the previous timestep's. It is kept in the type,
  so that choosing costs nothing.

# Examples
```jldoctest
julia> observation = (; reward = 1.0);

julia> state = (; expected_reward = 0.5);

julia> parameters = (; learning_rate = 0.1);

julia> previous_action = (; choice = BeforeFirstTimestep());

julia> ModelInput(observation, state, previous_action, parameters).timestep_size
1

julia> ModelInput(observation, state, previous_action, parameters, 0.5).timestep_size
0.5

julia> ModelInput(observation, state, previous_action, parameters).who_starts
UnknownWhoStarts()
```
"""
struct ModelInput{Tobservation, Tstate, Taction, Tparameters, Ttimestep_size, Trng, Twho_starts}
    observation::Tobservation
    previous_state::Tstate
    previous_action::Taction
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
    who_starts::Twho_starts
end

#Sets the timestep size to 1, the generator to the default one and who starts to unknown if they aren't specified
function ModelInput(observation, previous_state, previous_action, parameters, timestep_size = 1, rng::AbstractRNG = default_rng(), who_starts = UnknownWhoStarts())
    return ModelInput{typeof(observation), typeof(previous_state), typeof(previous_action), typeof(parameters), typeof(timestep_size), typeof(rng), typeof(who_starts)}(observation, previous_state, previous_action, parameters, timestep_size, rng, who_starts)
end

"""
    ModelOutput(action_distribution, new_state=(;))

The outputs of a model for a single timestep, which [`apply_model`](@ref) returns.

Each argument is kept in a field of the same name, such as `output.new_state`.

# Arguments
- `action_distribution::NamedTuple`: what the actions on this timestep are sampled from, in
  a `NamedTuple` with one entry for each action, such as `(; choice = Bernoulli(0.7))`.
  Each entry can be anything that works with `rand`, such as a distribution from
  Distributions.jl, or a [`Deterministic`](@ref) for an action that isn't random, and the
  sampled actions keep the names. An action that isn't taken on this timestep is
  [`NoAction`](@ref), and a model without actions gives an empty `NamedTuple`, `(;)`.
- `new_state=(;)`: the cognitive state of the model after this timestep, which is the
  `previous_state` of the next one.

# Examples
```jldoctest
julia> using Distributions: Bernoulli

julia> ModelOutput((; choice = Bernoulli(0.7)), (; expected_reward = 0.5)).new_state
(expected_reward = 0.5,)

julia> ModelOutput((; choice = Bernoulli(0.7))).new_state
NamedTuple()

julia> ModelOutput((; choice = NoAction(), bet = Bernoulli(0.2))).action_distribution.choice
NoAction()
```
"""
struct ModelOutput{Taction, Tstate}
    action_distribution::Taction
    new_state::Tstate

    #Only takes named action distributions, so that the model names every action
    function ModelOutput(action_distribution::NamedTuple, new_state = (;))
        return new{typeof(action_distribution), typeof(new_state)}(action_distribution, new_state)
    end
end

"""
    ModelAttributes(; parameters, states, observations, actions)

The names of the attributes of a model, which [`attribute_names`](@ref) returns.

An environment model returns one too, where the observations are those it gives and the
actions those it receives, the other way round from a cognitive model.

Each argument is a `Tuple` of `Symbol`s, which is kept in the type, so that the names are
known at compile time, and read as a property of the same name, such as
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
struct ModelAttributes{Tparameters, Tstates, Tobservations, Tactions} end

#Takes the names by keyword, puts a single name or a vector of names in a tuple, and keeps them in the type
function ModelAttributes(; parameters, states, observations, actions)
    return ModelAttributes{as_tuple_of_names(parameters), as_tuple_of_names(states), as_tuple_of_names(observations), as_tuple_of_names(actions)}()
end

"""
    as_tuple_of_names(names)

Return `names` as a `Tuple` of `Symbol`s, which is how a [`ModelAttributes`](@ref) keeps
them: a tuple as it is, a single `Symbol` in a tuple, and a `Vector` of them as a tuple.
"""
as_tuple_of_names(names::Tuple{Vararg{Symbol}}) = names
as_tuple_of_names(name::Symbol) = (name,)
as_tuple_of_names(names::AbstractVector{Symbol}) = Tuple(names)

#Reads the names in the type as properties
function Base.getproperty(attributes::ModelAttributes{Tparameters, Tstates, Tobservations, Tactions}, name::Symbol) where {Tparameters, Tstates, Tobservations, Tactions}
    if name === :parameters
        return Tparameters
    elseif name === :states
        return Tstates
    elseif name === :observations
        return Tobservations
    elseif name === :actions
        return Tactions
    else
        throw(FieldError(typeof(attributes), name))
    end
end

Base.propertynames(::ModelAttributes) = (:parameters, :states, :observations, :actions)

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
