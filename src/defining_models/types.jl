## Supertypes of models ##

"""
    AbstractTimestepModel

Supertype for timestep models: cognitive models and environment models, which every
function that a model implements or is used with takes. A model is a subtype of
[`AbstractCognitiveModel`](@ref) or of [`AbstractEnvironmentModel`](@ref), and generally not
of this type itself.

The two kinds work the same way, one timestep at a time: on each timestep a model receives
an input and gives an output, in a [`ModelContext`](@ref) and a [`ModelOutput`](@ref). In a
simulation or a fitted session, the outputs of the cognitive model are its actions, which
are the inputs of the environment model, and the outputs of the environment model are the
observations, which are the inputs of the cognitive model. A cognitive model can therefore
also serve as the environment of another.
"""
abstract type AbstractTimestepModel end

"""
    AbstractCognitiveModel

Supertype for cognitive models, whose inputs are observations and whose outputs are actions.

In the REPL, a cognitive model shows its type, the values of its fields if it has any, and
the names of its attributes, or `not specified` if it has no method of
[`attribute_names`](@ref). A kind of attribute with more than five names shows only how many
there are, and `attribute_names` gives them. On one line, as inside a vector, a model shows
its type and its kind.

# Implementation
Create a subtype and add methods of [`apply_timestep_model`](@ref),
[`initialize_state`](@ref) and [`attribute_names`](@ref) for it. A model can also add
methods of [`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
[`stopping_criterion`](@ref). Fitting with a PPL can need methods of its own, which the
documentation for that PPL describes.

# Examples
```jldoctest
julia> struct ExampleModel <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::ExampleModel)
           return ModelAttributes(;
               parameters = (:learning_rate, :noise),
               states = :expected_reward,
               inputs = :reward,
               outputs = (),
           )
       end;

julia> ExampleModel()
-- Cognitive model --
Type: ExampleModel
Parameters: learning_rate, noise
States: expected_reward
Inputs: reward
Outputs: none

julia> print(ExampleModel())
ExampleModel (cognitive model)
```
"""
abstract type AbstractCognitiveModel <: AbstractTimestepModel end

"""
    AbstractEnvironmentModel

Supertype for environment models, whose inputs are the actions of a cognitive model and
whose outputs are its observations.

An environment model shows in the REPL and on one line as a cognitive model does, with its
own kind.

# Implementation
Create a subtype and add the same methods as for an [`AbstractCognitiveModel`](@ref):
[`apply_timestep_model`](@ref), [`initialize_state`](@ref) and [`attribute_names`](@ref),
and optionally [`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
[`stopping_criterion`](@ref). Fitting with a PPL can need methods of its own, which the
documentation for that PPL describes.

See also [`simulate`](@ref), which has an example.

# Examples
```jldoctest
julia> struct ExampleEnvironment <: AbstractEnvironmentModel end

julia> function CognitiveModeling.attribute_names(environment::ExampleEnvironment)
           return ModelAttributes(;
               parameters = :reward_probability,
               states = (),
               inputs = :choice,
               outputs = :reward,
           )
       end;

julia> ExampleEnvironment()
-- Environment model --
Type: ExampleEnvironment
Parameters: reward_probability
States: none
Inputs: choice
Outputs: reward
```
"""
abstract type AbstractEnvironmentModel <: AbstractTimestepModel end

## Types for cases where there is no observation or action ##

"""
    AbstractNoValue

Supertype for what stands in for an observation or an action that isn't there:
[`NoValue`](@ref) and [`BeforeFirstTimestep`](@ref), and in the data of a fitted session
[`UnimputedMissingValue`](@ref). A model which treats them all the same can check for it, as
in `context.input.reward isa AbstractNoValue`.
"""
abstract type AbstractNoValue end

"""
    NoValue()

Stand-in for an observation that isn't made, or an action that isn't taken, on a timestep,
as one entry of the observations or the actions `NamedTuple`, such as
`(; reward = NoValue())` or `(; choice = NoValue())`. Whether it is an observation or an
action is given by where it is. Before the first timestep, an observation or an action is
[`BeforeFirstTimestep`](@ref) instead.
"""
struct NoValue <: AbstractNoValue end

"""
    BeforeFirstTimestep()

Stand-in for an observation or an action from before the first timestep, when there is none
yet, as one entry of a `NamedTuple`, such as `(; choice = BeforeFirstTimestep())`.

A model receives it for every previous output on its first timestep, and for every input
too when the other side hasn't moved yet: a cognitive model when the agent starts, and an
environment model when the environment starts. [`NoValue`](@ref) is different: it says that
nothing was observed or done on a timestep that did run.

Before the first timestep is timestep 0, which a simulation, a history and a fitted session
start with when their [`SaveHistory`](@ref) saves the initial values, as it does by
default. Its observation and action are this marker unless others are given, and so is its
size.
"""
struct BeforeFirstTimestep <: AbstractNoValue end

## Type for an action or an observation that isn't random ##

"""
    Deterministic(value)

An action or an observation that is always `value`, for a cognitive model or an environment
model which gives it without any randomness, as in `(; reward = Deterministic(1.0))`.

A simulation samples every action and every observation with `rand`, and sampling from a
`Deterministic` gives `value` itself. The value is kept in the field `value`.

A simulation throws an error for a number, a string or a symbol that isn't wrapped, since
none of them is something to sample from. A vector, a tuple or a range that isn't wrapped is
sampled from as it is: `rand` picks one of its elements, with no error.

# Examples
```jldoctest
julia> using Random: Xoshiro

julia> rand(Xoshiro(1), Deterministic(1.0))
1.0

julia> ModelOutput((; choice = Deterministic(true))).output_distribution.choice.value
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

## Types for who moves first on a timestep ##

"""
    AgentStarts()

Make the agent move first on each timestep of a simulation or a fitted session.

On a timestep, the cognitive model updates and acts on the previous observation, and the
environment then responds to its action. On the first timestep, the cognitive model receives
an observation with every entry [`BeforeFirstTimestep`](@ref). Both models receive it as the
`who_starts` of their [`ModelContext`](@ref), so that a cognitive model can act before it
updates when that is what it means.

See also [`EnvironmentStarts`](@ref), [`UnknownWhoStarts`](@ref), [`simulate`](@ref) and
[`SessionInfo`](@ref).
"""
struct AgentStarts end

"""
    EnvironmentStarts()

Make the environment move first on each timestep of a simulation or a fitted session.

On a timestep, the environment gives an observation in response to the previous action, and
the cognitive model then updates and acts on it. On the first timestep, the environment
receives an action with every entry [`BeforeFirstTimestep`](@ref). Both models receive it as
the `who_starts` of their [`ModelContext`](@ref).

See also [`AgentStarts`](@ref), [`UnknownWhoStarts`](@ref), [`simulate`](@ref) and
[`SessionInfo`](@ref).
"""
struct EnvironmentStarts end

"""
    UnknownWhoStarts()

The `who_starts` of a [`ModelContext`](@ref) when neither side is known to move first on a
timestep, as when a model runs on a single input with [`simulate`](@ref), or is used outside
a simulation. [`AgentStarts`](@ref) and [`EnvironmentStarts`](@ref) are the other
two, which a simulation or a fitted session passes on from its own `who_starts`.
"""
struct UnknownWhoStarts end

## What a timestep receives and returns ##

"""
    ModelContext(; input, previous_state, previous_output, parameters, timestep_size=1,
                 rng=default_rng(), who_starts=UnknownWhoStarts())
    ModelContext(input, previous_state, previous_output, parameters, timestep_size=1,
                 rng=default_rng(), who_starts=UnknownWhoStarts())

Everything a model needs to run a single timestep, which [`apply_timestep_model`](@ref)
receives. The arguments can be given by name, which says what each one is, or in this order.

Each argument is kept in a field of the same name, such as `context.input`. For a cognitive
model the input is its observation and the output its action, and for an environment model
the input is the action of the cognitive model and the output the observation it gives.

# Arguments
- `input`: what the model receives on this timestep, in a `NamedTuple` with the names the
  other side gave the entries, as in `context.input.reward`. An entry that isn't there on
  this timestep is [`NoValue`](@ref). On the first timestep, when the other side hasn't
  moved yet, every entry is [`BeforeFirstTimestep`](@ref).
- `previous_state`: the state of the model after the previous timestep. On the first
  timestep, this is the output of [`initialize_state`](@ref).
- `previous_output`: what the model gave on the previous timestep, in a `NamedTuple` with
  the names the model gave the entries, as in `context.previous_output.choice`. On the first
  timestep, each of them is [`BeforeFirstTimestep`](@ref).
- `parameters`: the parameters of the model, in a `NamedTuple`.
- `timestep_size=1`: how much time this timestep covers; the difference between the end of
  the previous and the end of the current timestep.
- `rng=default_rng()`: the random number generator to sample from if anything in the model
  is random, so that a simulation can be reproduced.
- `who_starts=UnknownWhoStarts()`: who moves first on the timestep, [`AgentStarts`](@ref) or
  [`EnvironmentStarts`](@ref) as the simulation or the fitted session was told, or
  [`UnknownWhoStarts`](@ref) when neither is known. A model which can update before it acts
  or act before it updates can choose by it: for a cognitive model, with
  `EnvironmentStarts()` the input is this timestep's observation, and with `AgentStarts()`
  the previous timestep's.

# Examples
```jldoctest
julia> input = (; reward = 1.0);

julia> previous_state = (; expected_reward = 0.5);

julia> previous_output = (; choice = BeforeFirstTimestep());

julia> parameters = (; learning_rate = 0.1);

julia> ModelContext(; input, previous_state, previous_output, parameters).timestep_size
1

julia> ModelContext(; input, previous_state, previous_output, parameters, timestep_size = 0.5).timestep_size
0.5

julia> ModelContext(input, previous_state, previous_output, parameters).who_starts
UnknownWhoStarts()
```
"""
struct ModelContext{Tinput,Tstate,Toutput,Tparameters,Ttimestep_size,Trng,Twho_starts}
    input::Tinput
    previous_state::Tstate
    previous_output::Toutput
    parameters::Tparameters
    timestep_size::Ttimestep_size
    rng::Trng
    who_starts::Twho_starts
end

#Sets the timestep size to 1, the generator to the default one and who starts to unknown if they aren't specified
function ModelContext(
    input,
    previous_state,
    previous_output,
    parameters,
    timestep_size=1,
    rng::AbstractRNG=default_rng(),
    who_starts=UnknownWhoStarts(),
)
    return ModelContext{
        typeof(input),
        typeof(previous_state),
        typeof(previous_output),
        typeof(parameters),
        typeof(timestep_size),
        typeof(rng),
        typeof(who_starts),
    }(
        input, previous_state, previous_output, parameters, timestep_size, rng, who_starts
    )
end

#Takes the arguments by name, with the same defaults as the positional form
function ModelContext(;
    input,
    previous_state,
    previous_output,
    parameters,
    timestep_size=1,
    rng::AbstractRNG=default_rng(),
    who_starts=UnknownWhoStarts(),
)
    return ModelContext(
        input, previous_state, previous_output, parameters, timestep_size, rng, who_starts
    )
end

"""
    ModelOutput(output_distribution, new_state=(;))

The outputs of a model for a single timestep, which [`apply_timestep_model`](@ref) returns.

Each argument is kept in a field of the same name, such as `output.new_state`.

# Arguments
- `output_distribution::NamedTuple`: what the model's outputs on this timestep are sampled
  from, in a `NamedTuple` with one entry for each output, such as
  `(; choice = Bernoulli(0.7))` for the action of a cognitive model, or
  `(; reward = Bernoulli(0.8))` for the observation an environment model gives. Each entry
  can be anything that works with `rand`, such as a distribution from Distributions.jl, or a
  [`Deterministic`](@ref) for an output that isn't random, and the sampled outputs keep the
  names. An output that isn't given on this timestep is [`NoValue`](@ref), and a model
  without outputs gives an empty `NamedTuple`, `(;)`.
- `new_state=(;)`: the state of the model after this timestep, which is the
  `previous_state` of the next one. The outputs are sampled after it is made, so it can't
  depend on them: the model receives them as `context.previous_output` on the next
  timestep.

# Examples
```jldoctest
julia> using Distributions: Bernoulli

julia> ModelOutput((; choice = Bernoulli(0.7)), (; expected_reward = 0.5)).new_state
(expected_reward = 0.5,)

julia> ModelOutput((; choice = Bernoulli(0.7))).new_state
NamedTuple()

julia> ModelOutput((; choice = NoValue(), bet = Bernoulli(0.2))).output_distribution.choice
NoValue()
```
"""
struct ModelOutput{Toutput,Tstate}
    output_distribution::Toutput
    new_state::Tstate

    #Only takes named output distributions, so that the model names every output
    function ModelOutput(output_distribution::NamedTuple, new_state=(;))
        return new{typeof(output_distribution),typeof(new_state)}(
            output_distribution, new_state
        )
    end
end

## The names of a model's attributes ##

"""
    ModelAttributes(; parameters, states, inputs, outputs)

The names of the attributes of a model, which [`attribute_names`](@ref) returns.

For a cognitive model the inputs are its observations and the outputs its actions, and for
an environment model the inputs are the actions it receives and the outputs the observations
it gives.

Each argument is a `Tuple` of `Symbol`s, read as a property of the same name, such as
`attributes.parameters`. A single `Symbol` is put in a tuple, and a `Vector` of them is made
into one. All four must be given, and `()` means that the model has none of that kind.
A name can only be used once across the four kinds, so that it says which attribute of the
model is meant wherever the attributes are named together, and an `ArgumentError` is thrown
otherwise.

# Examples
```jldoctest
julia> attributes = ModelAttributes(;
           parameters = (:learning_rate, :inverse_temperature),
           states = :expected_reward,
           inputs = [:reward],
           outputs = (),
       )
ModelAttributes
Parameters: learning_rate, inverse_temperature
States: expected_reward
Inputs: reward
Outputs: none

julia> attributes.states
(:expected_reward,)

julia> ModelAttributes(; parameters = :choice, states = (), inputs = (), outputs = :choice)
ERROR: ArgumentError: The name choice is used for both a parameter and an output, but a model can only use a name for one attribute.
[...]
```
"""
struct ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs} end

#Takes the names by keyword, puts a single name or a vector of names in a tuple, checks that each is used once, and keeps them in the type
function ModelAttributes(; parameters, states, inputs, outputs)
    attributes = ModelAttributes{
        as_tuple_of_names(parameters),
        as_tuple_of_names(states),
        as_tuple_of_names(inputs),
        as_tuple_of_names(outputs),
    }()
    check_unique_attribute_names(attributes)
    return attributes
end

"""
    as_tuple_of_names(names)

Return `names` as a `Tuple` of `Symbol`s, which is how a [`ModelAttributes`](@ref) keeps
them: a tuple as it is, a single `Symbol` in a tuple, and a `Vector` of them as a tuple.
Used by the constructors of [`ModelAttributes`](@ref) and [`MultipleSessionsInfo`](@ref).
"""
as_tuple_of_names(names::Tuple{Vararg{Symbol}}) = names
as_tuple_of_names(name::Symbol) = (name,)
as_tuple_of_names(names::AbstractVector{Symbol}) = Tuple(names)

"""
    check_unique_attribute_names(attributes)

Throw an `ArgumentError` if a name in `attributes`, a [`ModelAttributes`](@ref), is used for
more than one attribute, within a kind or across the kinds, and return nothing otherwise.
Whether the names are unique is decided by [`unique_names`](@ref), and the message is only
made when they aren't. Used by the constructor of `ModelAttributes`.
"""
function check_unique_attribute_names(attributes::ModelAttributes)
    all_names = (
        attributes.parameters...,
        attributes.states...,
        attributes.inputs...,
        attributes.outputs...,
    )
    unique_names(all_names) || throw_duplicate_attribute_name(attributes)
    return nothing
end

"""
    unique_names(names)

Return whether no name in the tuple `names` appears twice. It is unrolled over the names
with [`has_name`](@ref). Used by [`check_unique_attribute_names`](@ref).
"""
unique_names(::Tuple{}) = true
function unique_names(names::Tuple)
    return !has_name(first(names), Base.tail(names)) && unique_names(Base.tail(names))
end

"""
    throw_duplicate_attribute_name(attributes)

Throw the `ArgumentError` of [`check_unique_attribute_names`](@ref), naming the first name
of `attributes` that is used twice and the kinds it is used for. 
"""
@noinline function throw_duplicate_attribute_name(attributes::ModelAttributes)
    kinds = (
        ("a parameter", "parameters", attributes.parameters),
        ("a state", "states", attributes.states),
        ("an input", "inputs", attributes.inputs),
        ("an output", "outputs", attributes.outputs),
    )
    for (i, (kind, kinds_name, names)) in enumerate(kinds)
        allunique(names) || throw(
            ArgumentError(
                "The $kinds_name are named $names, but a model can only use a name for one attribute.",
            ),
        )
        for (other_kind, _, other_names) in kinds[(i + 1):end], name in names
            if name in other_names
                throw(
                    ArgumentError(
                        "The name $name is used for both $kind and $other_kind, but a model can only use a name for one attribute.",
                    ),
                )
            end
        end
    end
    return nothing
end

#Reads the names in the type as properties
function Base.getproperty(
    attributes::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}, name::Symbol
) where {Tparameters,Tstates,Tinputs,Toutputs}
    if name === :parameters
        return Tparameters
    elseif name === :states
        return Tstates
    elseif name === :inputs
        return Tinputs
    elseif name === :outputs
        return Toutputs
    else
        throw(FieldError(typeof(attributes), name))
    end
end

Base.propertynames(::ModelAttributes) = (:parameters, :states, :inputs, :outputs)

## Errors ##

"""
    InvalidParametersError(message)
    InvalidParametersError()

An error for a model to throw from [`apply_timestep_model`](@ref) or
[`initialize_state`](@ref) when its parameters aren't valid, as in
`throw(InvalidParametersError())`.

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
    return InvalidParametersError(
        "Model parameters are not valid, possibly dependent on observations, actions and states.",
    )
end

#Prints the message when the error isn't caught
function Base.showerror(io::IO, error::InvalidParametersError)
    print(io, "InvalidParametersError: ", error.message)
    return nothing
end
