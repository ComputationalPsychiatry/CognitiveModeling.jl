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
Create a subtype and add methods of [`run_timestep`](@ref), [`initialize_state`](@ref) and
[`attribute_names`](@ref) for it. A model can also add methods of
[`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
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
[`run_timestep`](@ref), [`initialize_state`](@ref) and [`attribute_names`](@ref), and
optionally [`prepare_state_for_saving`](@ref), [`default_parameters`](@ref) and
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
start with when their [`SaveHistory`](@ref) saves the initial timestep values, as it does by
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
a simulation. [`AgentStarts`](@ref) and [`EnvironmentStarts`](@ref) are the other two, which
a simulation or a fitted session passes on from its own `who_starts`.
"""
struct UnknownWhoStarts end

## What a timestep receives and returns ##

"""
    ModelContext(; input, previous_state, previous_output, parameters, timestep_size=1,
                 rng=default_rng(), who_starts=UnknownWhoStarts())
    ModelContext(input, previous_state, previous_output, parameters, timestep_size=1,
                 rng=default_rng(), who_starts=UnknownWhoStarts())

Everything a model needs to run a single timestep, which [`run_timestep`](@ref) receives.
The arguments can be given by name, which says what each one is, or in this order.

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

The outputs of a model for a single timestep, which [`run_timestep`](@ref) returns.

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
- `new_state::NamedTuple=(;)`: the state of the model after this timestep, which is the
  `previous_state` of the next one, with one entry for each state in the model's
  attributes, also when the state is a single object, as in `(; rl_env)`. A model without a
  state gives an empty `NamedTuple`, `(;)`. The outputs are sampled after the state is made,
  so it can't depend on them: the model receives them as `context.previous_output` on the
  next timestep.

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

    #Only takes named output distributions and a named state, so that the model names every output and state
    function ModelOutput(output_distribution::NamedTuple, new_state::NamedTuple=(;))
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
into one. All four must be given, and `()` means that the model has none of that kind. A
name can only be used once across the four kinds, so that it says which attribute of the
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
        as_tuple(parameters)::Tuple{Vararg{Symbol}},
        as_tuple(states)::Tuple{Vararg{Symbol}},
        as_tuple(inputs)::Tuple{Vararg{Symbol}},
        as_tuple(outputs)::Tuple{Vararg{Symbol}},
    }()
    check_unique_attribute_names(attributes)
    return attributes
end

"""
    as_tuple(values)

Return `values` as a `Tuple`: a tuple as it is, a vector as a tuple, and anything else in a
tuple of one. Used for the arguments that take one value or several: the names given to
[`ModelAttributes`](@ref), [`SaveHistory`](@ref), [`MultipleSessionsInfo`](@ref),
[`HierarchicalGroupedParameters`](@ref) and [`HierarchicalRegressionParameters`](@ref),
which are then asserted to be `Symbol`s, and the values that stand for a missing value in
[`SessionInfo`](@ref).
"""
as_tuple(values::Tuple) = values
as_tuple(values::AbstractVector) = Tuple(values)
as_tuple(value) = (value,)

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

## Kinds of attributes ##

"""
    AbstractAttributeKind

Supertype of the kinds of attributes of a model: [`ParameterKind`](@ref) for its parameters,
[`StateKind`](@ref) for its states, and the subtypes of
[`AbstractObservableAttributeKind`](@ref), [`ObservationKind`](@ref) for the observations
and [`ActionKind`](@ref) for the actions. Used to say which kind of attribute something is.
"""
abstract type AbstractAttributeKind end

"""
    AbstractObservableAttributeKind

Supertype of the kinds of attributes that the data of a session has:
[`ObservationKind`](@ref) for the observations, which are the inputs in the attributes of
its cognitive model, and [`ActionKind`](@ref) for the actions, which are its outputs. Used
when a table is split into sessions and when the data of a session is named, to say which
of the two a column or a vector of data is.
"""
abstract type AbstractObservableAttributeKind <: AbstractAttributeKind end

"""
    ObservationKind()

The kind of attribute that the observations of a session are, see
[`AbstractObservableAttributeKind`](@ref).
"""
struct ObservationKind <: AbstractObservableAttributeKind end

"""
    ActionKind()

The kind of attribute that the actions of a session are, see
[`AbstractObservableAttributeKind`](@ref).
"""
struct ActionKind <: AbstractObservableAttributeKind end

"""
    ParameterKind()

The kind of attribute that the parameters of a model are, see
[`AbstractAttributeKind`](@ref).
"""
struct ParameterKind <: AbstractAttributeKind end

"""
    StateKind()

The kind of attribute that the states of a model are, see [`AbstractAttributeKind`](@ref).
"""
struct StateKind <: AbstractAttributeKind end

## The values a parameter can take ##

"""
    ParameterDomain(; type, lower=-Inf, upper=Inf, possible_values=nothing,
                    constraint=nothing)

The values that a parameter of a cognitive model or an environment model can take, which
the model gives for it in [`parameter_domains`](@ref): values of `type`, with the numbers
in them between `lower` and `upper`, or among `possible_values`, which satisfy
`constraint`.

# Keyword arguments
- `type`: the type of the parameter's values, `Real` unless it is given, or the type of the
  possible values when there are some. An abstract type takes every type of its
  kind, so with `Real` a value can be a `Float32`, a `Float64` or a number of an automatic
  differentiation package alike, and no value is converted. A type of whole numbers, such as
  `Integer`, takes every whole number, also in another number type, such as `3.0`, and
  `Bool` takes `true`, `false`, 0 and 1. A type that isn't a type of numbers, such as
  `AbstractString` or `Symbol`, is for a parameter such as the name of a strategy. An array
  type, such as `AbstractVector{Real}` or `AbstractMatrix{Integer}`, takes every array with
  as many dimensions, and a tuple type, such as `NTuple{3,Real}` or `Tuple{Vararg{Real}}`,
  takes tuples of its length, with each element a value of the type of the elements, so
  that `AbstractVector{Real}` takes a `Vector{Float64}` and a vector of numbers of automatic
  differentiation alike. An array type or a tuple type that leaves the type of its elements
  open, such as `AbstractVector{<:Real}` or `NTuple{3}`, is an error: give the type of the
  elements, as `AbstractVector{Real}` does. A type of floating-point numbers, such as
  `Float64` or `AbstractVector{Float64}`, is an error too: give `Real` or
  `AbstractVector{Real}`, which take them all.
- `lower=-Inf` and `upper=Inf`: the smallest and the largest value of a parameter that is a
  real number, or of each element of an array or a tuple of real numbers, which are both in
  the domain. A parameter of another type has no bounds.
- `possible_values=nothing`: the values that the parameter, or each element of an array or a
  tuple, can take, in a tuple, such as `(:greedy, :random)` or `(2, 4, 8)`, or in a range,
  such as `1:4`. A number is compared by its value, so `4.0` is among `(2, 4, 8)`. Without
  `type`, the type is that of the possible values: `Symbol` for symbols, `Integer` for whole
  numbers, `Real` for other numbers and `AbstractString` for strings. A domain with possible
  values has no bounds.
- `constraint=nothing`: a function of the parameter's whole value which returns `true` when
  the value is in the domain, such as `sums_to_one(values) = sum(values) ≈ 1` for the
  probabilities of the options of a choice. It is called after the type, the bounds and the
  possible values are checked, and a named function names the constraint in errors and in
  the display of a model.

A parameter is discrete when it has possible values, or when its type, or the type of its
elements, is a type of whole numbers, or isn't a type of real numbers, and continuous
otherwise, as [`CognitiveModeling.discrete_parameter_names`](@ref) and
[`CognitiveModeling.continuous_parameter_names`](@ref) read it. The arguments are kept as
`domain.type`, `domain.lower`, `domain.upper`, `domain.possible_values` and
`domain.constraint`.

# Examples
```jldoctest
julia> learning_rate = ParameterDomain(; lower = 0, upper = 1)
ParameterDomain(; lower = 0, upper = 1)

julia> learning_rate.type, learning_rate.lower, learning_rate.upper
(Real, 0, 1)

julia> ParameterDomain(; type = Integer, lower = 1)
ParameterDomain(; type = Integer, lower = 1)

julia> ParameterDomain(; type = AbstractVector{Real}, lower = 0, upper = 1)
ParameterDomain(; type = AbstractVector{Real}, lower = 0, upper = 1)

julia> strategy = ParameterDomain(; possible_values = (:greedy, :random));

julia> strategy.type, strategy.possible_values
(Symbol, (:greedy, :random))

julia> sums_to_one(values) = sum(values) ≈ 1;

julia> ParameterDomain(; type = AbstractVector{Real}, constraint = sums_to_one)
ParameterDomain(; type = AbstractVector{Real}, constraint = sums_to_one)
```
"""
struct ParameterDomain{
    Ttype,Telement,Tlower<:Real,Tupper<:Real,Tpossible_values,Tconstraint
}
    lower::Tlower
    upper::Tupper
    possible_values::Tpossible_values
    constraint::Tconstraint

    #Takes the type of the values, the bounds, the possible values and the constraint, and checks that they fit together. The type and the type of its elements are kept in the type of the domain, so that a value is checked against them without looking them up when a model runs
    function ParameterDomain(;
        type::Union{Type,Nothing}=nothing,
        lower::Real=(-Inf),
        upper::Real=Inf,
        possible_values=nothing,
        constraint=nothing,
    )

        #The possible values are in a tuple or a range, which are the same objects each time parameter_domains runs. A vector would be made again each time, and its length isn't known when the code is compiled
        if !(possible_values isa Union{Nothing,Tuple,AbstractRange})
            throw(
                ArgumentError(
                    "A ParameterDomain takes its possible values in a tuple, such as (:greedy, :random), or in a range, such as 1:4, but it was given possible_values = $(repr(possible_values)).",
                ),
            )
        elseif possible_values !== nothing && isempty(possible_values)
            throw(
                ArgumentError(
                    "A ParameterDomain takes at least one possible value, but it was given possible_values = $(repr(possible_values)).",
                ),
            )
        end

        #Without a type, the type is that of the possible values, or Real
        type = something(type, type_of_possible_values(possible_values))

        #An array type or a tuple type that leaves the type of its elements open, as AbstractVector{<:Real} does, is a new type each time parameter_domains runs, which makes every check slow, and which Mooncake.jl can't differentiate through
        if has_open_element_type(type)
            throw(
                ArgumentError(
                    "A ParameterDomain was given type = $type, which leaves the type of its elements open. Give the type of the elements, as in AbstractVector{Real}, which takes arrays with elements of every real number type, NTuple{3,Real}, or AbstractVector{Any} for elements of any type.",
                ),
            )
        end

        #The type of the elements of an array or a tuple, and of a single value the type itself
        element_type = type_of_elements(type)

        #A type of floating-point numbers would leave out the other ones, and the numbers of automatic differentiation packages, also as the type of the elements
        if type <: AbstractFloat
            throw(
                ArgumentError(
                    "A ParameterDomain was given type = $type, but a type of floating-point numbers leaves out the others, and the numbers of automatic differentiation packages. Give Real, the default, for a parameter that is a real number.",
                ),
            )
        elseif element_type <: AbstractFloat
            throw(
                ArgumentError(
                    "A ParameterDomain was given type = $type, whose elements are floating-point numbers, but these leave out the others, and the numbers of automatic differentiation packages. Give elements of type Real, as in AbstractVector{Real} or NTuple{3,Real}.",
                ),
            )
        end

        #Only real numbers have bounds, also as the elements of an array or a tuple
        if !(element_type <: Real) && (lower != -Inf || upper != Inf)
            throw(
                ArgumentError(
                    "A ParameterDomain with type = $type was given the bounds lower = $lower and upper = $upper, but only a parameter that is a real number, or an array or a tuple of real numbers, has bounds.",
                ),
            )
        end

        #The bounds are numbers, with the lower one at most the upper one
        if isnan(lower) || isnan(upper) || lower > upper
            throw(
                ArgumentError(
                    "The lower bound of a ParameterDomain is at most its upper bound, but it was given lower = $lower and upper = $upper.",
                ),
            )
        end

        #Possible values say what the parameter takes, which bounds would only repeat or contradict
        if possible_values !== nothing && (lower != -Inf || upper != Inf)
            throw(
                ArgumentError(
                    "A ParameterDomain takes possible values or bounds, but it was given possible_values = $(repr(possible_values)) and the bounds lower = $lower and upper = $upper.",
                ),
            )
        end

        #Each possible value is a value of the type of the elements
        if possible_values !== nothing &&
            !all(value -> is_value_of_type(value, element_type), possible_values)
            throw(
                ArgumentError(
                    "A ParameterDomain with type = $type was given possible_values = $(repr(possible_values)), but each of them must be a value of type $element_type.",
                ),
            )
        end
        return new{
            type,
            element_type,
            typeof(lower),
            typeof(upper),
            typeof(possible_values),
            typeof(constraint),
        }(
            lower, upper, possible_values, constraint
        )
    end
end

"""
    type_of_possible_values(possible_values)

Return the type that a [`ParameterDomain`](@ref) with `possible_values` and no type has:
`Real` without possible values, and otherwise the type of the possible values, widened to
`Integer` for whole numbers, `Real` for other numbers and `AbstractString` for strings, so
that `(2, 4, 8)` is `Integer` and `(:greedy, :random)` is `Symbol`.
"""
type_of_possible_values(::Nothing) = Real
function type_of_possible_values(possible_values::Union{Tuple,AbstractRange})
    type = eltype(possible_values)
    type <: Bool && return Bool
    type <: Integer && return Integer
    type <: Real && return Real
    type <: AbstractString && return AbstractString
    return type
end

"""
    is_value_of_type(value, type)

Return whether `value` is a value of `type`, as a [`ParameterDomain`](@ref) takes it: a real
number as [`takes_value`](@ref) decides, and any other value when it is of the type. Used by
`ParameterDomain` for its possible values.
"""
is_value_of_type(value::Real, type::Type) = type <: Real && takes_value(type, value)
is_value_of_type(value, type::Type) = value isa type

"""
    type_of_elements(type)

Return the type of the elements of `type` when it is an array type or a tuple type, such as
`Real` for `AbstractVector{Real}` or `NTuple{3,Real}`, and `type` itself otherwise. Used by
[`ParameterDomain`](@ref).
"""
function type_of_elements(type::Type)
    type <: Union{AbstractArray,Tuple} && return eltype(type)
    return type
end

#Julia's eltype gives Any for an array type whose element type is open, as for AbstractVector{<:Real}, and also for AbstractVector{Any}, which is the only one of them whose elements are all of type Any
"""
    has_open_element_type(type)

Return whether `type` is an array type which leaves the type of its elements open, such as
`AbstractVector{<:Real}` or `AbstractVector`, or a tuple type with a type variable, such as
`NTuple{3}`. Used by [`ParameterDomain`](@ref).
"""
function has_open_element_type(type::Type)
    type <: AbstractArray && return eltype(type) === Any && !(type <: AbstractArray{Any})
    type <: Tuple && return !(type isa DataType)
    return false
end

#Reads the type of the values, which is in the type of the domain, as a property next to the bounds
function Base.getproperty(domain::ParameterDomain{Ttype}, name::Symbol) where {Ttype}
    name === :type && return Ttype
    return getfield(domain, name)
end

function Base.propertynames(::ParameterDomain)
    return (:type, :lower, :upper, :possible_values, :constraint)
end

#Shows the domain as it is written, with the arguments that aren't the defaults
function Base.show(io::IO, domain::ParameterDomain)
    arguments = String[]
    domain.type === type_of_possible_values(domain.possible_values) ||
        push!(arguments, "type = $(domain.type)")
    domain.lower == -Inf || push!(arguments, "lower = $(domain.lower)")
    domain.upper == Inf || push!(arguments, "upper = $(domain.upper)")
    domain.possible_values === nothing ||
        push!(arguments, "possible_values = $(repr(domain.possible_values))")
    if domain.constraint !== nothing
        name = something(name_of_constraint(domain.constraint), "an anonymous function")
        push!(arguments, "constraint = $name")
    end
    print(io, "ParameterDomain(")
    isempty(arguments) || print(io, "; ", join(arguments, ", "))
    print(io, ")")
    return nothing
end

"""
    is_discrete(domain)

Return whether a parameter with `domain`, a [`ParameterDomain`](@ref), is discrete: when it
has possible values, or when its type, or the type of its elements when it is an array type
or a tuple type, is a type of whole numbers, or isn't a type of real numbers. Used by
[`discrete_parameter_names`](@ref) and by the display of a model.
"""
function is_discrete(
    ::ParameterDomain{Ttype,Telement,Tlower,Tupper,Tpossible_values}
) where {Ttype,Telement,Tlower,Tupper,Tpossible_values}
    return Tpossible_values !== Nothing || !(Telement <: Real) || Telement <: Integer
end

"""
    name_of_constraint(constraint)

Return the name of `constraint`, the constraint of a [`ParameterDomain`](@ref), as errors and
the display of a model write it: the name of a function, such as `sums_to_one`, `nothing` for
an anonymous function, and the object as it is shown otherwise.
"""
function name_of_constraint(constraint::Function)
    name = string(nameof(constraint))
    return startswith(name, "#") ? nothing : name
end
name_of_constraint(constraint) = repr(constraint)

## Errors ##

"""
    InvalidParametersError(message)
    InvalidParametersError()

An error for a model to throw from [`run_timestep`](@ref) or [`initialize_state`](@ref) when
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
    return InvalidParametersError(
        "Model parameters are not valid, possibly dependent on observations, actions and states.",
    )
end

#Prints the message when the error isn't caught
function Base.showerror(io::IO, error::InvalidParametersError)
    print(io, "InvalidParametersError: ", error.message)
    return nothing
end

"""
    ParameterDomainError(message)

The error that is thrown when a parameter of a cognitive model or an environment model has a
value outside its domain in [`parameter_domains`](@ref): of another type than the domain's,
or below its lower bound or above its upper bound.

A simulation and a fit stop with it. A fit which catches an
[`InvalidParametersError`](@ref), to give the parameters of a sample a probability of zero,
doesn't catch this one. Give the parameters priors or transformations that stay within the
domains instead, or leave the check out with `check_parameter_domains = false` in
[`simulate`](@ref), [`InstantiatedModel`](@ref) or [`SessionInfo`](@ref).

# Examples
```jldoctest
julia> sprint(showerror, ParameterDomainError("The parameter learning_rate is 1.5, which is outside its domain, from 0 to 1."))
"ParameterDomainError: The parameter learning_rate is 1.5, which is outside its domain, from 0 to 1."
```
"""
struct ParameterDomainError <: Exception
    message::String
end

#Prints the message when the error isn't caught
function Base.showerror(io::IO, error::ParameterDomainError)
    print(io, "ParameterDomainError: ", error.message)
    return nothing
end
