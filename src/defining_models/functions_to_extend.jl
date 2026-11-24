## Functions that a model must implement ##

"""
    timestep_model(model::AbstractTimestepModel, context::ModelContext)::ModelOutput

Run a single timestep of `model`: update its state, and return what its outputs are sampled
from. For a cognitive model the outputs are its actions, and for an environment model the
observations it gives.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A
simulation samples the outputs itself, from the `output_distribution` of the
[`ModelOutput`](@ref) that the method returns, so an output that isn't random is wrapped in
a [`Deterministic`](@ref). In simulations, anything else that is random, such as a change of
the state, should be sampled from `context.rng`. If states are mutated in place, a
[`prepare_state_for_saving`](@ref) method must be added. For use within PPLs, this function
can also have special implementations, and there may be special constraints: see the
documentation for the specific PPL for details.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: BernoulliLogit

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.timestep_model(model::RescorlaWagner, context::ModelContext)

    ## Unpack the context ##
    #Reward observed on this timestep
    r = context.input.reward
    #Expected reward from the previous state
    V₋₁ = context.previous_state.expected_reward
    #Learning rate parameter
    α = context.parameters.learning_rate
    #Inverse temperature parameter
    β = context.parameters.inverse_temperature

    ## Update state ##
    #Update expected reward using the Rescorla-Wagner rule
    Vₜ = V₋₁ + α * (r - V₋₁)

    ## Compute action probabilities ##
    #Softmax function, instantiated as a Bernoulli logit distribution
    Pₜ = BernoulliLogit(β * (Vₜ - 0.5))

    return ModelOutput((; choice = Pₜ), (; expected_reward = Vₜ))
end

parameters = (; learning_rate = 0.1, inverse_temperature = 2.0)
context = ModelContext(; input = (; reward = 1.0), previous_state = (; expected_reward = 0.5), previous_output = (; choice = NoValue()), parameters)

CognitiveModeling.timestep_model(RescorlaWagner(), context).new_state

# output

(expected_reward = 0.55,)
```
"""
function timestep_model(model::AbstractTimestepModel, context)
    return error(
        "There is no timestep_model method for $(describe_model(model)) with a context of type $(typeof(context)). Each model must define CognitiveModeling.timestep_model(model::$(typeof(model)), context::CognitiveModeling.ModelContext).",
    )
end

"""
    initialize_state(model::AbstractSubmodel, parameters)

Create the initial state that `model` has before its first timestep, which can depend on
`parameters`. A submodel has one too, as an [`AbstractSubmodel`](@ref).

# Implementation
Every model must add a method of this function, which throws an error otherwise. It returns
a `NamedTuple` with one entry for each state in the model's attributes, also when the state
is a single object, as in `(; rl_env)`, and a model without a state returns an empty
`NamedTuple`, `(;)`. For use within PPLs, this function can also have special
implementations: see the documentation for the specific PPL for details.

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
function initialize_state(model::AbstractSubmodel, parameters)
    return error(
        "There is no initialize_state method for $(describe_model(model)). Each model must define CognitiveModeling.initialize_state(model::$(typeof(model)), parameters), which returns (;) if the model has no state.",
    )
end

"""
    attribute_names(model::AbstractSubmodel)::ModelAttributes

Return the names of the parameters, states, inputs and outputs of `model`, in a
[`ModelAttributes`](@ref). These are the names of the model's own attributes: those of its
submodels, which [`submodel_fields`](@ref) names, are their own.

The names define the model, so they must match what the model uses. The parameters are the
names that [`timestep_model`](@ref) and [`initialize_state`](@ref) read from the parameters.
The states are the names in the state that `initialize_state` returns and in `new_state`,
which `timestep_model` reads from `context.previous_state`. The inputs are the names that
`timestep_model` reads from `context.input`, and the outputs are the names in
`output_distribution`, which are also those in `context.previous_output`. For a cognitive
model the inputs are its observations and the outputs its actions, and for an environment
model the inputs are the actions it receives and the outputs the observations it gives.
Nothing checks that they match while the model runs.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A model
without one kind of attribute gives `()` for it. [`simulate_session`](@ref) and
[`InstantiatedModel`](@ref) use the output names for the output before the first timestep,
and check the parameters, the previous output and the state against the names. A simulation
with an environment also checks that the two models fit together: the outputs of each must
be the inputs of the other, unless a [`NameMap`](@ref) says which of them are passed on, and
under which names.

# Examples
```jldoctest
julia> struct RescorlaWagner <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::RescorlaWagner)
           return ModelAttributes(;
               parameters = (:learning_rate, :inverse_temperature),
               states = :expected_reward,
               inputs = :reward,
               outputs = :choice,
           )
       end;

julia> CognitiveModeling.attribute_names(RescorlaWagner()).parameters
(:learning_rate, :inverse_temperature)
```
"""
function attribute_names(model::AbstractSubmodel)
    return error(
        "There is no attribute_names method for $(describe_model(model)). Each model must define CognitiveModeling.attribute_names(model::$(typeof(model))), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs.",
    )
end

## Functions that a model can implement ##

"""
    prepare_state_for_saving(model::AbstractSubmodel, state)

Return what a simulation saves of `state`, the state of a cognitive model or of an
environment model, as it is about to save it. The state that the next timestep starts from
isn't affected.

# Implementation
A model can add a method of this function. The default returns `state` as it is, so nothing
is copied, apart from the state of each submodel, which the submodel's own method prepares. One which changes an object in its state in place must add a method which copies
it, or all the saved states show the final values, with no error. The method must return the
state with the names in the model's attributes, since a [`SaveHistory`](@ref) selects states
by those names. Nothing checks this while the model runs.

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
function prepare_state_for_saving(model::AbstractSubmodel, state)
    return prepare_submodel_states_for_saving(model, state)
end

"""
    default_parameters(model::AbstractSubmodel)

Return the default parameter values of a cognitive model or of an environment model, such as
`(; learning_rate = 0.1)`, which [`InstantiatedModel`](@ref) and [`simulate_session`](@ref)
use for the parameters they aren't given. The display of a model shows each default value
after the name of its parameter, and [`CognitiveModeling.get_default_parameters`](@ref)
gives them together with those of the model's submodels.

# Implementation
A model can add a method of this function. The default is an empty `NamedTuple`, `(;)`, so
there are no default values.
"""
function default_parameters(model::AbstractSubmodel)
    return (;)
end

"""
    parameter_domains(model::AbstractSubmodel)

Return the values that the parameters of `model`, a cognitive model or an environment
model, can take: a `NamedTuple` with a [`Domain`](@ref) under the name of each
parameter that has one. A parameter without a domain can take any value.

The parameters of a simulation and of a fitted session are checked against their domains
before anything runs, and a value outside its domain throws a
[`ParameterDomainError`](@ref), unless [`simulate_session`](@ref),
[`InstantiatedModel`](@ref) or [`SessionInfo`](@ref) is given
`check_parameter_domains = false`. [`CognitiveModeling.discrete_parameter_names`](@ref)
gives the discrete parameters, and [`CognitiveModeling.continuous_parameter_names`](@ref)
the continuous ones, which are sampled in different ways. The display of a model shows each
domain after the name of its parameter, and
[`CognitiveModeling.get_parameter_domains`](@ref) gives them together with those of the
model's submodels.

# Implementation
A model can add a method of this function. The default is an empty `NamedTuple`, `(;)`, so
that every parameter can take any value. A domain is best kept to the values the model
can't run without, since a value outside it stops a fit.

# Examples
```jldoctest
julia> struct CountsOptions <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::CountsOptions)
           return ModelAttributes(; parameters = (:learning_rate, :n_options, :bias), states = (), inputs = (), outputs = ())
       end;

julia> function CognitiveModeling.parameter_domains(model::CountsOptions)
           return (;
               learning_rate = Domain(; lower = 0, upper = 1),
               n_options = Domain(; type = Integer, lower = 1),
           )
       end;

julia> CognitiveModeling.discrete_parameter_names(CountsOptions())
(:n_options,)

julia> CountsOptions()
-- Cognitive model --
Type: CountsOptions
Parameters:
  learning_rate ∈ [0, 1]
  n_options ∈ {1, 2, …}
  bias
States: none
Inputs: none
Outputs: none
```
"""
function parameter_domains(model::AbstractSubmodel)
    return (;)
end

"""
    input_domains(model::AbstractTimestepModel)

Return the values that the inputs of `model`, a cognitive model or an environment model, can
take: a `NamedTuple` with a [`Domain`](@ref) under the name of each input that has one. An
input without a domain can take any value. [`output_domains`](@ref) gives the domains of
the outputs in the same way.

The data of a fitted session is checked against the domains of its cognitive model once,
when it is read: the observations against the input domains, and the actions against the
output domains. A value outside its domain throws an `ArgumentError` that names its row in
the table, or its timestep in the session, and markers such as [`ImputedMissingValue`](@ref)
aren't checked. The display of a model shows each domain after the name of its input or
output.

# Implementation
A model can add a method of this function and of `output_domains`. The defaults are an empty
`NamedTuple`, `(;)`, so that every input and every output can take any value.

# Examples
```jldoctest
julia> struct ChoosesOption <: AbstractCognitiveModel end

julia> function CognitiveModeling.attribute_names(model::ChoosesOption)
           return ModelAttributes(; parameters = (), states = (), inputs = :reward, outputs = :choice)
       end;

julia> CognitiveModeling.input_domains(model::ChoosesOption) = (; reward = Domain(; lower = 0, upper = 1));

julia> CognitiveModeling.output_domains(model::ChoosesOption) = (; choice = Domain(; possible_values = (1, 2)));

julia> ChoosesOption()
-- Cognitive model --
Type: ChoosesOption
Parameters: none
States: none
Inputs:
  reward ∈ [0, 1]
Outputs:
  choice ∈ {1, 2}

julia> SessionInfo(ChoosesOption(), [0.5, 1.0], [1, 3]; who_starts = EnvironmentStarts())
ERROR: ArgumentError: The action choice at timestep 2 is 3, which isn't one of its possible values, (1, 2).
```
"""
function input_domains(model::AbstractTimestepModel)
    return (;)
end

"""
    output_domains(model::AbstractTimestepModel)

Return the values that the outputs of `model`, a cognitive model or an environment model,
can take: a `NamedTuple` with a [`Domain`](@ref) under the name of each output that has one,
as [`input_domains`](@ref) does for the inputs, which also tells how the actions in the data
are checked against them and gives an example. The default is an empty `NamedTuple`, `(;)`,
so that every output can take any value.
"""
function output_domains(model::AbstractTimestepModel)
    return (;)
end

"""
    stopping_criterion(model::AbstractTimestepModel, state, parameters)

Return whether `model`, a cognitive model or an environment model, with `state` as its state
after a timestep and its `parameters`, has signaled for a simulation to stop there instead
of simulating all of `n_steps`.

[`simulate_session`](@ref) and [`simulate_session!`](@ref) check it for both models before
each timestep, with the states that the timestep before left. When it returns `true` for
either, no further timestep is simulated, and what is returned and added to the histories
has the timesteps that were simulated. When the environment starts, the cognitive model has
then made its last observation and acted on it, and the environment doesn't receive that
action. When the cognitive model starts, the environment has then responded to the last
action with an observation, and the cognitive model doesn't receive that observation. A
model whose criterion is already met before the first timestep can't be simulated, and
throws an error: a new simulation starts from the initial state, and [`reset!`](@ref) puts
an [`InstantiatedModel`](@ref) back there. A fitted session checks it before every timestep
too, and throws the same error when it is met before the first, as [`SessionInfo`](@ref)
describes. Under a compiler like Reactant.jl, a criterion on traced values returns a traced
Bool, and the simulation is then masked instead of stopped, as `simulate_session` describes.

# Implementation
A model can add a method of this function. The default returns `false`, so a simulation
never stops early.

# Examples
```jldoctest
julia> struct Maze <: AbstractEnvironmentModel end

julia> function CognitiveModeling.stopping_criterion(environment::Maze, state, parameters)
           return state.position == parameters.goal
       end;

julia> CognitiveModeling.stopping_criterion(Maze(), (; position = 4), (; goal = 4))
true
```
"""
function stopping_criterion(model::AbstractTimestepModel, state, parameters)
    return false
end

## Functions that a model with submodels implements ##

"""
    submodel_fields(model::AbstractSubmodel)::Tuple{Vararg{Symbol}}

Return the names of the fields of `model` that hold its submodels, each an
[`AbstractSubmodel`](@ref), as a tuple of `Symbol`s. The names are also those under which the
parameters and the state of each submodel are bundled. [`CognitiveModeling.submodels`](@ref)
reads the submodels from these fields.

# Implementation
A model with submodels adds a method of this function, which returns the names as a literal
tuple, so that they are known when the code is compiled. The default returns `()`, for a
model without submodels. A model whose submodels aren't fields of its own adds a method of
`submodels` instead.

# Examples
```jldoctest
julia> struct Perception <: AbstractSubmodel end

julia> struct Response <: AbstractSubmodel end

julia> struct PerceptionAndResponse <: AbstractCognitiveModel
           perception::Perception
           response::Response
       end

julia> CognitiveModeling.submodel_fields(model::PerceptionAndResponse) = (:perception, :response);

julia> CognitiveModeling.submodels(PerceptionAndResponse(Perception(), Response()))
(perception = Perception (submodel), response = Response (submodel))
```
"""
function submodel_fields(model::AbstractSubmodel)
    return ()
end

"""
    submodels(model::AbstractSubmodel)::NamedTuple

Return the submodels of `model` in a `NamedTuple`, each under the name that its parameters
and its state are bundled under. A model without submodels returns an empty `NamedTuple`,
`(;)`.

# Implementation
The default reads the fields that [`CognitiveModeling.submodel_fields`](@ref) names, which is
how most models declare their submodels. A model can add a method of this function instead,
when its submodels aren't fields of its own, such as a model that keeps them in a
`NamedTuple`. The method must return the same names whenever it is called on a model of the
same type, so that they are known when the code is compiled.
"""
function submodels(model::AbstractSubmodel)
    return fields_as_named_tuple(model, Val(submodel_fields(model)))
end

"""
    uses_submodel_prefixes(model::AbstractSubmodel)::Bool

Return whether the flat names of the parameters of the submodels of `model` start with the
name of their submodel and `__`, as in `perception__learning_rate`, so that two submodels can
have parameters of the same name. Without prefixes, the flat names are the submodels' own
names, as in `learning_rate`, and must then differ.

# Implementation
A model with submodels can add a method of this function. The default returns `false`. A
model whose submodels are chosen by its users, such as a generic combination of a perceptual
model and a response model, can take the choice as a keyword of its constructor and keep it
in its type, so that its own method returns it.
"""
function uses_submodel_prefixes(model::AbstractSubmodel)
    return false
end
