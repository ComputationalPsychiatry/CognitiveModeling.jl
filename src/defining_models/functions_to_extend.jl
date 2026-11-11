## Functions that a model must implement ##

"""
    apply_model(model::AbstractModel, context::ModelContext)::ModelOutput

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

function CognitiveModeling.apply_model(model::RescorlaWagner, context::ModelContext)

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
context = ModelContext(; input = (; reward = 1.0), previous_state = (; expected_reward = 0.5), previous_output = (; choice = NoAction()), parameters)

CognitiveModeling.apply_model(RescorlaWagner(), context).new_state

# output

(expected_reward = 0.55,)
```
"""
function apply_model(model::AbstractModel, context)
    return error(
        "There is no apply_model method for $(describe_model(model)) with a context of type $(typeof(context)). Each model must define CognitiveModeling.apply_model(model::$(typeof(model)), context::CognitiveModeling.ModelContext).",
    )
end

"""
    initialize_state(model::AbstractModel, parameters)

Create the initial state that `model` has before its first timestep, which can depend on
`parameters`.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A model
without a state returns an empty `NamedTuple`, `(;)`. For use within PPLs, this function
can also have special implementations: see the documentation for the specific PPL for
details.

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
function initialize_state(model::AbstractModel, parameters)
    return error(
        "There is no initialize_state method for $(describe_model(model)). Each model must define CognitiveModeling.initialize_state(model::$(typeof(model)), parameters), which returns (;) if the model has no state.",
    )
end

"""
    attribute_names(model::AbstractModel)::ModelAttributes

Return the names of the parameters, states, inputs and outputs of `model`, in a
[`ModelAttributes`](@ref).

The names define the model, so they must match what the model uses. The parameters are the
names that [`apply_model`](@ref) and [`initialize_state`](@ref) read from the parameters.
The states are the names in the state that `initialize_state` returns and in `new_state`,
which `apply_model` reads from `context.previous_state`. The inputs are the names that
`apply_model` reads from `context.input`, and the outputs are the names in
`output_distribution`, which are also those in `context.previous_output`. For a cognitive
model the inputs are its observations and the outputs its actions, and for an environment
model the inputs are the actions it receives and the outputs the observations it gives.
Nothing checks that they match while the model runs.

# Implementation
Every model must add a method of this function, which throws an error otherwise. A model
without one kind of attribute gives `()` for it, and so does one whose state isn't a
`NamedTuple`. [`simulate`](@ref) and [`InstantiatedModel`](@ref) use the output names for
the output before the first timestep, and check the parameters, the previous output and a
`NamedTuple` state against the names. A simulation with an environment also checks that the
two models fit together: the outputs of each must be the inputs of the other, unless a
[`NameMap`](@ref) says which of them are passed on, and under which names.

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
function attribute_names(model::AbstractModel)
    return error(
        "There is no attribute_names method for $(describe_model(model)). Each model must define CognitiveModeling.attribute_names(model::$(typeof(model))), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs.",
    )
end

## Functions that a model can implement ##

"""
    prepare_state_for_saving(model::AbstractModel, state)

Return what a simulation saves of `state`, the state of a cognitive model or of an
environment model, as it is about to save it. The state that the next timestep starts from
isn't affected.

# Implementation
A model can add a method of this function. The default returns `state` as it is, so nothing
is copied. One which changes an object in its state in place must add a method which copies
it, or all the saved states show the final values, with no error. The method must return the
state with the names that [`attribute_names`](@ref) lists for it, since a
[`SaveHistory`](@ref) selects states by those names. Nothing checks this while the model
runs.

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
function prepare_state_for_saving(model::AbstractModel, state)
    return state
end

"""
    default_parameters(model::AbstractModel)

Return the default parameter values of a cognitive model or of an environment model, such as
`(; learning_rate = 0.1)`, which [`InstantiatedModel`](@ref) and [`simulate`](@ref) use for
the parameters they aren't given.

# Implementation
A model can add a method of this function. The default is an empty `NamedTuple`, `(;)`, so
there are no default values.
"""
function default_parameters(model::AbstractModel)
    return (;)
end

"""
    parameter_domains(model::AbstractModel)

Return the values that the parameters of `model`, a cognitive model or an environment
model, can take: a `NamedTuple` from the name of a parameter to its domain, which is a
`NamedTuple` with any of `lower`, the smallest value it can have, `upper`, the largest, and
`discrete`, whether it must be a whole number. What a domain leaves out is unbounded and
continuous, and so is a parameter without a domain.

The parameters of a simulation and of a fitted session are checked against their domains
before anything runs, and a value outside its domain throws an
[`InvalidParametersError`](@ref). An inference backend can also warn before a fit when a
prior gives values outside a domain, and
[`CognitiveModeling.discrete_parameter_names`](@ref) gives the discrete parameters, which
are sampled in another way than the continuous ones.

# Implementation
A model can add a method of this function. The default is an empty `NamedTuple`, `(;)`, so
that every parameter can take any value. A domain is best kept to the values the model
can't run without, since a value outside it can't be fitted at all.

# Examples
```jldoctest
julia> struct CountsOptions <: AbstractCognitiveModel end

julia> function CognitiveModeling.parameter_domains(model::CountsOptions)
           return (;
               learning_rate = (; lower = 0, upper = 1),
               n_options = (; lower = 1, discrete = true),
           )
       end;

julia> CognitiveModeling.discrete_parameter_names(CountsOptions())
(:n_options,)
```
"""
function parameter_domains(model::AbstractModel)
    return (;)
end

"""
    stopping_criterion(model::AbstractModel, state, parameters)

Return whether `model`, a cognitive model or an environment model, with `state` as its state
after a timestep and its `parameters`, has signaled for a simulation to stop there instead
of running all of `n_steps`.

[`simulate`](@ref) and [`simulate!`](@ref) check it for both models before each timestep,
with the states that the timestep before left, unless they are given
`check_stopping_criterion = false`. When it returns `true` for either, no further timestep
runs, and what is returned and added to the histories has the timesteps that ran. When the
environment starts, the cognitive model has then made its last observation and acted on it,
and the environment doesn't receive that action. When the cognitive model starts, the
environment has then responded to the last action with an observation, and the cognitive
model doesn't receive that observation. A model whose criterion is already met
before the first timestep can't be simulated, and throws an error: a new simulation starts
from the initial state, and [`reset!`](@ref) puts an [`InstantiatedModel`](@ref) back there.
A fitted session checks it as [`SessionInfo`](@ref) describes.

# Implementation
A model can add a method of this function. The default returns `false`, so a simulation runs
all of its timesteps.

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
function stopping_criterion(model::AbstractModel, state, parameters)
    return false
end
