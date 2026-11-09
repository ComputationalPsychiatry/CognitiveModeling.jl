## Functions that a model or an environment model can implement ##

"""
    turing_apply_model(model::AbstractModel, context::ModelContext)

Create a Turing.jl model for a single timestep of `model`, a cognitive model or an
environment model, to be used as a submodel, which returns a [`ModelOutput`](@ref). This
needs Turing.jl to be loaded.

# Implementation
A model can add a method of this function. The default calls [`apply_model`](@ref). A
model which samples within a timestep needs to add a method, however; or if any other
changes are desired within Turing. It must sample with `~` instead of from `context.rng`.
Otherwise the model has a different log density each time it is evaluated, with no error.
The same holds for an environment model, which fitting uses to fill in observations that
are [`MissingValue`](@ref) in the data: what it samples itself, such as a change of its
state, must be sampled with `~`, while fitting samples the observations from what it
returns.

# Examples
```julia
using CognitiveModeling
using Turing

struct NoisyRescorlaWagner <: AbstractCognitiveModel end

Turing.@model function CognitiveModeling.turing_apply_model(
        model::NoisyRescorlaWagner,
        context::ModelContext,
    )

    ## Unpack the context ##
    #Reward observed on this timestep
    r = context.input.reward
    #Expected reward from the previous state
    V₋₁ = context.previous_state.expected_reward
    #Learning rate parameter
    α = context.parameters.learning_rate

    ## Update state ##
    #Sample the noise in the update
    ϵ ~ Normal(0, context.parameters.noise)
    #Update expected reward using the Rescorla-Wagner rule
    Vₜ = V₋₁ + α * (r - V₋₁) + ϵ

    return ModelOutput((; choice = BernoulliLogit(Vₜ - 0.5)), (; expected_reward = Vₜ))
end
```
"""
function turing_apply_model end

"""
    turing_initialize_state(model::AbstractModel, parameters)

Create a Turing.jl model which gives the state of `model`, a cognitive model or an
environment model, before the first timestep, to be used as a submodel, as
[`initialize_state`](@ref) does. This needs Turing.jl to be loaded.

# Implementation
A model can add a method of this function. The default calls `initialize_state`, which is
enough for most models, since a starting state that is fitted is simply one that is copied
from a parameter there: a fitted session runs it with the parameters of each evaluation, so
a prior on the parameter is a prior on the starting state. A model which samples something
of its own when it initializes needs to add a method, and sample it with `~`, as for
[`turing_apply_model`](@ref). Its variables are named under `initial_cognitive_state`, or
`initial_environment_state` for an environment model.

# Examples
```julia
using CognitiveModeling
using Turing

struct UncertainStart <: AbstractCognitiveModel end

Turing.@model function CognitiveModeling.turing_initialize_state(
        model::UncertainStart,
        parameters,
    )

    #Sample where the expected reward starts, around the parameter for it
    V₀ ~ Normal(parameters.initial_value, parameters.initial_noise)

    return (; expected_reward = V₀)
end
```
"""
function turing_initialize_state end

## Functions that the parts of a full model implement ##

"""
    turing_session_parameters_model(parameters_model, sessions_info::MultipleSessionsInfo)

Create the Turing.jl model which gives the parameters of every session in `sessions_info`,
from a description of how they are given or sampled, for the cognitive model or the
environment model of a [`turing_full_model`](@ref). This needs Turing.jl to be loaded.

The model must return a `NamedTuple` with one entry for each parameter, named as the model
names it, and a vector in each with one value for every session, in the order of the
sessions. Each subtype of [`AbstractSessionParametersModel`](@ref) that comes with the package
has a method.

# Implementation
A parameters model of another kind, such as a hierarchical model or a regression, is a
subtype of `AbstractSessionParametersModel` with a method of this function, which can read
[`n_sessions`](@ref) and the identifiers of the sessions from `sessions_info`, and anything
else from its own fields, such as the covariates of a regression. The full model treats it as
one that samples, so that with it the environment of every session runs, and the
observations in the data inform the environment's parameters.
"""
function turing_session_parameters_model end

"""
    turing_downstream_model(downstream_model, saved_behavioural_values, cognitive_parameters,
                            sessions_info::MultipleSessionsInfo)

Create the Turing.jl model which runs after the sessions in a [`turing_full_model`](@ref),
from a description of it, with what the sessions saved, the cognitive parameters of every
session as the parameters model returned them, and the sessions' `sessions_info`. This needs
Turing.jl to be loaded.

The model can return anything, which the full model returns as its `downstream_values`.
[`NoDownstreamModel`](@ref) is the default, with a method which samples nothing and returns
`nothing`.

# Implementation
A downstream model is a subtype of [`AbstractDownstreamModel`](@ref) with a method of this
function, which can ignore any of the arguments. A regression of some measurement on a state
of the cognitive model, for instance, reads the state from `saved_behavioural_values`, a
vector with the saved values of each session as [`turing_behavioural_session`](@ref) returns
them, so the `cognitive_save_history` of the sessions must select that state.
"""
function turing_downstream_model end
