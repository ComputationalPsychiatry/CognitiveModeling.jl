## Turing.jl models which users call, defined in the Turing extension ##

"""
    turing_behavioural_session(session_info::SessionInfo, cognitive_parameters)
    turing_behavioural_session(session_info::SessionInfo, cognitive_parameters,
                               environment_parameters=(;))

Create a Turing.jl model of one session: the cognitive model in `session_info` goes through
the timesteps of its data with `cognitive_parameters`, and the recorded actions are scored
against the action distributions it gives. The second form is for a session with an
environment model, which takes its parameters too. This needs Turing.jl to be loaded.

The model is a building block for a Turing.jl model which samples the parameters, as a
submodel with its own prefix: `to_submodel(prefix(turing_behavioural_session(...),
@varname(session)), false)`. [`turing_multiple_behavioural_sessions`](@ref) runs one for
each of several sessions, and [`turing_full_model`](@ref) samples their parameters too.

# Arguments
- `session_info::SessionInfo`: the data, the models and the settings of the session,
  prepared once with [`SessionInfo`](@ref).
- `cognitive_parameters::NamedTuple`: the parameters of the cognitive model, usually the
  random variables of the model around this one. Those it doesn't have take the values in
  [`default_parameters`](@ref), if the model provides defaults for them.
- `environment_parameters::NamedTuple=(;)`: the parameters of the environment model, with
  its default values for those that aren't given, if it provides any. A session without an
  environment takes none.

# What it does
The states where the session starts come from [`turing_initialize_state`](@ref) of each
model with the parameters, so that a starting state which is a parameter is fitted like any
other, and a model can sample its start. On each timestep, [`turing_apply_model`](@ref) of
the cognitive model runs as a submodel with the prefix `timestep[t]`, so a model which
samples within a timestep has its own variables for each timestep, such as
`timestep[3].noise`. An environment model runs on every timestep too, with the prefix
`timestep[t].environment`, and what it gives for an observation is used where the data has
[`MissingValue`](@ref): the missing observation is imputed from it, as a latent variable with
that distribution, or as the value itself if it is a [`Deterministic`](@ref). When no
observation in the data is missing, the environment doesn't run, and the session is the
same as one without an environment, with the same variables. A missing
action is a latent variable with the cognitive model's action distribution on that timestep,
named `timestep[3].choice.imputed`, and a missing observation imputed from a distribution is
named `timestep[3].environment.reward.imputed`. The sampler must handle the type of a latent
variable: a continuous action can be sampled with NUTS, a discrete one needs a sampler for
discrete variables. The recorded actions are scored together at the end, in one
`product_distribution` over every action and timestep, which is several times faster than
scoring each timestep on its own.

# What it returns
What the `save_history` of `session_info` selects of the observations, the actions and the
states of the session, as a `NamedTuple` of vectors with one element for timestep 0, before
the first timestep, and one for each timestep after it, which `returned(model, chain)` gives
for each sample. Timestep 0 has the previous observation and action, which are
[`BeforeFirstTimestep`](@ref) unless others are given, and the initial state, which can
depend on the parameters. Element `t + 1` is then timestep `t`, and a [`SaveHistory`](@ref)
with `save_initial_values = false` leaves timestep 0 out. The returned actions and
observations are those the model received, with the imputed values. Its `timestep_sizes`
has the size of each timestep, after `BeforeFirstTimestep()` for timestep 0 when it is
there, when anything is selected, and is empty otherwise. Any latent variable in a session,
imputed or from a model which samples within a timestep, makes evaluating the model much
slower in the current version of DynamicPPL.jl.

# Examples
```julia
using CognitiveModeling
using Turing

#The model to fit, a Rescorla-Wagner model with a softmax choice
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = parameters.initial_value)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    Pₜ = BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))
    return ModelOutput((; choice = Pₜ), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(;
        parameters = (:learning_rate, :inverse_temperature, :initial_value),
        states = :expected_reward,
        observations = :reward,
        actions = :choice,
    )
end

CognitiveModeling.default_parameters(model::RescorlaWagner) = (; initial_value = 0.5)

#The data of one session, where the third choice wasn't recorded, prepared once
session_info = SessionInfo(
    RescorlaWagner(),
    (; reward = [1.0, 0.0, 1.0, 1.0]),
    (; choice = [true, false, MissingValue(), true]);
    who_starts = EnvironmentStarts(),
    save_history = SaveHistory(; states = :expected_reward),
)

#Priors on the parameters, and the session as a submodel
@model function fit_one_session(session_info)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ LogNormal(0, 1)
    session ~ to_submodel(
        prefix(
            turing_behavioural_session(session_info, (; learning_rate, inverse_temperature)),
            @varname(session),
        ),
        false,
    )
    return session
end

chain = sample(fit_one_session(session_info), NUTS(), 1000)

#The expected rewards on each timestep, for each sample
trajectories = returned(fit_one_session(session_info), chain)
```
"""
function turing_behavioural_session end

"""
    turing_multiple_behavioural_sessions(sessions_info::MultipleSessionsInfo,
                                         cognitive_parameters, environment_parameters=(;);
                                         parallelisation=NoParallelisation())

Create a Turing.jl model which runs [`turing_behavioural_session`](@ref) for every session
in `sessions_info`, each with its own cognitive parameters, as a submodel with the prefix
`session[i]`, so that a latent variable of the third session is named like
`session[3].timestep[2].choice.imputed`. Which session `i` is stands in
`sessions_info.session_ids[i]`, and [`session_index`](@ref) gives `i` for an identifier. The
environment parameters are the same for every session, and can only be given when the
sessions have an environment model. With `parallelisation = ParallelSessions()`, the sessions run over
the threads Julia was started with. **The model that is evaluated must then be marked with
`setthreadsafe(model, true)` from DynamicPPL**, since the log density of a threaded model
that isn't is wrong with no error; the sessions throw an error when it isn't marked. See
[`ParallelSessions`](@ref). This needs Turing.jl to be loaded.

`cognitive_parameters` is a `NamedTuple` with one vector for each parameter, with one value
for every session in the order of `sessions_info`, which is what a population model returns,
see [`turing_population_model`](@ref). It returns a vector with what each session returns,
in the same order. The model is a building block for a Turing.jl model which samples the
parameters of the sessions in some way of its own, as a submodel without a prefix:
`to_submodel(turing_multiple_behavioural_sessions(sessions_info, parameters), false)`.
[`turing_full_model`](@ref) does this with a population model.

The session identifiers aren't in the names of the variables because each identifier would
give its sessions a prefix of its own type, and DynamicPPL compiles the code of a session
once for each: with 40 sessions, the first evaluation took 1.5 s instead of 0.2, and every
one after it 44 ns per timestep instead of 9.
"""
function turing_multiple_behavioural_sessions end

"""
    turing_full_model(population_model, sessions_info::MultipleSessionsInfo;
                      downstream_model=NoDownstreamModel(), environment_parameters=(;),
                      catch_invalid_parameters=false, parallelisation=NoParallelisation(),
                      include_submodel_prefixes=false)

Create the Turing.jl model which fits a cognitive model to several sessions, in three parts:
`population_model` samples the cognitive parameters of every session, the sessions in
`sessions_info` run with them as [`turing_multiple_behavioural_sessions`](@ref), and
`downstream_model` can do something with what they saved, such as a regression of a
measurement on a state. This needs Turing.jl to be loaded.

# Arguments
- `population_model`: a description of how the parameters of the sessions are sampled, which
  [`turing_population_model`](@ref) has a method for, such as [`IndependentSessions`](@ref)
  with a prior for each parameter.
- `sessions_info::MultipleSessionsInfo`: the sessions, prepared once with
  [`MultipleSessionsInfo`](@ref).
- `downstream_model=NoDownstreamModel()`: a description of what runs after the sessions,
  which [`turing_downstream_model`](@ref) has a method for. The default samples nothing.
- `environment_parameters::NamedTuple=(;)`: the parameters of the environment model of the
  sessions, the same for all of them, when they have one. Sessions without an environment
  take none.
- `catch_invalid_parameters::Bool=false`: whether the sessions run inside
  [`turing_catch_invalid_parameters`](@ref), so that an [`InvalidParametersError`](@ref)
  rejects the sample instead of stopping the sampler. The population model runs outside it,
  so that a rejected first draw keeps its parameters and the sampler can start, and the model
  returns `nothing` for a rejected sample, without running the downstream model. A
  downstream model with variables of its own then lacks them in a rejected draw, so a
  sampler can't start from one: give it valid `initial_params` in that case.
- `parallelisation=NoParallelisation()`: whether the sessions run over the threads Julia was
  started with, with `ParallelSessions()`. **The model must then be marked with
  `setthreadsafe(model, true)` from DynamicPPL before it is sampled**: without the mark, a
  threaded model has a wrong log density with no error, so the sessions throw an error
  instead. The gradient is right with ForwardDiff and ReverseDiff, and Mooncake and Enzyme
  throw. See [`ParallelSessions`](@ref).
- `include_submodel_prefixes::Bool=false`: whether the variables of the population model and
  of the downstream model are named under `population_model` and `downstream_model`. Without
  the prefixes, the names are shorter, but a variable of the same name in both is one
  variable, which `sample` refuses with an error and a log density silently uses once.

The variables of the population model are named as it names them, such as
`learning_rate.session[3]` for `IndependentSessions`, or under `population_model` with
`include_submodel_prefixes = true`. Those of the sessions are under `session[i]`, and those
of the downstream model as it names them, or under `downstream_model` with the prefixes. The
model returns `(; cognitive_parameters, saved_behavioural_values, downstream_values)`: the
parameters of every session as the population model gave them, what each session saved, as
a vector of what [`turing_behavioural_session`](@ref) returns, and what the downstream model
returned. `returned(model, chain)` gives all three for each sample, so a fitted state is read
from the chain without simulating again.

Saving states costs about 5 % of a gradient. Without a downstream model that needs them, a
fit is fastest with a `save_history` that saves nothing, and the states are then read from
the chain with `returned` on the model of a copy of the sessions that saves them,
`MultipleSessionsInfo(sessions_info; save_history = SaveHistory(; states = :expected_reward))`,
since the two models have the same variables.

# Examples
```julia
using CognitiveModeling
using Turing

#The data of two sessions, as a table with one vector for each column
table = (;
    id = ["S1", "S1", "S1", "S2", "S2", "S2"],
    reward = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    choice = [true, false, true, false, missing, true],
)

#The sessions, with the RescorlaWagner model of turing_behavioural_session
sessions_info = MultipleSessionsInfo(
    RescorlaWagner(),
    table;
    observation_columns = :reward,
    action_columns = :choice,
    session_columns = :id,
    who_starts = EnvironmentStarts(),
    save_history = SaveHistory(; states = :expected_reward),
)

#Each session has its own parameters, from the same priors
population_model = IndependentSessions((;
    learning_rate = Beta(2, 2),
    inverse_temperature = LogNormal(0, 1),
))

model = turing_full_model(population_model, sessions_info)
chain = sample(model, NUTS(), 1000)

#The parameters and the expected rewards of every session, for each sample
results = returned(model, chain)
results[1].cognitive_parameters.learning_rate
results[1].saved_behavioural_values[2].states

#The same fit without saving, and the states read from the chain afterwards
unsaved = turing_full_model(population_model, MultipleSessionsInfo(sessions_info; save_history = SaveHistory()))
chain = sample(unsaved, NUTS(), 1000)
results = returned(model, chain)
```
"""
function turing_full_model end

"""
    turing_catch_invalid_parameters(model)

Create a Turing.jl model which runs `model`, any Turing.jl model, inside a `try` block: an
[`InvalidParametersError`](@ref) thrown by a cognitive model or an environment model then
gives the parameters a log density of `-Inf`, so that a sampler rejects the sample instead
of stopping, and the model returns `nothing` for that sample. Otherwise it returns what
`model` returns. This needs Turing.jl to be loaded.

It goes around the part of a model that can throw, with the parameters sampled outside it,
as in

```julia
@model function fit_one_session(session_info)
    learning_rate ~ Beta(2, 2)
    session ~ to_submodel(
        turing_catch_invalid_parameters(
            turing_behavioural_session(session_info, (; learning_rate)),
        ),
        false,
    )
end
```

and [`turing_full_model`](@ref) does this with `catch_invalid_parameters = true`. The
variables sampled inside the `try` block are lost when it catches an error, so around a
whole model a sampler can't start when its first draw is rejected. Anything that uses what
it returns after a rejection, which is `nothing`, can't be differentiated with Enzyme, so
return or stop at once when it is `nothing`. Mooncake can't differentiate a model with a
`try` block at all, and a compiled ReverseDiff tape doesn't see the error, so the rejected
parameters get a finite log density with no warning. It works with ForwardDiff and with
Enzyme. Nothing in the package catches the error unless this is used, so a sampler stops
with it otherwise.
"""
function turing_catch_invalid_parameters end
