## Turing.jl models which users call, defined in the Turing extension ##

"""
    turing_behavioral_session(session_info::SessionInfo, cognitive_parameters)
    turing_behavioral_session(session_info::SessionInfo, cognitive_parameters,
                              environment_parameters=(;))

Create a Turing.jl model of one session: the cognitive model in `session_info` goes through
the timesteps of its data with `cognitive_parameters`, and the actions in the data are
scored against the action distributions it gives. The second form is for a session with an
environment model, which takes its parameters too. This needs Turing.jl to be loaded.

The model is a building block for a Turing.jl model which samples the parameters, as a
submodel with its own prefix: `to_submodel(prefix(turing_behavioral_session(...),
@varname(session)), false)`. [`turing_multiple_behavioral_sessions`](@ref) runs one for
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
In outline, the model of a session without an environment looks like this. It is not the
actual code, which also checks the names and the parameters, and saves what is selected:

```julia
#The state where the session starts, which a model can sample
state ~ to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))

for t in 1:n_timesteps

    #The cognitive model gives a distribution for each action, under the prefix timestep[t]
    output ~ to_submodel(turing_apply_timestep_model(cognitive_model, context))

    #An action that is missing in the data is a latent variable with that distribution
    choice.imputed ~ output.output_distribution.choice
end

#The actions in the data are scored once, against the distributions of their timesteps
actions_to_condition_on ~ product_distribution(action_distributions)

#The trajectories over the timesteps that the save histories select
return saved_trajectories
```

A session which includes its environment runs the environment model in the same way on each
timestep, before the cognitive model when the environment starts and after it when the agent
starts, and scores the observations in the data like the actions.

The states where the session starts come from [`turing_initialize_state`](@ref) of each
model with the parameters, so that a starting state which is a parameter is fitted like any
other, and a model can sample its start. Both models are treated the same way on each
timestep: [`turing_apply_timestep_model`](@ref) of the cognitive model runs as a submodel
with the prefix `timestep[t]`, so a model which samples within a timestep has its own
variables for each timestep, such as `timestep[3].noise`, and an environment model runs with
the prefix `timestep[t].environment`. The outputs of each, the actions and the observations,
are then taken from the data where they are recorded, and where the data has
[`MissingValue`](@ref) they are imputed from what the model gave: as a latent variable with
that distribution, named `timestep[3].choice.imputed` or
`timestep[3].environment.reward.imputed`, or as the value itself if it is a
[`Deterministic`](@ref). Where the data has [`UnimputedMissingValue`](@ref), or a marker for
nothing happening, the models receive the marker as it is, and nothing is sampled or scored
for it. The sampler must handle the type of a latent variable: a continuous value can be
sampled with NUTS, a discrete one needs a sampler for discrete variables. The actions in the
data are scored at the end as observed values, in one `product_distribution` over every
action and timestep, against the distributions the cognitive model gave for them. The
observations in the data are scored the same way against the distributions the environment
gave, so that they inform the missing values and the environment's parameters. A
`Deterministic` is scored as a point mass, with a log density of 0 for its value and `-Inf`
for any other, and only a number or a `Bool` can be, so a value in the data that the model
gives as another kind of `Deterministic` is an error, and so is one where the model gave
nothing, [`NoValue`](@ref). When the session doesn't include its environment, as the
`include_environment` of `session_info` decides, which by default is when nothing in the
data is `MissingValue()` and the environment's history isn't saved, the session is the same
as one without an environment, with the same variables. The [`stopping_criterion`](@ref) of
the models is checked as the `check_stopping_criterion` of `session_info` says: by default
only after the last timestep with data, so that a fit runs every timestep and a session from
[`prepare_for_prediction`](@ref) can end early.

# What it returns
The trajectories that the save histories of `session_info` select, in a `NamedTuple` with
the same entries as [`simulate`](@ref) returns: `states`, `observations`, `actions` and
`action_distributions` of the cognitive model, `environment` with the `states`,
`observations`, `actions` and `observation_distributions` of the environment when the
session has one, which are empty when it didn't run, and `timestep_sizes`. Each is a vector
with one element for timestep 0, before the first timestep, and one for each timestep that
ran, which `returned(model, chain)` gives for each sample. Timestep 0 has the previous
observation and action, which are [`BeforeFirstTimestep`](@ref) unless others are given,
`BeforeFirstTimestep()` for every distribution, and the initial states, which can depend on
the parameters. Element `t + 1` is then timestep `t`, and a [`SaveHistory`](@ref) with
`save_initial_timestep_values = false` leaves timestep 0 out. The returned actions and
observations are those the models received, with the imputed values. `timestep_sizes` has
the size of each timestep, after `BeforeFirstTimestep()` for timestep 0 when it is there,
when anything is selected, and is empty otherwise. Any latent variable in a session, imputed
or from a model which samples within a timestep, greatly increases the amount of estimated
variables, which will generally make fitting the model much slower.

# Examples
```julia
using CognitiveModeling
using Turing

#The model to fit, a Rescorla-Wagner model with a softmax choice
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = parameters.initial_value)
end

function CognitiveModeling.apply_timestep_model(model::RescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    Pₜ = BernoulliLogit(context.parameters.inverse_temperature * (Vₜ - 0.5))
    return ModelOutput((; choice = Pₜ), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(;
        parameters = (:learning_rate, :inverse_temperature, :initial_value),
        states = :expected_reward,
        inputs = :reward,
        outputs = :choice,
    )
end

CognitiveModeling.default_parameters(model::RescorlaWagner) = (; initial_value = 0.5)

#The data of one session, where the third choice wasn't recorded, prepared once
session_info = SessionInfo(
    RescorlaWagner(),
    (; reward = [1.0, 0.0, 1.0, 1.0]),
    (; choice = [true, false, MissingValue(), true]);
    who_starts = EnvironmentStarts(),
    cognitive_save_history = SaveHistory(; states = :expected_reward),
)

#Priors on the parameters, and the session as a submodel
@model function fit_one_session(session_info)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ LogNormal(0, 1)
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(session_info, (; learning_rate, inverse_temperature)),
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
function turing_behavioral_session end

"""
    turing_multiple_behavioral_sessions(sessions_info::MultipleSessionsInfo,
                                        cognitive_parameters, environment_parameters=(;);
                                        parallelization=SerialSessions())

Create a Turing.jl model which runs [`turing_behavioral_session`](@ref) for every session
in `sessions_info`, each with its own cognitive parameters, as a submodel with the prefix
`session[i]`, so that a latent variable of the third session is named like
`session[3].timestep[2].choice.imputed`. Which session `i` is stands in
`sessions_info.session_ids[i]`, and [`session_index`](@ref) gives `i` for an identifier.
Environment parameters can only be given when the sessions have an environment model. With
`parallelization = ParallelSessions()`, the sessions run over the threads Julia was
started with. **The model that is evaluated, which is the outermost one, must then be
marked with `setthreadsafe(model, true)` from DynamicPPL**, since the log density of a
threaded model that isn't is wrong with no error; the sessions throw an error when it isn't
marked. See `parallelization` in [`turing_full_model`](@ref). This needs Turing.jl to be
loaded.

`cognitive_parameters` is a `NamedTuple` with one vector for each parameter, with one value
for every session in the order of `sessions_info`, which is what a session parameter model
returns, see [`turing_session_parameters_model`](@ref), and `environment_parameters` is in
the same form. It returns a vector with what each session returns, in the same order. The
model is a building block for a Turing.jl model which samples the parameters of the sessions
in some way of its own, as a submodel without a prefix:
`to_submodel(turing_multiple_behavioral_sessions(sessions_info, parameters), false)`.
[`turing_full_model`](@ref) does this with session parameter models.
"""
function turing_multiple_behavioral_sessions end

"""
    turing_full_model(cognitive_parameters_model, sessions_info::MultipleSessionsInfo;
                      environment_parameters_model=NoSessionParameters(),
                      downstream_model=NoDownstreamModel(),
                      fixed_cognitive_parameters=(;), fixed_environment_parameters=(;),
                      catch_invalid_parameters=false, parallelization=SerialSessions(),
                      include_submodel_prefixes=false)

Create the Turing.jl model which fits a cognitive model to several sessions, in three parts:
`cognitive_parameters_model` gives or samples the cognitive parameters of every session, and
`environment_parameters_model` those of the environment model, the sessions in
`sessions_info` run with them as [`turing_multiple_behavioral_sessions`](@ref), and
`downstream_model` can do something with the trajectories they saved, such as a regression
of a measurement on a state. This needs Turing.jl to be loaded.

# Arguments
- `cognitive_parameters_model`: the session parameter model of the cognitive model, which
  gives or estimates its parameters for every session: a subtype of
  [`AbstractSessionParametersModel`](@ref), such as [`IndependentSessionParameters`](@ref)
  with a prior for each parameter.
- `sessions_info::MultipleSessionsInfo`: the sessions, prepared once with
  [`MultipleSessionsInfo`](@ref).
- `environment_parameters_model=NoSessionParameters()`: the session parameter model of the
  environment model of the sessions, when they have one. By default the environment's
  parameters take its default values, and [`FixedSessionParameters`](@ref) gives others.
  When they are estimated, every session includes its environment, also those with nothing
  to impute, so that the observations in the data inform them, which is an error for a
  session prepared with `include_environment = false`. Sessions without an environment take
  no parameters.
- `downstream_model=NoDownstreamModel()`: a final submodel which is run after the sessions,
  such as for correlations with other kinds of data: a subtype of
  [`AbstractDownstreamModel`](@ref). The default samples nothing.
- `fixed_cognitive_parameters::NamedTuple=(;)` and
  `fixed_environment_parameters::NamedTuple=(;)`: values which every session gets for the
  parameters they name, such as `(; initial_value = 0.3)`, while the session parameter
  models give the others. A parameter is either fixed here or given by its session
  parameter model, and a fixed parameter must be one in the model's attributes.
- `catch_invalid_parameters::Bool=false`: whether an [`InvalidParametersError`](@ref) in a
  session rejects the sample instead of stopping the sampler, by running the sessions inside
  [`turing_catch_invalid_parameters`](@ref). If the first draw is rejected and the
  downstream model has variables of its own, give the sampler valid `initial_params`.
- `parallelization=SerialSessions()`: whether the sessions run over the threads Julia was
  started with, with `ParallelSessions()`. **The model must then be marked with
  `setthreadsafe(model, true)` from DynamicPPL before it is sampled**: without the mark, a
  threaded model has a wrong log density with no error, so the sessions throw an error
  instead.
- `include_submodel_prefixes::Bool=false`: whether the variables of the two session
  parameter models and of the downstream model are named under `cognitive_parameters_model`,
  `environment_parameters_model` and `downstream_model`. Without the prefixes, the names are
  shorter, but a variable of the same name in two of them is one variable, which `sample`
  refuses with an error, and which a log density gives one value with both priors counted.
  A warning says so when the model is made, when the cognitive model and the environment
  model have a parameter of the same name and both session parameter models sample.

The variables of a session parameter model are named as it names them, or under the
session parameter model's name with `include_submodel_prefixes = true`. With
`IndependentSessionParameters`, each parameter is a vector with a value for each session,
such as `learning_rate.session`, so the learning rate of the third session is
`learning_rate.session[3]`. With `GroupedHierarchicalSessionParameters`, each parameter of
a group-level distribution is a vector with a value for each group, named after its prior,
such as `learning_rate.mean.group` and `learning_rate.sd.group`, and the values of the
sessions before the transformation are `learning_rate.standardized`, or
`learning_rate.session` for a parameter that is centered. With
`HierarchicalRegressionSessionParameters`, the intercept of a parameter is
`learning_rate.intercept`, and the coefficient of a predictor is under its name, such as
`learning_rate.age.coefficient`, or under its name and its level for a categorical
predictor, such as `learning_rate.diagnosis_clinical.coefficient`. A random intercept has
its variables under its name: its standard deviation, `learning_rate.participant.sd`, which
is a vector with a value for each group when it is grouped, and a standardized value for
each level, `learning_rate.participant.standardized`, or the deviation of each level,
`learning_rate.participant.deviation`, when it is centered. A part of a
[`CompositeSessionParameters`](@ref) that is given with a name has its variables under that
name, such as `noise.bias.session`. The variables of the sessions are under `session[i]`,
and those of the downstream model as it names them, or under `downstream_model` with the
prefixes.

The model returns a `NamedTuple` with `cognitive_parameters`, `environment_parameters`,
`saved_behavioral_trajectories` and `downstream_values`: the cognitive parameters and the
environment parameters of every session as the session parameter models gave them, the
trajectories each session saved, as a vector of what [`turing_behavioral_session`](@ref)
returns, and what the downstream model returned.
`returned(model, chain)` gives them for each sample, so a fitted state is read from the
chain without simulating again.

Saving states adds a cost to gradient calculations. Without a downstream model that needs
them, a fit is fastest with sessions that save nothing, and the trajectories are then read
from the chain afterwards with [`session_trajectories_chain`](@ref), or with `returned` on
the model of a copy of the sessions that saves them,
`MultipleSessionsInfo(sessions_info; cognitive_save_history = SaveHistory(; states = :expected_reward))`,
since the two models have the same variables. A warning says so when the sessions save
anything and there is no downstream model.

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

#The sessions, with the RescorlaWagner model of turing_behavioral_session
sessions_info = MultipleSessionsInfo(
    RescorlaWagner(),
    table;
    observation_columns = :reward,
    action_columns = :choice,
    session_columns = :id,
    who_starts = EnvironmentStarts(),
    cognitive_save_history = SaveHistory(; states = :expected_reward),
)

#Each session has its own parameters, from the same priors
cognitive_parameters_model = IndependentSessionParameters((;
    learning_rate = Beta(2, 2),
    inverse_temperature = LogNormal(0, 1),
))

model = turing_full_model(cognitive_parameters_model, sessions_info)
chain = sample(model, NUTS(), 1000)

#The parameters and the expected rewards of every session, for each sample
results = returned(model, chain)
results[1].cognitive_parameters.learning_rate
results[1].saved_behavioral_trajectories[2].states

#The same fit without saving, and the states read from the chain afterwards
sessions_without_saving = MultipleSessionsInfo(
    sessions_info; cognitive_save_history = SaveHistory(),
)
unsaved = turing_full_model(cognitive_parameters_model, sessions_without_saving)
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
            turing_behavioral_session(session_info, (; learning_rate)),
        ),
        false,
    )
end
```

and [`turing_full_model`](@ref) does this with `catch_invalid_parameters = true`. The
variables sampled inside the `try` block are lost when it catches an error, so around a
whole model a sampler can't start when its first draw is rejected. Nothing in the package
catches the error unless this is used, so a sampler stops with it otherwise. Not every
automatic differentiation backend can differentiate a model with it, as the page on
automatic differentiation says.
"""
function turing_catch_invalid_parameters end
