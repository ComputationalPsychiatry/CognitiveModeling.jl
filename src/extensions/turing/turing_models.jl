## Turing.jl models which users call, defined in the Turing extension ##

"""
    TuringBackend()

Marker of Turing.jl as the inference backend, which [`full_model`](@ref),
[`session_model`](@ref), [`multiple_sessions_model`](@ref) and
[`catch_invalid_parameters`](@ref) take to make Turing.jl models. This needs Turing.jl to be
loaded.
"""
struct TuringBackend <: AbstractInferenceBackend end

"""
    turing_behavioral_session(session_info::SessionInfo, cognitive_parameters)
    turing_behavioral_session(session_info::SessionInfo, cognitive_parameters,
                              environment_parameters=(;))

Create a Turing.jl model of one session: the cognitive model in `session_info` goes through
the timesteps of its data with `cognitive_parameters`, and the actions in the data are
scored against the action distributions it gives. The second form is for a session with an
environment model, which takes its parameters too. This needs Turing.jl to be loaded.

[`session_model`](@ref) with `TuringBackend()` makes this model. It is a building block for
a Turing.jl model which samples the parameters, as a submodel with its own prefix:
`to_submodel(prefix(session_model(TuringBackend(), ...), @varname(session)), false)`.
[`multiple_sessions_model`](@ref) runs one for each of several sessions, and
[`full_model`](@ref) samples their parameters too.

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
    output ~ to_submodel(turing_run_timestep(cognitive_model, context))

    #An action that is missing in the data is a latent variable with that distribution
    choice.imputed ~ output.output_distribution.choice
end

#The actions in the data are scored once, against the distributions of their timesteps
actions_to_condition_on ~ product_distribution(action_distributions)

#The parameters as they were given, and the trajectories over the timesteps that the save histories select
return (; cognitive_parameters, environment_parameters, saved_trajectories)
```

A session which includes its environment runs the environment model in the same way on each
timestep, before the cognitive model when the environment starts and after it when the agent
starts, and scores the observations in the data like the actions.

The states where the session starts come from [`turing_initialize_state`](@ref) of each
model with the parameters, so that a starting state which is a parameter is fitted like any
other, and a model can sample its start. Both models are treated the same way on each
timestep: [`turing_run_timestep`](@ref) of the cognitive model runs as a submodel with the
prefix `timestep[t]`, so a model which samples within a timestep has its own variables for
each timestep, such as `timestep[3].noise`, and an environment model runs with the prefix
`timestep[t].environment`. The outputs of each, the actions and the observations, are then
taken from the data where they are recorded, and where the data has [`MissingValue`](@ref)
they are imputed from what the model gave: as a latent variable with that distribution,
named `timestep[3].choice.imputed` or `timestep[3].environment.reward.imputed`, or as the
value itself if it is a [`Deterministic`](@ref). Where the data has
[`UnimputedMissingValue`](@ref), or a marker for nothing happening, the models receive the
marker as it is, and nothing is sampled or scored for it. The actions in the data are scored
at the end as observed values, in one `product_distribution` over every action and timestep,
against the distributions the cognitive model gave for them. The observations in the data
are scored the same way against the distributions the environment gave, so that they inform
the missing values and the environment's parameters. A `Deterministic` is scored as a point
mass, with a log density of 0 for its value and `-Inf` for any other, and only a number or a
`Bool` can be, so a value in the data that the model gives as another kind of
`Deterministic` is an error, and so is one where the model gave nothing, [`NoValue`](@ref).
When the session doesn't include its environment, as the `include_environment` of
`session_info` decides, which by default is when nothing in the data is `MissingValue()` and
the environment's history isn't saved, the session is the same as one without an
environment, with the same variables. The [`stopping_criterion`](@ref) of the models is
checked as the `check_stopping_criterion` of `session_info` says: by default only after the
last timestep with data, so that a fit runs every timestep and a session from
[`decondition_sessions`](@ref) can end early.

# What it returns
A `NamedTuple` with `cognitive_parameters` and `environment_parameters`, the parameters as
they were given, without the default values of the models, and `saved_trajectories`, the
trajectories that the save histories of `session_info` select, which
`returned(model, chain)` gives for each sample. The trajectories are a `NamedTuple` with the
same entries as [`simulate_session`](@ref) returns: `states`, `observations`, `actions` and
`action_distributions` of the cognitive model, `environment` with the `states`,
`observations`, `actions` and `observation_distributions` of the environment when the
session has one, which are empty when it didn't run, and `timestep_sizes`. Each is a vector
with one element for timestep 0, before the first timestep, and one for each timestep that
ran. Timestep 0 has the previous observation and action, which are
[`BeforeFirstTimestep`](@ref) unless others are given, `BeforeFirstTimestep()` for every
distribution, and the initial states, which can depend on the parameters. Element `t + 1` is
then timestep `t`, and a [`SaveHistory`](@ref) with `save_initial_timestep_values = false`
leaves timestep 0 out. The returned actions and observations are those the models received,
with the imputed values. `timestep_sizes` has the size of each timestep, after
`BeforeFirstTimestep()` for timestep 0 when it is there, when anything is selected, and is
empty otherwise. Any latent variable in a session, imputed or from a model which samples
within a timestep, greatly increases the amount of estimated variables, which will generally
make fitting the model much slower.

# Examples
```julia
using CognitiveModeling
using Turing

#The model to fit, a Rescorla-Wagner model with a softmax choice
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = parameters.initial_value)
end

function CognitiveModeling.run_timestep(model::RescorlaWagner, context::ModelContext)
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
            session_model(
                TuringBackend(), session_info, (; learning_rate, inverse_temperature)
            ),
            @varname(session),
        ),
        false,
    )
    return session
end

chain = sample(fit_one_session(session_info), NUTS(), 1000)

#The parameters and the expected rewards on each timestep, for each sample
returned_values = returned(fit_one_session(session_info), chain)
returned_values[1].saved_trajectories.states

#The same as chains, which merge into one
parameters = get_session_parameters(returned_values)
trajectories = get_session_trajectories(returned_values)
merge(parameters, trajectories)
```
"""
function turing_behavioral_session end

"""
    turing_multiple_behavioral_sessions(sessions_info::MultipleSessionsInfo,
                                        cognitive_parameters, environment_parameters=(;),
                                        parallelization=SerialSessions())

Create a Turing.jl model which runs [`turing_behavioral_session`](@ref) for every session in
`sessions_info`, each with its own cognitive parameters, as a submodel with the prefix
`session[i]`, so that a latent variable of the third session is named like
`session[3].timestep[2].choice.imputed`. Which session `i` is stands in
`sessions_info.session_ids[i]`, and [`session_index`](@ref) gives `i` for a data identifier.
Environment parameters can only be given when the sessions have an environment model. With
[`ParallelSessions`](@ref)`()` as `parallelization`, after the environment parameters, which
are `(;)` for sessions without an environment, the sessions run over the threads Julia was
started with. **The model that is evaluated, which is the outermost one, must then be marked
with `setthreadsafe(model, true)` from DynamicPPL**, since the log density of a threaded
model that isn't is wrong with no error; the sessions throw an error when it isn't marked.
See `parallelization` in [`full_model`](@ref) with `TuringBackend()`. This needs Turing.jl
to be loaded.

`cognitive_parameters` is a `NamedTuple` with one vector for each parameter, with one value
for every session in the order of `sessions_info`, which is what a parameter model returns,
see [`turing_parameters_model`](@ref), and `environment_parameters` is in the same form. It
returns a `NamedTuple` with `cognitive_parameters` and `environment_parameters` as they were
given, and `saved_trajectories`, a vector with the trajectories each session saved, in the
same order, as [`turing_behavioral_session`](@ref) returns them.
[`multiple_sessions_model`](@ref) with `TuringBackend()` makes this model, and takes
`parallelization` as a keyword. It is a building block for a Turing.jl model which samples
the parameters of the sessions in some way of its own, as a submodel without a prefix:
`to_submodel(multiple_sessions_model(TuringBackend(), sessions_info, parameters), false)`.
[`full_model`](@ref) does this with parameter models. A model of one's own can return
the same `NamedTuple`, and [`get_session_trajectories`](@ref) and
[`get_session_parameters`](@ref) then read what it returns too, also when it transforms its
variables into the parameters.
"""
function turing_multiple_behavioral_sessions end

"""
    full_model(::TuringBackend, cognitive_parameters_model,
               sessions_info::MultipleSessionsInfo;
               environment_parameters_model=NoParameters(),
               downstream_model=NoDownstreamModel(), fixed_cognitive_parameters=(;),
               fixed_environment_parameters=(;), catch_invalid_parameters=false,
               parallelization=SerialSessions(), include_submodel_prefixes=false)

Create the Turing.jl model which fits a cognitive model to several sessions, in three parts:
`cognitive_parameters_model` gives or samples the cognitive parameters of every session, and
`environment_parameters_model` those of the environment model, the sessions in
`sessions_info` run with them as [`turing_multiple_behavioral_sessions`](@ref), and
`downstream_model` can do something with the trajectories they saved, such as a regression
of a measurement on a state. The parts are prepared with [`prepare_full_model`](@ref), and
the model is [`turing_full_model`](@ref) of the prepared parts. This needs Turing.jl to be
loaded.

# Arguments
- `cognitive_parameters_model`: the parameter model of the cognitive model, which gives or
  estimates its parameters for every session: a subtype of
  [`AbstractParametersModel`](@ref), such as [`IndependentParameters`](@ref) with a prior
  for each parameter. A tuple of parameter models, or a `NamedTuple` of them, is made into a
  [`CompositeParameters`](@ref) of them, with each part of a `NamedTuple` under its name. So
  is a tuple or a `NamedTuple` given as `environment_parameters_model`.
- `sessions_info::MultipleSessionsInfo`: the sessions, prepared once with
  [`MultipleSessionsInfo`](@ref).
- `environment_parameters_model=NoParameters()`: the parameter model of the environment
  model of the sessions, when they have one. By default the environment's parameters take
  its default values, and [`FixedParameters`](@ref) gives others. When they are estimated,
  every session includes its environment, also those with nothing to impute, so that the
  observations in the data inform them, which is an error for a session prepared with
  `include_environment = false`. Sessions without an environment take no parameters.
- `downstream_model=NoDownstreamModel()`: a final submodel which is run after the sessions,
  such as for correlations with other kinds of data: a subtype of
  [`AbstractDownstreamModel`](@ref). The default samples nothing.
- `fixed_cognitive_parameters::NamedTuple=(;)` and
  `fixed_environment_parameters::NamedTuple=(;)`: values which every session gets for the
  parameters they name, such as `(; initial_value = 0.3)`, while the parameter models give
  the others. A parameter is either fixed here or given by its parameter model, and a fixed
  parameter must be one in the model's attributes.
- `catch_invalid_parameters::Bool=false`: whether an [`InvalidParametersError`](@ref) in a
  session rejects the sample instead of stopping the sampler, by running the sessions inside
  [`catch_invalid_parameters`](@ref), which gives the parameters a log density of `-Inf`. If the first draw is rejected and the downstream model has variables of its own,
  give the sampler valid `initial_params`. A [`ParameterDomainError`](@ref) isn't caught,
  and stops the sampler.
- `parallelization=SerialSessions()`: whether the sessions run over the threads Julia was
  started with, with `ParallelSessions()`. **The model must then be marked with
  `setthreadsafe(model, true)` from DynamicPPL before it is sampled**: without the mark, a
  threaded model has a wrong log density with no error, so the sessions throw an error
  instead.
- `include_submodel_prefixes::Bool=false`: whether the variables of the two parameter models
  and of the downstream model are named under `cognitive_parameters_model`,
  `environment_parameters_model` and `downstream_model`. Without the prefixes, the names are
  shorter, but a variable of the same name in two of them is one variable, which `sample`
  refuses with an error, and which a log density gives one value with both priors counted. A
  warning says so when the model is made, when the cognitive model and the environment model
  have a parameter of the same name and both parameter models sample.

The variables of a parameter model are named after the parameters, such as
`learning_rate.session` with `IndependentParameters`, where `learning_rate.session[3]` is
the learning rate of the third session, or under the parameter model's name with
`include_submodel_prefixes = true`. A part of a [`CompositeParameters`](@ref) that is given
with a name has its variables under that name. The variables of the sessions are under
`session[i]`, and those of the downstream model as it names them, or under
`downstream_model` with the prefixes. `rand(model)` shows every variable's name.

The model returns a `NamedTuple` with `cognitive_parameters`, `environment_parameters`,
`saved_trajectories` and `downstream_values`: the cognitive parameters and the environment
parameters of every session as the parameter models gave them, the trajectories each session
saved, as [`turing_multiple_behavioral_sessions`](@ref) returns them, and what the
downstream model returned. `returned(model, chain)` gives them for each sample, so a fitted
state is read from the chain without simulating again.

Saving states adds a cost to gradient calculations. Without a downstream model that needs
them, a fit is fastest with sessions that save nothing, and the trajectories are then read
from the chain afterwards with [`get_session_trajectories`](@ref), or with `returned` on
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
cognitive_parameters_model = IndependentParameters((;
    learning_rate = Beta(2, 2),
    inverse_temperature = LogNormal(0, 1),
))

model = full_model(TuringBackend(), cognitive_parameters_model, sessions_info)
chain = sample(model, NUTS(), 1000)

#The parameters and the expected rewards of every session, for each sample
returned_values = returned(model, chain)
returned_values[1].cognitive_parameters.learning_rate
returned_values[1].saved_trajectories[2].states

#The same fit without saving, and the states read from the chain afterwards
sessions_without_saving = MultipleSessionsInfo(
    sessions_info; cognitive_save_history = SaveHistory(),
)
unsaved = full_model(TuringBackend(), cognitive_parameters_model, sessions_without_saving)
chain = sample(unsaved, NUTS(), 1000)
returned_values = returned(model, chain)
```
"""
full_model(::TuringBackend, cognitive_parameters_model, sessions_info::MultipleSessionsInfo)

#The parts are prepared by full_model, outside the model's body, which is differentiated
"""
    turing_full_model(cognitive_parameters_model, sessions_info::MultipleSessionsInfo;
                      environment_parameters_model=NoParameters(),
                      downstream_model=NoDownstreamModel(),
                      catch_invalid_parameters=false, parallelization=SerialSessions(),
                      include_submodel_prefixes=false)

Create the Turing.jl model of a full model from parts that [`prepare_full_model`](@ref)
prepared: the parameter models give or sample the cognitive and the environment parameters
of every session, the sessions run with them as
[`turing_multiple_behavioral_sessions`](@ref), and the downstream model runs after them, as
[`full_model`](@ref) with [`TuringBackend`](@ref) describes, which prepares the parts and
returns this model. This needs Turing.jl to be loaded.
"""
function turing_full_model end

"""
    turing_catch_invalid_parameters(model)

Create a Turing.jl model which runs `model`, any Turing.jl model, inside a `try` block: an
[`InvalidParametersError`](@ref) thrown by a cognitive model or an environment model then
gives the parameters a log density of `-Inf`, so that a sampler rejects the sample instead
of stopping, and the model returns `nothing` for that sample. Otherwise it returns what
`model` returns. Any other error is thrown as it is, also a [`ParameterDomainError`](@ref)
for a parameter outside its domain. [`catch_invalid_parameters`](@ref) with
`TuringBackend()` makes this model. This needs Turing.jl to be loaded.

It goes around the part of a model that can throw, with the parameters sampled outside it,
as in

```julia
@model function fit_one_session(session_info)
    learning_rate ~ Beta(2, 2)
    session ~ to_submodel(
        catch_invalid_parameters(
            TuringBackend(),
            session_model(TuringBackend(), session_info, (; learning_rate)),
        ),
        false,
    )
end
```

and [`full_model`](@ref) does this with `catch_invalid_parameters = true`. The
variables sampled inside the `try` block are lost when it catches an error, so around a
whole model a sampler can't start when its first draw is rejected. Nothing in the package
catches the error unless this is used, so a sampler stops with it otherwise. Not every
automatic differentiation backend can differentiate a model with it, as the page on
automatic differentiation says.
"""
function turing_catch_invalid_parameters end
