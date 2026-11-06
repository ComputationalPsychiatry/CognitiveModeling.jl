## Turing.jl models which users call, defined in the Turing extension ##

"""
    turing_single_session(cognitive_model, cognitive_parameters, observations, actions;
                          <keyword arguments>)

Create a Turing.jl model of one session: `cognitive_model` with `cognitive_parameters` goes
through the timesteps of the data, and the recorded actions are scored against the action
distributions it gives. This needs Turing.jl to be loaded.

The model is a building block for a Turing.jl model which samples the parameters, as a
submodel with its own prefix: `to_submodel(prefix(turing_single_session(...),
@varname(session)), false)`. It returns what `save_history` selects of the observations, the
actions and the states of the session, as a `NamedTuple` of vectors with one element for
timestep 0, before the first timestep, and one for each timestep after it, which
`returned(model, chain)` gives for each sample. Timestep 0 has the previous observation and
action, which are [`BeforeFirstTimestep`](@ref) unless others are given, and the initial
state, which can depend on the parameters. Element `t + 1` is then timestep `t`, and a
[`SaveHistory`](@ref) with `save_initial_values = false` leaves timestep 0 out. Its
`timestep_sizes` has the size of each timestep, after `BeforeFirstTimestep()` for timestep 0
when it is there, when anything is selected, and is empty otherwise. Nothing is selected by
default.

On each timestep, [`turing_apply_model`](@ref) of `cognitive_model` runs as a submodel with
the prefix `timestep[t]`, so a model which samples within a timestep has its own variables
for each timestep, such as `timestep[3].noise`. The recorded actions are scored together at
the end, in one `product_distribution` over every action and timestep, which is several
times faster than scoring each timestep on its own.

# Arguments
- `cognitive_model::AbstractCognitiveModel`: the cognitive model.
- `cognitive_parameters::NamedTuple`: its parameters, usually the random variables of the
  model around this one. Those it doesn't have take the values in
  [`default_parameters`](@ref), if the model provides defaults for them.
- `observations::NamedTuple`: the observations in the data, with one vector for each name
  and one entry for each timestep, such as `(; reward = [1.0, 0.0, 1.0])`. The data must have
  every observation the model lists in [`attribute_names`](@ref), and can have more. An
  observation that wasn't recorded is [`MissingObservation`](@ref), which `environment`
  fills in.
- `actions::NamedTuple`: the actions in the data, in the same form, with exactly the actions
  the model lists. An action that was taken but not recorded is [`MissingAction`](@ref), and
  is imputed unless `impute_missing_actions` is `false`. An action that wasn't taken is
  [`NoAction`](@ref), and isn't scored.
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: whether the model acts on the
  observation of the same timestep, `EnvironmentStarts()`, or on the observation of the
  timestep before, `AgentStarts()`, as in [`simulate`](@ref). It has no default, because a
  wrong choice shifts the model by a timestep without any error.
- `environment::Union{Nothing, AbstractEnvironmentModel}=nothing`: an environment model
  which fills in the observations that are `MissingObservation()`. When it is given, it runs
  on every timestep as a submodel with the prefix `timestep[t].environment`, and what it
  gives for an observation is only used where the data has none: the missing observation is
  then imputed from it, as a latent variable with that distribution, or as the value itself
  if it is a [`Deterministic`](@ref). Its `turing_apply_model` must sample with `~` if
  anything else in it is random. It must list the same observations and the same actions as
  `cognitive_model`, since a session takes no maps of names. Without it, the data can't have
  missing observations.
- `environment_parameters::NamedTuple=(;)`: the parameters of `environment`, with its
  default values for those that aren't given, if it provides any.
- `timestep_sizes=1`: how much time each timestep covers, as in `simulate`.
- `impute_missing_actions::Bool=true`: whether an action that is `MissingAction()` is
  imputed, as a latent variable. When it is `false`, nothing is sampled for it, and it is
  passed on as `NoAction()` instead: `cognitive_model` receives `NoAction()` as that
  previous action on the next timestep, and `environment` as the action it responds to.
- `save_history::SaveHistory=SaveHistory()`: which observations, actions and states are
  returned, by name, and whether timestep 0 is. The returned actions and observations are
  those the model received, with the imputed and filled in values.
- `catch_invalid_parameters::Bool=false`: whether an [`InvalidParametersError`](@ref) from
  the model gives the parameters a log density of `-Inf`, so that a sampler rejects them,
  instead of stopping the sampler. It is off by default because the `try` block it needs
  can't be differentiated with Mooncake, and a compiled ReverseDiff tape doesn't see the
  error, so the parameters get a finite log density with no warning. It works with
  ForwardDiff and Enzyme.
- `previous_cognitive_state`, `previous_action`, `previous_observation`,
  `previous_environment_state`: where the session starts, with the same defaults as in
  `simulate`.

An imputed action is a latent variable named after its timestep and action, such as
`timestep[3].choice.imputed`, with the model's action distribution on that timestep as its
distribution. An observation imputed from a distribution is one in the same way, under the
environment's prefix, such as `timestep[3].environment.reward.imputed`. The sampler must
handle its type: a continuous action can be sampled with NUTS, a discrete one needs a
sampler for discrete variables. Any latent variable in a session, imputed or from a model
which samples within a timestep, makes evaluating the model much slower in the current
version of DynamicPPL.jl.

# Examples
```julia
using CognitiveModeling
using Turing

#The model to fit, a Rescorla-Wagner model with a softmax choice
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    Pₜ = BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))
    return ModelOutput((; choice = Pₜ), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(;
        parameters = (:learning_rate, :inverse_temperature),
        states = :expected_reward,
        observations = :reward,
        actions = :choice,
    )
end

#The data of one session, where the third choice wasn't recorded
observations = (; reward = [1.0, 0.0, 1.0, 1.0])
actions = (; choice = [true, false, MissingAction(), true])

#Priors on the parameters, and the session as a submodel
@model function fit_one_session(observations, actions)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ LogNormal(0, 1)
    session ~ to_submodel(
        prefix(
            turing_single_session(
                RescorlaWagner(),
                (; learning_rate, inverse_temperature),
                observations,
                actions;
                who_starts = EnvironmentStarts(),
                save_history = SaveHistory(; states = :expected_reward),
            ),
            @varname(session),
        ),
        false,
    )
    return session
end

chain = sample(fit_one_session(observations, actions), NUTS(), 1000)

#The expected rewards on each timestep, for each sample
trajectories = returned(fit_one_session(observations, actions), chain)
```
"""
function turing_single_session end
