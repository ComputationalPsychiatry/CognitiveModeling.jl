# # Predictive checks and parameter recovery
#
# A model that is fitted to the data of some sessions can also simulate that data. With
# parameters drawn from their priors, the simulated data shows what the model expects before
# it is fitted, a prior predictive check, and with the parameters of a fit, how well the
# fitted model describes the data, a posterior predictive check. Data simulated with known
# parameters can also be fitted again, to see whether the fit finds the parameters that
# simulated it: a parameter recovery. This page does all three with Turing.jl.
#
# The package's models of the sessions take their data as arguments, and Turing.jl's
# `decondition` and `condition` work on the variables of a model, so they leave that data as
# it is. [`decondition_sessions`](@ref) takes the data out of the sessions, so that the
# model simulates every action as a variable, and [`condition_sessions`](@ref) puts values
# of those variables back in the sessions as data. Turing.jl's own functions can then
# simulate, and its `condition` can condition the model on the same variables. The page does
# each step in both ways, and compares them at the end.

using CognitiveModeling
using Distributions: Bernoulli, BernoulliLogit, Beta, LogNormal
using Random: Xoshiro
using Statistics: mean
using Turing
using Turing: DynamicPPL

# ## The model and the data
#
# The Rescorla-Wagner model learns the expected reward of an option from the rewards it
# gets, and chooses the option with a probability that grows with its expected reward:

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward=0.5)
end

function CognitiveModeling.timestep_model(model::RescorlaWagner, context::ModelContext)
    #The expected reward before this timestep, and after its reward
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)

    #The higher the expected reward, the more likely the option is chosen
    Pₜ = BernoulliLogit(context.parameters.inverse_temperature * (Vₜ - 0.5))
    return ModelOutput((; choice=Pₜ), (; expected_reward=Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# The rewards come from an environment which gives a reward with a fixed probability on
# every timestep:

struct BernoulliRewards <: AbstractEnvironmentModel end

CognitiveModeling.initialize_state(environment::BernoulliRewards, parameters) = (;)

function CognitiveModeling.timestep_model(
    environment::BernoulliRewards, context::ModelContext
)
    return ModelOutput((; reward=Bernoulli(context.parameters.reward_probability)))
end

function CognitiveModeling.attribute_names(environment::BernoulliRewards)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

# Three participants, each with a learning rate of their own, are simulated for 40
# timesteps, which gives a table with a row for each timestep:

simulated_learning_rates = [0.2, 0.5, 0.8]
simulations = map(enumerate(simulated_learning_rates)) do (participant, learning_rate)
    return simulate_session(
        RescorlaWagner(),
        BernoulliRewards();
        who_starts=EnvironmentStarts(),
        n_steps=40,
        cognitive_parameters=(; learning_rate, inverse_temperature=5.0),
        environment_parameters=(; reward_probability=0.7),
        cognitive_save_history=SaveHistory(;
            inputs=:reward, outputs=:choice, save_initial_timestep_values=false
        ),
        rng=Xoshiro(participant),
    )
end
table = (;
    participant=repeat(["P1", "P2", "P3"]; inner=40),
    reward=[
        float(observation.reward) for s in simulations for observation in s.observations
    ],
    choice=[action.choice for s in simulations for action in s.actions],
)

# The full model fits a learning rate and an inverse temperature to each participant. The
# rewards are observations in the data, so the sessions need no environment:

sessions_info = MultipleSessionsInfo(
    RescorlaWagner(),
    table;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
)
model = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Beta(2, 2), inverse_temperature=LogNormal(1, 0.5)
    )),
    sessions_info,
)

# ## Prior predictive checks
#
# The sessions of [`decondition_sessions`](@ref) have [`SimulatedMissingValue`](@ref) in
# place of every choice, so that the model simulates each choice as a variable. They have
# the rewards of the data, since they have no environment that could simulate the rewards:

decondition_sessions(sessions_info).sessions[1].actions.choice[1:3]

# Given the full model, it returns a copy whose sessions are deconditioned. Turing.jl's
# `sample` with `Prior()` then simulates a data set for each draw of the parameters from
# their priors. The chain has the parameters and the simulated choices, as variables named
# such as `session[1].timestep[3].choice.simulated`:

deconditioned_model = decondition_sessions(model)
prior_predictions = sample(Xoshiro(1), deconditioned_model, Prior(), 200; progress=false)

# The share of the timesteps on which each participant chose the option, in the data and
# in each simulated data set:

participants = (:P1, :P2, :P3)
share_in_data = [mean(table.choice[(40 * (i - 1) + 1):(40 * i)]) for i in 1:3]
prior_shares = [
    vec(
        mean(
            prior_predictions[@varname(session[i].timestep[t].choice.simulated)] for
            t in 1:40
        ),
    ) for i in 1:3
]
[
    (; participant, data=share_in_data[i], simulated=extrema(prior_shares[i])) for
    (i, participant) in enumerate(participants)
]

# [`predict_session_trajectories`](@ref) does the same from a chain of the parameters'
# priors, and returns the trajectories that the sessions save, named after the
# participants, as [`get_session_trajectories`](@ref) names them, here the choices:

prior_chain = sample(Xoshiro(2), model, Prior(), 200; progress=false)
prior_trajectories = predict_session_trajectories(
    Xoshiro(3),
    model,
    prior_chain;
    cognitive_save_history=SaveHistory(;
        outputs=:choice, save_initial_timestep_values=false
    ),
)
[
    extrema(mean.(prior_trajectories[@varname($participant.choice)])) for
    participant in participants
]

# ## Posterior predictive checks
#
# A posterior predictive check simulates the data with the parameters of each sample of a
# fit. Turing.jl's `predict` does it with the deconditioned model, and
# [`predict_session_trajectories`](@ref) with the model itself:

chain = sample(Xoshiro(4), model, NUTS(), 300; progress=false)
posterior_predictions = predict(Xoshiro(5), deconditioned_model, chain)
posterior_trajectories = predict_session_trajectories(
    Xoshiro(6),
    model,
    chain;
    cognitive_save_history=SaveHistory(;
        outputs=:choice, save_initial_timestep_values=false
    ),
)
[
    (;
        participant,
        data=share_in_data[i],
        with_predict=mean(
            mean(posterior_predictions[@varname(session[i].timestep[t].choice.simulated)])
            for t in 1:40
        ),
        with_predict_session_trajectories=mean(
            mean.(posterior_trajectories[@varname($participant.choice)])
        ),
    ) for (i, participant) in enumerate(participants)
]

# ## Parameter recovery
#
# A draw from the prior of the deconditioned model has parameters, and choices simulated
# with them. `DynamicPPL.subset` keeps the variables within the sessions, the simulated
# choices, and leaves out the parameters, which the fit is to find:

simulated = rand(Xoshiro(7), deconditioned_model)
simulated_choices = DynamicPPL.subset(simulated, [@varname(session)])
simulated_parameters = DynamicPPL.subset(
    simulated, [@varname(learning_rate), @varname(inverse_temperature)]
)

# Turing.jl's `condition` conditions the deconditioned model on the simulated choices, as
# variables, and [`condition_sessions`](@ref) puts them in the data of its sessions, where
# they are scored as data from a table is. The two models give the same log joint at any
# parameters, here at those that simulated the choices:

turing_recovery_model = condition(deconditioned_model, simulated_choices)
package_recovery_model = condition_sessions(deconditioned_model, simulated_choices)
(
    logjoint(turing_recovery_model, simulated_parameters),
    logjoint(package_recovery_model, simulated_parameters),
)

# Either is then fitted as any model is. The learning rate that the fit finds for each
# participant, next to the one that simulated the choices:

recovered = sample(Xoshiro(8), package_recovery_model, NUTS(), 300; progress=false)
recovered_parameters = get_session_parameters(package_recovery_model, recovered)
[
    (;
        participant,
        simulated=simulated[@varname(learning_rate.session)][i],
        recovered=mean(recovered_parameters[@varname($participant.learning_rate)]),
    ) for (i, participant) in enumerate(participants)
]

# ## Turing.jl's functions or the package's
#
# | | Turing.jl's functions | The package's functions |
# |:-- |:-- |:-- |
# | Prior predictive check | `sample(decondition_sessions(model), Prior(), n)` | `predict_session_trajectories(model, sample(model, Prior(), n))` |
# | Posterior predictive check | `predict(decondition_sessions(model), chain)` | `predict_session_trajectories(model, chain)` |
# | Parameter recovery | `condition(decondition_sessions(model), values)` | `condition_sessions(decondition_sessions(model), values)` |
#
# They give the same results in most cases:
# - The predictive checks simulate every action and observation of the sessions again, also
#   those that a fit imputed, with the same distributions, so only the random numbers
#   differ, unless the model samples something of its own within a timestep.
# - The two models of a parameter recovery have the same log joint, and so the same
#   posterior, when the sessions have no environment, or one whose parameters are fitted.
#
# They differ in what they return, and in a few cases:
# - Turing.jl's functions return the simulated actions and observations as variables of a
#   chain, and [`predict_session_trajectories`](@ref) returns what the sessions save, such
#   as the states of the models, as [`get_session_trajectories`](@ref) does.
# - Turing.jl's `predict` keeps the values of the chain for what a model samples within a
#   timestep, such as `session[1].timestep[3].noise`, since the deconditioned model has the
#   same variable, while [`predict_session_trajectories`](@ref) samples those again too.
# - Turing.jl's `predict` fails for a fit that imputed values at different timesteps in
#   different sessions, where [`predict_session_trajectories`](@ref) works.
# - A session with an environment whose parameters are fixed scores the conditioned
#   observations against the environment's distributions with Turing.jl's `condition`, and
#   leaves its environment out with [`condition_sessions`](@ref), since nothing in it is
#   simulated. The log joints then differ by the same amount at every value of the
#   parameters, so the posteriors are the same.
# - Turing.jl's `condition` leaves out a value whose variable the model doesn't have, such
#   as one with a misspelled name, while [`condition_sessions`](@ref) throws an error for
#   it, and for a value of any variable that isn't an action or an observation of the
#   sessions, such as a parameter.
# - With a [`CognitiveModeling.stopping_criterion`](@ref), a deconditioned session with an
#   environment checks it before every timestep, also with Turing.jl's `condition`. Such a
#   session then ends where the criterion is met at the parameters of each evaluation:
#   before its last value, which leaves out the values after it with no error, or after it,
#   with a variable for every action and observation that follows.
#   [`condition_sessions`](@ref) ends each session after its last timestep with a value, and
#   doesn't check the criterion, as a fit to a table of that data doesn't.
#
# How fast they are differs more. With 20 sessions of 100 timesteps, and the values given
# with a concrete type each, as `rand` and `DynamicPPL.subset` give them:
# - Fitting: with Turing.jl's `condition`, each evaluation of the log density took about
#   4 times as long as with [`condition_sessions`](@ref), and each gradient about 1 to 10
#   times as long, depending on the automatic differentiation backend. With an environment,
#   whose observations it conditioned on too, each evaluation took about 10 times as long,
#   and each gradient about 1.5 to 20 times.
# - Simulating: `sample` with `Prior()` on the deconditioned model took about 50 times as
#   long per draw as [`predict_session_trajectories`](@ref), and its chain has a variable
#   for every simulated choice. `predict` took about one and a half times as long as
#   [`predict_session_trajectories`](@ref). [`simulate_session`](@ref) took about a
#   hundredth of that, with the parameters given for each session instead of read from a
#   chain.
