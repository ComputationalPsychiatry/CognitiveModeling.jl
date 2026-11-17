# # Custom Turing.jl models
#
# *This page is a draft.*
#
# [`turing_full_model`](@ref) fits several sessions with parameter models, which
# give the parameters of every session in one of the ways the package offers. A Turing.jl
# model of one's own can give them in any other way, and run the sessions inside it as a
# submodel: [`turing_multiple_behavioral_sessions`](@ref) runs several sessions, and
# [`turing_behavioral_session`](@ref) runs one. Both return the parameters they were given,
# with the trajectories the sessions saved. So when a model of one's own returns what they
# return, [`session_parameters_chain`](@ref) and [`session_trajectories_chain`](@ref) read
# its fit as they read that of a full model, also when the parameters are transformations of
# the model's variables, which the chain itself doesn't have.
#
# This page fits a Rescorla-Wagner model to several participants with a hierarchical model
# written by hand, and then to a single participant.

using CognitiveModeling
using Distributions: Bernoulli, BernoulliLogit, Exponential, LogNormal, Normal
using LogExpFunctions: logistic
using Random: Xoshiro
using Statistics: mean
using Turing

# ## The model and the data
#
# The Rescorla-Wagner model learns the expected reward of an option from the rewards it
# gets, and chooses the option with a probability that grows with its expected reward:

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward=0.5)
end

function CognitiveModeling.run_timestep(model::RescorlaWagner, context::ModelContext)
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

function CognitiveModeling.run_timestep(
    environment::BernoulliRewards, context::ModelContext
)
    return ModelOutput((; reward=Bernoulli(context.parameters.reward_probability)))
end

function CognitiveModeling.attribute_names(environment::BernoulliRewards)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

# Five participants, each with a learning rate of their own, are simulated for 50
# timesteps, which gives a table with a row for each timestep:

simulated_learning_rates = [0.2, 0.35, 0.5, 0.65, 0.8]
simulations = map(enumerate(simulated_learning_rates)) do (participant, learning_rate)
    return simulate(
        RescorlaWagner(),
        BernoulliRewards();
        who_starts=EnvironmentStarts(),
        n_steps=50,
        cognitive_parameters=(; learning_rate, inverse_temperature=5.0),
        environment_parameters=(; reward_probability=0.7),
        cognitive_save_history=SaveHistory(;
            inputs=:reward, outputs=:choice, save_initial_timestep_values=false
        ),
        rng=Xoshiro(participant),
    )
end
table = (;
    participant=repeat(["P1", "P2", "P3", "P4", "P5"]; inner=50),
    reward=[
        float(observation.reward) for s in simulations for observation in s.observations
    ],
    choice=[action.choice for s in simulations for action in s.actions],
)
sessions_info = MultipleSessionsInfo(
    RescorlaWagner(),
    table;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
)

# ## A hierarchical model of one's own
#
# The learning rates of the participants come from a population on the logit scale: each
# participant's is the mean of the population plus a standardized deviation scaled by the
# population's standard deviation, which `logistic` then takes to between 0 and 1. The
# participants share one inverse temperature. The model gives
# [`turing_multiple_behavioral_sessions`](@ref) the parameters as a `NamedTuple` with a
# vector for each parameter, with a value for every session in the order of
# `sessions_info.session_ids`, and returns what it returns:

@model function hierarchical_learning_rates(sessions_info)
    n_participants = CognitiveModeling.n_sessions(sessions_info)

    #The population of the learning rates, on the logit scale
    population_mean ~ Normal(0, 1)
    population_sd ~ Exponential(0.5)

    #The learning rate of each participant, from its standardized deviation from the mean
    standardized ~ filldist(Normal(), n_participants)
    learning_rate = logistic.(population_mean .+ population_sd .* standardized)

    #One inverse temperature for every participant
    inverse_temperature ~ LogNormal(1, 0.5)

    #The sessions, each with its own parameters, as a submodel without a prefix of its own
    sessions ~ to_submodel(
        turing_multiple_behavioral_sessions(
            sessions_info,
            (;
                learning_rate, inverse_temperature=fill(inverse_temperature, n_participants)
            ),
        ),
        false,
    )
    return sessions
end

model = hierarchical_learning_rates(sessions_info)
chain = sample(Xoshiro(1), model, NUTS(), 500; progress=false)

# ## Reading the fit
#
# The chain has the variables of the model: the population's mean and standard deviation,
# the standardized deviations and the inverse temperature, but not the learning rates, which
# the model computes from them. `returned` gives what the model returned for each sample,
# and [`session_parameters_chain`](@ref) makes a chain of it with the parameters of each
# session, named after the session, such as `P3.learning_rate`:

parameters = session_parameters_chain(returned(model, chain), sessions_info)

# The mean learning rate of each participant, next to the one they were simulated with:

[
    (;
        participant,
        simulated,
        fitted=mean(parameters[@varname($participant.learning_rate)]),
    ) for (participant, simulated) in
    zip(Symbol.(sessions_info.session_ids), simulated_learning_rates)
]

# The trajectories are read the same way. A fit is fastest when the sessions save nothing,
# and the same model with a copy of the sessions that save the expected rewards has the same
# variables, so `returned` runs it on the samples of the chain, and
# [`session_trajectories_chain`](@ref) makes a chain of what the sessions saved:

saving_sessions = MultipleSessionsInfo(
    sessions_info; cognitive_save_history=SaveHistory(; states=:expected_reward)
)
trajectories = session_trajectories_chain(
    returned(hierarchical_learning_rates(saving_sessions), chain), sessions_info
)

# Each entry holds, for every sample, the expected rewards of a participant over the
# timesteps, from timestep 0. The two chains have the samples of the fit, and merge into one:

merge(parameters, trajectories)

# ## A single session
#
# A single session goes in a model of one's own with
# [`turing_behavioral_session`](@ref), with a prefix of its own, here with the learning rate
# sampled on the logit scale:

session_info = SessionInfo(
    RescorlaWagner(),
    (; reward=table.reward[101:150]),
    (; choice=table.choice[101:150]);
    who_starts=EnvironmentStarts(),
    cognitive_save_history=SaveHistory(; states=:expected_reward),
)

@model function one_session(session_info)
    #The learning rate on the logit scale, and the inverse temperature
    learning_rate_logit ~ Normal(0, 1)
    inverse_temperature ~ LogNormal(1, 0.5)

    #The session, with its variables under the prefix session
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(
                session_info,
                (; learning_rate=logistic(learning_rate_logit), inverse_temperature),
            ),
            @varname(session),
        ),
        false,
    )
    return session
end

single_model = one_session(session_info)
single_chain = sample(Xoshiro(2), single_model, NUTS(), 500; progress=false)

# Without a session info, the chains name the parameters and the trajectories without a
# session, such as `learning_rate` and `expected_reward`:

single_results = returned(single_model, single_chain)
merge(session_parameters_chain(single_results), session_trajectories_chain(single_results))

# ## What a model of one's own needs
#
# - It returns what [`turing_multiple_behavioral_sessions`](@ref) or
#   [`turing_behavioral_session`](@ref) returns, as it is, for `returned` to give it to
#   [`session_parameters_chain`](@ref) and [`session_trajectories_chain`](@ref).
# - The parameters of several sessions are a `NamedTuple` with one vector for each
#   parameter, with a value for every session in the order of `sessions_info.session_ids`.
#   [`CognitiveModeling.session_index`](@ref) gives the place of a session's data identifier.
# - The variables of the sessions are under `session[i]`, or under the prefix of a single
#   session, so the model's own variables need other names.
# - The forms of the two functions which take a model and a chain only take the package's
#   own models, which know their sessions, so a model of one's own gives them its results.
