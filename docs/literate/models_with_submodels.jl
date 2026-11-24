# # Models with submodels
#
# A cognitive model can be built from parts that are models themselves, such as a perception
# that learns and a response that chooses. This page shows how a model holds such submodels,
# how their parameters and their states are named, and how the model is simulated and
# fitted. An environment model holds submodels in the same way.

using CognitiveModeling
using Distributions: BernoulliLogit, Beta, LogNormal
using Random: Xoshiro
using Statistics: mean
using Turing

# ## The submodels
#
# A submodel is an [`AbstractSubmodel`](@ref). Like a cognitive model, it names its
# parameters and its states in [`CognitiveModeling.attribute_names`](@ref), and it can have
# an initial state, default parameters and domains. It has no inputs or outputs of its own,
# and no timestep: the model that holds it decides when it is used, and with what.
#
# This perception learns the expected reward of an option, and keeps its last prediction
# error. Its initial value has a default, and its learning rate has a domain:

struct Perception <: AbstractSubmodel end

function CognitiveModeling.attribute_names(perception::Perception)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value),
        states=(:expected_reward, :prediction_error),
        inputs=(),
        outputs=(),
    )
end

CognitiveModeling.default_parameters(perception::Perception) = (; initial_value=0.5)

function CognitiveModeling.parameter_domains(perception::Perception)
    return (; learning_rate=Domain(; lower=0, upper=1))
end

function CognitiveModeling.initialize_state(perception::Perception, parameters)
    return (; expected_reward=parameters.initial_value, prediction_error=0.0)
end

# What a submodel does is written as functions of one's own, with any names and arguments.
# This one gives the state of the perception after a reward:

function update_perception(perception::Perception, parameters, state, reward)
    α = parameters.learning_rate
    δ = reward - state.expected_reward
    return (; expected_reward=state.expected_reward + α * δ, prediction_error=δ)
end

# The response chooses the option with a probability that grows with its expected reward.
# It has a parameter, and no state:

struct Response <: AbstractSubmodel end

function CognitiveModeling.attribute_names(response::Response)
    return ModelAttributes(;
        parameters=:inverse_temperature, states=(), inputs=(), outputs=()
    )
end

function choice_distribution(response::Response, parameters, expected_reward)
    return BernoulliLogit(parameters.inverse_temperature * (expected_reward - 0.5))
end

# ## The model
#
# The model holds the two in fields, and names those fields in
# [`CognitiveModeling.submodel_fields`](@ref). Its own attributes are only what is its own:
# here the reward it receives, the choice it gives, and a count of its rewards as a state.

struct PerceivesAndResponds <: AbstractCognitiveModel
    perception::Perception
    response::Response
end

CognitiveModeling.submodel_fields(model::PerceivesAndResponds) = (:perception, :response)

function CognitiveModeling.attribute_names(model::PerceivesAndResponds)
    return ModelAttributes(;
        parameters=(), states=:n_rewards, inputs=:reward, outputs=:choice
    )
end

# The model reads the parameters of each submodel under the name of its field, as in
# `parameters.perception`, and keeps the state of each submodel that has one under that
# name too:

function CognitiveModeling.initialize_state(model::PerceivesAndResponds, parameters)
    return (;
        n_rewards=0,
        perception=CognitiveModeling.initialize_state(
            model.perception, parameters.perception
        ),
    )
end

function CognitiveModeling.timestep_model(
    model::PerceivesAndResponds, context::ModelContext
)
    perception = update_perception(
        model.perception,
        context.parameters.perception,
        context.previous_state.perception,
        context.input.reward,
    )
    choice = choice_distribution(
        model.response, context.parameters.response, perception.expected_reward
    )
    n_rewards = context.previous_state.n_rewards + (context.input.reward > 0)
    return ModelOutput((; choice), (; n_rewards, perception))
end

model = PerceivesAndResponds(Perception(), Response())

# ## The names of the parameters
#
# Everywhere outside the model, its parameters are flat: those of the model and of its
# submodels in one `NamedTuple`. [`CognitiveModeling.get_parameter_names`](@ref) gives
# their names:

parameter_names = CognitiveModeling.get_parameter_names(model)

# With `with_default_parameters = false`, it gives those without a default value, which a
# simulation or a fit must be given:

names_to_give = CognitiveModeling.get_parameter_names(model; with_default_parameters=false)

# The default values and the domains of the submodels are read with the model's in the same
# way, by [`CognitiveModeling.get_default_parameters`](@ref) and
# [`CognitiveModeling.get_parameter_domains`](@ref):

default_parameters = CognitiveModeling.get_default_parameters(model)

# Two submodels with a parameter of the same name need prefixes. When
# [`CognitiveModeling.uses_submodel_prefixes`](@ref) returns `true` for a model, the flat
# names of its submodels' parameters start with the name of the submodel, as in
# `perception__learning_rate`.
#
# [`CognitiveModeling.bundle_parameters`](@ref) gives flat parameters as the model itself
# reads them, which is useful for calling its functions by hand:

bundled_parameters = CognitiveModeling.bundle_parameters(
    model, (; learning_rate=0.3, initial_value=0.5, inverse_temperature=3.0)
)

# ## Simulating
#
# A simulation takes the flat parameters. Two participants with different learning rates
# each see 80 rewards, which come with a probability of 0.8:

learning_rates = [0.1, 0.6]
simulations = map(1:2) do participant
    rewards = Float64.(rand(Xoshiro(participant), 80) .< 0.8)
    return simulate_session(
        model,
        FixedObservationsEnvironment((; reward=rewards));
        who_starts=EnvironmentStarts(),
        n_steps=80,
        cognitive_parameters=(;
            learning_rate=learning_rates[participant], inverse_temperature=3.0
        ),
        environment_parameters=(;),
        action_map=NameMap(),
        save_cognitive_trajectories=SaveTrajectories(;
            states=(:n_rewards, :perception => :expected_reward),
            inputs=:reward,
            outputs=:choice,
            save_initial_timestep_values=false,
        ),
        rng=Xoshiro(participant),
    )
end

# The state of a submodel is saved under the name of the submodel. The name alone selects
# all of it, and here the [`SaveTrajectories`](@ref) selects one of its two states with
# `:perception => :expected_reward`, so the prediction errors aren't saved:

last_saved_state = simulations[1].states[end]

# ## Fitting
#
# The priors of a fit are given under the flat names too:

table = (;
    participant=repeat(["P1", "P2"]; inner=80),
    reward=[observation.reward for s in simulations for observation in s.observations],
    choice=[action.choice for s in simulations for action in s.actions],
)
sessions_info = MultipleSessionsInfo(
    model,
    table;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
)
priors = (; learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5))
fitted_model = full_model(TuringBackend(), IndependentParameters(priors), sessions_info)
chain = sample(Xoshiro(3), fitted_model, NUTS(), 300; progress=false)

# The trajectories of the fit have an entry for each state of a submodel that is saved,
# named after the session, the submodel and the state, beside whether the parameters of
# each session stayed valid:

fitted_trajectories = get_session_trajectories(
    fitted_model,
    chain;
    save_cognitive_trajectories=SaveTrajectories(; states=:perception => :expected_reward),
)
trajectory_names = string.(Turing.FlexiChains.get_name.(keys(fitted_trajectories)))

# The expected reward of the first participant after the last timestep, averaged over the
# samples:

final_expected_reward = mean(
    last, fitted_trajectories[@varname(P1.perception.expected_reward)]
)
