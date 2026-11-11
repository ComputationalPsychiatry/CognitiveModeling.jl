using Turing
using Turing.DynamicPPL: LogDensityFunction, LogDensityProblems

# Test cases for the log density of Turing.jl models of a session, which only this suite runs,
# since Turing.jl isn't loaded by the AD and Reactant suites of the package itself, in
# test/ad_and_reactant. Each case
# differentiates the log density the way a sampler does: through a LogDensityFunction made
# once, with the parameters and the latents in one vector in the order of the model. The
# parameters have flat priors, so the log density is the session's own.

# A Rescorla-Wagner model with a softmax choice, which doesn't learn when there is no observation
struct TuringRescorlaWagner <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringRescorlaWagner, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_model(model::TuringRescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    r = context.input.reward
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + context.parameters.learning_rate * (r - V₋₁)
    return ModelOutput(
        (; choice=BernoulliLogit(context.parameters.inverse_temperature * (Vₜ - 0.5))),
        (; expected_reward=Vₜ),
    )
end
function CognitiveModeling.attribute_names(model::TuringRescorlaWagner)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# A continuous action, so that missing ones can be imputed
struct TuringContinuousAction <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringContinuousAction, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_model(model::TuringContinuousAction, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput(
        (; choice=Normal(Vₜ, context.parameters.noise_scale)), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::TuringContinuousAction)
    return ModelAttributes(;
        parameters=(:learning_rate, :noise_scale),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# Samples noise within each timestep in Turing
struct TuringNoisyModel <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringNoisyModel, parameters)
    return (; expected_reward=0.5)
end
Turing.@model function CognitiveModeling.turing_apply_model(
    model::TuringNoisyModel, context::ModelContext
)
    noise ~ Normal(0, context.parameters.noise_scale)
    Vₜ =
        context.previous_state.expected_reward +
        context.parameters.learning_rate *
        (context.input.reward - context.previous_state.expected_reward) +
        noise
    return ModelOutput((; choice=Normal(Vₜ, 1.0)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::TuringNoisyModel)
    return ModelAttributes(;
        parameters=(:learning_rate, :noise_scale),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# Gives the previous choice as the reward, scaled by a parameter
struct TuringRewardsPreviousChoice <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::TuringRewardsPreviousChoice, parameters
)
    return (;)
end
function CognitiveModeling.apply_model(
    environment::TuringRewardsPreviousChoice, context::ModelContext
)
    reward = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    return ModelOutput((; reward=Deterministic(reward * context.parameters.scale)))
end
function CognitiveModeling.attribute_names(environment::TuringRewardsPreviousChoice)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end

# Gives a distribution around the previous choice to sample the reward from, so that a missing reward is a latent
struct TuringNoisyRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::TuringNoisyRewards, parameters) = (;)
function CognitiveModeling.apply_model(
    environment::TuringNoisyRewards, context::ModelContext
)
    reward = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    return ModelOutput((; reward=Normal(reward * context.parameters.scale, 1.0)))
end
function CognitiveModeling.attribute_names(environment::TuringNoisyRewards)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end

const TURING_REWARDS = (; reward=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
const TURING_CHOICES = (;
    choice=[true, false, true, true, false, false, true, true, false, true]
)
const TURING_CONTINUOUS_CHOICES = (;
    choice=[0.3, 0.6, MissingValue(), 0.4, 0.7, MissingValue(), 0.5, 0.2, 0.8, 0.6]
)
const TURING_REWARDS_NOT_ALWAYS_GIVEN = (;
    reward=[
        1.0, 0.0, NoObservation(), 1.0, 0.0, NoObservation(), 0.0, 0.0, NoObservation(), 1.0
    ]
)
const TURING_MISSING_REWARDS = (;
    reward=[
        MissingValue(),
        0.0,
        MissingValue(),
        1.0,
        MissingValue(),
        1.0,
        0.0,
        MissingValue(),
        1.0,
        1.0,
    ]
)
# The rewards TuringRewardsPreviousChoice gives with a scale of 1, the previous choice and 0.5 before the first, with some missing, so that
# the recorded ones match what it gives, which they must, since a Deterministic is scored as a point mass
const TURING_MISSING_PREVIOUS_CHOICE_REWARDS = (;
    reward=[
        MissingValue(),
        1.0,
        MissingValue(),
        1.0,
        MissingValue(),
        0.0,
        0.0,
        MissingValue(),
        1.0,
        0.0,
    ]
)

# The sessions, prepared once, with flat priors on the parameters
Turing.@model function fit_rescorla_wagner(session_info)
    learning_rate ~ Turing.Flat()
    inverse_temperature ~ Turing.Flat()
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(session_info, (; learning_rate, inverse_temperature)),
            @varname(session),
        ),
        false,
    )
    return nothing
end

Turing.@model function fit_with_imputed_actions(session_info)
    learning_rate ~ Turing.Flat()
    noise_scale ~ Turing.Flat()
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(session_info, (; learning_rate, noise_scale)),
            @varname(session),
        ),
        false,
    )
    return nothing
end

# A session with an environment, whose parameter is fitted too
Turing.@model function fit_with_environment(session_info)
    learning_rate ~ Turing.Flat()
    inverse_temperature ~ Turing.Flat()
    scale ~ Turing.Flat()
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(
                session_info, (; learning_rate, inverse_temperature), (; scale)
            ),
            @varname(session),
        ),
        false,
    )
    return nothing
end

# The same with the environment's scale fixed at 1, for an environment which gives its rewards without randomness: the recorded rewards
# are scored as point masses, which a fitted scale would make -Inf everywhere but at one value
Turing.@model function fit_with_fixed_environment(session_info)
    learning_rate ~ Turing.Flat()
    inverse_temperature ~ Turing.Flat()
    session ~ to_submodel(
        prefix(
            turing_behavioral_session(
                session_info, (; learning_rate, inverse_temperature), (; scale=1.0)
            ),
            @varname(session),
        ),
        false,
    )
    return nothing
end

# Three sessions of the Rescorla-Wagner model, through the full model with independent priors on each session's parameters
const TURING_TABLE = (;
    id=repeat([1, 2, 3]; inner=10),
    reward=[TURING_REWARDS.reward; reverse(TURING_REWARDS.reward); TURING_REWARDS.reward],
    choice=[TURING_CHOICES.choice; TURING_CHOICES.choice; reverse(TURING_CHOICES.choice)],
)
const TURING_SESSIONS = MultipleSessionsInfo(
    TuringRescorlaWagner(),
    TURING_TABLE;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)
const TURING_FULL_MODEL = turing_full_model(
    IndependentSessionParameters((;
        learning_rate=Turing.Flat(), inverse_temperature=Turing.Flat()
    )),
    TURING_SESSIONS,
)

# The same sessions with an environment whose parameter is sampled once for all of them, so that every session scores its rewards
const TURING_SESSIONS_WITH_ENVIRONMENT = MultipleSessionsInfo(
    TuringRescorlaWagner(),
    TURING_TABLE;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
    environment=TuringNoisyRewards(),
)
const TURING_FULL_MODEL_WITH_ENVIRONMENT_PARAMETERS = turing_full_model(
    IndependentSessionParameters((;
        learning_rate=Turing.Flat(), inverse_temperature=Turing.Flat()
    )),
    TURING_SESSIONS_WITH_ENVIRONMENT;
    environment_parameters_model=IdenticalSessionParameters((; scale=Turing.Flat())),
)

# The same sessions with a column that puts the first and the third in one group, for parameters from the distributions of groups
const TURING_GROUPED_SESSIONS = MultipleSessionsInfo(
    TuringRescorlaWagner(),
    merge(TURING_TABLE, (; group=repeat(["a", "b", "a"]; inner=10)));
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)
turing_to_unit_interval(x) = 1 / (1 + exp(-x))
const TURING_GROUP_PRIORS = (; mean=Normal(0, 1), sd=Exponential(1))

# Both parameters from the distributions of groups, as standardized values: the learning rate with two groups, the inverse temperature with one
const TURING_FULL_MODEL_WITH_GROUPS = turing_full_model(
    GroupedHierarchicalSessionParameters((;
        learning_rate=HierarchicalParameter(
            TURING_GROUP_PRIORS; transformation=turing_to_unit_interval, group_by=:group
        ),
        inverse_temperature=HierarchicalParameter(TURING_GROUP_PRIORS; transformation=exp),
    )),
    TURING_GROUPED_SESSIONS,
)

# A composite: the learning rate from the distributions of two groups directly, and the inverse temperature independent for each session
const TURING_FULL_MODEL_WITH_COMPOSITE = turing_full_model(
    CompositeSessionParameters(
        GroupedHierarchicalSessionParameters((;
            learning_rate=HierarchicalParameter(
                TURING_GROUP_PRIORS;
                transformation=turing_to_unit_interval,
                group_by=:group,
                centered=true,
            ),
        )),
        IndependentSessionParameters((; inverse_temperature=Turing.Flat())),
    ),
    TURING_GROUPED_SESSIONS,
)

#The session of each case, prepared once
function rescorla_wagner_session(who_starts)
    return SessionInfo(TuringRescorlaWagner(), TURING_REWARDS, TURING_CHOICES; who_starts)
end
function rewards_not_always_given_session(who_starts)
    return SessionInfo(
        TuringRescorlaWagner(), TURING_REWARDS_NOT_ALWAYS_GIVEN, TURING_CHOICES; who_starts
    )
end
function imputed_actions_session()
    return SessionInfo(
        TuringContinuousAction(),
        TURING_REWARDS,
        TURING_CONTINUOUS_CHOICES;
        who_starts=EnvironmentStarts(),
    )
end
function environment_session()
    return SessionInfo(
        TuringRescorlaWagner(),
        TURING_MISSING_PREVIOUS_CHOICE_REWARDS,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
        environment=TuringRewardsPreviousChoice(),
    )
end
function complete_environment_session()
    return SessionInfo(
        TuringRescorlaWagner(),
        TURING_REWARDS,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
        environment=TuringRewardsPreviousChoice(),
    )
end
function imputed_observations_session()
    return SessionInfo(
        TuringRescorlaWagner(),
        TURING_MISSING_REWARDS,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
        environment=TuringNoisyRewards(),
    )
end
function noise_in_timesteps_session()
    return SessionInfo(
        TuringNoisyModel(),
        TURING_REWARDS,
        (; choice=TURING_REWARDS.reward .- 0.5);
        who_starts=EnvironmentStarts(),
    )
end

# The log density as a function of the vector of parameters and latents, through a LogDensityFunction made once
function log_density_of(model)
    log_density_function = LogDensityFunction(model)
    return x -> LogDensityProblems.logdensity(log_density_function, x)
end

const TURING_CASES = [
    (;
        name="turing session, environment starts",
        f=log_density_of(fit_rescorla_wagner(rescorla_wagner_session(EnvironmentStarts()))),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session, agent starts",
        f=log_density_of(fit_rescorla_wagner(rescorla_wagner_session(AgentStarts()))),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with imputed actions",
        f=log_density_of(fit_with_imputed_actions(imputed_actions_session())),
        x=[0.3, 0.8, 0.2, -0.1],
        reactant=false,
    ),
    (;
        name="turing session with an environment",
        f=log_density_of(fit_with_fixed_environment(environment_session())),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with an environment which doesn't run, since nothing in the data is missing",
        f=log_density_of(fit_with_environment(complete_environment_session())),
        x=[0.3, 2.0, 0.9],
        reactant=false,
    ),
    (;
        name="turing session with imputed observations",
        f=log_density_of(fit_with_environment(imputed_observations_session())),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
        reactant=false,
    ),
    (;
        name="turing session with a reward that isn't always given, environment starts",
        f=log_density_of(
            fit_rescorla_wagner(rewards_not_always_given_session(EnvironmentStarts()))
        ),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with a reward that isn't always given, agent starts",
        f=log_density_of(
            fit_rescorla_wagner(rewards_not_always_given_session(AgentStarts()))
        ),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with noise in each timestep",
        f=log_density_of(fit_with_imputed_actions(noise_in_timesteps_session())),
        x=[0.3, 0.8, 0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0, -0.3, 0.15, 0.1],
        reactant=false,
    ),
    (;
        name="turing full model with three sessions and independent priors",
        f=log_density_of(TURING_FULL_MODEL),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
        reactant=false,
    ),
    (;
        name="turing full model with an environment parameter shared by the sessions",
        f=log_density_of(TURING_FULL_MODEL_WITH_ENVIRONMENT_PARAMETERS),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.9],
        reactant=false,
    ),
    (;
        name="turing full model with parameters from the distributions of groups",
        f=log_density_of(TURING_FULL_MODEL_WITH_GROUPS),
        x=[0.2, -0.4, 0.8, 1.2, 0.3, -0.2, 0.1, 0.5, 0.6, 0.4, -0.3, 0.2],
        reactant=false,
    ),
    (;
        name="turing full model with a composite of centered group parameters and independent ones",
        f=log_density_of(TURING_FULL_MODEL_WITH_COMPOSITE),
        x=[0.2, -0.4, 0.8, 1.2, 0.3, -0.5, 0.1, 1.0, 2.0, 3.0],
        reactant=false,
    ),
]
