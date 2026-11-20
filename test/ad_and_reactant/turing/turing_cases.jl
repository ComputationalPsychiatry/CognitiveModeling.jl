using Turing
using Turing.DynamicPPL:
    LogDensityFunction, LogDensityProblems, getlogjoint_internal, LinkAll

# Test cases for the log density of Turing.jl models of a session, which only this suite runs,
# since Turing.jl isn't loaded by the AD and Reactant suites of the package itself, in
# test/ad_and_reactant/ad and test/ad_and_reactant/reactant. Each case
# differentiates the log density the way a sampler does: through a LogDensityFunction made
# once, with the parameters and the latents in one vector in the order of the model. The
# parameters have flat priors, so the log density is the session's own.

# A Rescorla-Wagner model with a softmax choice, which doesn't learn when there is no observation
struct TuringRescorlaWagner <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringRescorlaWagner, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.run_timestep(model::TuringRescorlaWagner, context::ModelContext)
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

# The same model with the expected reward it starts from as a third parameter
struct TuringRescorlaWagnerWithInitialValue <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(
    model::TuringRescorlaWagnerWithInitialValue, parameters
)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.run_timestep(
    model::TuringRescorlaWagnerWithInitialValue, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput(
        (; choice=BernoulliLogit(context.parameters.inverse_temperature * (Vₜ - 0.5))),
        (; expected_reward=Vₜ),
    )
end
function CognitiveModeling.attribute_names(model::TuringRescorlaWagnerWithInitialValue)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :initial_value),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# The Rescorla-Wagner model with domains for its parameters, which a session checks them against on every evaluation, with the number types of each backend
struct TuringRescorlaWagnerWithinDomains <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(
    model::TuringRescorlaWagnerWithinDomains, parameters
)
    return (; expected_reward=0.5)
end
function CognitiveModeling.run_timestep(
    model::TuringRescorlaWagnerWithinDomains, context::ModelContext
)
    return CognitiveModeling.run_timestep(TuringRescorlaWagner(), context)
end
function CognitiveModeling.attribute_names(model::TuringRescorlaWagnerWithinDomains)
    return CognitiveModeling.attribute_names(TuringRescorlaWagner())
end
function CognitiveModeling.parameter_domains(model::TuringRescorlaWagnerWithinDomains)
    return (;
        learning_rate=Domain(; lower=0, upper=1), inverse_temperature=Domain(; lower=0)
    )
end

# Three observations, so that a row of the data has three columns, of which the bonus can be NoValue
struct TuringThreeObservations <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringThreeObservations, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.run_timestep(
    model::TuringThreeObservations, context::ModelContext
)
    #The reward updates the expected reward, and the cue and the bonus, when it is given, shift the choice
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    bonus = context.input.bonus isa NoValue ? 0.0 : context.input.bonus
    logit =
        context.parameters.inverse_temperature * (Vₜ - 0.5) +
        0.5 * context.input.cue +
        bonus
    return ModelOutput((; choice=BernoulliLogit(logit)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::TuringThreeObservations)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=(:reward, :cue, :bonus),
        outputs=:choice,
    )
end

# A continuous action, so that missing ones can be imputed
struct TuringContinuousAction <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringContinuousAction, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.run_timestep(
    model::TuringContinuousAction, context::ModelContext
)
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
Turing.@model function CognitiveModeling.turing_run_timestep(
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
function CognitiveModeling.run_timestep(
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
function CognitiveModeling.run_timestep(
    environment::TuringNoisyRewards, context::ModelContext
)
    reward = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    return ModelOutput((; reward=Normal(reward * context.parameters.scale, 1.0)))
end
function CognitiveModeling.attribute_names(environment::TuringNoisyRewards)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end

# The same environment with a domain for its scale
struct TuringNoisyRewardsWithinDomain <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::TuringNoisyRewardsWithinDomain, parameters
)
    return (;)
end
function CognitiveModeling.run_timestep(
    environment::TuringNoisyRewardsWithinDomain, context::ModelContext
)
    return CognitiveModeling.run_timestep(TuringNoisyRewards(), context)
end
function CognitiveModeling.attribute_names(environment::TuringNoisyRewardsWithinDomain)
    return CognitiveModeling.attribute_names(TuringNoisyRewards())
end
function CognitiveModeling.parameter_domains(environment::TuringNoisyRewardsWithinDomain)
    return (; scale=Domain(; lower=0),)
end

const TURING_REWARDS = (; reward=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
const TURING_CHOICES = (;
    choice=[true, false, true, true, false, false, true, true, false, true]
)
const TURING_CONTINUOUS_CHOICES = (;
    choice=[0.3, 0.6, MissingValue(), 0.4, 0.7, MissingValue(), 0.5, 0.2, 0.8, 0.6]
)
const TURING_REWARDS_NOT_ALWAYS_GIVEN = (;
    reward=[1.0, 0.0, NoValue(), 1.0, 0.0, NoValue(), 0.0, 0.0, NoValue(), 1.0]
)
const TURING_THREE_OBSERVATIONS = (;
    TURING_REWARDS...,
    cue=[1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
    bonus=[0.2, 0.0, 0.1, 0.3, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1],
)
const TURING_THREE_OBSERVATIONS_BONUS_NOT_ALWAYS_GIVEN = (;
    TURING_THREE_OBSERVATIONS...,
    bonus=Union{NoValue,Float64}[
        0.2, NoValue(), 0.1, 0.3, NoValue(), 0.2, 0.1, NoValue(), 0.3, 0.1
    ],
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
# the ones in the data match what it gives, which they must, since a Deterministic is scored as a point mass
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

# The same with the environment's scale fixed at 1, for an environment which gives its rewards without randomness: the rewards in the data
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
const TURING_FULL_MODEL = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Turing.Flat(), inverse_temperature=Turing.Flat()
    )),
    TURING_SESSIONS,
)

# The same sessions with the model of three parameters. Each session's parameters are read from a NamedTuple with a vector for
# each parameter, and from three on, Enzyme stopped the process on a row read with a recursive merge, which select_row avoids
const TURING_FULL_MODEL_WITH_THREE_PARAMETERS = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Turing.Flat(),
        inverse_temperature=Turing.Flat(),
        initial_value=Turing.Flat(),
    )),
    MultipleSessionsInfo(
        TuringRescorlaWagnerWithInitialValue(),
        TURING_TABLE;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
)

# The same sessions with the model whose parameters have domains, and priors that stay within them
const TURING_FULL_MODEL_WITH_PARAMETER_DOMAINS = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Uniform(0, 1), inverse_temperature=Turing.FlatPos(0.0)
    )),
    MultipleSessionsInfo(
        TuringRescorlaWagnerWithinDomains(),
        TURING_TABLE;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
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
const TURING_FULL_MODEL_WITH_ENVIRONMENT_PARAMETERS = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Turing.Flat(), inverse_temperature=Turing.Flat()
    )),
    TURING_SESSIONS_WITH_ENVIRONMENT;
    environment_parameters_model=IdenticalParameters((; scale=Turing.Flat())),
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

# Both parameters from the distributions of groups, non-centered, as standard normal values: the learning rate with two groups, the inverse temperature with one
const TURING_FULL_MODEL_WITH_GROUPS = full_model(
    TuringBackend(),
    HierarchicalGroupedParameters((;
        learning_rate=HierarchicalParameter(
            TURING_GROUP_PRIORS; transformation=turing_to_unit_interval, group_by=:group
        ),
        inverse_temperature=HierarchicalParameter(TURING_GROUP_PRIORS; transformation=exp),
    )),
    TURING_GROUPED_SESSIONS,
)

# A composite: the learning rate from the distributions of two groups directly, and the inverse temperature independent for each session
const TURING_FULL_MODEL_WITH_COMPOSITE = full_model(
    TuringBackend(),
    CompositeParameters(
        HierarchicalGroupedParameters((;
            learning_rate=HierarchicalParameter(
                TURING_GROUP_PRIORS;
                transformation=turing_to_unit_interval,
                group_by=:group,
                centered=true,
            ),
        )),
        IndependentParameters((; inverse_temperature=Turing.Flat())),
    ),
    TURING_GROUPED_SESSIONS,
)

# The same sessions with the columns of a regression: an age and a group for each session, and a participant with two of them
const TURING_REGRESSION_SESSIONS = MultipleSessionsInfo(
    TuringRescorlaWagner(),
    merge(
        TURING_TABLE,
        (;
            group=repeat(["a", "b", "a"]; inner=10),
            age=repeat([-1.0, 0.5, 1.0]; inner=10),
            participant=repeat(["P1", "P2", "P1"]; inner=10),
        ),
    );
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)

# Both parameters from regressions: the learning rate on the age, the group and their interaction, with a varying intercept
# and a varying slope of the age for each participant, and a varying intercept for each session, whose standard deviation
# is one for each group, and the inverse temperature with centered varying intercepts of the participants
const TURING_FULL_MODEL_WITH_REGRESSION = full_model(
    TuringBackend(),
    HierarchicalRegressionParameters((;
        learning_rate=RegressionParameter(
            Intercept(Normal(0, 1)),
            ContinuousPredictor(:age, Normal(0, 1)),
            CategoricalPredictor(:group, Normal(0, 1)),
            Interaction((:age, :group), Normal(0, 1)),
            VaryingEffects(
                :participant,
                Intercept(Exponential(1)),
                ContinuousPredictor(:age, Exponential(1)),
            ),
            VaryingEffects(:id, Intercept(Exponential(1)); group_by=:group, name=:session);
            transformation=turing_to_unit_interval,
        ),
        inverse_temperature=RegressionParameter(
            Intercept(Normal(0, 1)),
            VaryingEffects(:participant, Intercept(Exponential(1)); centered=true);
            transformation=exp,
        ),
    )),
    TURING_REGRESSION_SESSIONS,
)

# The inverse temperature of each session sampled from the distribution of a family, a log-normal one whose mean
# the regression gives and whose standard deviation has a prior, next to a learning rate from a regression without one
const TURING_FULL_MODEL_WITH_FAMILY_DISTRIBUTION = full_model(
    TuringBackend(),
    HierarchicalRegressionParameters((;
        learning_rate=RegressionParameter(
            Intercept(Normal(0, 1)),
            ContinuousPredictor(:age, Normal(0, 1));
            transformation=turing_to_unit_interval,
        ),
        inverse_temperature=RegressionParameter(
            Intercept(Normal(0, 1));
            family=RegressionFamily(
                (; mean=RegressionOutcome(), sd=Exponential(1)); distribution=LogNormal
            ),
        ),
    )),
    TURING_REGRESSION_SESSIONS,
)

# Sets of sessions where only some have latent variables. The first and the third session of three impute their missing
# actions, which the model gives continuously, and the second has all of its own
const TURING_COMPLETE_CONTINUOUS_CHOICES = (;
    choice=[0.3, 0.6, 0.5, 0.4, 0.7, 0.4, 0.5, 0.2, 0.8, 0.6]
)
const TURING_FULL_MODEL_WITH_SOME_SESSIONS_IMPUTING_ACTIONS = full_model(
    TuringBackend(),
    IndependentParameters((; learning_rate=Turing.Flat(), noise_scale=Turing.Flat())),
    MultipleSessionsInfo(
        TuringContinuousAction(),
        (;
            id=repeat([1, 2, 3]; inner=10),
            reward=TURING_TABLE.reward,
            choice=[
                TURING_CONTINUOUS_CHOICES.choice
                TURING_COMPLETE_CONTINUOUS_CHOICES.choice
                reverse(TURING_CONTINUOUS_CHOICES.choice)
            ],
        );
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
)

# The first and the third session run their environment, whose scale is fixed, to impute their missing rewards, and the
# second, which has all of its rewards, doesn't run it
const TURING_FULL_MODEL_WITH_SOME_SESSIONS_RUNNING_THE_ENVIRONMENT = full_model(
    TuringBackend(),
    IndependentParameters((;
        learning_rate=Turing.Flat(), inverse_temperature=Turing.Flat()
    )),
    MultipleSessionsInfo(
        TuringRescorlaWagner(),
        (;
            id=repeat([1, 2, 3]; inner=10),
            reward=[
                TURING_MISSING_REWARDS.reward
                reverse(TURING_REWARDS.reward)
                reverse(TURING_MISSING_REWARDS.reward)
            ],
            choice=TURING_TABLE.choice,
        );
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        environment=TuringNoisyRewards(),
    );
    fixed_environment_parameters=(; scale=1.0),
)

# Sessions of three, six and ten timesteps, where only the second and the third impute missing actions, at the same
# timesteps
const TURING_FULL_MODEL_WITH_SESSIONS_OF_DIFFERENT_LENGTHS = full_model(
    TuringBackend(),
    IndependentParameters((; learning_rate=Turing.Flat(), noise_scale=Turing.Flat())),
    MultipleSessionsInfo(
        TuringContinuousAction(),
        (;
            id=[fill(1, 3); fill(2, 6); fill(3, 10)],
            reward=[
                TURING_REWARDS.reward[1:3]
                TURING_REWARDS.reward[1:6]
                TURING_REWARDS.reward
            ],
            choice=[
                TURING_COMPLETE_CONTINUOUS_CHOICES.choice[1:3]
                TURING_CONTINUOUS_CHOICES.choice[1:6]
                TURING_CONTINUOUS_CHOICES.choice
            ],
        );
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
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
function three_observations_session(observations)
    return SessionInfo(
        TuringThreeObservations(),
        observations,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
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
function parameter_domains_session()
    return SessionInfo(
        TuringRescorlaWagnerWithinDomains(),
        TURING_REWARDS,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
    )
end
function parameter_domains_with_environment_session()
    return SessionInfo(
        TuringRescorlaWagnerWithinDomains(),
        TURING_MISSING_REWARDS,
        TURING_CHOICES;
        who_starts=EnvironmentStarts(),
        environment=TuringNoisyRewardsWithinDomain(),
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
# A model which chooses one of three options with probabilities that are a parameter, a vector
struct TuringChoosesWithProbabilities <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(
    model::TuringChoosesWithProbabilities, parameters
)
    return (;)
end
function CognitiveModeling.run_timestep(
    model::TuringChoosesWithProbabilities, context::ModelContext
)
    return ModelOutput((; choice=Categorical(context.parameters.choice_probabilities)))
end
function CognitiveModeling.attribute_names(model::TuringChoosesWithProbabilities)
    return ModelAttributes(;
        parameters=:choice_probabilities, states=(), inputs=:trial, outputs=:choice
    )
end
function CognitiveModeling.parameter_domains(model::TuringChoosesWithProbabilities)
    return (; choice_probabilities=Domain(; type=AbstractVector{Real}, lower=0, upper=1))
end

# Three sessions of ten choices, with the trial as the observation, and the first and the third in one group
const TURING_CHOICE_SESSIONS = MultipleSessionsInfo(
    TuringChoosesWithProbabilities(),
    (;
        id=repeat(["S1", "S2", "S3"]; inner=10),
        trial=repeat(1:10, 3),
        choice=repeat([1, 2, 3, 3, 2, 1, 1, 3, 2, 1], 3),
        group=repeat(["a", "b", "a"]; inner=10),
    );
    observation_columns=:trial,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)

# A vector of choice probabilities for each session, from a Dirichlet prior, and from a hierarchical Dirichlet whose concentrations two groups have each
const TURING_FULL_MODEL_WITH_DIRICHLET_PRIORS = full_model(
    TuringBackend(),
    IndependentParameters((; choice_probabilities=Dirichlet(ones(3)))),
    TURING_CHOICE_SESSIONS,
)
const TURING_FULL_MODEL_WITH_HIERARCHICAL_DIRICHLET = full_model(
    TuringBackend(),
    HierarchicalGroupedParameters((;
        choice_probabilities=HierarchicalParameter(
            (; concentration=filldist(Gamma(2, 2), 3));
            distribution=Dirichlet,
            group_by=:group,
        ),
    )),
    TURING_CHOICE_SESSIONS,
)

# A stand-in for the distributions of SequentialSamplingModels.jl, whose values are NamedTuples: a choice of 1 with a probability and of 2 otherwise, and an exponential response time with a rate
struct TuringChoiceAndTime{T} <: Turing.Distributions.Distribution{
    Turing.Distributions.Multivariate,Turing.Distributions.ValueSupport
}
    probability::T
    rate::T
end
function Turing.Distributions.logpdf(distribution::TuringChoiceAndTime, value::NamedTuple)
    choice_density = if value.choice == 1
        log(distribution.probability)
    else
        log1p(-distribution.probability)
    end
    return choice_density + log(distribution.rate) - distribution.rate * value.rt
end

# A model which responds with a choice and a time from the stand-in, and with a confidence, a number
struct TuringRespondsWithChoiceAndTime <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(
    model::TuringRespondsWithChoiceAndTime, parameters
)
    return (;)
end
function CognitiveModeling.run_timestep(
    model::TuringRespondsWithChoiceAndTime, context::ModelContext
)
    probability = context.parameters.choice_probability
    return ModelOutput((;
        response=TuringChoiceAndTime(probability, context.parameters.rate),
        confidence=Normal(probability, 0.1),
    ))
end
function CognitiveModeling.attribute_names(model::TuringRespondsWithChoiceAndTime)
    return ModelAttributes(;
        parameters=(:choice_probability, :rate),
        states=(),
        inputs=:trial,
        outputs=(:response, :confidence),
    )
end

# Three sessions of ten responses, with the choice and the response time in two columns, which make one action
const TURING_FULL_MODEL_WITH_RESPONSES = full_model(
    TuringBackend(),
    IndependentParameters((; choice_probability=Beta(2, 2), rate=LogNormal(0, 0.5))),
    MultipleSessionsInfo(
        TuringRespondsWithChoiceAndTime(),
        (;
            id=repeat(["S1", "S2", "S3"]; inner=10),
            trial=repeat(1:10, 3),
            choice=repeat([1, 2, 1, 1, 2, 1, 2, 2, 1, 1], 3),
            rt=repeat([0.4, 0.7, 0.3, 0.5, 0.2, 0.9, 1.1, 0.6, 0.8, 0.5], 3),
            confidence=repeat([0.7, 0.4, 0.8, 0.9, 0.8, 0.3, 0.2, 0.6, 0.5, 0.7], 3),
        );
        observation_columns=:trial,
        action_columns=((; choice=:choice, rt=:rt) => :response, :confidence),
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
)

# A model which reports two numbers at once, from a normal distribution over vectors around a mean that is a parameter
struct TuringReportsTwoNumbers <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TuringReportsTwoNumbers, parameters) = (;)
function CognitiveModeling.run_timestep(
    model::TuringReportsTwoNumbers, context::ModelContext
)
    return ModelOutput((;
        report=MvNormal(fill(context.parameters.mean, 2), [1.0 0.0; 0.0 1.0])
    ))
end
function CognitiveModeling.attribute_names(model::TuringReportsTwoNumbers)
    return ModelAttributes(; parameters=:mean, states=(), inputs=:trial, outputs=:report)
end

# Three sessions of ten reports, each made from two columns
const TURING_FULL_MODEL_WITH_VECTOR_REPORTS = full_model(
    TuringBackend(),
    IndependentParameters((; mean=Normal(0, 1))),
    MultipleSessionsInfo(
        TuringReportsTwoNumbers(),
        (;
            id=repeat(["S1", "S2", "S3"]; inner=10),
            trial=repeat(1:10, 3),
            first=repeat([0.1, 0.3, -0.2, 0.5, 0.0, 0.4, -0.1, 0.2, 0.6, 0.3], 3),
            second=repeat([0.5, 0.2, 0.0, -0.3, 0.1, 0.2, 0.4, -0.2, 0.3, 0.1], 3),
        );
        observation_columns=:trial,
        action_columns=(:first, :second) => ((first, second) -> [first, second]),
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    ),
)

# A cognitive model which learns the expected payoff of each of three options, a vector, with a learning rate for each
# option, a vector, from the payoffs of every option, a vector, beside the number of the trial, a number. It rates each
# option, a vector from a normal distribution over vectors, and chooses an option with a probability in proportion to
# its expected payoff, a number
struct TuringRatesAndChoosesOptions <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TuringRatesAndChoosesOptions, parameters)
    return (; expected_payoffs=0.5 .* one.(parameters.learning_rates), n_updates=0)
end
function CognitiveModeling.run_timestep(
    model::TuringRatesAndChoosesOptions, context::ModelContext
)
    (; expected_payoffs, n_updates) = context.previous_state
    payoffs = context.input.payoffs
    if !(payoffs isa AbstractNoValue)
        expected_payoffs =
            expected_payoffs .+
            context.parameters.learning_rates .* (payoffs .- expected_payoffs)
        n_updates += 1
    end
    return ModelOutput(
        (;
            ratings=MvNormal(
                expected_payoffs,
                context.parameters.rating_noise^2 .*
                [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0],
            ),
            choice=Categorical(expected_payoffs ./ sum(expected_payoffs)),
        ),
        (; expected_payoffs, n_updates),
    )
end
function CognitiveModeling.attribute_names(model::TuringRatesAndChoosesOptions)
    return ModelAttributes(;
        parameters=(:learning_rates, :rating_noise),
        states=(:expected_payoffs, :n_updates),
        inputs=(:payoffs, :trial),
        outputs=(:ratings, :choice),
    )
end
function CognitiveModeling.parameter_domains(model::TuringRatesAndChoosesOptions)
    return (;
        learning_rates=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        rating_noise=Domain(; lower=0),
    )
end

# The payoffs, ratings and choices of three sessions of different lengths, with the payoffs and the ratings in a column for
# each option, and the same table with the second rating of the fourth row missing, which makes its whole vector missing
const TURING_OPTIONS_TABLE = (;
    id=["S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", "S3"],
    trial=[1, 2, 1, 2, 3, 1, 2, 3, 4],
    payoff_1=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    payoff_2=[0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0],
    payoff_3=[3.0, 3.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
    rating_1=[0.7, 0.6, 0.4, 0.8, 0.9, 0.3, 0.2, 0.7, 0.5],
    rating_2=Union{Missing,Float64}[0.2, 1.1, 1.4, 0.9, 1.5, 1.6, 0.8, 0.6, 1.3],
    rating_3=[2.4, 2.8, 0.3, 1.9, 0.9, 2.5, 2.9, 1.2, 2.2],
    choice=[3, 3, 2, 1, 2, 2, 3, 3, 3],
)
const TURING_OPTIONS_TABLE_WITH_A_MISSING_RATING = (;
    TURING_OPTIONS_TABLE...,
    rating_2=Union{Missing,Float64}[0.2, 1.1, 1.4, missing, 1.5, 1.6, 0.8, 0.6, 1.3],
)
function turing_full_model_of_options(table)
    return full_model(
        TuringBackend(),
        IndependentParameters((;
            learning_rates=filldist(Beta(2, 2), 3), rating_noise=LogNormal(-1, 0.5)
        )),
        MultipleSessionsInfo(
            TuringRatesAndChoosesOptions(),
            table;
            observation_columns=(
                (:payoff_1, :payoff_2, :payoff_3) =>
                    ((first, second, third) -> [first, second, third]) => :payoffs,
                :trial,
            ),
            action_columns=(
                (:rating_1, :rating_2, :rating_3) =>
                    ((first, second, third) -> [first, second, third]) => :ratings,
                :choice,
            ),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        ),
    )
end
const TURING_FULL_MODEL_WITH_VECTORS_IN_EVERY_ATTRIBUTE = turing_full_model_of_options(
    TURING_OPTIONS_TABLE
)
const TURING_FULL_MODEL_WITH_AN_IMPUTED_VECTOR_ACTION = turing_full_model_of_options(
    TURING_OPTIONS_TABLE_WITH_A_MISSING_RATING
)

function log_density_of(model)
    log_density_function = LogDensityFunction(model)
    return x -> LogDensityProblems.logdensity(log_density_function, x)
end

# The log density on the unconstrained values that samplers move in, for a model whose variables are constrained together, such as a vector of probabilities, which finite differences on the values themselves would step out of
function linked_log_density_of(model)
    log_density_function = LogDensityFunction(model, getlogjoint_internal, LinkAll())
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
        name="turing session with three observations",
        f=log_density_of(
            fit_rescorla_wagner(three_observations_session(TURING_THREE_OBSERVATIONS))
        ),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with three observations, of which the third isn't always given",
        f=log_density_of(
            fit_rescorla_wagner(
                three_observations_session(TURING_THREE_OBSERVATIONS_BONUS_NOT_ALWAYS_GIVEN)
            ),
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
        name="turing session with parameter domains",
        f=log_density_of(fit_rescorla_wagner(parameter_domains_session())),
        x=[0.3, 2.0],
        reactant=false,
    ),
    (;
        name="turing session with parameter domains and an environment",
        f=log_density_of(
            fit_with_environment(parameter_domains_with_environment_session())
        ),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
        reactant=false,
    ),
    (;
        name="turing full model with parameter domains",
        f=log_density_of(TURING_FULL_MODEL_WITH_PARAMETER_DOMAINS),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
        reactant=false,
    ),
    (;
        name="turing full model with three sessions and independent priors",
        f=log_density_of(TURING_FULL_MODEL),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
        reactant=false,
    ),
    (;
        name="turing full model with three parameters",
        f=log_density_of(TURING_FULL_MODEL_WITH_THREE_PARAMETERS),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6],
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
    (;
        name="turing full model with a regression with a family distribution",
        f=log_density_of(TURING_FULL_MODEL_WITH_FAMILY_DISTRIBUTION),
        x=[0.2, 0.3, 0.5, 0.4, 0.6, 0.9, 1.3],
        reactant=false,
    ),
    (;
        name="turing full model with a vector parameter for each session, from a Dirichlet prior",
        f=linked_log_density_of(TURING_FULL_MODEL_WITH_DIRICHLET_PRIORS),
        x=[0.3, -0.2, 0.1, 0.5, -0.4, 0.2],
        reactant=false,
    ),
    (;
        name="turing full model with a vector parameter for each session, from a hierarchical Dirichlet with two groups",
        f=linked_log_density_of(TURING_FULL_MODEL_WITH_HIERARCHICAL_DIRICHLET),
        x=[0.4, 0.1, -0.3, -0.2, 0.3, 0.6, 0.3, -0.2, 0.1, 0.5, -0.4, 0.2],
        reactant=false,
    ),
    (;
        name="turing full model with an output whose values are NamedTuples, made from two columns",
        f=log_density_of(TURING_FULL_MODEL_WITH_RESPONSES),
        x=[0.6, 0.7, 0.3, 2.0, 3.0, 1.5],
        reactant=false,
    ),
    (;
        name="turing full model with an output whose values are vectors, made from two columns",
        f=log_density_of(TURING_FULL_MODEL_WITH_VECTOR_REPORTS),
        x=[0.2, -0.1, 0.4],
        reactant=false,
    ),
    # The learning rates of the three sessions, a column each, the rating noises, and then the imputed ratings. The learning
    # rates stay below 1 where the gradient is prepared, at x .+ 0.25
    (;
        name="turing full model with observations, actions, states and parameters that are vectors",
        f=log_density_of(TURING_FULL_MODEL_WITH_VECTORS_IN_EVERY_ATTRIBUTE),
        x=[0.2, 0.6, 0.7, 0.5, 0.3, 0.7, 0.4, 0.45, 0.1, 0.3, 0.5, 0.4],
        reactant=false,
    ),
    (;
        name="turing full model with a missing action that is a vector, imputed",
        f=log_density_of(TURING_FULL_MODEL_WITH_AN_IMPUTED_VECTOR_ACTION),
        x=[0.2, 0.6, 0.7, 0.5, 0.3, 0.7, 0.4, 0.45, 0.1, 0.3, 0.5, 0.4, 0.8, 0.9, 1.9],
        reactant=false,
    ),
    (;
        name="turing full model with parameters from regressions",
        f=log_density_of(TURING_FULL_MODEL_WITH_REGRESSION),
        x=[
            0.2,
            0.3,
            -0.4,
            0.1,
            0.8,
            0.3,
            -0.2,
            0.5,
            0.4,
            -0.6,
            0.6,
            1.1,
            0.1,
            -0.3,
            0.2,
            0.5,
            0.7,
            0.2,
            -0.1,
        ],
        reactant=false,
    ),
    (;
        name="turing full model where only the first and the third session impute missing actions",
        f=log_density_of(TURING_FULL_MODEL_WITH_SOME_SESSIONS_IMPUTING_ACTIONS),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.5, 0.4, 0.6, 0.3],
        reactant=false,
    ),
    (;
        name="turing full model where only the first and the third session run their environment",
        f=log_density_of(TURING_FULL_MODEL_WITH_SOME_SESSIONS_RUNNING_THE_ENVIRONMENT),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.6, 0.2, 0.8, -0.1, 0.5, 0.3, 0.9, 0.1],
        reactant=false,
    ),
    (;
        name="turing full model of sessions of different lengths, where only the second and the third impute missing actions",
        f=log_density_of(TURING_FULL_MODEL_WITH_SESSIONS_OF_DIFFERENT_LENGTHS),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.5, 0.4, 0.6, 0.3],
        reactant=false,
    ),
]
