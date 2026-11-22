# # Discrete parameters
#
# *This page is a sketch of a tutorial. Its samplers take very few samples, and the sections
# that are marked as planned describe what isn't in the package yet.*
#
# Some parameters of a cognitive model aren't continuous: whether a participant uses a
# strategy at all, which of several strategies they use, or how many items they can hold in
# memory. This page shows how a model says that a parameter is discrete, and how such a
# parameter is fitted.

using CognitiveModeling
using LogExpFunctions: logistic
using Random: Xoshiro
using Turing
using Turing: Bijectors, DynamicPPL

# ## A model with a discrete parameter
#
# This model learns the expected reward of an option only when `uses_rewards` is `true`, and
# otherwise keeps expecting a half. The domain of the parameter says that it is discrete, by
# its type:

struct LearnsOrNot <: AbstractCognitiveModel end

CognitiveModeling.initialize_state(model::LearnsOrNot, parameters) = (; expected_reward=0.5)

function CognitiveModeling.timestep_model(model::LearnsOrNot, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = if Bool(context.parameters.uses_rewards)
        V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    else
        V₋₁
    end
    return ModelOutput((; choice=BernoulliLogit(5 * (Vₜ - 0.5))), (; expected_reward=Vₜ))
end

function CognitiveModeling.attribute_names(model::LearnsOrNot)
    return ModelAttributes(;
        parameters=(:learning_rate, :uses_rewards),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

function CognitiveModeling.parameter_domains(model::LearnsOrNot)
    return (; learning_rate=Domain(; lower=0, upper=1), uses_rewards=Domain(; type=Bool))
end

# [`CognitiveModeling.discrete_parameter_names`](@ref) and
# [`CognitiveModeling.continuous_parameter_names`](@ref) then give the parameters of each
# kind.

# ## The data
#
# Four participants are simulated for 40 timesteps, each with rewards that come with a
# probability of 0.7, as the rewards of a [`FixedObservationsEnvironment`](@ref). The first
# two use their rewards, and the other two don't. Their ages go into the table for the
# regression further down:

ages = [20.0, 35.0, 50.0, 65.0]
uses_rewards = [true, true, false, false]
simulations = map(1:4) do participant
    rewards = Float64.(rand(Xoshiro(participant), 40) .< 0.7)
    return simulate_session(
        LearnsOrNot(),
        FixedObservationsEnvironment((; reward=rewards));
        who_starts=EnvironmentStarts(),
        n_steps=40,
        cognitive_parameters=(; learning_rate=0.3, uses_rewards=uses_rewards[participant]),
        environment_parameters=(;),
        action_map=NameMap(),
        cognitive_save_history=SaveHistory(;
            inputs=:reward, outputs=:choice, save_initial_timestep_values=false
        ),
        rng=Xoshiro(participant),
    )
end
table = (;
    participant=repeat(["P1", "P2", "P3", "P4"]; inner=40),
    age=repeat(ages; inner=40),
    reward=[observation.reward for s in simulations for observation in s.observations],
    choice=[action.choice for s in simulations for action in s.actions],
)
sessions_info = MultipleSessionsInfo(
    LearnsOrNot(),
    table;
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
)

# ## Sampling a discrete parameter
#
# A sampler which uses gradients, such as NUTS, can't sample a discrete parameter. With
# `Gibbs`, each kind of variable gets a sampler of its own. Here every session has its own
# parameters:

parameters_model = IndependentParameters((;
    learning_rate=Beta(2, 2), uses_rewards=Bernoulli(0.5)
))
model = full_model(TuringBackend(), parameters_model, sessions_info)

# The names of the variables of each kind can be read from the model. DynamicPPL.jl, which
# Turing.jl is built on, gives the distribution that every variable is sampled from, and a
# variable is discrete when its distribution is:

priors = DynamicPPL.extract_priors(model)
names = collect(keys(priors))
#The names of the discrete variables, such as uses_rewards.session
discrete = filter(name -> priors[name] isa DiscreteDistribution, names)
#The names of the continuous variables, such as learning_rate.session
continuous = filter(name -> !(priors[name] isa DiscreteDistribution), names)

# This finds every discrete variable of the model, also a choice that wasn't recorded and
# is imputed, which is a variable without being a parameter. Each kind then gets its
# sampler:

sampler = Gibbs(continuous => NUTS(), discrete => MH())
chain = sample(Xoshiro(1), model, sampler, 20; nadapts=10, discard_initial=10)

# Inside a `Gibbs` sampler, NUTS only adapts its step size when `sample` is given `nadapts`,
# and the samples of those steps are kept unless `discard_initial` is given too. Without
# `nadapts`, it may never move, see
# [Turing.jl's issue 2874](https://github.com/TuringLang/Turing.jl/issues/2874). `Gibbs`
# also takes no sampler without variables, so a model without discrete variables is sampled
# with NUTS alone.
#
# The share of the samples in which a session's `uses_rewards` is `true` is the posterior
# probability that the session learned from its rewards.

# ## A regression on a discrete parameter
#
# A [`RegressionParameter`](@ref) gives a number for each session, which a discrete
# parameter can't be. With a `family`, the value of each session is sampled from a
# distribution instead, as a variable of its own, and the regression gives a parameter of
# that distribution. Here it gives the log odds that a session uses its rewards, which
# rise or fall with age:

parameters_model = HierarchicalRegressionParameters((;
    learning_rate=RegressionParameter(Intercept(Normal(0, 1)); transformation=logistic),
    uses_rewards=RegressionParameter(
        Intercept(Normal(0, 1)),
        ContinuousPredictor(:age, Normal(0, 0.5));
        family=BernoulliLogit,
    ),
))
model = full_model(TuringBackend(), parameters_model, sessions_info)

# A distribution which takes the sum of the regression as it is, such as `BernoulliLogit`,
# is the safe choice. With `transformation = logistic` and `Bernoulli`, a sampler which
# uses ForwardDiff.jl for its gradients stops with an error as soon as it tries a sum that
# is high enough for the probability to be exactly 1.

# The coefficients are continuous and the values of the sessions discrete, so each kind
# gets its sampler again:

priors = DynamicPPL.extract_priors(model)
names = collect(keys(priors))
#The names of the discrete variables, such as uses_rewards.session
discrete = filter(name -> priors[name] isa DiscreteDistribution, names)
#The names of the continuous variables, such as learning_rate.coef and uses_rewards.coef
continuous = filter(name -> !(priors[name] isa DiscreteDistribution), names)
sampler = Gibbs(continuous => NUTS(), discrete => MH())
chain = sample(Xoshiro(1), model, sampler, 20; nadapts=10, discard_initial=10)

# ## Ordered levels and counts
#
# A parameter with ordered levels, such as a number of planning steps from one to four,
# needs one regression and a cutpoint between each pair of neighboring levels, as in an
# ordinal regression. The cutpoints are other parameters of the family's distribution, with
# a prior of their own, and are the same for every session. A [`RegressionFamily`](@ref) has
# the parameters of the distribution in the order in which it takes them, with
# [`RegressionOutcome`](@ref) for the one that the regression gives and a prior for each of
# the others. The regression has no intercept, since the cutpoints take its place:

planning_steps = RegressionParameter(
    ContinuousPredictor(:age, Normal(0, 1));
    family=RegressionFamily(
        (;
            location=RegressionOutcome(),
            cutpoints=Bijectors.ordered(filldist(Normal(0, 2), 3)),
        );
        distribution=OrderedLogistic,
    ),
)

# A count needs no other parameters: a number of items out of seven is
# `family = logit -> BinomialLogit(7, logit)`, and a count without an upper bound is
# `family = LogPoisson`.

# ## Unordered categories (planned)
#
# A parameter with more than two unordered categories, such as which of three strategies a
# participant uses, needs a regression for each category but one, with a softmax across
# them. This may come through BayesianRegressionModels.jl.

# ## Without sampling the discrete values (planned)
#
# A discrete parameter with few values can be summed out: the session is run once for each
# value, and the results are weighted by the probabilities of the values. The model then has
# no discrete variables, so NUTS can fit all of it, and the probability of each value for a
# session is computed afterwards.
