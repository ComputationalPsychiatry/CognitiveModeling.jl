# # Discrete parameters
#
# *This page is a sketch of a tutorial. Its code isn't run when the documentation is built,
# and the sections that are marked as planned describe what isn't in the package yet.*
#
# Some parameters of a cognitive model aren't continuous: whether a participant uses a
# strategy at all, which of several strategies they use, or how many items they can hold in
# memory. This page shows how a model says that a parameter is discrete, and how such a
# parameter is fitted.

using CognitiveModeling
using LogExpFunctions: logistic
using Turing
using Turing: Bijectors, DynamicPPL

# ## A model with a discrete parameter
#
# This model learns the expected reward of an option only when `uses_rewards` is `true`, and
# otherwise keeps expecting a half. The domain of the parameter says that it is discrete, by
# its type:

struct LearnsOrNot <: AbstractCognitiveModel end

CognitiveModeling.initialize_state(model::LearnsOrNot, parameters) = (; expected_reward=0.5)

function CognitiveModeling.run_timestep(model::LearnsOrNot, context::ModelContext)
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
    return (;
        learning_rate=ParameterDomain(; lower=0, upper=1),
        uses_rewards=ParameterDomain(; type=Bool),
    )
end

# [`CognitiveModeling.discrete_parameter_names`](@ref) and
# [`CognitiveModeling.continuous_parameter_names`](@ref) then give the parameters of each
# kind.

# ## Sampling a discrete parameter
#
# A sampler which uses gradients, such as NUTS, can't sample a discrete parameter. With
# `Gibbs`, each kind of variable gets a sampler of its own. Here every session has its own
# parameters, with `sessions_info` made from a table as on the other pages:

parameters_model = IndependentParameters((;
    learning_rate=Beta(2, 2), uses_rewards=Bernoulli(0.5)
))
model = full_model(TuringBackend(), parameters_model, sessions_info)

# The names of the variables of each kind can be read from the model. DynamicPPL.jl, which
# Turing.jl is built on, gives the distribution that every variable is sampled from, and a
# variable is discrete when its distribution is. A function of one's own keeps the lines
# together:

function discrete_and_continuous_variable_names(model)
    distributions = DynamicPPL.extract_priors(model)
    variable_names = collect(keys(distributions))
    is_discrete = map(name -> distributions[name] isa DiscreteDistribution, variable_names)
    return (;
        discrete=variable_names[is_discrete], continuous=variable_names[.!is_discrete]
    )
end

variable_names = discrete_and_continuous_variable_names(model)

# This finds every discrete variable of the model, also a choice that wasn't recorded and
# is imputed, which is a variable without being a parameter. Each kind then gets its
# sampler:

sampler = Gibbs(variable_names.continuous => NUTS(), variable_names.discrete => MH())
chain = sample(model, sampler, 2000; nadapts=1000, discard_initial=1000)

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

variable_names = discrete_and_continuous_variable_names(model)
sampler = Gibbs(variable_names.continuous => NUTS(), variable_names.discrete => MH())
chain = sample(model, sampler, 2000; nadapts=1000, discard_initial=1000)

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
