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
using Turing: Bijectors

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
# `Gibbs`, each kind of parameter gets a sampler of its own. Here every session has its own
# parameters, with `sessions_info` made from a table as on the other pages:

parameters_model = IndependentParameters((;
    learning_rate=Beta(2, 2), uses_rewards=Bernoulli(0.5)
))
model = turing_full_model(parameters_model, sessions_info)

sampler = Gibbs(
    @varname(learning_rate.session) => NUTS(), @varname(uses_rewards.session) => MH()
)
chain = sample(model, sampler, 1000)

# The share of the samples in which a session's `uses_rewards` is `true` is the posterior
# probability that the session learned from its rewards.

# ## A regression on a discrete parameter
#
# A [`RegressionParameter`](@ref) gives a number for each session, which a discrete
# parameter can't be. With a `family_distribution`, the value of each session is sampled
# from a distribution around that number instead, as a variable of its own. Here the
# regression gives the probability that a session uses its rewards, and a session is more
# or less likely to use them with age:

parameters_model = HierarchicalRegressionParameters((;
    learning_rate=RegressionParameter(Intercept(Normal(0, 1)); transformation=logistic),
    uses_rewards=RegressionParameter(
        Intercept(Normal(0, 1)),
        ContinuousPredictor(:age, Normal(0, 0.5));
        transformation=logistic,
        family_distribution=Bernoulli,
    ),
))
model = turing_full_model(parameters_model, sessions_info)

# The coefficients are continuous and the values of the sessions discrete, so each kind
# gets its sampler again:

sampler = Gibbs(
    (@varname(learning_rate.coef), @varname(uses_rewards.coef)) => NUTS(),
    @varname(uses_rewards.session) => MH(),
)
chain = sample(model, sampler, 1000)

# When the model is made, the values that the family distribution can give are checked
# against the domain of the parameter, so a `Poisson` for a parameter that is `true` or
# `false` is an error before the fit.

# ## Ordered levels and counts
#
# A parameter with ordered levels, such as a number of planning steps from one to four,
# needs one regression and a cutpoint between each pair of neighboring levels, as in an
# ordinal regression. The cutpoints are other parameters of the family distribution, with a
# prior of their own, and are the same for every session. The regression has no intercept,
# since the cutpoints take its place:

planning_steps = RegressionParameter(
    ContinuousPredictor(:age, Normal(0, 1));
    family_distribution=OrderedLogistic,
    family_distribution_priors=(; cutpoints=Bijectors.ordered(filldist(Normal(0, 2), 3))),
)

# A count works in the same way: a number of items out of seven as
# `value -> Binomial(7, value)` with `transformation = logistic`, and a count without an
# upper bound as `Poisson` with `transformation = exp`.

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
