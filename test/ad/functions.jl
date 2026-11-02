# The functions to check. runtests.jl differentiates each one with every backend and
# compiles it with Reactant. To check new code, add an entry to AD_TEST_FUNCTIONS, where
# f takes a vector x and returns a number.

# Placeholder until the package has real models
function rescorla_wagner_loglikelihood(parameters, observations, actions)
    learning_rate, inverse_temperature = parameters
    expected_value = zero(eltype(parameters))
    loglikelihood = zero(eltype(parameters))
    for trial in eachindex(observations)
        action_probability = 1 / (1 + exp(-inverse_temperature * expected_value))
        loglikelihood += actions[trial] * log(action_probability)
        loglikelihood += (1 - actions[trial]) * log(1 - action_probability)
        expected_value += learning_rate * (observations[trial] - expected_value)
    end
    return loglikelihood
end

const OBSERVATIONS = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
const ACTIONS = [1, 0, 1, 1, 1, 0, 1, 1]

const AD_TEST_FUNCTIONS = [
    (;
        name = "Rescorla-Wagner log likelihood",
        f = parameters -> rescorla_wagner_loglikelihood(parameters, OBSERVATIONS, ACTIONS),
        x = [0.3, 2.0],
    ),
]
