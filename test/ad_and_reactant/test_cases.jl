# What the AD and Reactant tests check, with one entry per premade model. In each entry,
# f takes a vector x and returns a number, such as a model's log likelihood for example data.
const TEST_CASES = [
    # Placeholder until the first premade model exists
    (; name = "Sine", f = x -> sum(sin, x), x = [0.3, 2.0]),
]
