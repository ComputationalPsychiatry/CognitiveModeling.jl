#What the AD and Reactant tests check, with one entry per premade model. In each entry,
#f takes a vector x and returns a number, such as a model's log likelihood for example data.
#Set reactant = false for models whose action distributions can't run on Reactant, such as
#Distributions.jl ones.
#Placeholder until the first premade model exists
const TEST_CASES = [(; name="Sine", f=x -> sum(sin, x), x=[0.3, 2.0], reactant=true)]
