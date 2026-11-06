# # Compiling simulations with Reactant.jl
#
# Placeholder tutorial. It will show how to compile a simulation with Reactant.jl, and how to
# take its gradient with Enzyme.jl. Until then, this is what to know before trying.
#
# Short simulations can be compiled with Reactant.jl, if the action distributions work with
# traced numbers, which those from Distributions.jl don't. Compile a function that takes a
# `Reactant.ReactantRNG()` as a positional argument and passes it on as `rng`, because `@jit`
# doesn't trace keyword arguments, and the default generator repeats its numbers on every
# run. Compile times grow quickly with `n_steps`, so this is for a few hundred timesteps at
# most.
#
# The page [Automatic differentiation and Reactant](@ref) lists what is tested, and the
# pitfalls a model can run into.

using CognitiveModeling
