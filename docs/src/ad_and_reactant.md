# Automatic differentiation and Reactant

Simulations can be differentiated with ForwardDiff, ReverseDiff, Enzyme and Mooncake, and
compiled with Reactant. Every commit is tested against every form of each backend, on the
cases in `test/ad_and_reactant/test_cases.jl`: simulations with an environment, with sampled
actions and rewards, with timestep sizes and with gradients of the sizes themselves, without
states, with a state changed in place, with a selection of what is saved, and agents
simulated with an environment or on single observations, with and without a history. Every
gradient is compared with finite differences. Zygote isn't supported, because the package
uses for loops.

| Backend | Forms tested |
|---|---|
| ForwardDiff | default, chunk size 1 |
| ReverseDiff | default, compiled tape |
| Enzyme | forward, reverse, and both with runtime activity |
| Mooncake | forward, reverse |
| Reactant | compiled value, and compiled Enzyme reverse gradient, on the cases without a `Xoshiro` |

## Known failures

These are bugs in the backends, which the package doesn't work around. Their tests are
marked broken in `test/ad_and_reactant/ad/runtests.jl`, so that the suite passes while the
bug is there, and a fix shows up as an unexpected pass. Last checked with Julia 1.13.3 and
Enzyme 0.13.216.

| Backend | When | What happens | Status |
|---|---|---|---|
| Enzyme reverse, with and without runtime activity | `simulate` when the agent starts, actions and rewards are sampled from one seeded `Xoshiro`, and the number of timesteps is a constant | wrong gradient, no error | reproduced without the package, to be reported |
| Enzyme forward with runtime activity | `simulate!` of an agent with an environment, when the model's state starts as a constant such as `(; expected_reward = 0.5)` | wrong gradient, no error | reproduced without the package, to be reported |
| Enzyme reverse with runtime activity | the same | throws a `MethodError` from `create_shadow_ret` | reproduced without the package, to be reported |

A state which depends on the parameters from the start isn't affected, and neither is
`simulate!` on single observations. The other forms of Enzyme, and the other backends, are
right on all of these cases.

## Pitfalls

These aren't failures of the package's own code, but a model can run into them.

- **ReverseDiff and `deepcopy`:** `deepcopy` of anything that holds a tracked number cuts it
  off the tape, so both forms of ReverseDiff give zero gradients, with no error. A method of
  `prepare_state_for_saving` which uses `deepcopy` has this problem, so copy only what
  changes, such as `copy(state.values)`.
- **Enzyme reverse and `SVector` states:** updating an `SVector` in a state with `setindex`
  gives wrong gradients for some sizes, with no error. A `Vector` changed in place or by
  broadcasting is right.
- **Compiled ReverseDiff tapes:** a tape records one path through the code, so a model with
  a branch on a value gives wrong gradients at points where the other branch is taken, with
  no error. The tests prepare the tape at a different point to catch this.
- **Reactant:** the constructors of Distributions.jl don't accept traced numbers, so a model
  built on them can't be compiled, and sets `reactant = false` in its test case. An `if` on a
  traced value throws, where `ifelse` works. A `Xoshiro` can't run in a compiled function,
  and the default generator repeats its numbers on every run, so pass a
  `Reactant.ReactantRNG()` as a positional argument. Compile times grow with the number of
  timesteps, because the loop is unrolled, so a few hundred timesteps is the practical limit.
