# Automatic differentiation and Reactant

Simulations can be differentiated with ForwardDiff, ReverseDiff, Enzyme and Mooncake, and
compiled with Reactant. Every commit is tested against every form of each backend, on the
cases in `test/ad_and_reactant/test_cases.jl`: simulations with an environment, with sampled
actions and rewards, with timestep sizes and with gradients of the sizes themselves, without
states, with a state changed in place, with a selection of what is saved, and agents
simulated with an environment or on single observations, with and without a history. The
AD suite also differentiates the log density of Turing.jl models of a session, in
`test/ad_and_reactant/turing_cases.jl`: with the agent or the environment starting, with
imputed actions, with an environment filling in observations, and with a model which samples
within each timestep. Every gradient is compared with finite differences. Zygote isn't supported, because the package
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
bug is there, and a fix shows up as an unexpected pass. Last checked with Julia 1.13.3,
Enzyme 0.13.218, Mooncake 0.5.67, Turing 0.49.0 and DynamicPPL 0.42.14.

| Backend | When | What happens | Status |
|---|---|---|---|
| Enzyme reverse, with and without runtime activity | `simulate` when the agent starts, actions and rewards are sampled from one seeded `Xoshiro`, and the number of timesteps is a constant | wrong gradient, no error | reproduced without the package, to be reported |
| Enzyme forward with runtime activity | `simulate!` of an agent with an environment, when the model's state starts as a constant such as `(; expected_reward = 0.5)` | wrong gradient, no error | reproduced without the package, to be reported |
| Enzyme reverse with runtime activity | the same | throws a `MethodError` from `create_shadow_ret` | reproduced without the package, to be reported |
| Enzyme forward and reverse without runtime activity | the log density of any Turing.jl model of a session | throws an `EnzymeRuntimeActivityError` | a limitation Turing.jl documents: use runtime activity |
| Enzyme reverse with runtime activity | a session with imputed actions, or with an environment filling in missing observations | throws an `EnzymeNonScalarReturnException` from Enzyme's rule for building a struct of unknown type | to be reported |
| Enzyme forward with runtime activity | a session with an environment filling in missing observations | wrong gradient, no error | to be reported |

A state which depends on the parameters from the start isn't affected, and neither is
`simulate!` on single observations. The other forms of Enzyme, and the other backends, are
right on all of these cases. ForwardDiff, ReverseDiff and Mooncake are right on every
Turing.jl case, and so are both forms of Enzyme with runtime activity on a session without
missing data.

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
- **Turing.jl models and Enzyme:** Enzyme needs runtime activity to differentiate a Turing.jl
  model, as Turing's own documentation says; without it, it throws an
  `EnzymeRuntimeActivityError`.
- **Turing.jl models and Mooncake:** Mooncake can't differentiate `logjoint(model, values)`
  when the model has latent variables under an indexed prefix, such as the imputed actions of
  a session or a model which samples within a timestep, because building the variables runs
  DynamicPPL's warning about growable arrays, and a `@warn` holds a `try`. Samplers go through
  a `LogDensityFunction`, where the variables exist already, and that is fine. The same `try`
  is why `catch_invalid_parameters` of [`turing_single_session`](@ref) is off by default, and
  a compiled ReverseDiff tape doesn't see the error it catches either, so the rejected
  parameters get a finite log density with no error.
- **Latent variables in a session:** in DynamicPPL 0.42.14, one latent variable anywhere
  inside a session, imputed or from a model which samples within a timestep, makes every
  timestep of the session about 450 ns slower to evaluate, and a Mooncake gradient about
  10 µs slower per timestep, whatever the number of latents. A session without them costs
  about 5 ns per timestep. This is DynamicPPL's cost of looking variables up under a prefix
  that holds some, not the package's, and is to be raised with the Turing developers.
- **Reactant:** the constructors of Distributions.jl don't accept traced numbers, so a model
  built on them can't be compiled, and sets `reactant = false` in its test case. An `if` on a
  traced value throws, where `ifelse` works. A `Xoshiro` can't run in a compiled function,
  and the default generator repeats its numbers on every run, so pass a
  `Reactant.ReactantRNG()` as a positional argument. Compile times grow with the number of
  timesteps, because the loop is unrolled, so a few hundred timesteps is the practical limit.
