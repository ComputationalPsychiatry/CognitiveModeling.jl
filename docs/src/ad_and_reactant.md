# Automatic differentiation and Reactant

Simulations can be differentiated with ForwardDiff, ReverseDiff, Enzyme and Mooncake, and
compiled with Reactant. Every commit is tested against every form of each backend, on the
cases in `test/ad_and_reactant/test_cases.jl`: simulations with an environment, with sampled
actions and rewards, with timestep sizes and with gradients of the sizes themselves, without
states, with a state changed in place, with a selection of what is saved, with maps of names
between the model and the environment, with an environment which ends the simulation, with
a reward that isn't given on every timestep, and on single observations, and agents
simulated with an environment or on single observations, with and without a history, and
with a state saved with `deepcopy`. Code in the package's extensions is differentiated in
suites of its own, each with its own environment, beside the other tests of the extension.
The one for Turing.jl, in `test/extensions/turing/ad`, differentiates the log density of
Turing.jl models of a session, in its `turing_cases.jl`: with the agent or the environment
starting, with imputed actions, with an environment filling in observations, with an
environment which doesn't run because nothing in the data is missing, with observations
imputed from an environment's distributions, with a reward that isn't given on every
timestep, and with a model which samples within each timestep, and of the full model of
several sessions, also with an environment parameter shared by the sessions. That keeps the
package's own code tested without Turing.jl, since the backends work without it and with
other probabilistic programming languages. Every gradient is compared with finite
differences. Zygote isn't supported, because the package uses for loops.

| Backend | Forms tested |
|---|---|
| ForwardDiff | default, chunk size 1 |
| ReverseDiff | default, compiled tape |
| Enzyme | forward, reverse, and both with runtime activity |
| Mooncake | forward, reverse |
| Reactant | compiled value, and compiled Enzyme reverse gradient, on the cases without a `Xoshiro` |

## Known failures

These are bugs in the backends, which the package doesn't work around. Their tests are
marked broken in `runtests.jl` of `test/ad_and_reactant/ad` and of
`test/extensions/turing/ad`, so that the suites pass while the bug is there,
and a fix shows up as an unexpected pass. Last checked with Julia 1.13.3, ReverseDiff
1.18.9, Enzyme 0.13.219, Mooncake 0.5.70, Turing 0.49.0 and DynamicPPL 0.42.14.

| Backend | When | What happens | Status |
|---|---|---|---|
| Enzyme reverse, with and without runtime activity | `simulate` when the agent starts, actions and rewards are sampled from one seeded `Xoshiro`, and the number of timesteps is a constant | wrong gradient, no error | reported as Enzyme.jl issue 3912 |
| Enzyme forward with runtime activity | `simulate!` of an agent with an environment, when the model's state starts as a constant such as `(; expected_reward = 0.5)` | wrong gradient, no error | reported as Enzyme.jl issue 3911 |
| Enzyme reverse with runtime activity | the same | throws a `MethodError` from `create_shadow_ret` | reported as Enzyme.jl issue 3911 |
| Enzyme reverse, with and without runtime activity | numbers copied into a vector whose element type is a `Union`: in `simulate` when an observation is `NoValue` on some timesteps and a number on the others, and the observations are saved, and when a vector of timestep sizes is returned after `BeforeFirstTimestep()` for timestep 0, which it is by default | throws an `EnzymeNoTypeError` | reported as Enzyme.jl issue 4032; with a single timestep size, or `save_initial_values = false`, the sizes are right |
| Enzyme forward and reverse without runtime activity | the log density of a Turing.jl model of a session with an imputed action, an environment, or a model which samples within a timestep, and of the full model | throws an `EnzymeRuntimeActivityError` | a limitation Turing.jl documents: use runtime activity. A plain session passes |
| Enzyme reverse, with and without runtime activity, and Enzyme forward when the agent starts | a session whose observations are `NoValue` on some timesteps, so that a row of the data is one of two `NamedTuple`s | throws an `IllegalTypeAnalysisException` | reported as Enzyme.jl issue 4031; an action that is `MissingValue` on some timesteps is fine |

A state which depends on the parameters from the start isn't affected, and neither is
`simulate!` on single observations. A simulation where an observation isn't made on every
timestep is right on every backend when the observations aren't saved, and on Enzyme's
forward mode when they are. The other forms of
Enzyme, and the other backends, are right on all of these cases. ForwardDiff, ReverseDiff
and Mooncake are right on every Turing.jl case, and so are both forms of Enzyme with runtime
activity on a session where every column of the data has one type.

## Pitfalls

These aren't failures of the package's own code, but a model can run into them.

- **Enzyme reverse with runtime activity and a softmax in one line:** a softmax written as
  `exp.(v .- maximum(v)) ./ sum(exp.(v .- maximum(v)))` gives wrong gradients, with no
  error, when it runs in a loop on values that differ between the timesteps. The `softmax`
  of LogExpFunctions.jl is right, and so is any softmax which keeps its exponentials in a
  variable and divides them by their sum. It is reported as Enzyme.jl issue 4030.
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
  is why [`turing_catch_invalid_parameters`](@ref) is only used when asked for, and
  a compiled ReverseDiff tape doesn't see the error it catches either, so the rejected
  parameters get a finite log density with no error.
- **Catching invalid parameters:** [`turing_catch_invalid_parameters`](@ref), and a full
  model with `catch_invalid_parameters = true`, work with ForwardDiff and with Enzyme.
  Anything that uses what the model returns after a rejection, which is `nothing`, can't be
  differentiated with Enzyme, so return or stop at once when it is `nothing`. Mooncake can't
  differentiate a model with a `try` block at all, and a compiled ReverseDiff tape doesn't
  see the error, so the rejected parameters get a finite log density with no warning.
- **Types known when a session is compiled:** a session costs about 10 ns per timestep to
  evaluate, with or without latent variables, as long as every type is known when it is
  compiled. Earlier versions of the package built a `NamedTuple` from names only known at run
  time, which made every timestep about a hundred times slower, and that cost was taken for
  one of DynamicPPL.jl's. The rows of the data get the element types of their columns for the
  same reason, so a column which holds two types costs nothing extra. A model's own code can
  lose the same factor in the same way, so check a slow model with JET's `@report_opt`.
- **Reactant:** the constructors of Distributions.jl don't accept traced numbers, so a model
  built on them can't be compiled, and sets `reactant = false` in its test case. An `if` on a
  traced value throws, where `ifelse` works. A `Xoshiro` can't run in a compiled function,
  and the default generator repeats its numbers on every run, so pass a
  `Reactant.ReactantRNG()` as a positional argument. Compile times grow with the number of
  timesteps, because the loop is unrolled, so a few hundred timesteps is the practical limit.
