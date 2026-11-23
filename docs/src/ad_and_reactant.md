# Automatic differentiation and Reactant

Simulations can be differentiated with ForwardDiff, ReverseDiff, Enzyme and Mooncake, and
compiled with Reactant. Every commit is tested against every form of each backend, on the
simulation cases that the test suites share, in `test/test_cases/simulation_cases.jl`: each
example model of `test/example_models` simulated with its environment, with either starting,
the baseline model with each setting of a simulation, such as timestep sizes and gradients
of the sizes themselves, a selection of what is saved, an environment of fixed observations,
given previous states, actions and observations, and a stopping criterion that isn't
checked, simulations that end before their last timestep, at a timestep that depends on the
parameters, by a stopping criterion or by parameters that turn invalid, agents instantiated
and simulated with `simulate_session!`, also twice and after `reset!`, and models simulated
on single observations with `simulate_timestep` and `simulate_timestep!`. The models give
the means of their distributions, so that a simulation is deterministic and the gradient
goes all the way round its loop, and a few cases also sample from a seeded `Xoshiro`. The
gradient is that of the sum of every real number that the simulation saved, and for a
simulation that ends early, of those of the timesteps it kept, since Reactant masks it
instead of stopping it.
Code in the package's extensions is differentiated in suites of its own, each with its own
environment, in `test/ad_and_reactant`. The one for Turing.jl, in
`test/ad_and_reactant/turing`, differentiates the log density of the inference cases that
the test suites share, in `test/test_cases/inference_cases.jl`, the way a sampler does:
through a `LogDensityFunction`, on the unconstrained values of every variable. The cases are
those whose variables are all continuous: a session of each example model with its
environment, also with the agent starting where that changes what is differentiated, a
session with an imputed action or observation of each continuous kind, several sessions in
one table, of different lengths and of which only some include their environment, and the
full model with each parameter model and its features, also with imputed values. The
parameters of a session are variables with flat priors, so that the gradient is that of the
session's own log density. That keeps the package's own code tested without Turing.jl,
since the backends work without it and with other probabilistic programming languages. Every
gradient is compared with finite differences. Zygote isn't supported, because the package
uses for loops.

The Turing.jl models can't be compiled with Reactant, so the functions that the model of a
session calls within an inference backend are compiled in sessions that are run by hand, in
`test/ad_and_reactant/session_by_hand.jl`: the log density is added up directly, and
everything else is done by the package's functions, in the steps of the Turing.jl model of a
session. They run with the agent or the environment starting, with a reward that isn't
given on every timestep, with imputed actions, with an environment whose observations are
imputed, which doesn't run, or which scores every observation, with maps of names, with
saved histories, with a size for each timestep, with a session that ends when a stopping
criterion is met, with parameters that turn invalid and a criterion met while the data
continues, which give a log density of `-Inf`, and with several sessions. A session whose
flags Reactant traces is masked instead of stopped, as a simulation is, and a stand-in for a
traced flag masks one in plain Julia too, so that the compiled session has a reference.

| Backend | Forms tested |
|---|---|
| ForwardDiff | default, chunk size 1 |
| ReverseDiff | default, compiled tape |
| Enzyme | forward, reverse, and both with runtime activity |
| Mooncake | forward, reverse |
| Reactant | compiled value, and compiled Enzyme reverse gradient, on the cases without a `Xoshiro` and on the sessions run by hand |

## Known failures

These are bugs and limitations of the backends, which the package doesn't work around. Their
tests are marked broken in `runtests.jl` of `test/ad_and_reactant/ad` and of
`test/ad_and_reactant/turing`, so that the suites pass while the bug is there,
and a fix shows up as an unexpected pass. Those that kill the process aren't run, so a fix of
theirs must be checked by hand. Last checked with Julia 1.13.4, ReverseDiff
1.18.11, Enzyme 0.13.221, Mooncake 0.5.72, Turing 0.49.0 and DynamicPPL 0.42.14.

| Backend | When | What happens | Status |
|---|---|---|---|
| Enzyme reverse, with and without runtime activity | `simulate_session` when the agent starts, actions and rewards are sampled from one seeded `Xoshiro`, and the number of timesteps is a constant | wrong gradient, no error | reported as Enzyme.jl issue 3912 |
| Enzyme forward with runtime activity | `simulate_session!` of an agent with an environment, when the model's state starts as a constant such as `(; expected_reward = 0.5)` | wrong gradient, no error | reported as Enzyme.jl issue 3911 |
| Enzyme reverse with runtime activity | the same | throws a `MethodError` from `create_shadow_ret` | reported as Enzyme.jl issue 3911 |
| Enzyme forward without runtime activity | `simulate_session` with a [`FixedObservationsEnvironment`](@ref) whose observations have two types, such as a reward that is `NoValue` on some timesteps | throws an `EnzymeRuntimeActivityError` | use runtime activity, which is right. Observations of one type pass |
| Enzyme reverse, with and without runtime activity | `simulate_session` where an output of either model, an action or an observation, is `NoValue` on some timesteps, so that it has two types in the loop over the timesteps | throws a `TypeError`, that a `RefValue` was given where a value was expected, or an `EnzymeNoTypeError` | reported as Enzyme.jl issues 4044 and 4032; forward mode is right, but for the row below |
| Enzyme forward, with and without runtime activity | `simulate_session` where the cognitive model's action is `NoValue` on some timesteps and the environment starts | wrong gradient, no error | to be reported; right when the agent starts |
| Enzyme, every form | `simulate_session` where a state changes type, such as an integer before the first timestep and a number after it, and a model with every feature, whose choice is `NoValue` on some timesteps among outputs of other kinds | throws an `IllegalTypeAnalysisException` | related to Enzyme.jl issue 4031; a state that keeps its type is right |
| ReverseDiff with a compiled tape | `simulate_session` where the model stops once a value passes a threshold, or its parameters turn invalid on a timestep that depends on them, or it samples its choices and rewards from a seeded `Xoshiro`, and a Turing.jl session whose stopping criterion is met once a value passes a threshold | wrong gradient, no error, at points where the other branch is taken | a limitation of compiled tapes, see the pitfalls below |
| Enzyme forward and reverse without runtime activity | the log density of a Turing.jl full model, and of many sessions: with imputed values, of models whose outputs are numbers from a normal distribution, vectors or `NamedTuple`s, of a model which uses the sizes of the timesteps, and with an environment which samples within a timestep or has names of its own | throws an `EnzymeRuntimeActivityError` | a limitation of Enzyme: use runtime activity, which Turing.jl's own tests run Enzyme with. The sessions of the baseline and of most models whose outputs are choices pass |
| Enzyme, every form | a Turing.jl session whose observation is a vector, missing on one timestep and imputed: the imputed vector is a view of the parameters, which meets the recorded vectors in the loop over the timesteps | throws an `IllegalTypeAnalysisException` | reported as Enzyme.jl issue 4106 |
| Enzyme, every form | a Turing.jl session or full model of the model with every feature, whose choice is `NoValue` on some timesteps among outputs of other kinds, and sessions where the agent starts and an observation is imputed, so that a value of one of two types goes from one timestep to the next | throws an `IllegalTypeAnalysisException` | related to Enzyme.jl issue 4031 |
| Enzyme reverse, with and without runtime activity | a Turing.jl session where an output of either model is `NoValue` on some timesteps | throws a `TypeError`, that a `RefValue` was given where a value was expected | reported as Enzyme.jl issue 4044 |
| Enzyme forward with runtime activity | the log density of a Turing.jl model with an output from a normal distribution over vectors, `MvNormal`, whose covariance depends on a parameter | throws an `EnzymeNoDerivativeError`, since Enzyme has no forward rule for LAPACK's triangular solve, `trtrs` | reported as Enzyme.jl issue 1991; use reverse mode, which is right |
| Enzyme reverse with runtime activity, on macOS without bounds checks | the log density of a Turing.jl full model with a parameter that is a vector, with and without missing values | throws an `LLVMException`, "Referring to an instruction in another function!", when Enzyme links its code, since LLVM's GVN pass makes a load in one function refer to a load in another | to be reported; right with `--check-bounds=yes` and on Linux. LLVM fixed the pass in llvm/llvm-project#223456, after the LLVM 20.1.8 of Julia 1.13.4 |
| Enzyme reverse with runtime activity | a Turing.jl session with an imputed value among several outputs, such as a missing confidence beside a choice, or a missing cue among the observations | throws an `EnzymeNoShadowError`: no shadow for the values of the timestep before, which the model of each timestep takes | to be reported |
| ReverseDiff, also with a compiled tape | Turing.jl sessions of a model whose actions are numbers from a normal distribution, when the agent starts, so that the first action has a constant mean and its distribution another type of tracked numbers than the later ones | throws an `InexactError`, from a `TrackedReal{Real}` that Distributions.jl's `product_distribution` makes | to be reported to ReverseDiff.jl |
| Mooncake forward | the log density of several Turing.jl sessions where only some have latent variables, such as imputed values, differentiated as a function that holds the model's `LogDensityFunction` | throws an `UndefRefError` | reported as Mooncake.jl issue 1476; Turing.jl's samplers aren't affected, since they differentiate through `LogDensityFunction(model; adtype)`, which is right |

A state which depends on the parameters from the start isn't affected, and neither is
`simulate_timestep!`. The other forms of Enzyme, and the other backends, are right on all of
these cases. ForwardDiff is right on every Turing.jl case, ReverseDiff and Mooncake on every
case but those above, and both forms of Enzyme with runtime activity on most sessions and
full models.

## Pitfalls

These aren't failures of the package's own code, but a model can run into them.

- **Enzyme reverse with runtime activity and a softmax in one line:** a softmax written as
  `exp.(v .- maximum(v)) ./ sum(exp.(v .- maximum(v)))` gives wrong gradients, with no
  error, when it runs in a loop on values that differ between the timesteps. The `softmax`
  of LogExpFunctions.jl is right, and so is any softmax which keeps its exponentials in a
  variable and divides them by their sum. It is reported as Enzyme.jl issue 4030.
- **Enzyme without runtime activity and objects that hold data:** an object with an array in
  it that is made within the function that is differentiated, or captured by it, such as a
  [`FixedObservationsEnvironment`](@ref) made from the observations there, throws an
  `EnzymeRuntimeActivityError`. Make it once outside the function, as a constant, or use
  runtime activity.
- **Compiled ReverseDiff tapes:** a tape records one path through the code, so a model with
  a branch on a value gives wrong gradients at points where the other branch is taken, with
  no error. The tests prepare the tape at a different point to catch this.
- **Turing.jl models and Enzyme:** Enzyme needs runtime activity to differentiate most
  Turing.jl models, and Turing.jl's own tests run Enzyme only with it; without it, it throws
  an `EnzymeRuntimeActivityError`.
- **Turing.jl models and Mooncake:** Mooncake can't differentiate `logjoint(model, values)`
  when the model has latent variables under an indexed prefix, such as the imputed actions of
  a session or a model which samples within a timestep, because building the variables runs
  DynamicPPL's warning about growable arrays, and a `@warn` holds a `try`. Samplers go through
  a `LogDensityFunction`, where the variables exist already, and that is fine.
- **Parameters that turn invalid:** a session whose model sets `parameters_valid` to `false`
  in its [`ModelOutput`](@ref) gets a log density of `-Inf` and scores none of its data, so
  its gradient is zero with every backend, apart from that of any value it imputes or
  samples. A compiled ReverseDiff tape follows the branches taken at the point where it was
  compiled, so at parameters that turn invalid on another timestep it gives a finite log
  density with no error. A model should keep its values finite on the
  timestep that turns invalid, as with `ifelse(parameters_valid, new_value, previous_value)`,
  since a `NaN` there makes the gradients of the reverse modes `NaN`, which an optimizer can't
  use.
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
  A `parameters_valid` or a stopping criterion that depends on traced values is a traced
  Bool, which nothing can branch on, so a simulation is masked from the first timestep where
  one is, as [`simulate_session`](@ref) describes: every timestep is simulated, and
  `n_timesteps` says how many were kept.
  A function can't be compiled when it is given, or holds, an object with a field declared
  as `NamedTuple` without its names and types, such as a model with `settings::NamedTuple`:
  give the field a type parameter instead.
