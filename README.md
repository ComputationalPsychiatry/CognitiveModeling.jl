# CognitiveModeling.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ComputationalPsychiatry.github.io/CognitiveModeling.jl/dev/)
[![CI](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml/badge.svg?branch=dev)](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A Julia package for computational modeling of cognition and behavior (cognitive modeling, or cognitive modelling in British English). It is a remake of [ActionModels.jl](https://github.com/ComputationalPsychiatry/ActionModels.jl) and is under development.

Pre-created models live in a package of their own in this repository, [PremadeCognitiveModels](PremadeCognitiveModels).

What is planned is in the [roadmap](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/issues/11).

## Use of generative AI
This package has been written with assistance from Large Language Models, but under close supervision, and all generated code has been hand-reviewed. 

## Development

Work happens on `dev`, and `main` holds released code, so open pull requests against `dev`.

The code is formatted with [JuliaFormatter](https://github.com/JuliaEditorSupport/JuliaFormatter.jl) 2.14.10 in [Blue style](https://github.com/JuliaDiff/BlueStyle). Install its `jlfmt` app once per machine, and the formatting hook once per clone. The hook then formats the files of every `git commit`:

```bash
julia -e 'import Pkg; Pkg.Apps.add(name="JuliaFormatter", version="2.14.10")'
pre-commit install
```

The version is pinned in three places, which change together: this command, the `rev` of the hook in `.pre-commit-config.yaml`, and the `version` in `.github/workflows/Format.yml`. The hook runs the `jlfmt` that is installed, so a different version of it can format differently from the check on GitHub.

Run the main tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The automatic differentiation and Reactant tests each have their own environment, and both check the simulation cases that the suites share, in `test/common/src/simulation_cases.jl`:

```bash
julia --project=test/ad_and_reactant/ad -e 'using Pkg; Pkg.instantiate(); include("test/ad_and_reactant/ad/runtests.jl")'
```

```bash
julia --project=test/ad_and_reactant/reactant -e 'using Pkg; Pkg.instantiate(); include("test/ad_and_reactant/reactant/runtests.jl")'
```

The Turing.jl extension is tested in an environment of its own, since Turing.jl holds FlexiChains.jl at an older version than the main tests use:

```bash
julia --project=test/extensions/turing -e 'using Pkg; Pkg.instantiate(); include("test/extensions/turing/runtests.jl")'
```

Its two parts, the tests of the extension's features and the shared inference cases, can be run on their own, as CI runs them in a job each:

```bash
julia --project=test/extensions/turing test/extensions/turing/runtests.jl shared_cases
```

Automatic differentiation of the Turing.jl extension is tested in another environment, with its own cases:

```bash
julia --project=test/extensions/turing/ad -e 'using Pkg; Pkg.instantiate(); include("test/extensions/turing/ad/runtests.jl")'
```

Both suites of automatic differentiation test every backend when they are run without arguments, and otherwise the backends whose names start with one of their arguments, as CI runs each group of backends in a job of its own:

```bash
julia --project=test/ad_and_reactant/ad test/ad_and_reactant/ad/runtests.jl "Enzyme forward" Mooncake
```

The scripts of the documentation are tested with the checks of their results, as `docs/how_scripts_are_run.toml` describes. Each runner takes the tier of the tests as its argument, `fast` or `long`, and runs the scripts of that tier. `CI.yml` runs the fast tier on every push, and `LongTests.yml` the long tier on pull requests into `main` and when started by hand. The tutorials that the documentation's build runs are tested in its environment:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs test/documentation/tutorials/runtests.jl fast
```

The tutorials that the build shows without running have an environment of their own each, with the packages they use:

```bash
julia --project=test/documentation/tutorials/interfacing_with_reinforcement_learning -e 'using Pkg; Pkg.instantiate()'
julia --project=test/documentation/tutorials/interfacing_with_reinforcement_learning test/documentation/tutorials/interfacing_with_reinforcement_learning/runtests.jl fast
```

```bash
julia --project=test/documentation/tutorials/interfacing_with_pomdps -e 'using Pkg; Pkg.instantiate()'
julia --project=test/documentation/tutorials/interfacing_with_pomdps test/documentation/tutorials/interfacing_with_pomdps/runtests.jl fast
```

```bash
julia --project=test/documentation/tutorials/interfacing_with_juliacontrol -e 'using Pkg; Pkg.instantiate()'
julia --project=test/documentation/tutorials/interfacing_with_juliacontrol test/documentation/tutorials/interfacing_with_juliacontrol/runtests.jl fast
```

```bash
julia --project=test/documentation/tutorials/interfacing_with_sequential_sampling_models -e 'using Pkg; Pkg.instantiate()'
julia --project=test/documentation/tutorials/interfacing_with_sequential_sampling_models test/documentation/tutorials/interfacing_with_sequential_sampling_models/runtests.jl fast
```

The examples in `docs/examples` have an environment of their own:

```bash
julia --project=test/documentation/examples -e 'using Pkg; Pkg.instantiate()'
julia --project=test/documentation/examples test/documentation/examples/runtests.jl fast
```

Run the benchmarks of simulating and of fitting with each automatic differentiation backend. The `Benchmarks` workflow runs them on every push to `dev` and `main`, and keeps a history of the results with charts at <https://ComputationalPsychiatry.github.io/CognitiveModeling.jl/benchmarks/>:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate(); include("benchmark/run_benchmarks.jl")'
```

Build the documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

The build shows only the code of the tutorials that only the builds of a release run, which take long. Build the documentation of a release, which runs them too:

```bash
DOCUMENTATION_FOR_A_RELEASE=true julia --project=docs docs/make.jl
```
