# CognitiveModeling.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ComputationalPsychiatry.github.io/CognitiveModeling.jl/dev/)
[![CI](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml/badge.svg?branch=dev)](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A Julia package for computational modeling of cognition and behavior. It is a remake of [ActionModels.jl](https://github.com/ComputationalPsychiatry/ActionModels.jl) and is under development.

Pre-created models live in a separate package, [PremadeCognitiveModels.jl](https://github.com/ComputationalPsychiatry/PremadeCognitiveModels.jl).

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

The automatic differentiation and Reactant tests each have their own environment, and both check every entry in `test/ad_and_reactant/test_cases.jl`:

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

Automatic differentiation of the Turing.jl extension is tested in another environment, with its own cases:

```bash
julia --project=test/extensions/turing/ad -e 'using Pkg; Pkg.instantiate(); include("test/extensions/turing/ad/runtests.jl")'
```

The two tutorials that the documentation shows without running are run by tests of their own, each in an environment with the packages it uses:

```bash
julia --project=test/tutorials/reinforcement_learning -e 'using Pkg; Pkg.instantiate(); include("test/tutorials/reinforcement_learning/runtests.jl")'
```

```bash
julia --project=test/tutorials/pomdps -e 'using Pkg; Pkg.instantiate(); include("test/tutorials/pomdps/runtests.jl")'
```

Build the documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
