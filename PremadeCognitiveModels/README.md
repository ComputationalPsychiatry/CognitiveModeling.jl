# PremadeCognitiveModels.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ComputationalPsychiatry.github.io/PremadeCognitiveModels.jl/dev/)
[![CI](https://github.com/ComputationalPsychiatry/PremadeCognitiveModels.jl/actions/workflows/CI.yml/badge.svg?branch=dev)](https://github.com/ComputationalPsychiatry/PremadeCognitiveModels.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ComputationalPsychiatry/PremadeCognitiveModels.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/ComputationalPsychiatry/PremadeCognitiveModels.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A collection of pre-created cognitive models for [CognitiveModeling.jl](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl). It is under development.

## Development

Work happens on `dev`, and `main` holds released code, so open pull requests against `dev`.

Clone [CognitiveModeling.jl](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl) next to this repo. `Project.toml` uses the copy at `../CognitiveModeling.jl`, so changes to it take effect here straight away.

The code is formatted with [JuliaFormatter](https://github.com/JuliaEditorSupport/JuliaFormatter.jl) in [Blue style](https://github.com/JuliaDiff/BlueStyle). Install its `jlfmt` app once per machine, and the formatting hook once per clone. The hook then formats the files of every `git commit`:

```bash
julia -e 'import Pkg; Pkg.Apps.add("JuliaFormatter")'
pre-commit install
```

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

Build the documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
