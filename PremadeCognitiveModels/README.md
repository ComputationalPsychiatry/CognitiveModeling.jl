# PremadeCognitiveModels.jl

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

A collection of pre-created cognitive models for [CognitiveModeling.jl](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl). It is under development.

## Development

The package lives in the repository of CognitiveModeling.jl, as a package of its own, and its work goes through that repository: its `dev` branch, its pull requests and its CI, which runs the tests below on every pull request. `Project.toml` uses the copy of CognitiveModeling.jl in the folder above, so changes to it take effect here straight away. Formatting is set up as the [README of CognitiveModeling.jl](../README.md) describes.

Run the main tests, from this folder:

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
