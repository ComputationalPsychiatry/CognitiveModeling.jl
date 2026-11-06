# CognitiveModeling.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://ComputationalPsychiatry.github.io/CognitiveModeling.jl/dev/)
[![CI](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml/badge.svg?branch=dev)](https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl/branch/dev/graph/badge.svg)](https://codecov.io/gh/ComputationalPsychiatry/CognitiveModeling.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

A Julia package for computational modeling of cognition and behaviour. It is a remake of [ActionModels.jl](https://github.com/ComputationalPsychiatry/ActionModels.jl) and is under development.

Pre-created models live in a separate package, [PremadeCognitiveModels.jl](https://github.com/ComputationalPsychiatry/PremadeCognitiveModels.jl).

## Development

Work happens on `dev`, and `main` holds released code, so open pull requests against `dev`.

Install the formatting hook once per clone. It runs [Runic](https://github.com/fredrikekre/Runic.jl) on every `git commit`:

```bash
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

Automatic differentiation of the Turing.jl extension is tested in an environment of its own, with its own cases, beside the extension's other tests:

```bash
julia --project=test/extensions/turing/ad -e 'using Pkg; Pkg.instantiate(); include("test/extensions/turing/ad/runtests.jl")'
```

Build the documentation:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```
