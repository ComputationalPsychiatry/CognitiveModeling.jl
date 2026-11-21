window.BENCHMARK_DATA = {
  "lastUpdate": 1795260574825,
  "repoUrl": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl",
  "entries": {
    "CognitiveModeling.jl": [
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "4123972ac0c3ece702faa537015cccbc437871e2",
          "message": "Add FixedObservationsEnvironment, an environment which gives the observations of each timestep from vectors, takes no actions and ends a simulation after its last entry, add a benchmark folder with a BenchmarkTools suite of simulations and of the log density and its gradient of Turing.jl models with ForwardDiff, ReverseDiff, Mooncake and Enzyme, and a Benchmarks workflow which keeps a history of the results on the gh-pages branch, compile sessions that are run by hand in the Reactant suite, so that the functions for use within the model of an inference backend are tested with Reactant, which can't compile the Turing.jl models, give the table of MultipleSessionsInfo a type parameter, since Reactant can't compile a function which holds an object with a field declared as NamedTuple, mark Enzyme without runtime activity as a known failure on fixed observations of two types, and add extended benchmarks with full data sets to the planned page",
          "timestamp": "2026-11-17T08:47:48+01:00",
          "tree_id": "2c7cf18c9144b2be531964b0d785eb04d20df7d4",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/4123972ac0c3ece702faa537015cccbc437871e2"
        },
        "date": 1794902164588,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3773.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1852.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 143706,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200144\nallocs=697\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 148539,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 147638,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1713673,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 26930,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9590.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5256.166666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9013.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 195629,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2412.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "06e9be21a47fa2d062b93d97a4d7a4a3879699da",
          "message": "Say in the docstrings of the regression that an interaction comes with the main effects of its predictors, that the name of an interaction with _by_ is its default, that varying effects are also called random effects, and that a regression commonly has varying effects over the sessions, without which sessions with the same values in the columns share a value of the parameter, and shorten the error of a ContinuousPredictor that is given several columns",
          "timestamp": "2026-11-17T10:50:02+01:00",
          "tree_id": "7fa0d48dfbd2cb26f440bb1f6cf8f4fd5992e945",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/06e9be21a47fa2d062b93d97a4d7a4a3879699da"
        },
        "date": 1794909258315,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3746.0555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1851.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 144634,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200144\nallocs=697\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 170723,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 144959.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1714753,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27180.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9695.3,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6089,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9296.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 195334.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2392.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "24d9cfc09b395236de88ac6d628919aea80980fa",
          "message": "Keep the docstring of distribution_support free of Turing.jl, as the rest of the package's own docstrings are, and say on the page on fitting with Turing.jl that the extension gives it the values of the univariate distributions of Distributions.jl",
          "timestamp": "2026-11-17T14:09:56+01:00",
          "tree_id": "23dc67ff43c6978f960a82615e05a522c9866d01",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/24d9cfc09b395236de88ac6d628919aea80980fa"
        },
        "date": 1794930781911,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4667.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2396.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 181448,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200144\nallocs=697\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 186858.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 197639,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2320700,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34214,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10813.333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6930.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 10119,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 249831,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3068,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "b1bc0e621994964a1a012d889f9d4a0a96ad7c04",
          "message": "Show in the sketch of the tutorial on discrete parameters how to read the names of the discrete and of the continuous variables of a Turing.jl model from the distributions that DynamicPPL.jl records, and how to give each kind a sampler in Gibbs, with nadapts for a sampler which adapts, since NUTS doesn't adapt inside Gibbs without it, and run the same lines in the tests of the Turing.jl extension, on a model of one's own, on a full model with a discrete parameter and an imputed choice, and on a regression with a family, so that a change in Turing.jl or DynamicPPL.jl shows there",
          "timestamp": "2026-11-17T19:37:49+01:00",
          "tree_id": "ce3d68c5c25bca3f0cd25b9374dd87baf6c3b0ad",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/b1bc0e621994964a1a012d889f9d4a0a96ad7c04"
        },
        "date": 1794940916549,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3065.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1450.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 135008.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200144\nallocs=697\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 115299.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 106551,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1388297,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 19820,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7823.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 3741.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 6768.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 146381,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1755.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "d6237c8f67f5cdce2fab82cb88b0887822254b21",
          "message": "Add an MCMCChains.jl extension for reading and simulating fitted sessions to the planned interfaces, and remove a comment left from the check at preparation",
          "timestamp": "2026-11-17T20:26:03+01:00",
          "tree_id": "c059850aa0d76728b239d49a8e827529a3f629c7",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/d6237c8f67f5cdce2fab82cb88b0887822254b21"
        },
        "date": 1794944038803,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3192.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2077.9444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 141833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200144\nallocs=697\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 106119.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 113892.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1699365.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 22061,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7740.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4159.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 7512,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 168428,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1942.15,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "a89830eb7a59da6134f087e641ad89aca2f404fa",
          "message": "Run the sessions one after the other or in parallel in two methods of turing_multiple_behavioral_sessions, chosen by its last positional argument, parallelization, which comes after the environment parameters and was a keyword, instead of in a second model, turing_run_sessions, which is gone, with the environment parameters checked in both methods",
          "timestamp": "2026-11-17T20:45:42+01:00",
          "tree_id": "6a3f79286f9973cbb848285388b5c04a59e2a491",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/a89830eb7a59da6134f087e641ad89aca2f404fa"
        },
        "date": 1794945263636,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3774.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1853.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 151233,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200176\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 151809,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 147733,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1791201,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27391,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9778.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5329.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9168.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 195089,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2422.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "de084ae80f664434b08cdce266ab95a4bafcb7de",
          "message": "Combine turing_impute_missing_values and turing_impute_each_value into one Turing.jl model, turing_impute_missing_values, which returns the data as it is when nothing in it is missing, and otherwise imputes the first value under its name and goes on with the rest, building the data again from its values in order, so that the values keep their types and sessions with a missing action or observation are no longer many times slower",
          "timestamp": "2026-11-18T09:36:34+01:00",
          "tree_id": "7c44fcf02fa4afe46477602c24133a602125a435",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/de084ae80f664434b08cdce266ab95a4bafcb7de"
        },
        "date": 1794991384511,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3790.222222222222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1892.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 142880.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200176\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 147753,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 146310,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1715660,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27191,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9794.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5273,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9171.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 192805,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2388.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "19954ec44c2787bc1d6e24dec600b1a322750b26",
          "message": "Rename turing_hierarchical_group_parameters and turing_hierarchical_group_parameter, which sample the parameters of the group-level distribution of each group, to turing_group_level_parameters and turing_group_level_parameter, and turing_regression_parameters and turing_regression_parameter to turing_hierarchical_regression_parameters and turing_hierarchical_regression_parameter, after HierarchicalRegressionParameters",
          "timestamp": "2026-11-18T10:24:13+01:00",
          "tree_id": "ac8518d600c2ec9d9a1d6f143c021d1883b075ec",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/19954ec44c2787bc1d6e24dec600b1a322750b26"
        },
        "date": 1794998226620,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4651.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2387.777777777778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 179746.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200176\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 173846,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 198552,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2230166,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34038.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10756.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6850.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 10018,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 246076,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3059.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "510371c03299169217f07bc4f689b23c2665afba",
          "message": "Describe the cause of Enzyme's IllegalTypeAnalysisException for an imputed action that is a vector correctly, in the known failures of the tests of the Turing.jl extension and on the page on automatic differentiation: the imputed vector is a view of the vector of parameters, which meets the recorded vectors in the loop over the timesteps, and not the Union of the column of actions",
          "timestamp": "2026-11-18T23:19:11+01:00",
          "tree_id": "1d93244800c6a59074e517783dae1e51d41a66dd",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/510371c03299169217f07bc4f689b23c2665afba"
        },
        "date": 1795040953585,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4208.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2692.222222222222,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 174442,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200176\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 150049,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 144987.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2203736,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27507,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9881,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5426.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9254.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 227168,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2450.6000000000004,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "f0e9346c8732a71823325c8f5a005671e8aa9629",
          "message": "Check the data of a fitted session against the domains of the cognitive model's inputs and outputs: input_domains and output_domains, public beside parameter_domains, give a Domain for an input or an output, and when a session is read its observations are checked against the input domains and its actions against the output domains, once, apart from markers such as MissingValue. A value outside its domain is an ArgumentError that names its row of the table, or its timestep in a session. The checks of parameters and of values in the data share their code, which is told what is checked: a parameter of a model, whose error is a ParameterDomainError as before, or a DataValue. The display of a model shows the domains of its inputs and outputs as it does those of its parameters",
          "timestamp": "2026-11-19T00:06:57+01:00",
          "tree_id": "6ae473212e5549bd18f2ce2c8ab66334e97ab1a2",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/f0e9346c8732a71823325c8f5a005671e8aa9629"
        },
        "date": 1795043864758,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4609.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4160\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2467.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 186868,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200176\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 178753,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339488\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 195353,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92496\nallocs=980\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2333594,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1556528\nallocs=42516\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 33983,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22752\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10643,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7221,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 10240,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 249143.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=155296\nallocs=4245\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3040.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "1e385204cefc56781082d86b46cc7c4c81f4befe",
          "message": "Use WithoutOutputs of the example models in the tests of simulate in place of OnlyLearns, a copy of it. The other models of those tests stay, since none of them is the same as an example model. The tests pass as before, and cover the same lines of the package",
          "timestamp": "2026-11-20T18:06:10+01:00",
          "tree_id": "65890a09437fdcbb347e37bb2c114156b0538e13",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/1e385204cefc56781082d86b46cc7c4c81f4befe"
        },
        "date": 1795197937484,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3536.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2123.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 140878,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 105235,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 113189.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93312\nallocs=982\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1789756,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 21860,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7773.166666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4207.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 7372.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 178047.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1917.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1007996,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7290409,
            "unit": "ns",
            "extra": "gctime=2322470\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1105490,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1047128\nallocs=4108\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 19748101,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 227414,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324160\nallocs=887\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "8bdbf6784293985ddedf042f5f19b74559cfb672",
          "message": "Test every script of the documentation, and run the tutorials that take long only in the builds of a release. docs/how_scripts_are_run.toml says for each tutorial whether every build runs it, \"always\", only the builds of a release, which are those of a tag and of a pull request into main, \"for_releases\", or none, \"never\", and for each script the tier of its tests: \"fast\" on every push, \"long\" on pull requests into main and when started by hand, in the new workflow LongTests.yml, or \"none\". Other builds show only the code of a tutorial for releases, with a note, and Docs.yml sets DOCUMENTATION_FOR_A_RELEASE for tags and pull requests into main. Every script has a folder for its tests named after it, whose results.jl checks what it gave. The tutorials that the build runs are tested in the documentation's environment by test/documentation/tutorials/runtests.jl, those that it doesn't in environments of their own, whose folders are renamed after their tutorials, and the examples by test/documentation/examples/runtests.jl. Each runner takes the tier as its argument, and one with an environment of its own fails when it is run for another tier than its script's. The placeholder tutorials have a results.jl with nothing to check yet, the tutorial on discrete parameters simulates its sessions and samples very little, so that it runs, also in the build, and a check in the main suite keeps the settings, the scripts and the folders of their tests in step",
          "timestamp": "2026-11-20T20:10:02+01:00",
          "tree_id": "90aeb013bf433875630be1581ac0d0e61fc3408b",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/8bdbf6784293985ddedf042f5f19b74559cfb672"
        },
        "date": 1795203857713,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3365,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1506.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 133569.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 121661,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 105262.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93312\nallocs=982\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1473682,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 19830,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7756.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 3809.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 6551.333333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 147751,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1744.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 882457,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 8429840.5,
            "unit": "ns",
            "extra": "gctime=2135659\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 755598.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1047128\nallocs=4108\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 17857957,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 187159,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324160\nallocs=887\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "e163df7daa7127e814a5ae0dd2f1156450e3d89f",
          "message": "Declare TOML in the environment of the main suite, whose check of the scripts of the documentation reads docs/how_scripts_are_run.toml with it. Pkg.test, which CI runs the main suite with, only loads the packages that the environment declares, also those of the standard library",
          "timestamp": "2026-11-20T20:56:31+01:00",
          "tree_id": "4df887ff27418275871775b0faceaa8f7adab33c",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/e163df7daa7127e814a5ae0dd2f1156450e3d89f"
        },
        "date": 1795205277103,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4268.555555555556,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1953.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 169637,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 184738,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 145005,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93312\nallocs=982\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1979461,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27140.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9840.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5287.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9086,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5936\nallocs=51\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 202214,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2421.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1070602,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10284564,
            "unit": "ns",
            "extra": "gctime=1971223\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 991786,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1047128\nallocs=4108\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 23635230,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 208383,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324160\nallocs=887\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "33e0bc22c2eeef01bd3fbc5dd04894351aa6b020",
          "message": "Use the example models in the tests of simulate, simulate! and instantiated models wherever one can serve a test: Baseline with the distributions written by hand in place of LearnsRewards, and the example environments, with FixedObservationsEnvironment for given rewards and the means of the distributions where a test checks values exactly, so that every test keeps its exact expectations. 28 of the 48 local models and distributions remain: those that test errors, such as a model without names or a method that returns the wrong type, those whose feature no example model has, such as a function that prepares the state for saving, an environment with default parameters or one that changes its state in place, and the two distributions they give. The tests of FixedObservationsEnvironment and of displaying instantiated models use the example models too. The suite covers the same lines of the package as before, and the test of vectors checks twelve values fewer, since VectorAttributes has no number beside its vectors",
          "timestamp": "2026-11-20T21:46:03+01:00",
          "tree_id": "12bb1f337ed9dbea7494bf97a586658e8fb424e7",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/33e0bc22c2eeef01bd3fbc5dd04894351aa6b020"
        },
        "date": 1795208786143,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5486.875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2189.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 215708,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 193045,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 182420,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92352\nallocs=952\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2530425.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34511,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 11847.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6665.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9674,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 258171.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3090.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1409190,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11020212,
            "unit": "ns",
            "extra": "gctime=2330711\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1251758,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1042328\nallocs=3958\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28625912,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 268928,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324160\nallocs=887\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "peter@waade.net",
            "name": "PTWaade",
            "username": "PTWaade"
          },
          "distinct": true,
          "id": "fd8b4814e57e40c49faaeeaabff22cdc546a07da",
          "message": "Say that Turing.jl's own tests run Enzyme only with runtime activity, where the page on automatic differentiation, a known failure and a comment of the benchmarks said that Turing.jl's documentation says Enzyme needs it. Turing.jl's page on automatic differentiation doesn't mention runtime activity, and its tests of Enzyme, in test/integration/enzyme, set runtime activity in both modes",
          "timestamp": "2026-11-21T03:12:27+01:00",
          "tree_id": "ac7f5a27f624cef9f43773709c54ec94bc4a9329",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/fd8b4814e57e40c49faaeeaabff22cdc546a07da"
        },
        "date": 1795260573510,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4343.055555555556,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1946.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 173945,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 172142.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 144262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92352\nallocs=952\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1974902,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 26964.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9882.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5234.333333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8825.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 211310,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2416.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1077902,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10026577,
            "unit": "ns",
            "extra": "gctime=2084888\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 977484,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1042328\nallocs=3958\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 22995221,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 206332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324160\nallocs=887\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}