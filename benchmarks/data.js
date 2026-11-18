window.BENCHMARK_DATA = {
  "lastUpdate": 1794998227234,
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
      }
    ]
  }
}