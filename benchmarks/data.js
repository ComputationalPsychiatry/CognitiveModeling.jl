window.BENCHMARK_DATA = {
  "lastUpdate": 1795557099774,
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
          "id": "e12d7be3e7047986b6dcc322ce9f966bb08cf01f",
          "message": "Give Turing.jl's predict a method for the models that the package makes around sessions, for a FlexiChain, which leaves the variables within the timesteps out of the chain before Turing.jl's own predict runs, as predict_session_trajectories does, so that predict(decondition(model), chain) simulates again the actions and observations that the fit imputed, as a predictive check needs. Turing.jl's own predict keeps the value that a chain has for a variable, and throws a BoundsError when the fit imputed values in two sessions at different timesteps, from FlexiChains.jl's rebuilding of a sample, which DynamicPPL.jl pull request 1542 fixes. The parameters, any start that a model samples, and the statistics of the sampler are kept from the chain, as Turing.jl's own predict keeps them",
          "timestamp": "2026-11-21T12:52:28+01:00",
          "tree_id": "b1b3bea862b5875189ab4ed251582ac62fb644e2",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/e12d7be3e7047986b6dcc322ce9f966bb08cf01f"
        },
        "date": 1795305739457,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3078,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1421.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 126210,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 111092.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 94052,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92352\nallocs=952\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1366526.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 18398,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7367.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 3382.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 6067.714285714285,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 139320.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1594.4,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 777763,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7545760,
            "unit": "ns",
            "extra": "gctime=1787741\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 659035,
            "unit": "ns",
            "extra": "gctime=0\nmemory=976752\nallocs=3956\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 15586248,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 172771,
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
          "id": "6e94652946d9f0adf0115003577fa17cdb4242a6",
          "message": "Add to the plan compatibility with KernelAbstractions.jl, and with GPUs in general, through the packages of JuliaGPU, such as CUDA.jl, AMDGPU.jl, Metal.jl and oneAPI.jl",
          "timestamp": "2026-11-22T15:48:15+01:00",
          "tree_id": "3935536a7ab84c0816b7943c2207e8029df8dee3",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/6e94652946d9f0adf0115003577fa17cdb4242a6"
        },
        "date": 1795362961034,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4065,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2375.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 167786.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200272\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 129603,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339584\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 126925,
            "unit": "ns",
            "extra": "gctime=0\nmemory=92352\nallocs=952\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2005781,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1544944\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 24053,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22800\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 8986.833333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4675.857142857143,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8107.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 204061.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2135.3,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1100803,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076416\nallocs=2561\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 8406720,
            "unit": "ns",
            "extra": "gctime=2383503\nmemory=34279408\nallocs=8975\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1169107,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1042328\nallocs=3958\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 23076373,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15300720\nallocs=413250\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 262724,
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
          "id": "7852657d0221e0a39211537a68f3f063c62f3291",
          "message": "Add SessionTrajectories, an exported type for the trajectories of one session and how it ended, which simulate_session and simulate_session! return in place of a NamedTuple, and which a fitted session returns under saved_trajectories, so that a downstream model gets a vector of them. Its fields are those the NamedTuple had: states, observations, actions and action_distributions of the cognitive model, environment with the trajectories of the environment, which is NoEnvironment() for a fitted session without an environment, timestep_sizes, parameters_valid, n_timesteps and last_timestep, which fitted sessions now have too, with the number of timesteps that ran and nothing as the last timestep. NamedTuple(trajectories) gives the fields in a NamedTuple, two are equal when their fields are, and in the REPL they show the number of timesteps, the names of what was saved of each kind for both models, and whether the parameters stayed valid, and on one line the number of timesteps. The internal saved_trajectories builds it, simulation_result is gone, and with_empty_environment_entry is with_empty_environment_trajectories. get_session_trajectories and get_session_parameters read the type, and returned values without it are an error. The docstrings of the Turing.jl models now say how sampling can fail to start when parameters turn invalid: a sample whose session ends early lacks the variables after that timestep, the imputed or simulated values of later timesteps and those of a downstream model, and a sampler of Turing.jl takes the variables of the model from one draw from the prior, made with the default random number generator, so it throws a KeyError when the parameters of that draw turn invalid, and an ArgumentError about indices that were not set when the parameters it starts from do, which valid initial_params prevent. The test helpers sum_of_reals and without_distributions read the type through NamedTuple, since a value of a type they don't know adds nothing to what the AD and Reactant suites differentiate. The plan has a format for the trajectories and the parameters of fitted sessions that every inference backend returns, and a version of the simulation for GPUs",
          "timestamp": "2026-11-22T22:54:40+01:00",
          "tree_id": "ca6c7e1b57320dfd0a75118874f24e35c05ecc89",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/7852657d0221e0a39211537a68f3f063c62f3291"
        },
        "date": 1795385085379,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3227.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1370,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 132417,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200496\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 115002,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339776\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 97270,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93184\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1403602,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545040\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 18688,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22896\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7541.166666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 3440.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 6073.285714285715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 141306,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1667.45,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 872717.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076984\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7757307,
            "unit": "ns",
            "extra": "gctime=1908516\nmemory=34283368\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 702508,
            "unit": "ns",
            "extra": "gctime=0\nmemory=977928\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 16887194,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301160\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 180324,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324600\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "ce43818b75a876e17dcfd21e2c59b50ec5953e19",
          "message": "Rename the markers MissingValue to ImputedMissingValue and SimulatedValue to SimulatedMissingValue, beside UnimputedMissingValue, the internal simulate_session_timestep to simulate_both_models and simulate_model_on_input to simulate_one_model, and the Turing.jl extension's returned_with_simulation to returned_with_prior_fallback, after DynamicPPL.jl's word for drawing from the prior what a chain doesn't have. condition_sessions now ends each session after its last timestep with a value, with the sizes of its timesteps and its rows in the table cut with it, so that a simulation which a stopping criterion ended is fitted with the timesteps it had, as from a table. By default the criterion is checked after the last timestep with data, which is then never: before, such a session checked it from its first timestep without a value, and sampled every action and observation of the timesteps that followed, so that the number of variables changed with the parameters. A session without any value keeps every timestep. The documentation of condition_sessions and the tutorial on predictive checks say how Turing.jl's condition differs for a model with a stopping criterion. The test helpers that simulate are named after simulating instead of running (simulate_with, simulate_with_sizes, simulate_until_end, typed_simulation, first_simulation, simulate_bandit), and prose where run meant simulate now says simulate, as in the docstring of stopping_criterion, so that run is the word for applying a model without sampling its outputs",
          "timestamp": "2026-11-22T23:12:22+01:00",
          "tree_id": "0a874513a63a0824b7f9b68005bbd41cfd8364a6",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/ce43818b75a876e17dcfd21e2c59b50ec5953e19"
        },
        "date": 1795386339046,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5223.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2375.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 203996.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200496\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 182781.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339776\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 203641,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93184\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2461302,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545040\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 33914,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22896\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10960.333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6975,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 254551,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3082.3888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1321652,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076984\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10919113,
            "unit": "ns",
            "extra": "gctime=1737048\nmemory=34283368\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1318144,
            "unit": "ns",
            "extra": "gctime=0\nmemory=977928\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27362678.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301160\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 325368,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324600\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "04c0ea7106c8fc0296b06a9b3b8acaaa9d1da62d",
          "message": "Explain in the documentation of fitting with Turing.jl, and in the docstrings of decondition_sessions and condition_sessions, why the package has its own versions of Turing.jl's decondition and condition, what the difference is, and what they allow. The models take the data of their sessions as arguments, not as observed variables, since the data is what the models receive on each timestep, decides how each session runs, is checked before a fit, includes observations without a distribution when a session has no environment, and is scored all at once, which is faster, so Turing.jl's decondition leaves it as it is. decondition_sessions turns it into variables that Turing.jl's own functions simulate, named apart from those of a fit, and condition_sessions puts values of them back in the data, where Turing.jl's condition keeps them as variables, which are looked up by name on every evaluation, left out without an error when misspelled, and scored where a stopping criterion ends a session at the parameters of each evaluation. Together they allow predictive checks and parameter recoveries with Turing.jl's own functions on the package's models, without writing the simulated data to a table first",
          "timestamp": "2026-11-22T23:34:16+01:00",
          "tree_id": "b4c5b21b5e8fc9552a79ecffe9b017f0ac0e6137",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/04c0ea7106c8fc0296b06a9b3b8acaaa9d1da62d"
        },
        "date": 1795387633058,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4383.8125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2750.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 196591,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200496\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 151087,
            "unit": "ns",
            "extra": "gctime=0\nmemory=339776\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 146990.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=93184\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2317411,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545040\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27385,
            "unit": "ns",
            "extra": "gctime=0\nmemory=22896\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10072.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5425.333333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 245460.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2487.55,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1496174,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2076984\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10711301.5,
            "unit": "ns",
            "extra": "gctime=2934521.5\nmemory=34283368\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1354924,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1043504\nallocs=3988\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 26414724,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301160\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 287684.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324600\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "f35fe7a670bf1f3d00842873e6c22dbcadc79da1",
          "message": "Keep the keyword arguments that read a table into sessions in the field column_settings of MultipleSessionsInfo: observation_columns, action_columns, session_columns, timestep_sizes, the values that stand for a missing one and whether those are imputed, which is nothing for sessions given already prepared, unless they are given. Its copies keep them, through the new internal with_sessions, which decondition_sessions, condition_sessions, the copies with other save histories and the full model use. condition_sessions now also takes the data in the forms that sessions are made from: a table of several sessions, which it reads with those settings and the other settings of the sessions, and which must have the same sessions in the same order, since what a full model was prepared with for each session must still fit, and the observations and the actions of a single session, as SessionInfo takes them, with timestep_sizes as a keyword argument. In both, the values that stand for a missing one become ImputedMissingValue, as when the sessions were made. The same forms take a MultipleSessionsInfo or a SessionInfo and return the sessions with the data. The Turing.jl extension also takes values as pairs of a VarName and a value, as Turing.jl's condition does, and says which form fits when a model has one session or several. with_data reads the settings that a session was prepared with through the new internal session_settings",
          "timestamp": "2026-11-22T23:46:41+01:00",
          "tree_id": "64c4f79db64a3c6e0982911e7ec3eb7e118e7100",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/f35fe7a670bf1f3d00842873e6c22dbcadc79da1"
        },
        "date": 1795388451177,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5261.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2374.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 219943,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201072\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 184176,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340448\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 205976.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95024\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2473913.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545456\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34635,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23232\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10893.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7769.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9959,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 256656.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3081.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1317407.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077560\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10711413,
            "unit": "ns",
            "extra": "gctime=1664092.5\nmemory=34286392\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1289419.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=979880\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27648873.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301592\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 328040.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324936\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "168522d31b3aa6b586e160adacc9cc7f36fa2ff9",
          "message": "When the parameters of a fitted session turn invalid, run its remaining timesteps instead of ending the session there, each from the values before the timestep that turned invalid, so that every draw has the variables that a valid one has. A sampler of Turing.jl takes the variables of a model from one draw from the prior, made with Julia's default random number generator, so sampling could fail to start for a session with latent variables after the timestep that turned invalid, or a downstream model with variables of its own: with a KeyError when the parameters of that draw turned invalid, and with an ArgumentError about indices that were not set when those the sampler started from did. The session still gives the parameters a log density of -Inf, keeps nothing of that timestep or of those after it, and scores none of its data, so that its trajectories end before the timestep, with parameters_valid false, as before. The remaining timesteps sample what valid ones do, such as imputed actions and what a model samples within a timestep, neither their flags nor a stopping criterion is checked, and they cost the time that ending the session saved. The full model runs its downstream model on every draw, also on trajectories that ended early, the docstring of turing_downstream_model says that it must not rely on every session having all of its timesteps, and the internal parameters_valid_in_every_session is gone. A test samples, with four seeds of the default random number generator, a model whose first draws mostly turn invalid and whose downstream model has a variable of its own, and an action imputed after the timestep that turned invalid is a variable, also when the first timestep turns invalid. The docstrings of the Turing.jl models no longer describe how sampling could fail to start, the page on AD backends says that such a session scores none of its data, and the docstring of predict_session_trajectories says that a predictive session also ends early when the parameters turn invalid on the simulated data",
          "timestamp": "2026-11-23T00:14:48+01:00",
          "tree_id": "fff4b19400fecafdd7a9e45c0f62de8387bb3e5f",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/168522d31b3aa6b586e160adacc9cc7f36fa2ff9"
        },
        "date": 1795389696101,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4307.555555555556,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1868.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 167940,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200944\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 170223,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340448\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 141335.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=94256\nallocs=949\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1964833,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545456\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23232\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9752,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5259.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8828,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 204084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2412.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1032996,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077432\nallocs=2562\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10257622.5,
            "unit": "ns",
            "extra": "gctime=2082736.5\nmemory=34286392\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 978896,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1044688\nallocs=3956\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 22333381,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301592\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 207108,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324936\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "83d7fc2f61f7bc36969bbe78da337124c1c33be4",
          "message": "Mask a simulation instead of stopping it from the first timestep whose parameters_valid or stopping criterion must be masked, such as a Bool that a compiler like Reactant.jl traces, whose value isn't known when the code is compiled, so that nothing can branch on it. The internal masks_flag tells such a flag from a Bool and from ValidByDefault by whether ifelse takes it as a condition, so that the package needs nothing of Reactant.jl, and anything else is an ArgumentError, and masks_bool does the same for what a stopping criterion returns, without a method for ValidByDefault, so that JET and the compiler know that a value it doesn't mask is a Bool. The masked loop, simulate_remaining_timesteps_masked in the new file mask_timesteps.jl, simulates every one of the n_steps timesteps and carries a running flag, which turns false when the stopping criterion of either model is met before a timestep or the parameters of either model turn invalid on one. From then on, each timestep starts from the values of the last timestep that was kept, which the internal select_values selects with ifelse through NamedTuples, tuples and arrays, and nothing of it counts. n_timesteps, parameters_valid and last_timestep are selected in the same way, so that they are what the stopping loop returns, as traced values, and the vectors keep every timestep, of which the first n_timesteps were kept, with the sizes of every timestep. A stopping criterion that is met before the first timestep isn't an error then, but leaves no timestep kept, and there is no warning. The stopping loop switches to the masked one at the first timestep where a flag must be masked, before it for a stopping criterion and after it for parameters_valid, since a state can be a constant on the first timestep, as when the agent starts, and traced only after it. The timesteps after a first timestep that turned invalid start from its values, since the values from before it can have other types. simulate_session! and simulate_timestep! throw an ArgumentError for such a flag, since the histories and the counts of instantiated models, and whether they are updated, depend on it, and simulate_timestep returns it without a warning. Reactant.jl now compiles WithStoppingCriterion and WithInvalidParameters, and new simulation cases that end before their last timestep, at a timestep that depends on the parameters, by a stopping criterion that a learning rate of 0.9 meets after a few rewards or by parameters that are invalid from the start, return the sum of what they kept, whose values and gradients match finite differences under Reactant.jl and the AD backends, with ReverseDiff's compiled tape marked broken when it is prepared elsewhere, as for the other simulations whose path depends on the parameters. In the main suite, a stand-in for a traced Bool, and models that give their flags as stand-ins, check that a masked simulation keeps what the stopping one keeps, for sessions that stay valid, turn invalid on the first or a later timestep, or are ended by the stopping criterion of either model, that it switches at the right timestep, and what the other simulate functions do with such a flag. The docstrings of simulate_session, the other simulate functions, SessionTrajectories, ModelOutput and stopping_criterion, and the page on automatic differentiation and Reactant, say what a masked simulation returns",
          "timestamp": "2026-11-23T00:47:22+01:00",
          "tree_id": "1c011f7f0c34521987d7e2d0a59ab255dcc2fe2a",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/83d7fc2f61f7bc36969bbe78da337124c1c33be4"
        },
        "date": 1795392059462,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4508.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2825,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 193232,
            "unit": "ns",
            "extra": "gctime=0\nmemory=200944\nallocs=698\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 163478.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340448\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 148497.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=94256\nallocs=949\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2413467,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545456\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 28459.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23232\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10333.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5671,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9506,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 241705,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2528.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1472067,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077432\nallocs=2562\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11234844,
            "unit": "ns",
            "extra": "gctime=3365285\nmemory=34286392\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1419692,
            "unit": "ns",
            "extra": "gctime=0\nmemory=979112\nallocs=3954\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27344212,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301592\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 312599.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324936\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
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
          "id": "d9e1daaea9ff5a5e724333fbe6874080e05520bc",
          "message": "Add a section on comparing models to the tutorial on predictive checks and parameter recovery. Turing.jl's pointwise_loglikelihoods with factorize = true gives the log-likelihood of each choice with the parameters of each sample, since DynamicPPL.jl splits the product that a session scores its choices with into a value for each, in the order of the timesteps that have one. The tutorial puts them in an array of samples by chains by data points, from which the loo of PosteriorStats.jl, which smooths with PSIS.jl, gives the PSIS-LOO estimate, and compares the Rescorla-Wagner model with a model that chooses with the same probability on every timestep with PosteriorStats.jl's compare. It computes WAIC by hand from the same array, since PosteriorStats.jl no longer does, and AIC and BIC from the maximum of the log-likelihood that Turing.jl's maximum_likelihood finds. The checks of the tutorial's results test that the log-likelihoods of the choices of a sample add up to its log-likelihood, that the PSIS-LOO and WAIC estimates agree, that the model that learns ranks first, and that the BIC is above the AIC. PosteriorStats.jl is added to the documentation's environment, and the page of what is planned lists leave-one-session-out cross-validation, which needs the parameters of a session integrated over their population distribution",
          "timestamp": "2026-11-23T12:07:57+01:00",
          "tree_id": "4b437bd4b591c41cf9270d35bd914593d3dde049",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/d9e1daaea9ff5a5e724333fbe6874080e05520bc"
        },
        "date": 1795432663484,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4237.333333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1867.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 184982,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201072\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 174637,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340448\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 148529,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95024\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1994275.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545456\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27831,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23232\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9754.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5456.333333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8970.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 207155,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2685.9,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1207987,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077560\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 13206486,
            "unit": "ns",
            "extra": "gctime=3526580\nmemory=34286392\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1031046,
            "unit": "ns",
            "extra": "gctime=0\nmemory=979880\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 26185447.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301592\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 221867,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324936\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0cb232350fb79ac421428b91c189ba1e85efcf7a",
          "message": "Merge pull request #3 from ComputationalPsychiatry/work/required-checks\n\nAdd the checks that the protection of main and dev requires",
          "timestamp": "2026-11-23T17:16:34+01:00",
          "tree_id": "49efa6e87787da0472d3e7d983ddabb9cb1e806a",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/0cb232350fb79ac421428b91c189ba1e85efcf7a"
        },
        "date": 1795451509139,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5278.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2380,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 215094,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201072\nallocs=704\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 186640.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340448\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 207129,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95024\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2524515.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545456\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23232\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10950.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7807.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 10189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 259166.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3067.9444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1281175,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077560\nallocs=2568\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10739681.5,
            "unit": "ns",
            "extra": "gctime=1908233\nmemory=34286392\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1314392,
            "unit": "ns",
            "extra": "gctime=0\nmemory=979880\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28654189,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301592\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 332499.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=324936\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75064a1c3faa95bbb9b24d504c43b5a935a2b80c",
          "message": "Merge pull request #2 from ComputationalPsychiatry/work/always-check-stopping\n\nCheck the stopping criterion on every timestep, and stop with an error before the first",
          "timestamp": "2026-11-23T18:30:22+01:00",
          "tree_id": "9ad1feb7ed551c501290608bd9d36d73bf3429eb",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/75064a1c3faa95bbb9b24d504c43b5a935a2b80c"
        },
        "date": 1795455535202,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5549.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2422.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 235954,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201184\nallocs=705\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 188043,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 184838.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2604992,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1545536\nallocs=42152\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 35382,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 11564,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6806,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9584,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 272793,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154144\nallocs=4209\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3075.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1425763,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077976\nallocs=2569\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 12375251,
            "unit": "ns",
            "extra": "gctime=3202927\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1250784.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 30054733,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15301976\nallocs=413251\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 260766,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9d94d63af56c81c5580ad885e6c812375c1ea1f0",
          "message": "Merge pull request #4 from ComputationalPsychiatry/work/pointwise-product-type\n\nSplit the log-likelihood of every kind of output for pointwise log-likelihoods",
          "timestamp": "2026-11-23T19:35:18+01:00",
          "tree_id": "61abdaf8d900f5bb7aba177143253f537e159daf",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/9d94d63af56c81c5580ad885e6c812375c1ea1f0"
        },
        "date": 1795459386629,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5301.1875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2383.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 218631,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 183935,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 199034,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2510096,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 35457,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10987.333333333334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7125.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9698,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 258220,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1317375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 13429952,
            "unit": "ns",
            "extra": "gctime=1850586\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1290234,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27929886,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 325371,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "64ac5817998804fa831da768a00f7b1b2a4e9bcc",
          "message": "Merge pull request #5 from ComputationalPsychiatry/work/test-layout\n\nShare the test code as a package, and gather the tests of each extension in test/extensions",
          "timestamp": "2026-11-23T20:34:04+01:00",
          "tree_id": "bbc84e26e9d2b84e575b84262e38d9618b615e23",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/64ac5817998804fa831da768a00f7b1b2a4e9bcc"
        },
        "date": 1795463296102,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5396.375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2380,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 233403,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 189084,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 206177,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2542478,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 11114.166666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7160.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9729,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 265282.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3105.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1359669,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 12347628.5,
            "unit": "ns",
            "extra": "gctime=2778259\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1339566.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28693171,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 330514.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "97b9cd53c71d743eecf9df340df14b2f65df8e03",
          "message": "Merge pull request #6 from ComputationalPsychiatry/work/parameter-names\n\nAdd get_parameter_names, with an option for the parameters without defaults",
          "timestamp": "2026-11-23T21:31:36+01:00",
          "tree_id": "1338f0d6a40396e58acdc3c633c960540fd95cde",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/97b9cd53c71d743eecf9df340df14b2f65df8e03"
        },
        "date": 1795466411897,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4224.111111111111,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1871.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 176024.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 172309.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 145158,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1992307,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27331,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9672,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5284.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8834.625,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 221318,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2384.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1145723,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11737144,
            "unit": "ns",
            "extra": "gctime=2814644\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 972738.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 23924191,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 207452,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "20bffc2d061a64b4224be9eb417306de475c735d",
          "message": "Merge pull request #7 from ComputationalPsychiatry/work/ssm-model-comparison\n\nCompare models with PSIS-LOO in the SequentialSamplingModels tutorial",
          "timestamp": "2026-11-24T00:28:31+01:00",
          "tree_id": "bebea1a79ccd5bb61323b4f2643d86f55d90a4d8",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/20bffc2d061a64b4224be9eb417306de475c735d"
        },
        "date": 1795477041192,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5308.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2384.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 221094,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 187351,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 198412,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2470534.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34645,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 11141,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7029,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 255749,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3088,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1354420,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11879399,
            "unit": "ns",
            "extra": "gctime=2316954.5\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1295613,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28495632,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 329949,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "505aa1b285daab769485f81a0a79331634840682",
          "message": "Merge pull request #8 from ComputationalPsychiatry/work/parameter-getters\n\nGetters that read a model with its submodels, and public bundling of parameters",
          "timestamp": "2026-11-24T09:56:23+01:00",
          "tree_id": "4ce8e395c317479f1a8d5324922564f3a3ccd3dd",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/505aa1b285daab769485f81a0a79331634840682"
        },
        "date": 1795511372092,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3868.9444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2416.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 170400,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 145275.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 132334,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2093976,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 25352.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9048.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5063,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8561,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 211934,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2208.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1146293,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7810025,
            "unit": "ns",
            "extra": "gctime=2428897\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1211617.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 24496674,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 267838.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9ed6685c19e7037cf02bfd563a7ad2fbbbd25c0",
          "message": "Merge pull request #9 from ComputationalPsychiatry/work/save-submodel-states\n\nSelect some of the states of a submodel in a SaveHistory",
          "timestamp": "2026-11-24T10:19:54+01:00",
          "tree_id": "358dcfaaa7d827cb41dd58c79cd62177e2c3f356",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/c9ed6685c19e7037cf02bfd563a7ad2fbbbd25c0"
        },
        "date": 1795513023402,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5298.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2387.277777777778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 218761.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 195618,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 200297,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2455334.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34735,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10947,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7171,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9658,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 251126.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3110.6666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1313315.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10953986.5,
            "unit": "ns",
            "extra": "gctime=1773702.5\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1282382,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27550059,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 323128,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a6ea0a9f5fed0fd189c7e6a5310872bfecc10dc3",
          "message": "Merge pull request #10 from ComputationalPsychiatry/work/save-trajectories\n\nRename SaveHistory to SaveTrajectories, with the keywords and fields that follow",
          "timestamp": "2026-11-24T10:30:39+01:00",
          "tree_id": "ccc1fbaa5ea6d09de8d69879e66333247bd6df35",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/a6ea0a9f5fed0fd189c7e6a5310872bfecc10dc3"
        },
        "date": 1795513681438,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5237.3125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2375.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 223392.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 186358,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 199794,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2481599,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34455,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10853.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6943,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9788,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 255512.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3096.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1375443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 12400889.5,
            "unit": "ns",
            "extra": "gctime=2877365.5\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1309629,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28811501,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 328494,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7772e6ee4679ed0a950687e10e889cb419beba06",
          "message": "Merge pull request #13 from ComputationalPsychiatry/work/stochastic-timestep-models\n\nAdd is_stochastic, and an error from Turing.jl's default timestep for a model that samples without a Turing version",
          "timestamp": "2026-11-24T11:42:51+01:00",
          "tree_id": "c8b6c78f85e8e608d49633d82cfe186a57cf50a3",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/7772e6ee4679ed0a950687e10e889cb419beba06"
        },
        "date": 1795517587334,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3285.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2075.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 150920.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 105683,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 113142,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1803564,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 22323.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7916.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4508.285714285715,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 7314.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 182466,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1909.2,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1075304.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7244594,
            "unit": "ns",
            "extra": "gctime=2342017\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1082267,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 20421265.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 233476,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2e423c5ba803bbded419a9f699daf9471d00ec14",
          "message": "Merge pull request #14 from ComputationalPsychiatry/work/roadmap-issue\n\nKeep the plan of work in the roadmap issue instead of a docs page",
          "timestamp": "2026-11-24T11:42:57+01:00",
          "tree_id": "29daafc90b98164e58a5018157720e27959b1587",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/2e423c5ba803bbded419a9f699daf9471d00ec14"
        },
        "date": 1795518333920,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4326.444444444444,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1868.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 172744,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 170842,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 147317,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1972767,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27741,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9671.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5234.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8733,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 207901,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2411.05,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1051559,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 9850191,
            "unit": "ns",
            "extra": "gctime=1983983\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 997469,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 22735170.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 207570.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e04709545f999609efd77b7bb01dbe758b064671",
          "message": "Merge pull request #15 from ComputationalPsychiatry/work/submodel-test-cases\n\nModels with submodels in the shared test cases, and a tutorial on them",
          "timestamp": "2026-11-24T12:08:29+01:00",
          "tree_id": "9119fb39ab7c4b2038f57ae520d96972fb9f8445",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/e04709545f999609efd77b7bb01dbe758b064671"
        },
        "date": 1795519344461,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5331.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2377.777777777778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 216565,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 184565,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 197409,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2484099,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34475,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6816.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9578,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 257321,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3250.5555555555557,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1306443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11670817.5,
            "unit": "ns",
            "extra": "gctime=2309204\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1272129.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28264586,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 324607,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "802dc503d5938402fb55e24b414dbb40aac6380b",
          "message": "Merge pull request #16 from ComputationalPsychiatry/work/premade-in-repository\n\nMove PremadeCognitiveModels.jl into this repository, as a package of its own",
          "timestamp": "2026-11-24T13:09:40+01:00",
          "tree_id": "6686495350a696eb1465370846f8dff80ba535d0",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/802dc503d5938402fb55e24b414dbb40aac6380b"
        },
        "date": 1795523020966,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4589.6875,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2757.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 218642,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 165214,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 176349,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2484971.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 30973.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10415.833333333332,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5856.916666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9628.666666666668,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 245375,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2725.3333333333335,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1670785,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 13349109,
            "unit": "ns",
            "extra": "gctime=4327480.5\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1387262,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 28094415.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 309267,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "34dce217b42e5de71a430e691e9c4b8fb09e2220",
          "message": "Merge pull request #17 from ComputationalPsychiatry/work/full-model-of-one-session\n\nFit a single session with full_model, given its SessionInfo and then the parameter model",
          "timestamp": "2026-11-24T14:22:04+01:00",
          "tree_id": "e93f448e73a077a0bca53314615ab3afa07c59db",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/34dce217b42e5de71a430e691e9c4b8fb09e2220"
        },
        "date": 1795527181925,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3286.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2089.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 151695.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 106671,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 114756.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1815174.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 21838,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7826.666666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4186.125,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 7162.299999999999,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 180654.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1930.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1118288.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7748628,
            "unit": "ns",
            "extra": "gctime=2280697.5\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1098063,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 20599100.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 231258,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dbe4546fe12ecdd0e9605e5b0c7767743821e10b",
          "message": "Merge pull request #18 from ComputationalPsychiatry/work/premade-docs-subfolder\n\nPlaceholder documentation of PremadeCognitiveModels in a folder of the repository's site",
          "timestamp": "2026-11-24T15:03:07+01:00",
          "tree_id": "5db811a24e816d35836fb8db87dc518b26b62307",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/dbe4546fe12ecdd0e9605e5b0c7767743821e10b"
        },
        "date": 1795529790587,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5224.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2377.777777777778,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 215472,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 187030,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 197970,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2464810,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34404,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 10960.666666666666,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 7133.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9638,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 250643,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3067.8888888888887,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1299277,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10878978.5,
            "unit": "ns",
            "extra": "gctime=1745139\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1277320,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1046640\nallocs=3988\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 27473761.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 326400,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6793eb16cd9a567ac4379e564cbc877d6c597943",
          "message": "Merge pull request #19 from ComputationalPsychiatry/work/full-model-sessions-first\n\nfull_model takes the session data before the parameter model, for several sessions too",
          "timestamp": "2026-11-24T15:47:18+01:00",
          "tree_id": "20d82ae85ddbca3e0ecb56b99bb46f4f346e42cf",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/6793eb16cd9a567ac4379e564cbc877d6c597943"
        },
        "date": 1795532302340,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 3369.9444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2104.1111111111113,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 157458,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 112270.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 115675.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1862274.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 22410,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 7931.166666666667,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 4158.142857142857,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":7,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 7247.8,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 177724,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 1949.1,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1229620,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 7663964,
            "unit": "ns",
            "extra": "gctime=2581311\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1167098,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1046640\nallocs=3988\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 21018882,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 237278.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "59eef6af510982d46f817aa44bf58b8cabca1bcd",
          "message": "Merge pull request #20 from ComputationalPsychiatry/work/full-model-internals-sessions-first\n\nGive prepare_full_model and turing_full_model the session data first",
          "timestamp": "2026-11-24T19:45:39+01:00",
          "tree_id": "236f46bf1ce4a0d7cd98ef9d856e5208af696a62",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/59eef6af510982d46f817aa44bf58b8cabca1bcd"
        },
        "date": 1795546759182,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 5473.25,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":8,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 2424.1666666666665,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 215755,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 191904.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 186291,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 2555029,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 34682,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 11701,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":3,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 6566,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":5,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 9735,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 271120.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 3092.4444444444443,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1351840,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 10525975,
            "unit": "ns",
            "extra": "gctime=2001401\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1244442,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 29047589,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 259461,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pthestrup@ethz.ch",
            "name": "Peter Thestrup Waade",
            "username": "PTWaade"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9ad7768a18d1cfe9aec1100cb6a1145eb6e138f9",
          "message": "Merge pull request #21 from ComputationalPsychiatry/work/premade-checks-required\n\nMake failures of the premade models fail the build",
          "timestamp": "2026-11-24T22:44:30+01:00",
          "tree_id": "943c166362cac3b573ab55c10e2c9eaf86d5ce34",
          "url": "https://github.com/ComputationalPsychiatry/CognitiveModeling.jl/commit/9ad7768a18d1cfe9aec1100cb6a1145eb6e138f9"
        },
        "date": 1795557098497,
        "tool": "julia",
        "benches": [
          {
            "name": "simulate/saving everything",
            "value": 4443.222222222223,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5024\nallocs=16\nparams={\"evals\":9,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "simulate/saving nothing",
            "value": 1871.7,
            "unit": "ns",
            "extra": "gctime=0\nmemory=512\nallocs=16\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Enzyme",
            "value": 183650.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=201024\nallocs=695\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ForwardDiff",
            "value": 171923,
            "unit": "ns",
            "extra": "gctime=0\nmemory=340608\nallocs=392\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, Mooncake",
            "value": 146636,
            "unit": "ns",
            "extra": "gctime=0\nmemory=95264\nallocs=981\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/gradient, ReverseDiff",
            "value": 1985052,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1547616\nallocs=42212\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/full model, independent sessions/log density",
            "value": 27555.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=23312\nallocs=175\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Enzyme",
            "value": 9774.5,
            "unit": "ns",
            "extra": "gctime=0\nmemory=16688\nallocs=29\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ForwardDiff",
            "value": 5302.833333333333,
            "unit": "ns",
            "extra": "gctime=0\nmemory=4904\nallocs=16\nparams={\"evals\":6,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, Mooncake",
            "value": 8747.75,
            "unit": "ns",
            "extra": "gctime=0\nmemory=5840\nallocs=48\nparams={\"evals\":4,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/gradient, ReverseDiff",
            "value": 205097,
            "unit": "ns",
            "extra": "gctime=0\nmemory=154352\nallocs=4215\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/single session/log density",
            "value": 2425.6,
            "unit": "ns",
            "extra": "gctime=0\nmemory=1904\nallocs=13\nparams={\"evals\":10,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Enzyme",
            "value": 1145639,
            "unit": "ns",
            "extra": "gctime=0\nmemory=2077176\nallocs=2519\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ForwardDiff",
            "value": 11871316,
            "unit": "ns",
            "extra": "gctime=2982976\nmemory=34289848\nallocs=8984\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, Mooncake",
            "value": 1001937,
            "unit": "ns",
            "extra": "gctime=0\nmemory=981064\nallocs=3986\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/gradient, ReverseDiff",
            "value": 24075059,
            "unit": "ns",
            "extra": "gctime=0\nmemory=15312376\nallocs=413551\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          },
          {
            "name": "turing/study of 50 sessions of 200 timesteps, parameters by group/log density",
            "value": 210801,
            "unit": "ns",
            "extra": "gctime=0\nmemory=325320\nallocs=888\nparams={\"evals\":1,\"evals_set\":false,\"gcsample\":false,\"gctrial\":true,\"memory_tolerance\":0.01,\"overhead\":0,\"samples\":10000,\"seconds\":5,\"time_tolerance\":0.05}"
          }
        ]
      }
    ]
  }
}