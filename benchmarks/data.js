window.BENCHMARK_DATA = {
  "lastUpdate": 1794930783298,
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
      }
    ]
  }
}