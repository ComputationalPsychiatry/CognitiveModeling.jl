# Runs the benchmarks in benchmarks.jl and writes the median of each to results.json in this folder, which the Benchmarks
# workflow adds to the history on the gh-pages branch. Run it with
#     julia --project=benchmark benchmark/run_benchmarks.jl
using BenchmarkTools: BenchmarkTools, median, tune!
using InteractiveUtils: versioninfo

include("benchmarks.jl")

# The machine and its processor, since GitHub's runners come with processors of different speeds, so that the timings of
# two runs can differ with no change to the code
versioninfo()

tune!(SUITE)
results = run(SUITE; verbose=true)
display(median(results))
BenchmarkTools.save(joinpath(@__DIR__, "results.json"), median(results))
