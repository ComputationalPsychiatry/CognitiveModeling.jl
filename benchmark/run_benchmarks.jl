# Runs the benchmarks in benchmarks.jl and writes the median of each to results.json in this folder, which the Benchmarks
# workflow adds to the history on the gh-pages branch. Run it with
#     julia --project=benchmark benchmark/run_benchmarks.jl
using BenchmarkTools: BenchmarkTools, median, tune!

include("benchmarks.jl")

tune!(SUITE)
results = run(SUITE; verbose=true)
display(median(results))
BenchmarkTools.save(joinpath(@__DIR__, "results.json"), median(results))
