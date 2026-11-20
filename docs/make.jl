using CognitiveModeling
using Documenter
using Literate
using TOML

# How each script of the documentation is run, from docs/how_scripts_are_run.toml: whether the
# build runs a tutorial or only shows its code, and when the tests run it. A tutorial whose code
# is only shown uses packages too large for the documentation's environment, takes long to run,
# or is a sketch of what isn't built yet
how_scripts_are_run = TOML.parsefile(joinpath(@__DIR__, "how_scripts_are_run.toml"))

# Every script in docs/literate and docs/examples must have an entry, and every entry a script, so
# that a new script is given its settings when it is added
for (kind, folder) in (("tutorials", "literate"), ("examples", "examples"))
    script_names = sort(filter(endswith(".jl"), readdir(joinpath(@__DIR__, folder))))
    listed_names = sort(collect(keys(how_scripts_are_run[kind])))
    if script_names != listed_names
        error(
            "The $kind in docs/how_scripts_are_run.toml aren't the scripts in docs/$folder. Without an entry: $(setdiff(script_names, listed_names)). Without a script: $(setdiff(listed_names, script_names)).",
        )
    end
end

# Turn the scripts in docs/literate/ into markdown pages in docs/src/generated/
literate_dir = joinpath(@__DIR__, "literate")
generated_dir = joinpath(@__DIR__, "src", "generated")
for file in readdir(literate_dir; join=true)
    if how_scripts_are_run["tutorials"][basename(file)]["run_in_build"]
        Literate.markdown(file, generated_dir)
    else
        Literate.markdown(file, generated_dir; codefence="```julia" => "```")
    end
end

DocMeta.setdocmeta!(
    CognitiveModeling, :DocTestSetup, :(using CognitiveModeling); recursive=true
)

makedocs(;
    modules=[CognitiveModeling],
    authors="Peter Thestrup Waade <peter@waade.net>",
    sitename="CognitiveModeling.jl",
    format=Documenter.HTML(;
        canonical="https://ComputationalPsychiatry.github.io/CognitiveModeling.jl",
        edit_link="dev",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "generated/getting_started.md",
            "generated/custom_turing_models.md",
            "generated/discrete_parameters.md",
            "generated/cognitive_abm_with_agents.md",
            "generated/interfacing_with_common_rl.md",
            "generated/interfacing_with_reinforcement_learning.md",
            "generated/interfacing_with_pomdps.md",
            "generated/interfacing_with_juliacontrol.md",
            "generated/interfacing_with_actr_models.md",
            "generated/interfacing_with_sequential_sampling_models.md",
            "generated/compiling_with_reactant.md",
        ],
        "Planned tutorials and interfaces" => "planned.md",
        "Automatic differentiation and Reactant" => "ad_and_reactant.md",
        "API" => [
            "Overview" => "api/index.md",
            "api/defining_models.md",
            "api/simulation.md",
            "api/saving_histories.md",
            "api/fitting.md",
            "api/parameter_models.md",
            "api/fitting_with_turing.md",
        ],
        "Internals" => [
            "Overview" => "internals/index.md",
            "Defining models" => "internals/defining_models.md",
            "Simulation" => "internals/simulation.md",
            "Saving histories" => "internals/saving_histories.md",
            "Fitting" => [
                "Sessions" => "internals/fitting.md",
                "Parameter models" => "internals/parameter_models.md",
                "Preparing models" => "internals/preparing_models.md",
                "Within models" => "internals/within_models.md",
            ],
        ],
    ],
)

deploydocs(;
    repo="github.com/ComputationalPsychiatry/CognitiveModeling.jl",
    devbranch="dev",
    push_preview=true,
)
