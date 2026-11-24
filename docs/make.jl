using CognitiveModeling
using Documenter
using Literate
using TOML

# How each script of the documentation is run, from docs/how_scripts_are_run.toml, which says
# what each setting means: whether every build runs a tutorial, only the builds of a release, or
# none, and when the tests run it
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

# Every tutorial must say how the build runs it with one of these values
for (name, settings) in how_scripts_are_run["tutorials"]
    if !(settings["run_in_build"] in ("always", "for_releases", "never"))
        error(
            "The tutorial $name has run_in_build = $(repr(settings["run_in_build"])) in docs/how_scripts_are_run.toml, but it must be \"always\", \"for_releases\" or \"never\".",
        )
    end
end

# Whether this is a build of a release, which runs the tutorials that only releases run. The
# documentation's workflow, .github/workflows/Docs.yml, sets DOCUMENTATION_FOR_A_RELEASE to true
# for a tag and for a pull request into main. Other builds, such as one on one's own machine,
# show only the code of those tutorials, unless it is set, as in
#     DOCUMENTATION_FOR_A_RELEASE=true julia --project=docs docs/make.jl
for_a_release = get(ENV, "DOCUMENTATION_FOR_A_RELEASE", "false") == "true"

# The note on a tutorial which only the builds of a release run, when this build shows only its
# code. It goes after the title, the first line of the script that is a heading
function with_note_on_releases(content)
    note = """
    # !!! note "Only the code"
    #     This tutorial takes long to run, so only the documentation of a release runs it and
    #     shows its output. This version of the documentation shows its code.
    """
    lines = split(content, "\n")
    title = findfirst(line -> startswith(line, "# # "), lines)
    title === nothing && return note * "\n" * content
    return join(
        [lines[1:title]; ""; split(chomp(note), "\n"); lines[(title + 1):end]], "\n"
    )
end

# Turn the scripts in docs/literate/ into markdown pages in docs/src/generated/: a tutorial that
# this build runs with its code in blocks that Documenter runs, and one that it doesn't in plain
# blocks of Julia code
literate_dir = joinpath(@__DIR__, "literate")
generated_dir = joinpath(@__DIR__, "src", "generated")
for file in readdir(literate_dir; join=true)
    run_in_build = how_scripts_are_run["tutorials"][basename(file)]["run_in_build"]
    if run_in_build == "always" || (run_in_build == "for_releases" && for_a_release)
        Literate.markdown(file, generated_dir)
    elseif run_in_build == "for_releases"
        Literate.markdown(
            file,
            generated_dir;
            codefence="```julia" => "```",
            preprocess=with_note_on_releases,
        )
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
            "generated/predictive_checks_and_parameter_recovery.md",
            "generated/discrete_parameters.md",
            "generated/models_with_submodels.md",
            "generated/cognitive_abm_with_agents.md",
            "generated/interfacing_with_common_rl.md",
            "generated/interfacing_with_reinforcement_learning.md",
            "generated/interfacing_with_pomdps.md",
            "generated/interfacing_with_juliacontrol.md",
            "generated/interfacing_with_actr_models.md",
            "generated/interfacing_with_sequential_sampling_models.md",
            "generated/compiling_with_reactant.md",
        ],
        "Automatic differentiation and Reactant" => "ad_and_reactant.md",
        "API" => [
            "Overview" => "api/index.md",
            "api/defining_models.md",
            "api/simulation.md",
            "api/saving_trajectories.md",
            "api/fitting.md",
            "api/parameter_models.md",
            "api/fitting_with_turing.md",
        ],
        "Internals" => [
            "Overview" => "internals/index.md",
            "Defining models" => "internals/defining_models.md",
            "Simulation" => "internals/simulation.md",
            "Saving trajectories" => "internals/saving_trajectories.md",
            "Fitting" => [
                "Sessions" => "internals/fitting.md",
                "Parameter models" => "internals/parameter_models.md",
                "Preparing models" => "internals/preparing_models.md",
                "Within models" => "internals/within_models.md",
                "Workflow and postprocessing" => "internals/workflow_and_postprocessing.md",
            ],
        ],
    ],
)

deploydocs(;
    repo="github.com/ComputationalPsychiatry/CognitiveModeling.jl",
    devbranch="dev",
    push_preview=true,
)
