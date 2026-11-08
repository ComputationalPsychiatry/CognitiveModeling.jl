using CognitiveModeling
using Documenter
using Literate

# Tutorials whose code is shown without being run when the documentation is built, because
# the packages they use are too large for the documentation's environment
const NOT_RUN = ["interfacing_with_reinforcement_learning.jl", "interfacing_with_pomdps.jl"]

# Turn the scripts in docs/literate/ into markdown pages in docs/src/generated/
literate_dir = joinpath(@__DIR__, "literate")
generated_dir = joinpath(@__DIR__, "src", "generated")
for file in readdir(literate_dir; join=true)
    if basename(file) in NOT_RUN
        Literate.markdown(file, generated_dir; codefence="```julia" => "```")
    else
        Literate.markdown(file, generated_dir)
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
            "generated/cognitive_abm_with_agents.md",
            "generated/interfacing_with_common_rl.md",
            "generated/interfacing_with_reinforcement_learning.md",
            "generated/interfacing_with_pomdps.md",
            "generated/interfacing_with_actr_models.md",
            "generated/interfacing_with_sequential_sampling_models.md",
            "generated/compiling_with_reactant.md",
        ],
        "Planned tutorials and interfaces" => "planned.md",
        "Automatic differentiation and Reactant" => "ad_and_reactant.md",
        "API" => "api.md",
        "Internals" => "internals.md",
    ],
)

deploydocs(;
    repo="github.com/ComputationalPsychiatry/CognitiveModeling.jl",
    devbranch="dev",
    push_preview=true,
)
