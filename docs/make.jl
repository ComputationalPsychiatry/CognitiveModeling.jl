using CognitiveModeling
using Documenter
using Literate

# Turn the scripts in docs/literate/ into markdown pages in docs/src/generated/
literate_dir = joinpath(@__DIR__, "literate")
generated_dir = joinpath(@__DIR__, "src", "generated")
for file in readdir(literate_dir; join = true)
    Literate.markdown(file, generated_dir)
end

DocMeta.setdocmeta!(CognitiveModeling, :DocTestSetup, :(using CognitiveModeling); recursive = true)

makedocs(;
    modules = [CognitiveModeling],
    authors = "Peter Thestrup Waade <peter@waade.net>",
    sitename = "CognitiveModeling.jl",
    format = Documenter.HTML(;
        canonical = "https://ComputationalPsychiatry.github.io/CognitiveModeling.jl",
        edit_link = "dev",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => ["generated/getting_started.md", "generated/cognitive_abm_with_agents.md"],
        "Automatic differentiation and Reactant" => "ad_and_reactant.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/ComputationalPsychiatry/CognitiveModeling.jl",
    devbranch = "dev",
    push_preview = true,
)
