using PremadeCognitiveModels
using Documenter

DocMeta.setdocmeta!(
    PremadeCognitiveModels, :DocTestSetup, :(using PremadeCognitiveModels); recursive=true
)

makedocs(;
    modules=[PremadeCognitiveModels],
    authors="Peter Thestrup Waade <peter@waade.net>",
    sitename="PremadeCognitiveModels.jl",
    format=Documenter.HTML(;
        canonical="https://ComputationalPsychiatry.github.io/PremadeCognitiveModels.jl",
        edit_link="dev",
        assets=String[],
    ),
    pages=["Home" => "index.md", "API" => "api.md"],
)

deploydocs(;
    repo="github.com/ComputationalPsychiatry/PremadeCognitiveModels.jl",
    devbranch="dev",
    push_preview=true,
)
