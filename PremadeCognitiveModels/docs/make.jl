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
        canonical="https://ComputationalPsychiatry.github.io/CognitiveModeling.jl/PremadeCognitiveModels",
        edit_link="dev",
        assets=String[],
    ),
    pages=["Home" => "index.md", "API" => "api.md"],
)

# The documentation goes to the site of the repository, in a folder of its own beside that of
# the core, with its own versions from the tags of the package's releases, such as
# PremadeCognitiveModels-v0.1.0
deploydocs(;
    repo="github.com/ComputationalPsychiatry/CognitiveModeling.jl",
    dirname="PremadeCognitiveModels",
    tag_prefix="PremadeCognitiveModels-",
    devbranch="dev",
)
