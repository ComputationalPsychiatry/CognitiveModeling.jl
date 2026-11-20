#How each script of the documentation is run, from docs/how_scripts_are_run.toml, which says what each setting means,
#and the functions that the runners in test/documentation share, which include this file
using TOML: TOML

const DOCS_DIRECTORY = joinpath(@__DIR__, "..", "..", "docs")
const HOW_SCRIPTS_ARE_RUN = TOML.parsefile(
    joinpath(DOCS_DIRECTORY, "how_scripts_are_run.toml")
)

#The tier of tests that a runner is run for: fast when it is run without arguments, or the one argument it is given, as
#in julia --project=docs test/documentation/tutorials/runtests.jl long
function tier_to_test(arguments)
    tier = isempty(arguments) ? "fast" : only(arguments)
    if !(tier in ("fast", "long"))
        throw(
            ArgumentError(
                "The tier of the tests is fast or long, but $(repr(tier)) was given."
            ),
        )
    end
    return tier
end

#Check that the tier of a script with an environment of its own is the one that its runner is run for, so that a runner
#in the workflow of another tier stops with an error which says where it belongs
function check_tier(kind, name, arguments)
    tier = HOW_SCRIPTS_ARE_RUN[kind][name]["test_tier"]
    if tier != tier_to_test(arguments)
        error(
            "The test tier of $name in docs/how_scripts_are_run.toml is $tier, but its tests were run for the $(tier_to_test(arguments)) tier. Run them in the workflow of their tier: .github/workflows/CI.yml for fast, and .github/workflows/LongTests.yml for long.",
        )
    end
    return nothing
end

#Run a script of the documentation to the end in a module of its own, and then the checks of its results in the same
#module, so that they read what the script made, and the scripts of one runner don't share their names
function run_script_and_check_results(script_path, results_path)
    script = Module(Symbol(splitext(basename(script_path))[1]))
    Core.eval(script, :(using Test))
    Base.include(script, script_path)
    #It ran to the end
    @test true
    Base.include(script, results_path)
    return nothing
end
