## Displaying save histories ##

#Shows the selections on one line each in the REPL and notebooks, and whether the initial timestep values are saved
function Base.show(io::IO, ::MIME"text/plain", save_history::SaveHistory)
    print(io, "SaveHistory")
    for (kind, selection) in (
        ("States", save_history.states),
        ("Inputs", save_history.inputs),
        ("Outputs", save_history.outputs),
        ("Output distributions", save_history.output_distributions),
    )
        print(io, "\n", kind, ": ")
        if selection isa SaveEverything
            print(io, "everything")
        elseif isempty(selection)
            print(io, "none")
        else
            join(io, selection, ", ")
        end
    end
    print(
        io,
        "\nInitial timestep values: ",
        save_history.save_initial_timestep_values ? "saved" : "not saved",
    )
    return nothing
end
