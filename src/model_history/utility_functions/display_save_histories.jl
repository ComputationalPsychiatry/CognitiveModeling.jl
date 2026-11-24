## Displaying save histories ##

"""
    selected_names_to_show(selection::Tuple)

Return the names that a [`SaveHistory`](@ref) selects of one kind, as they are shown: each
name as it is, and each state of a submodel with the name of the submodel and a dot before
it, as in `perception.expected_reward`, down through its submodels. Used by the display of
a `SaveHistory`.
"""
function selected_names_to_show(selection::Tuple)
    shown = String[]
    for entry in selection
        if entry isa Symbol
            push!(shown, string(entry))
        else
            for name in selected_names_to_show(last(entry))
                push!(shown, string(first(entry), ".", name))
            end
        end
    end
    return shown
end

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
            join(io, selected_names_to_show(selection), ", ")
        end
    end
    print(
        io,
        "\nInitial timestep values: ",
        save_history.save_initial_timestep_values ? "saved" : "not saved",
    )
    return nothing
end
