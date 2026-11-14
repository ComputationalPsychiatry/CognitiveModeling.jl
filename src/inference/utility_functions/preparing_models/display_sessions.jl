## Displaying sessions ##

"""
    print_model_name(io, model)

Print the name of the type of `model`, or `none` for [`NoEnvironment`](@ref). Used by the
`show` methods of a [`SessionInfo`](@ref) and a [`MultipleSessionsInfo`](@ref).
"""
print_model_name(io::IO, model::AbstractTimestepModel) = print(io, nameof(typeof(model)))
print_model_name(io::IO, ::NoEnvironment) = print(io, "none")

"""
    print_data_names(io, kind, names)

Print a line with `kind` and `names`, the names of one kind of value in the data, or `none`
when there are none. Used by the `show` methods of a [`SessionInfo`](@ref) and a
[`MultipleSessionsInfo`](@ref).
"""
function print_data_names(io::IO, kind, names)
    print(io, "\n", kind, ": ")
    isempty(names) ? print(io, "none") : join(io, names, ", ")
    return nothing
end

#Shows a summary of a session in the REPL and notebooks, since the full type and the data would fill the screen
function Base.show(io::IO, ::MIME"text/plain", session_info::SessionInfo)
    print(io, "SessionInfo\nCognitive model: ")
    print_model_name(io, session_info.cognitive_model)
    print(io, "\nEnvironment: ")
    print_model_name(io, session_info.environment)
    print(io, "\nTimesteps: ", session_info.n_timesteps)
    print_data_names(io, "Observations", keys(session_info.observations))
    print_data_names(io, "Actions", keys(session_info.actions))
    print(
        io,
        "\nWho starts: ",
        session_info.who_starts isa AgentStarts ? "agent" : "environment",
    )
    return nothing
end

#Shows a summary of several sessions, with the models and the names of the first
function Base.show(io::IO, ::MIME"text/plain", sessions_info::MultipleSessionsInfo)
    first_session = first(sessions_info.sessions)
    print(
        io,
        "MultipleSessionsInfo\nSessions: ",
        n_sessions(sessions_info),
        "\nCognitive model: ",
    )
    print_model_name(io, first_session.cognitive_model)
    print(io, "\nEnvironment: ")
    print_model_name(io, first_session.environment)
    print_data_names(io, "Observations", keys(first_session.observations))
    print_data_names(io, "Actions", keys(first_session.actions))
    return nothing
end
