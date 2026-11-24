## Displaying the trajectories of a session ##

"""
    saved_names_to_show(saved::NamedTuple, submodel_states::NamedTuple)

Return the names of what was saved of one kind on a timestep, `saved`, as they are shown:
each name as it is, and for a state that holds the state of a submodel, which
`submodel_states` names with those of its own submodels, as [`SessionTrajectories`](@ref)
keeps them, the names of what was saved of it instead, each with the name of the submodel
and a dot before it, as in `perception.expected_reward`, down through its submodels. Used
by the display of a `SessionTrajectories`.
"""
function saved_names_to_show(saved::NamedTuple, submodel_states::NamedTuple)
    shown = String[]
    for name in keys(saved)
        value = getfield(saved, name)
        if haskey(submodel_states, name) && value isa NamedTuple
            for name_within in saved_names_to_show(value, getfield(submodel_states, name))
                push!(shown, string(name, ".", name_within))
            end
        else
            push!(shown, string(name))
        end
    end
    return shown
end

#Shows a summary of the trajectories in the REPL and notebooks, since the vectors would fill the screen: the number of timesteps, the names of what was saved of each kind, and whether the parameters stayed valid
function Base.show(io::IO, ::MIME"text/plain", trajectories::SessionTrajectories)

    #Print the number of timesteps, and that timestep 0 comes before them when the vectors have an element for it
    n_timesteps = trajectories.n_timesteps
    print(io, "SessionTrajectories\nTimesteps: ", n_timesteps)
    if n_timesteps isa Integer &&
        trajectories.states isa AbstractVector &&
        length(trajectories.states) == n_timesteps + 1
        print(io, ", after timestep 0")
    end

    #Print the names of what was saved of each kind, for the cognitive model and for an environment when there is one,
    #with the states of a submodel under its name
    submodel_states = trajectories.submodel_states
    kinds = if trajectories.environment isa NamedTuple
        environment = trajectories.environment
        (
            ("States", trajectories.states, submodel_states.cognitive_model),
            ("Observations", trajectories.observations, (;)),
            ("Actions", trajectories.actions, (;)),
            ("Action distributions", trajectories.action_distributions, (;)),
            ("Environment states", environment.states, submodel_states.environment),
            ("Environment observations", environment.observations, (;)),
            ("Environment actions", environment.actions, (;)),
            (
                "Environment observation distributions",
                environment.observation_distributions,
                (;),
            ),
        )
    else
        (
            ("States", trajectories.states, submodel_states.cognitive_model),
            ("Observations", trajectories.observations, (;)),
            ("Actions", trajectories.actions, (;)),
            ("Action distributions", trajectories.action_distributions, (;)),
        )
    end
    for (kind, trajectory, submodel_states_of_kind) in kinds
        print(io, "\n", kind, ": ")
        if trajectory isa AbstractVector &&
            !isempty(trajectory) &&
            first(trajectory) isa NamedTuple &&
            !isempty(first(trajectory))
            join(io, saved_names_to_show(first(trajectory), submodel_states_of_kind), ", ")
        else
            print(io, "none")
        end
    end
    trajectories.environment isa NamedTuple || print(io, "\nEnvironment: none")

    #Print whether the parameters stayed valid, and where the timestep that turned invalid is
    parameters_valid = parameters_are_valid(trajectories.parameters_valid)
    print(io, "\nParameters valid: ", parameters_valid)
    if parameters_valid === false && trajectories.last_timestep !== nothing
        print(io, " (see last_timestep)")
    end

    return nothing
end

#Shows the number of timesteps on one line, as inside a vector
function Base.show(io::IO, trajectories::SessionTrajectories)
    n_timesteps = trajectories.n_timesteps
    print(
        io,
        "SessionTrajectories (",
        n_timesteps,
        n_timesteps == 1 ? " timestep)" : " timesteps)",
    )
    return nothing
end
