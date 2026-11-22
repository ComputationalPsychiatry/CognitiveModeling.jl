### NAMING THE SESSIONS ###

#Chosen by the types of the sessions' data, so that the name has one type. Sessions whose data is simulated are named apart from fitted ones, so that Turing.jl's predict draws every variable within them again, whatever a chain of a fit to the sessions holds
"""
    name_of_session(sessions_info, i)

Return the prefix of the variables of session `i` of `sessions_info`, a
`MultipleSessionsInfo`: `simulated_sessions[i]` when the model simulates the data of the
sessions, as [`simulates_values`](@ref) says, as for the sessions that
`decondition_sessions` gives, and `sessions[i]` otherwise. Used by
[`turing_multiple_behavioral_sessions`](@ref) and `condition_sessions`.
"""
function name_of_session(sessions_info::MultipleSessionsInfo, i::Integer)
    return name_of_session(Val(simulates_values(sessions_info)), i)
end
name_of_session(::Val{true}, i::Integer) = @varname(simulated_sessions[i])
name_of_session(::Val{false}, i::Integer) = @varname(sessions[i])

### COLLECTING WHAT SESSIONS THAT RUN IN PARALLEL SAVED ###

"""
    include_other_types(saved_trajectories, saved_with_other_types)

Return `saved_trajectories`, what each session saved, with the trajectories in
`saved_with_other_types` put in their places, widening the vector as [`store_value!`](@ref)
does. A session whose saved trajectories didn't fit the type of `saved_trajectories` has
them in its place in `saved_with_other_types`, and is left unassigned in
`saved_trajectories`. When there are none, `saved_trajectories` is returned as it is. Used
by [`turing_multiple_behavioral_sessions`](@ref) with [`ParallelSessions`](@ref), after the
threads are done.
"""
function include_other_types(
    saved_trajectories::Vector, saved_with_other_types::Vector{Any}
)
    #In the common case, every session saved trajectories of one type
    if !any(i -> isassigned(saved_with_other_types, i), eachindex(saved_with_other_types))
        return saved_trajectories
    end

    #Otherwise, store each session's trajectories again, in order, so that the vector widens as it goes
    widened = initialize_storage(saved_trajectories[1], length(saved_trajectories))
    for i in 2:length(saved_trajectories)
        saved = if isassigned(saved_with_other_types, i)
            saved_with_other_types[i]
        else
            saved_trajectories[i]
        end
        widened = store_value!(widened, saved, i)
    end
    return widened
end
