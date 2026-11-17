### COLLECTING WHAT SESSIONS THAT RUN IN PARALLEL SAVED ###

"""
    include_other_types(saved_trajectories, saved_with_other_types)

Return `saved_trajectories`, what each session saved, with the trajectories in
`saved_with_other_types` put in their places, widening the vector as [`store_value!`](@ref)
does. A session whose saved trajectories didn't fit the type of `saved_trajectories` has
them in its place in `saved_with_other_types`, and is left unassigned in
`saved_trajectories`. When there are none, `saved_trajectories` is returned as it is. Used
by [`turing_run_sessions`](@ref) with [`ParallelSessions`](@ref), after the threads are
done.
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
