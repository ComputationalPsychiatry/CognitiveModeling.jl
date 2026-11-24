## The sizes of the timesteps that are saved ##

"""
    each_timestep_size(timestep_sizes, n_steps)

Return the size of each of `n_steps` timesteps as a vector, for the trajectories of an
instantiated model. A vector of sizes is returned as it is, and anything else is repeated
for every timestep. Used by [`simulate_session!`](@ref), for the sizes it adds to the
trajectories of the instantiated models, and by [`saved_timestep_sizes`](@ref).
"""
function each_timestep_size(timestep_sizes::AbstractVector, n_steps)
    return timestep_sizes
end

function each_timestep_size(timestep_size, n_steps)
    return fill(timestep_size, n_steps)
end

"""
    returned_timestep_sizes(timestep_sizes, n_timesteps, save_initial_timestep_values::Val)

Return the sizes of the `n_timesteps` timesteps that were simulated, as a simulation returns
them. A vector of sizes is cut to those timesteps, since a model can stop a simulation
early, and starts with [`BeforeFirstTimestep`](@ref) for timestep 0 when
`save_initial_timestep_values` is `Val(true)`, so that it has an element for each element of
what is saved. Anything else is returned as it is, since it is the size of every timestep.
Used by [`simulate_session`](@ref), [`simulate_session!`](@ref) and
[`saved_timestep_sizes`](@ref).
"""
function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{false})
    return timestep_sizes[1:n_timesteps]
end

function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{true})

    #Create the vector with room for timestep 0, and put the marker and the sizes in it
    sizes = Vector{Union{BeforeFirstTimestep,eltype(timestep_sizes)}}(
        undef, n_timesteps + 1
    )
    sizes[1] = BeforeFirstTimestep()
    copyto!(sizes, 2, timestep_sizes, 1, n_timesteps)

    return sizes
end

function returned_timestep_sizes(timestep_size, n_timesteps, save_initial_timestep_values)
    return timestep_size
end

"""
    saved_timestep_sizes(timestep_sizes, n_steps, save_trajectories)

Return the sizes of `n_steps` timesteps as they are returned with what a session saves: one
size for each timestep when `save_trajectories` saves anything, after
[`BeforeFirstTimestep`](@ref) for timestep 0 when it saves the initial timestep values, and
an empty vector when it saves nothing. Used by [`SessionInfo`](@ref), for the sizes a fitted
session returns.
"""
function saved_timestep_sizes(timestep_sizes, n_steps, save_trajectories::SaveTrajectories)
    return returned_timestep_sizes(
        each_timestep_size(timestep_sizes, n_steps),
        n_steps,
        Val(saves_initial_timestep_values(save_trajectories)),
    )
end
function saved_timestep_sizes(
    timestep_sizes, n_steps, ::SaveTrajectories{(),(),(),(),Tsave_initial_timestep_values}
) where {Tsave_initial_timestep_values}
    return empty(each_timestep_size(timestep_sizes, 0))
end
