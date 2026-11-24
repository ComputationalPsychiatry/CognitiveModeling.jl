## Reading the selection of a SaveTrajectories ##

"""
    state_selection(save_trajectories)
    input_selection(save_trajectories)
    output_selection(save_trajectories)
    output_distribution_selection(save_trajectories)

Return the selection of one kind in a [`SaveTrajectories`](@ref) as a `Val`. Used wherever
something is saved, by simulations, instantiated models and fitted sessions.
"""
function state_selection(
    ::SaveTrajectories{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Tstates)
end
function input_selection(
    ::SaveTrajectories{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Tinputs)
end
function output_selection(
    ::SaveTrajectories{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Toutputs)
end
function output_distribution_selection(
    ::SaveTrajectories{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Toutput_distributions)
end

"""
    saves_initial_timestep_values(save_trajectories)

Return whether `save_trajectories` saves the values from before the first timestep, as
timestep 0. Used by [`simulate_session`](@ref), [`InstantiatedModel`](@ref),
[`saved_timestep_sizes`](@ref) and fitted sessions, to decide whether timestep 0 is saved.
"""
function saves_initial_timestep_values(
    ::SaveTrajectories{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Tsave_initial_timestep_values
end

"""
    saves_nothing(save_trajectories)

Return whether `save_trajectories` selects nothing of any kind. Whether it saves the initial
timestep values doesn't matter, since there is nothing to save of them. Used by
[`simulate_session!`](@ref), [`InstantiatedModel`](@ref) and fitted sessions, which save
nothing at all then.
"""
function saves_nothing(
    ::SaveTrajectories{(),(),(),(),Tsave_initial_timestep_values}
) where {Tsave_initial_timestep_values}
    return true
end
saves_nothing(::SaveTrajectories) = false
