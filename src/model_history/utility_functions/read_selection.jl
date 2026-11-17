## Reading the selection of a save history ##

"""
    state_selection(save_history)
    input_selection(save_history)
    output_selection(save_history)
    output_distribution_selection(save_history)

Return the selection of one kind in a [`SaveHistory`](@ref) as a `Val`. Used wherever
something is saved, by simulations, instantiated models and fitted sessions.
"""
function state_selection(
    ::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Tstates)
end
function input_selection(
    ::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Tinputs)
end
function output_selection(
    ::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Toutputs)
end
function output_distribution_selection(
    ::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Val(Toutput_distributions)
end

"""
    saves_initial_timestep_values(save_history)

Return whether `save_history` saves the values from before the first timestep, as timestep
0. Used by [`simulate`](@ref), [`InstantiatedModel`](@ref), [`saved_timestep_sizes`](@ref)
and fitted sessions, to decide whether timestep 0 is saved.
"""
function saves_initial_timestep_values(
    ::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    return Tsave_initial_timestep_values
end

"""
    saves_nothing(save_history)

Return whether `save_history` selects nothing of any kind. Whether it saves the initial
timestep values doesn't matter, since there is nothing to save of them. Used by
[`simulate!`](@ref), [`InstantiatedModel`](@ref) and fitted sessions, which save nothing at
all then.
"""
function saves_nothing(
    ::SaveHistory{(),(),(),(),Tsave_initial_timestep_values}
) where {Tsave_initial_timestep_values}
    return true
end
saves_nothing(::SaveHistory) = false
