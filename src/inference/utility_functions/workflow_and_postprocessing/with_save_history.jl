## Sessions that save other trajectories, for reading or simulating a fit again ##

"""
    with_save_history(info, cognitive_save_history, environment_save_history)

Return a copy of `info`, a [`SessionInfo`](@ref) or a [`MultipleSessionsInfo`](@ref), whose
sessions save what the two save histories select. Used by the methods of
[`get_session_trajectories`](@ref) and [`predict_session_trajectories`](@ref) which take a
model, to read or simulate the sessions again with what is to be saved.
"""
function with_save_history(
    info::SessionInfo,
    cognitive_save_history::SaveHistory,
    environment_save_history::SaveHistory,
)
    return SessionInfo(info; cognitive_save_history, environment_save_history)
end
function with_save_history(
    info::MultipleSessionsInfo,
    cognitive_save_history::SaveHistory,
    environment_save_history::SaveHistory,
)
    return MultipleSessionsInfo(info; cognitive_save_history, environment_save_history)
end
