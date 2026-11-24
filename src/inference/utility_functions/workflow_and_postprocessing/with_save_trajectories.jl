## Sessions that save other trajectories, for reading or simulating a fit again ##

"""
    with_save_trajectories(info, save_cognitive_trajectories, save_environment_trajectories)

Return a copy of `info`, a [`SessionInfo`](@ref) or a [`MultipleSessionsInfo`](@ref), whose
sessions save what the two `SaveTrajectories` select. Used by the methods of
[`get_session_trajectories`](@ref) and [`predict_session_trajectories`](@ref) which take a
model, to read or simulate the sessions again with what is to be saved.
"""
function with_save_trajectories(
    info::SessionInfo,
    save_cognitive_trajectories::SaveTrajectories,
    save_environment_trajectories::SaveTrajectories,
)
    return SessionInfo(info; save_cognitive_trajectories, save_environment_trajectories)
end
function with_save_trajectories(
    info::MultipleSessionsInfo,
    save_cognitive_trajectories::SaveTrajectories,
    save_environment_trajectories::SaveTrajectories,
)
    return MultipleSessionsInfo(
        info; save_cognitive_trajectories, save_environment_trajectories
    )
end
