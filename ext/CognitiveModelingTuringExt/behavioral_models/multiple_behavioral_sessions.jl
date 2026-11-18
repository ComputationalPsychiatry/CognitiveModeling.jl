### MODEL FOR MULTIPLE SESSIONS ###

#Turing model for running several behavioral sessions, each with its own cognitive parameters, as a submodel with the prefix session[i], and collecting the trajectories each saved. The sessions run one after the other. The first session runs outside the loop, so that the vector of what the sessions saved has its type, and a session whose trajectories have another type widens it
#The environment parameters are given in the same form, one vector for each parameter with a value for each session, and only when the sessions have an environment
Turing.@model function CognitiveModeling.turing_multiple_behavioral_sessions(
    sessions_info::MultipleSessionsInfo,
    cognitive_parameters,
    environment_parameters=(;),
    parallelization::SerialSessions=SerialSessions(),
)
    check_environment_parameters(sessions_info, environment_parameters)
    sessions = sessions_info.sessions
    first_session ~ to_submodel(
        prefix(
            turing_behavioral_session(
                sessions[1],
                select_row(cognitive_parameters, 1),
                select_row(environment_parameters, 1),
            ),
            @varname(session[1]),
        ),
        false,
    )
    saved_trajectories = initialize_storage(
        first_session.saved_trajectories, length(sessions)
    )
    for i in 2:length(sessions)
        session ~ to_submodel(
            prefix(
                turing_behavioral_session(
                    sessions[i],
                    select_row(cognitive_parameters, i),
                    select_row(environment_parameters, i),
                ),
                @varname(session[i]),
            ),
            false,
        )
        #A session whose saved trajectories have another type widens the vector
        saved_trajectories = store_value!(saved_trajectories, session.saved_trajectories, i)
    end

    #The parameters as they were given, so that a model around the sessions which transforms its variables into them has them for each sample, with the trajectories each session saved
    return (; cognitive_parameters, environment_parameters, saved_trajectories)
end

#The same with the sessions over the threads Julia was started with, which needs the model that is evaluated to be marked thread-safe, since its log density is silently wrong otherwise
Turing.@model function CognitiveModeling.turing_multiple_behavioral_sessions(
    sessions_info::MultipleSessionsInfo,
    cognitive_parameters,
    environment_parameters,
    parallelization::ParallelSessions,
)
    check_environment_parameters(sessions_info, environment_parameters)
    if !(__varinfo__ isa DynamicPPL.ThreadSafeVarInfo)
        throw(
            ArgumentError(
                "The sessions run in parallel, so the model that is evaluated must be marked thread-safe with setthreadsafe(model, true) from DynamicPPL. Without it, the log density is wrong with no error.",
            ),
        )
    end
    sessions = sessions_info.sessions
    first_session ~ to_submodel(
        prefix(
            turing_behavioral_session(
                sessions[1],
                select_row(cognitive_parameters, 1),
                select_row(environment_parameters, 1),
            ),
            @varname(session[1]),
        ),
        false,
    )
    saved_trajectories = initialize_storage(
        first_session.saved_trajectories, length(sessions)
    )
    #A vector can't be widened while other threads write to it, so saved trajectories of another type are put aside until the threads are done
    saved_with_other_types = Vector{Any}(undef, length(sessions))
    #Written as Base.Threads.@threads, which is the same macro, so that DynamicPPL doesn't warn about thread safety when the package loads. The check above takes the warning's place
    Base.Threads.@threads for i in 2:length(sessions)
        session ~ to_submodel(
            prefix(
                turing_behavioral_session(
                    sessions[i],
                    select_row(cognitive_parameters, i),
                    select_row(environment_parameters, i),
                ),
                @varname(session[i]),
            ),
            false,
        )
        saved = session.saved_trajectories
        if saved isa eltype(saved_trajectories)
            saved_trajectories[i] = saved
        else
            saved_with_other_types[i] = saved
        end
    end
    saved_trajectories = include_other_types(saved_trajectories, saved_with_other_types)

    #The parameters as they were given, with the trajectories each session saved
    return (; cognitive_parameters, environment_parameters, saved_trajectories)
end
