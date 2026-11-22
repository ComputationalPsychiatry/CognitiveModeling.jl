### MODEL OF THE FULL MODEL ###

#The Turing.jl model of a full model, from the parts that full_model prepared, outside this model's body. Its docstring is in the package, beside full_model
Turing.@model function CognitiveModeling.turing_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    parallelization=SerialSessions(),
    include_submodel_prefixes::Bool=false,
)

    #Give or sample the cognitive parameters of every session, with the variables under cognitive_parameters_model when the prefixes are included
    if include_submodel_prefixes
        cognitive_parameters ~ to_submodel(
            prefix(
                turing_parameters_model(cognitive_parameters_model, sessions_info),
                @varname(cognitive_parameters_model),
            ),
            false,
        )
    else
        cognitive_parameters ~ to_submodel(
            turing_parameters_model(cognitive_parameters_model, sessions_info), false
        )
    end

    #Give or sample the parameters of the environment of every session, with the variables under environment_parameters_model when the prefixes are included
    if include_submodel_prefixes
        environment_parameters ~ to_submodel(
            prefix(
                turing_parameters_model(environment_parameters_model, sessions_info),
                @varname(environment_parameters_model),
            ),
            false,
        )
    else
        environment_parameters ~ to_submodel(
            turing_parameters_model(environment_parameters_model, sessions_info), false
        )
    end

    #Run all behavioral sessions
    sessions_result ~ to_submodel(
        turing_multiple_behavioral_sessions(
            sessions_info, cognitive_parameters, environment_parameters, parallelization
        ),
        false,
    )

    #The sessions return the parameters they were given too, which are those above
    saved_trajectories = sessions_result.saved_trajectories

    #A draw whose parameters turned invalid in a session has a log density of -Inf already, so the downstream model doesn't run on the trajectories that ended early. Its variables then don't exist for that draw, and a sampler takes the variables of the model from one draw from the prior, so sampling can fail to start when the downstream model has variables of its own, as the docstring describes
    if !parameters_valid_in_every_session(saved_trajectories)
        return (;
            cognitive_parameters,
            environment_parameters,
            saved_trajectories,
            downstream_values=nothing,
        )
    end

    #Do the optional downstream model, with its variables under downstream_model when the prefixes are included
    if include_submodel_prefixes
        downstream_values ~ to_submodel(
            prefix(
                turing_downstream_model(
                    downstream_model,
                    saved_trajectories,
                    cognitive_parameters,
                    sessions_info,
                ),
                @varname(downstream_model),
            ),
            false,
        )
    else
        downstream_values ~ to_submodel(
            turing_downstream_model(
                downstream_model, saved_trajectories, cognitive_parameters, sessions_info
            ),
            false,
        )
    end

    return (;
        cognitive_parameters, environment_parameters, saved_trajectories, downstream_values
    )
end
