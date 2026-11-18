### MODEL OF THE FULL MODEL ###

#The Turing.jl model of a full model, from the parts that full_model prepared, outside this model's body. Its docstring is in the package, beside full_model
Turing.@model function CognitiveModeling.turing_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    catch_invalid_parameters::Bool=false,
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

    #Run all behavioral sessions, inside a try block when invalid parameters are caught. The parameters are sampled outside it, so that they are kept when a draw is rejected, which a sampler needs to start: it needs every variable to exist after its first draw, also when that draw is rejected
    sessions_submodel = turing_multiple_behavioral_sessions(
        sessions_info, cognitive_parameters, environment_parameters, parallelization
    )
    if catch_invalid_parameters
        sessions_result ~
        to_submodel(turing_catch_invalid_parameters(sessions_submodel), false)

        #A rejected draw leaves nothing for the downstream model to run on, so the model returns nothing at once, which Enzyme can differentiate where a value of two types can't be. The variables of the downstream model then don't exist for that draw, so a sampler whose first draw is rejected can't start when the downstream model has variables of its own, and needs valid initial parameters
        if sessions_result === nothing
            return nothing
        end
    else
        sessions_result ~ to_submodel(sessions_submodel, false)
    end

    #The sessions return the parameters they were given too, which are those above
    saved_trajectories = sessions_result.saved_trajectories

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
