### THE FULL MODEL, MADE FROM ITS PARTS ###

#Creates the Turing model for fitting a cognitive model to several sessions, after warning when both parameter models can sample variables of the same name, and when the sessions save trajectories no downstream model uses, and after preparing its parts once
#The warnings and the preparation are done here, when the model is made, and not in the model, whose body is differentiated and runs on every evaluation: a warning holds a try block, which Mooncake can't differentiate
function CognitiveModeling.turing_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    fixed_cognitive_parameters::NamedTuple=(;),
    fixed_environment_parameters::NamedTuple=(;),
    catch_invalid_parameters::Bool=false,
    parallelization=SerialSessions(),
    include_submodel_prefixes::Bool=false,
)

    #A tuple or a NamedTuple of parameter models is made into a composite of them, before the warnings look at the models
    cognitive_parameters_model = as_parameters_model(cognitive_parameters_model)
    environment_parameters_model = as_parameters_model(environment_parameters_model)
    warn_if_parameter_names_overlap(
        cognitive_parameters_model,
        environment_parameters_model,
        sessions_info,
        include_submodel_prefixes,
    )
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

    #Prepare the parts once, as every inference backend does: the parameter models for the sessions, with the values that are fixed, and the sessions with their environment running when its parameters are sampled
    prepared = prepare_full_model(
        cognitive_parameters_model,
        sessions_info;
        environment_parameters_model,
        downstream_model,
        fixed_cognitive_parameters,
        fixed_environment_parameters,
    )
    return turing_run_full_model(
        prepared.cognitive_parameters_model,
        prepared.sessions_info;
        environment_parameters_model=prepared.environment_parameters_model,
        downstream_model=prepared.downstream_model,
        catch_invalid_parameters,
        parallelization,
        include_submodel_prefixes,
    )
end

"""
    warn_if_parameter_names_overlap(cognitive_parameters_model,
                                    environment_parameters_model, sessions_info,
                                    include_submodel_prefixes)

Warn when the attributes of the cognitive model and of the environment model of the sessions
have a parameter of the same name, both parameter models sample, and the variables have no
prefixes, since their variables can then have the same names, and a name that both use is
one variable, which `sample` rejects. Whether both sample is decided by their types. Used by
`turing_full_model` when it creates the model, outside the model's body, which is
differentiated.
"""
function warn_if_parameter_names_overlap(
    cognitive_parameters_model,
    environment_parameters_model,
    sessions_info::MultipleSessionsInfo,
    include_submodel_prefixes::Bool,
)
    include_submodel_prefixes && return nothing
    estimates_parameters(cognitive_parameters_model) || return nothing
    estimates_parameters(environment_parameters_model) || return nothing
    session_info = first(sessions_info.sessions)
    has_environment(session_info) || return nothing
    cognitive_names = attribute_names(session_info.cognitive_model).parameters
    environment_names = attribute_names(session_info.environment).parameters
    shared_names = Tuple(name for name in cognitive_names if name in environment_names)
    if !isempty(shared_names)
        @warn "The cognitive model and the environment model both have the parameters $shared_names, and both parameter models sample, so their variables can have the same names. A name that both use is one variable for both, which sample rejects. Give turing_full_model include_submodel_prefixes = true to keep them apart."
    end
    return nothing
end

"""
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

Warn when a session in `sessions_info` saves anything and `downstream_model` is
`NoDownstreamModel`, since saving adds a cost to each gradient and nothing in the model
uses the saved trajectories: they are read after the fit with `get_session_trajectories`,
from sessions that save nothing. Used by `turing_full_model` when it creates the model.
"""
function warn_if_saving_without_downstream_model(
    sessions_info::MultipleSessionsInfo, downstream_model
)
    downstream_model isa NoDownstreamModel || return nothing
    sessions_save_anything(sessions_info) || return nothing
    @warn "The sessions save trajectories, which adds a cost to each gradient, but there is no downstream model to use them. A fit is fastest with sessions that save nothing, and get_session_trajectories(model, chain; cognitive_save_history, environment_save_history) then gives the trajectories of every sample afterwards."
    return nothing
end
