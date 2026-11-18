### THE FULL MODEL, MADE FROM ITS PARTS ###

#Creates the Turing model for fitting a cognitive model to several sessions, after warning when both parameter models can sample variables of the same name, and after preparing its parts once, which warns when the sessions save trajectories that no downstream model uses
#The warnings and the preparation are done here, when the model is made, and not in the model, whose body is differentiated and runs on every evaluation: a warning holds a try block, which Mooncake can't differentiate
function CognitiveModeling.full_model(
    ::TuringBackend,
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

    #Prepare the parts once, as every inference backend does: the parameter models for the sessions, with the values that are fixed, and the sessions with their environment running when its parameters are sampled
    prepared = prepare_full_model(
        cognitive_parameters_model,
        sessions_info;
        environment_parameters_model,
        downstream_model,
        fixed_cognitive_parameters,
        fixed_environment_parameters,
    )
    return CognitiveModeling.turing_full_model(
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
`full_model` with `TuringBackend()` when it creates the model, outside the model's body,
which is differentiated.
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
        @warn "The cognitive model and the environment model both have the parameters $shared_names, and both parameter models sample, so their variables can have the same names. A name that both use is one variable for both, which sample rejects. Give full_model include_submodel_prefixes = true to keep them apart."
    end
    return nothing
end
