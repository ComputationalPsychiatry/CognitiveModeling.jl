## Preparing the parts of a full model ##

"""
    prepare_full_model(cognitive_parameters_model, sessions_info::MultipleSessionsInfo;
                       environment_parameters_model=NoParameters(),
                       downstream_model=NoDownstreamModel())

Prepare the parts of a full model once, before an inference backend makes its model of
them, and return them in a `NamedTuple` with `cognitive_parameters_model`,
`environment_parameters_model`, `downstream_model` and `sessions_info`. Useful for an
inference backend, whose full model calls it, so that what doesn't change between the
evaluations of the model is worked out once.

It checks that the two session parameter models and the downstream model are of the right
kind. It checks the parameters that each session parameter model gives, when
[`CognitiveModeling.parameter_names`](@ref) knows them, against those of the cognitive model
and of the environment model: a parameter that the model doesn't list is an error, and so is
one that the model lists which is neither given nor has a default value, unless it is a
parameter of an environment that no session runs. It prepares each session parameter model
for the sessions with
[`CognitiveModeling.prepare_parameters_model`](@ref). When the environment's parameters are
sampled, every session with an environment runs it, so that the observations in the data
inform them.
"""
function prepare_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
)

    #Check that each part is of the right kind
    check_is_session_parameters_model(cognitive_parameters_model)
    check_is_session_parameters_model(environment_parameters_model)
    check_is_downstream_model(downstream_model)

    #When the environment's parameters are sampled, every session runs its environment, so that its observations inform them
    sessions_info = sessions_for_environment_parameters(
        environment_parameters_model, sessions_info
    )

    #Check the parameters that the session parameter models give against those of the models. An environment that no session runs doesn't need its parameters
    first_session = first(sessions_info.sessions)
    check_given_parameters(cognitive_parameters_model, first_session.cognitive_model, true)
    check_given_parameters(
        environment_parameters_model,
        first_session.environment,
        any(session -> session.includes_environment, sessions_info.sessions),
    )

    #Prepare each session parameter model for the sessions
    return (;
        cognitive_parameters_model=prepare_parameters_model(
            cognitive_parameters_model, sessions_info
        ),
        environment_parameters_model=prepare_parameters_model(
            environment_parameters_model, sessions_info
        ),
        downstream_model,
        sessions_info,
    )
end

"""
    check_is_session_parameters_model(parameters_model)
    check_is_downstream_model(downstream_model)

Throw an informative error if `parameters_model` isn't a subtype of
[`AbstractSessionParametersModel`](@ref), such as the priors on their own, or if
`downstream_model` isn't a subtype of [`AbstractDownstreamModel`](@ref). Used by
[`prepare_full_model`](@ref).
"""
check_is_session_parameters_model(::AbstractSessionParametersModel) = nothing
function check_is_session_parameters_model(parameters_model)
    return throw(
        ArgumentError(
            "A session parameter model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it has type $(typeof(parameters_model)).",
        ),
    )
end
check_is_downstream_model(::AbstractDownstreamModel) = nothing
function check_is_downstream_model(downstream_model)
    return throw(
        ArgumentError(
            "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type $(typeof(downstream_model)).",
        ),
    )
end

"""
    check_given_parameters(parameters_model, model, model_runs)

Throw an informative error if `parameters_model`, a session parameter model, gives a
parameter that `model`, a cognitive model or an environment model, doesn't list in
[`attribute_names`](@ref), or, when `model_runs` is `true`, doesn't give one that the model
lists and has no default value for in [`default_parameters`](@ref). For sessions without an
environment, `model` is `NoEnvironment()`, and any parameter given is an error. Nothing is
checked when [`parameter_names`](@ref) doesn't know the parameters. Used by
[`prepare_full_model`](@ref).
"""
function check_given_parameters(
    parameters_model::AbstractSessionParametersModel, model::AbstractModel, model_runs::Bool
)
    given_names = parameter_names(parameters_model)
    given_names === nothing && return nothing
    listed_names = check_type_attribute_names(attribute_names(model), model).parameters

    #Every parameter that is given must be one the model lists
    unknown_names = Tuple(name for name in given_names if !(name in listed_names))
    if !isempty(unknown_names)
        throw(
            ArgumentError(
                "The session parameter model, a $(nameof(typeof(parameters_model))), gives the parameters $unknown_names, which $(describe_model(model)) doesn't list. It lists the parameters $listed_names.",
            ),
        )
    end

    #Every parameter the model lists must be given, or have a default value, when the model runs
    model_runs || return nothing
    default_names = keys(default_parameters(model))
    missing_names = Tuple(
        name for name in listed_names if !(name in given_names) && !(name in default_names)
    )
    if !isempty(missing_names)
        throw(
            ArgumentError(
                "The session parameter model, a $(nameof(typeof(parameters_model))), doesn't give the parameters $missing_names of $(describe_model(model)), which have no default values.",
            ),
        )
    end
    return nothing
end

function check_given_parameters(
    parameters_model::AbstractSessionParametersModel, ::NoEnvironment, model_runs::Bool
)
    given_names = parameter_names(parameters_model)
    (given_names === nothing || isempty(given_names)) && return nothing
    return throw(
        ArgumentError(
            "The sessions have no environment model, but the session parameter model of the environment, a $(nameof(typeof(parameters_model))), gives the parameters $given_names.",
        ),
    )
end
