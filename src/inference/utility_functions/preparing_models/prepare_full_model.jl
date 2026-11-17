## Preparing the parts of a full model ##

"""
    prepare_full_model(cognitive_parameters_model, sessions_info::MultipleSessionsInfo;
                       environment_parameters_model=NoParameters(),
                       downstream_model=NoDownstreamModel(),
                       fixed_cognitive_parameters=(;),
                       fixed_environment_parameters=(;))

Prepare the parts of a full model once, before an inference backend makes its model of
them, and return them in a `NamedTuple` with `cognitive_parameters_model`,
`environment_parameters_model`, `downstream_model` and `sessions_info`. Useful for an
inference backend, whose full model calls it, so that what doesn't change between the
evaluations of the model is worked out once.

A parameter model can be given as a tuple of parameter models, or a `NamedTuple` of them,
which is made into a [`CompositeParameters`](@ref) of them by [`as_parameters_model`](@ref),
with each part of a `NamedTuple` under its name. It checks that the two parameter models and
the downstream model are of the right kind. The values in `fixed_cognitive_parameters` and
`fixed_environment_parameters`, such as `(; initial_value = 0.3)`, are given to every
session for the parameters they name: each parameter model is combined with them in a
[`CompositeParameters`](@ref), so a parameter is either fixed or given by its parameter
model, and a fixed parameter must be one in the model's attributes. It checks the parameters
that each parameter model gives, when [`CognitiveModeling.parameter_names`](@ref) knows
them, against those of the cognitive model and of the environment model: a parameter that
isn't in the model's attributes is an error, and so is one in its attributes which is
neither given nor has a default value, unless it is a parameter of an environment that no
session runs. It prepares each parameter model for the sessions with
[`CognitiveModeling.prepare_parameters_model`](@ref). When `environment_parameters_model`
estimates parameters, every session with an environment includes it in the returned
`sessions_info`. It warns when the sessions save anything and there is no downstream model
to use what they save.
"""
function prepare_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    fixed_cognitive_parameters::NamedTuple=(;),
    fixed_environment_parameters::NamedTuple=(;),
)

    #A tuple or a NamedTuple of parameter models is made into a composite of them
    cognitive_parameters_model = as_parameters_model(cognitive_parameters_model)
    environment_parameters_model = as_parameters_model(environment_parameters_model)

    #Check that each part is of the right kind
    check_is_parameters_model(cognitive_parameters_model)
    check_is_parameters_model(environment_parameters_model)
    check_is_downstream_model(downstream_model)

    #Combine each parameter model with the values that are fixed, which every session then gets next to the parameters the model gives
    cognitive_parameters_model = with_fixed_parameters(
        cognitive_parameters_model,
        fixed_cognitive_parameters,
        first(sessions_info.sessions).cognitive_model,
        "fixed_cognitive_parameters",
    )
    environment_parameters_model = with_fixed_parameters(
        environment_parameters_model,
        fixed_environment_parameters,
        first(sessions_info.sessions).environment,
        "fixed_environment_parameters",
    )

    #When the environment's parameters are sampled, every session runs its environment, so that its observations inform them
    sessions_info = include_environments_if_estimated(
        environment_parameters_model, sessions_info
    )

    #Check the parameters that the parameter models give against those of the models. An environment that no session runs doesn't need its parameters
    first_session = first(sessions_info.sessions)
    check_given_parameters(cognitive_parameters_model, first_session.cognitive_model, true)
    check_given_parameters(
        environment_parameters_model,
        first_session.environment,
        any(session -> session.includes_environment, sessions_info.sessions),
    )

    #Saving adds a cost to every evaluation of a model, which nothing in it uses without a downstream model
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

    #Prepare each parameter model for the sessions
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

#The trajectories are read after a fit with get_session_trajectories instead, from sessions that save nothing
"""
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

Warn when a session in `sessions_info` saves anything and `downstream_model` is
[`NoDownstreamModel`](@ref). Used by [`prepare_full_model`](@ref).
"""
function warn_if_saving_without_downstream_model(
    sessions_info::MultipleSessionsInfo, downstream_model
)
    downstream_model isa NoDownstreamModel || return nothing
    sessions_save_anything(sessions_info) || return nothing
    @warn "The sessions save trajectories, which adds a cost to each gradient, but there is no downstream model to use them. A fit is fastest with sessions that save nothing, and get_session_trajectories(model, chain; cognitive_save_history, environment_save_history) then gives the trajectories of every sample afterwards."
    return nothing
end

"""
    with_fixed_parameters(parameters_model, fixed_parameters, model, keyword)

Return `parameters_model`, a parameter model, combined with the values in
`fixed_parameters`, which every session gets for the parameters they name: a
[`CompositeParameters`](@ref) of its parts, which keep their names, and a
[`FixedParameters`](@ref) without a name, or `parameters_model` itself when no values are
fixed. It is an error when a fixed parameter isn't in the attributes of `model`, and when
`parameters_model` gives it too, which is checked here when [`parameter_names`](@ref) knows
its parameters, and when the full model runs otherwise. `keyword` is the name of the keyword
argument, for the errors. Used by [`prepare_full_model`](@ref).
"""
function with_fixed_parameters(
    parameters_model, fixed_parameters::NamedTuple, model, keyword::String
)
    isempty(fixed_parameters) && return parameters_model

    #A fixed parameter must be one in the model's attributes
    check_fixed_parameter_names(fixed_parameters, model, keyword)

    #A parameter is either fixed or given by the parameter model
    given_names = parameter_names(parameters_model)
    if given_names !== nothing
        shared_names = Tuple(name for name in keys(fixed_parameters) if name in given_names)
        if !isempty(shared_names)
            throw(
                ArgumentError(
                    "$keyword fixes the parameters $shared_names, which the parameter model, a $(nameof(typeof(parameters_model))), gives too. A parameter is either fixed or given by the parameter model.",
                ),
            )
        end
    end

    #Without parameters of its own, the fixed values are all there is. Otherwise they join the model, or the parts of a composite, which keep their names
    parameters_model isa NoParameters && return FixedParameters(fixed_parameters)
    return add_part_without_name(parameters_model, FixedParameters(fixed_parameters))
end

"""
    check_fixed_parameter_names(fixed_parameters, model, keyword)

Throw an informative error if `fixed_parameters`, the values given with the keyword argument
`keyword`, name a parameter that isn't in the attributes of `model`, a cognitive model or an
environment model, from [`attribute_names`](@ref), or if they are given for sessions without
an environment, where `model` is `NoEnvironment()`. Used by [`with_fixed_parameters`](@ref).
"""
function check_fixed_parameter_names(
    fixed_parameters::NamedTuple, model::AbstractTimestepModel, keyword::String
)
    names_in_attributes =
        check_type_attribute_names(attribute_names(model), model).parameters
    unknown_names = Tuple(
        name for name in keys(fixed_parameters) if !(name in names_in_attributes)
    )
    if !isempty(unknown_names)
        throw(
            ArgumentError(
                "$keyword fixes the parameters $unknown_names, which aren't in the attributes of $(describe_model(model)). Its attributes have the parameters $names_in_attributes.",
            ),
        )
    end
    return nothing
end

function check_fixed_parameter_names(
    fixed_parameters::NamedTuple, ::NoEnvironment, keyword::String
)
    return throw(
        ArgumentError(
            "$keyword fixes the parameters $(keys(fixed_parameters)), but the sessions have no environment model.",
        ),
    )
end

"""
    check_is_parameters_model(parameters_model)
    check_is_downstream_model(downstream_model)

Throw an informative error if `parameters_model` isn't a subtype of
[`AbstractParametersModel`](@ref), such as the priors on their own, or if
`downstream_model` isn't a subtype of [`AbstractDownstreamModel`](@ref). Used by
[`prepare_full_model`](@ref).
"""
check_is_parameters_model(::AbstractParametersModel) = nothing
function check_is_parameters_model(parameters_model)
    return throw(
        ArgumentError(
            "A parameter model must be a subtype of AbstractParametersModel, such as IndependentParameters(priors) with the priors in a NamedTuple, or FixedParameters(values) for given values, but it has type $(typeof(parameters_model)).",
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

Throw an informative error if `parameters_model`, a parameter model, gives a parameter that
isn't in the attributes of `model`, a cognitive model or an environment model, from
[`attribute_names`](@ref), or, when `model_runs` is `true`, doesn't give one in its
attributes that has no default value in [`default_parameters`](@ref). For sessions without
an environment, `model` is `NoEnvironment()`, and any parameter given is an error. Nothing
is checked when [`parameter_names`](@ref) doesn't know the parameters. Used by
[`prepare_full_model`](@ref).
"""
function check_given_parameters(
    parameters_model::AbstractParametersModel,
    model::AbstractTimestepModel,
    model_runs::Bool,
)
    given_names = parameter_names(parameters_model)
    given_names === nothing && return nothing
    names_in_attributes =
        check_type_attribute_names(attribute_names(model), model).parameters

    #Every parameter that is given must be one in the model's attributes
    unknown_names = Tuple(name for name in given_names if !(name in names_in_attributes))
    if !isempty(unknown_names)
        throw(
            ArgumentError(
                "The parameter model, a $(nameof(typeof(parameters_model))), gives the parameters $unknown_names, which aren't in the attributes of $(describe_model(model)). Its attributes have the parameters $names_in_attributes.",
            ),
        )
    end

    #Every parameter in the model's attributes must be given, or have a default value, when the model runs
    model_runs || return nothing
    default_names = keys(default_parameters(model))
    missing_names = Tuple(
        name for
        name in names_in_attributes if !(name in given_names) && !(name in default_names)
    )
    if !isempty(missing_names)
        throw(
            ArgumentError(
                "The parameter model, a $(nameof(typeof(parameters_model))), doesn't give the parameters $missing_names of $(describe_model(model)), which have no default values.",
            ),
        )
    end
    return nothing
end

function check_given_parameters(
    parameters_model::AbstractParametersModel, ::NoEnvironment, model_runs::Bool
)
    given_names = parameter_names(parameters_model)
    (given_names === nothing || isempty(given_names)) && return nothing
    return throw(
        ArgumentError(
            "The sessions have no environment model, but the parameter model of the environment, a $(nameof(typeof(parameters_model))), gives the parameters $given_names.",
        ),
    )
end
