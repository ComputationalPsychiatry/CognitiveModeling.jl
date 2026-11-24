### DEFAULT METHODS OF THE FUNCTIONS THAT MODELS AND THE PARTS OF A FULL MODEL IMPLEMENT ###

#Default implementation of turing_timestep_model for both kinds of model, which just calls the normal timestep_model function
#Can be extended for specific cognitive models or environment models that sample within a timestep, and must be for those
#that say they do, which is decided by their type, so the check costs nothing for the others
Turing.@model function CognitiveModeling.turing_timestep_model(
    model::AbstractTimestepModel, context
)
    is_stochastic(model) && throw_without_turing_timestep_model(model)
    return timestep_model(model, context)
end

"""
    throw_without_turing_timestep_model(model)

Throw an informative error for `model`, a cognitive model or an environment model which
samples within its timesteps, as its method of [`CognitiveModeling.is_stochastic`](@ref)
says, but which has no method of [`CognitiveModeling.turing_timestep_model`](@ref) of its
own. Used by the default of `turing_timestep_model`.
"""
@noinline function throw_without_turing_timestep_model(model)
    return throw(
        ArgumentError(
            "$(uppercasefirst(describe_model(model))) samples within its timesteps, as its method of CognitiveModeling.is_stochastic says, but it has no method of CognitiveModeling.turing_timestep_model. Give it one which samples with ~ what its timestep_model samples from context.rng, so that each of those values is a variable of the fit.",
        ),
    )
end

#Default implementation of turing_initialize_state for both kinds of model, which just calls the normal initialize_state function
#Can be extended for specific models that sample when they initialize
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::AbstractTimestepModel, parameters
)
    return initialize_state(model, parameters)
end

#Without a downstream model, nothing is sampled after the sessions
Turing.@model function CognitiveModeling.turing_downstream_model(
    ::NoDownstreamModel,
    saved_trajectories,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return nothing
end

#Informative error for a parameter model without a method of its own
function CognitiveModeling.turing_parameters_model(
    parameters_model::AbstractParametersModel, sessions_info::MultipleSessionsInfo
)
    return error(
        "There is no turing_parameters_model method for a parameter model of type $(typeof(parameters_model)). Each parameter model must define Turing.@model function CognitiveModeling.turing_parameters_model(parameters_model::$(typeof(parameters_model)), sessions_info::MultipleSessionsInfo), which returns a NamedTuple with one vector for each parameter.",
    )
end

#Informative error for something that isn't a parameter model, such as the priors on their own
function CognitiveModeling.turing_parameters_model(
    parameters_model, sessions_info::MultipleSessionsInfo
)
    return throw(
        ArgumentError(
            "A parameter model must be a subtype of AbstractParametersModel, such as IndependentParameters(priors) with the priors in a NamedTuple, or FixedParameters(values) for given values, but it has type $(typeof(parameters_model)).",
        ),
    )
end

#Informative error for a downstream model without a method of its own
function CognitiveModeling.turing_downstream_model(
    downstream_model::AbstractDownstreamModel,
    saved_trajectories,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return error(
        "There is no turing_downstream_model method for a downstream model of type $(typeof(downstream_model)). Each downstream model must define Turing.@model function CognitiveModeling.turing_downstream_model(downstream_model::$(typeof(downstream_model)), saved_trajectories, cognitive_parameters, sessions_info::MultipleSessionsInfo).",
    )
end

#Informative error for something that isn't a downstream model
function CognitiveModeling.turing_downstream_model(
    downstream_model,
    saved_trajectories,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return throw(
        ArgumentError(
            "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type $(typeof(downstream_model)).",
        ),
    )
end
