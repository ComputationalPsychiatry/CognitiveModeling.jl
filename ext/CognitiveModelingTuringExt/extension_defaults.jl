### DEFAULT METHODS OF THE FUNCTIONS THAT MODELS AND THE PARTS OF A FULL MODEL IMPLEMENT ###

#Default implementation of turing_apply_model for both kinds of model, which just calls the normal apply_model function
#Can be extended for specific cognitive models or environment models that sample within a timestep
Turing.@model function CognitiveModeling.turing_apply_model(model::AbstractModel, input)
    return apply_model(model, input)
end

#Default implementation of turing_initialize_state for both kinds of model, which just calls the normal initialize_state function
#Can be extended for specific models that sample when they initialize
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::AbstractModel, parameters
)
    return initialize_state(model, parameters)
end

#Without a downstream model, nothing is sampled after the sessions
Turing.@model function CognitiveModeling.turing_downstream_model(
    ::NoDownstreamModel,
    saved_behavioural_values,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return nothing
end

#Informative error for a population model without a method of its own
function CognitiveModeling.turing_population_model(
    population_model::AbstractPopulationModel, sessions_info::MultipleSessionsInfo
)
    return error(
        "There is no turing_population_model method for a population model of type $(typeof(population_model)). Each population model must define Turing.@model function CognitiveModeling.turing_population_model(population_model::$(typeof(population_model)), sessions_info::MultipleSessionsInfo), which returns a NamedTuple with one vector for each parameter.",
    )
end

#Informative error for something that isn't a population model, such as the priors on their own
function CognitiveModeling.turing_population_model(
    population_model, sessions_info::MultipleSessionsInfo
)
    return throw(
        ArgumentError(
            "The population model must be a subtype of AbstractPopulationModel, such as IndependentSessions(priors) with the priors in a NamedTuple, but it has type $(typeof(population_model)).",
        ),
    )
end

#Informative error for a downstream model without a method of its own
function CognitiveModeling.turing_downstream_model(
    downstream_model::AbstractDownstreamModel,
    saved_behavioural_values,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return error(
        "There is no turing_downstream_model method for a downstream model of type $(typeof(downstream_model)). Each downstream model must define Turing.@model function CognitiveModeling.turing_downstream_model(downstream_model::$(typeof(downstream_model)), saved_behavioural_values, cognitive_parameters, sessions_info::MultipleSessionsInfo).",
    )
end

#Informative error for something that isn't a downstream model
function CognitiveModeling.turing_downstream_model(
    downstream_model,
    saved_behavioural_values,
    cognitive_parameters,
    sessions_info::MultipleSessionsInfo,
)
    return throw(
        ArgumentError(
            "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type $(typeof(downstream_model)).",
        ),
    )
end
