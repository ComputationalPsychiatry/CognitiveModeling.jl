## What the models sampled in a session of an inference backend ##
#What a model sampled of its own in a session, read with `variable`, which gives the value of a variable of the session by
#its name within the session, in the form that log_likelihood_by_hand and log_density_of_latents_by_hand take it. Most
#models sample nothing of their own
latents_from_variables(model, variable, n_timesteps) = (;)
function latents_from_variables(model::SamplingWithinTimestep, variable, n_timesteps)
    return (; noise=[variable("timesteps[$t].noise") for t in 1:n_timesteps])
end
function latents_from_variables(model::SampledInitialState, variable, n_timesteps)
    return (; initial_offset=variable("initial_cognitive_state.initial_offset"))
end
function latents_from_variables(model::EveryFeature, variable, n_timesteps)
    return (;
        initial_offset=variable("initial_cognitive_state.initial_offset"),
        noise=[variable("timesteps[$t].noise") for t in 1:n_timesteps],
    )
end
function latents_from_variables(model::EveryFeatureWithoutSampling, variable, n_timesteps)
    return (; initial_offset=0.0, noise=zeros(n_timesteps))
end
function latents_from_variables(
    environment::Union{SamplingWithinTimestepEnvironment,EveryFeatureEnvironment},
    variable,
    n_timesteps,
)
    return (; drift=[variable("timesteps[$t].environment.drift") for t in 1:n_timesteps])
end

#A function which gives the value of a variable of a model by the name of the variable, from values drawn from the model,
#with a prefix, such as "sessions[2].", for the variables of a session among several
function variables_of(values; prefix="")
    by_name = Dict(string(name) => values[name] for name in keys(values))
    return name -> by_name[prefix * name]
end
