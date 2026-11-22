### THE TRAJECTORIES OF FITTED SESSIONS SIMULATED AGAIN ###

#The copy of the model and the numbering of the samples are made by the helpers in get_session_trajectories.jl

#Makes a copy of the model whose sessions are simulated and save what the save histories select, simulates them for every sample of the chain, as Turing.jl's predict would, and makes a chain of the trajectories they saved
function CognitiveModeling.predict_session_trajectories(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    chain::Turing.FlexiChains.FlexiChain;
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
)
    predictive_model = with_session_info(
        model,
        info -> decondition_sessions(
            with_save_history(info, cognitive_save_history, environment_save_history)
        ),
        "predict_session_trajectories",
    )
    #Everything within the timesteps is simulated again, with the parameters of each sample
    returned_values = returned_with_simulation(
        rng, predictive_model, without_timestep_variables(chain)
    )
    numbering = sample_numbering(chain)
    session_info = info_of(predictive_model)
    if session_info isa SessionInfo
        return get_session_trajectories(returned_values; numbering...)
    else
        return get_session_trajectories(returned_values, session_info; numbering...)
    end
end

#A chain of another kind is an error, which says what the function takes
function CognitiveModeling.predict_session_trajectories(
    rng::AbstractRNG, model::DynamicPPL.Model, chain; keywords...
)
    return throw_not_flexichain("predict_session_trajectories(model, chain)", chain)
end

function CognitiveModeling.predict_session_trajectories(
    model::DynamicPPL.Model, chain; keywords...
)
    return predict_session_trajectories(default_rng(), model, chain; keywords...)
end

#Everything within the timesteps is simulated again, as the data is in a predictive check. Leaving those variables out of the chain also keeps FlexiChains.jl from rebuilding a sample whose sessions have variables at different timesteps, which fails: DynamicPPL.jl's skeleton gives the timesteps of every session the shape of the first session's (DynamicPPL 0.42)
"""
    without_timestep_variables(chain)

Return `chain` without the variables within the timesteps of a session, such as imputed
actions and observations and what a model samples within a timestep, named like
`session[2].timestep[3].choice.imputed`, or `timestep[3].choice.imputed` for a single
session. The parameters, and any start that a model samples, are kept. Used by
`predict_session_trajectories`.
"""
function without_timestep_variables(chain::Turing.FlexiChains.FlexiChain)
    kept = filter(!is_within_timestep, Turing.FlexiChains.parameters(chain))
    return chain[[Turing.FlexiChains.Parameter(name) for name in kept]]
end

"""
    is_within_timestep(name)

Return whether the variable `name` is within a timestep of a session, which its name says:
`timestep[t]` at its start, or after `session[i]`. Used by `without_timestep_variables`.
"""
is_within_timestep(name) = occursin(r"^(session\[[^\]]*\]\.)?timestep\[", string(name))

#Turing.jl's predict gives a chain of the simulated variables, not what the model returns, which has the trajectories of the sessions. Reading those from that chain with returned would run the model a second time, and fails when the sessions have simulated actions at different timesteps, as sessions of different lengths always do in a predictive check, for the reason above. So the model runs once for every sample, as predict runs it, and what it returns is kept
"""
    returned_with_simulation(rng, model, chain)

Return a matrix with what `model` returns for every sample of `chain`, iterations down and
chains across, with the values of the sample for the variables in the chain, and every other
variable drawn from its prior, as Turing.jl's `predict` draws them. Used by
`predict_session_trajectories`.
"""
function returned_with_simulation(
    rng::AbstractRNG, model::DynamicPPL.Model, chain::Turing.FlexiChains.FlexiChain
)
    n_iterations, n_chains = size(chain)
    return [
        first(
            DynamicPPL.init!!(
                rng,
                model,
                DynamicPPL.OnlyAccsVarInfo(),
                DynamicPPL.InitFromParams(chain, i, j, DynamicPPL.InitFromPrior()),
                DynamicPPL.UnlinkAll(),
            ),
        ) for i in 1:n_iterations, j in 1:n_chains
    ]
end
