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
    #The sessions are simulated under names apart from the fit's, so that everything within them is drawn again, with the parameters of each sample
    predictive_model = with_sessions_simulated(
        model,
        info -> decondition_sessions(
            with_save_history(info, cognitive_save_history, environment_save_history)
        ),
        "predict_session_trajectories",
    )
    returned_values = returned_with_prior_fallback(rng, predictive_model, chain)
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

#Turing.jl's predict gives a chain of the simulated variables, not what the model returns, which has the trajectories of the sessions. Reading those from that chain with returned would run the model a second time, and fails when the sessions have simulated actions at different timesteps, as sessions of different lengths always do in a predictive check: FlexiChains.jl rebuilds such a sample wrongly (DynamicPPL 0.42). So the model runs once for every sample, as predict runs it, and what it returns is kept
"""
    returned_with_prior_fallback(rng, model, chain)

Return a matrix with what `model` returns for every sample of `chain`, iterations down and
chains across, with the values of the sample for the variables in the chain, and every other
variable drawn from its prior, as Turing.jl's `predict` draws them. Used by
`predict_session_trajectories`.
"""
function returned_with_prior_fallback(
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
