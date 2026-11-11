### THE TRAJECTORIES OF FITTED SESSIONS, READ FROM A CHAIN WITH A MODEL THAT SAVES THEM ###

#Makes a copy of the model whose sessions save what the save histories select, reruns it on every sample of the chain, and makes a chain of the trajectories the sessions saved
function CognitiveModeling.session_trajectories_chain(
    model::DynamicPPL.Model,
    chain;
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
)
    saving_model = with_session_info(
        model,
        info -> with_save_history(info, cognitive_save_history, environment_save_history),
        "session_trajectories_chain",
    )
    results = returned(saving_model, chain)
    numbering = sample_numbering(chain)
    session_info = info_of(saving_model)
    if session_info isa SessionInfo
        return session_trajectories_chain(results; numbering...)
    else
        return session_trajectories_chain(results, session_info; numbering...)
    end
end

#Reruns a full model on every sample of the chain, and makes a chain of the parameters it gave each session
function CognitiveModeling.session_parameters_chain(model::DynamicPPL.Model, chain)
    if model.f !== turing_run_full_model
        throw(
            ArgumentError(
                "session_parameters_chain(model, chain) takes a model made by turing_full_model, which gives every session its parameters, but this one is made by $(model.f).",
            ),
        )
    end
    results = returned(model, chain)
    return session_parameters_chain(
        results, model.args.sessions_info; sample_numbering(chain)...
    )
end

#Makes a copy of the model whose sessions are simulated and save what the save histories select, simulates them with predict for every sample of the chain, and makes a chain of the trajectories they saved
function CognitiveModeling.predict_session_trajectories(
    rng::AbstractRNG,
    model::DynamicPPL.Model,
    chain;
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
)
    predictive_model = with_session_info(
        model,
        info -> prepare_for_prediction(
            with_save_history(info, cognitive_save_history, environment_save_history)
        ),
        "predict_session_trajectories",
    )
    #Everything within the timesteps is simulated again, with the parameters of each sample
    results = returned_with_simulation(
        rng, predictive_model, without_timestep_variables(chain)
    )
    numbering = sample_numbering(chain)
    session_info = info_of(predictive_model)
    if session_info isa SessionInfo
        return session_trajectories_chain(results; numbering...)
    else
        return session_trajectories_chain(results, session_info; numbering...)
    end
end

function CognitiveModeling.predict_session_trajectories(
    model::DynamicPPL.Model, chain; keywords...
)
    return predict_session_trajectories(default_rng(), model, chain; keywords...)
end

"""
    without_timestep_variables(chain)

Return `chain` without the variables within the timesteps of a session, such as imputed
actions and observations and what a model samples within a timestep, named like
`session[2].timestep[3].choice.imputed`, or `timestep[3].choice.imputed` for a single
session. The parameters, and any start that a model samples, are kept. Used by
`predict_session_trajectories`, so that everything within the timesteps is simulated again,
as the data is in a predictive check, which also keeps FlexiChains.jl from rebuilding the
variables of sessions of different lengths, which it can't do.
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

"""
    returned_with_simulation(rng, model, chain)

Return a matrix with what `model` returns for every sample of `chain`, iterations down and
chains across, with the values of the sample for the variables in the chain, and every other
variable drawn from its prior, as Turing.jl's `predict` draws them. Used by
`predict_session_trajectories`, which needs only what the sessions return, so that the
simulated values are never read back from a chain, which FlexiChains.jl can't do for
sessions of different lengths.
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

"""
    with_save_history(info, cognitive_save_history, environment_save_history)

Return a copy of `info`, a `SessionInfo` or a `MultipleSessionsInfo`, whose sessions save
what the two save histories select. Used by `session_trajectories_chain` and
`predict_session_trajectories`.
"""
function with_save_history(
    info::SessionInfo, cognitive_save_history, environment_save_history
)
    return SessionInfo(info; cognitive_save_history, environment_save_history)
end
function with_save_history(
    info::MultipleSessionsInfo, cognitive_save_history, environment_save_history
)
    return MultipleSessionsInfo(info; cognitive_save_history, environment_save_history)
end

"""
    with_session_info(model, change, function_name)

Return a copy of `model`, a Turing.jl model made by `turing_full_model`,
`turing_multiple_behavioral_sessions` or `turing_behavioral_session`, or one of them
inside `turing_catch_invalid_parameters`, whose session info is what `change` makes of its
own: a `SessionInfo` for a single session, and a `MultipleSessionsInfo` for several. Its
other arguments, its keyword arguments and whether it is marked thread-safe stay as they
are. Any other model is an error, which names `function_name`, the function that was called.
Used by `session_trajectories_chain` and `predict_session_trajectories`.
"""
function with_session_info(
    model::DynamicPPL.Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Threaded},
    change,
    function_name::AbstractString,
) where {F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Threaded}
    args = model.args
    if model.f === CognitiveModeling.turing_behavioral_session
        new_args = merge(args, (; session_info=change(args.session_info)))
    elseif model.f === CognitiveModeling.turing_multiple_behavioral_sessions
        new_args = merge(args, (; sessions_info=change(args.sessions_info)))
    elseif model.f === turing_run_full_model
        #The sessions of a full model run their environment when its parameters are sampled, as when the model was made
        new_args = merge(
            args,
            (;
                sessions_info=sessions_for_environment_parameters(
                    model.defaults.environment_parameters_model, change(args.sessions_info)
                ),
            ),
        )
    elseif model.f === CognitiveModeling.turing_catch_invalid_parameters
        new_args = merge(
            args, (; model=with_session_info(args.model, change, function_name))
        )
    else
        throw(
            ArgumentError(
                "$function_name(model, chain; cognitive_save_history) takes a model made by turing_full_model, turing_multiple_behavioral_sessions or turing_behavioral_session, but this one is made by $(model.f). For a model of one's own, give session_trajectories_chain the results of returned(model, chain) instead.",
            ),
        )
    end
    return DynamicPPL.Model{Threaded,missings}(
        model.f, new_args, model.defaults, model.context
    )
end

"""
    info_of(model)

Return the session info a model made by `with_session_info` runs: the `SessionInfo` of a
single session, or the `MultipleSessionsInfo` of several. Used by
`session_trajectories_chain`.
"""
function info_of(model::DynamicPPL.Model)
    if model.f === CognitiveModeling.turing_behavioral_session
        return model.args.session_info
    elseif model.f === CognitiveModeling.turing_catch_invalid_parameters
        return info_of(model.args.model)
    else
        return model.args.sessions_info
    end
end

"""
    sample_numbering(chain)

Return the numbers of the iterations and the chains of `chain` as keyword arguments, so that
the chain of the trajectories is numbered the same way, or none for a chain that isn't a
`FlexiChain`. Used by `session_trajectories_chain`.
"""
function sample_numbering(chain::Turing.FlexiChains.FlexiChain)
    return (;
        iter_indices=Turing.FlexiChains.iter_indices(chain),
        chain_indices=Turing.FlexiChains.chain_indices(chain),
    )
end
sample_numbering(chain) = (;)
