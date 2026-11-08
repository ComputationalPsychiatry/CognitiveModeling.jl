### WHAT FITTED SESSIONS SAVED, READ FROM A CHAIN WITH A MODEL THAT SAVES IT ###

#Makes a copy of the model whose sessions save what save_history selects, reruns it on every sample of the chain, and makes a chain of what the sessions saved
function CognitiveModeling.session_values_chain(
    model::DynamicPPL.Model, chain; save_history::SaveHistory
)
    saving_model = with_save_history(model, save_history)
    results = returned(saving_model, chain)
    numbering = sample_numbering(chain)
    session_info = info_of(saving_model)
    if session_info isa SessionInfo
        return session_values_chain(results; numbering...)
    else
        return session_values_chain(results, session_info; numbering...)
    end
end

"""
    with_save_history(model, save_history)

Return a copy of `model`, a Turing.jl model made by `turing_full_model`,
`turing_multiple_behavioural_sessions` or `turing_behavioural_session`, or one of them inside
`turing_catch_invalid_parameters`, whose sessions save what `save_history` selects. Its other
arguments, its keyword arguments and whether it is marked thread-safe stay as they are. Used
by `session_values_chain`.
"""
function with_save_history(
    model::DynamicPPL.Model{F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Threaded},
    save_history::SaveHistory,
) where {F,argnames,defaultnames,missings,Targs,Tdefaults,Ctx,Threaded}
    args = model.args
    if model.f === CognitiveModeling.turing_behavioural_session
        new_args = merge(
            args, (; session_info=SessionInfo(args.session_info; save_history))
        )
    elseif model.f === CognitiveModeling.turing_multiple_behavioural_sessions ||
        model.f === CognitiveModeling.turing_full_model
        new_args = merge(
            args, (; sessions_info=MultipleSessionsInfo(args.sessions_info; save_history))
        )
    elseif model.f === CognitiveModeling.turing_catch_invalid_parameters
        new_args = merge(args, (; model=with_save_history(args.model, save_history)))
    else
        throw(
            ArgumentError(
                "session_values_chain(model, chain; save_history) takes a model made by turing_full_model, turing_multiple_behavioural_sessions or turing_behavioural_session, but this one is made by $(model.f). For a model of one's own, give session_values_chain the results of returned(model, chain) instead.",
            ),
        )
    end
    return DynamicPPL.Model{Threaded,missings}(
        model.f, new_args, model.defaults, model.context
    )
end

"""
    info_of(model)

Return the session info a model made by `with_save_history` runs: the `SessionInfo` of a
single session, or the `MultipleSessionsInfo` of several. Used by `session_values_chain`.
"""
function info_of(model::DynamicPPL.Model)
    if model.f === CognitiveModeling.turing_behavioural_session
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
the chain of what the sessions saved is numbered the same way, or none for a chain that isn't
a `FlexiChain`. Used by `session_values_chain`.
"""
function sample_numbering(chain::Turing.FlexiChains.FlexiChain)
    return (;
        iter_indices=Turing.FlexiChains.iter_indices(chain),
        chain_indices=Turing.FlexiChains.chain_indices(chain),
    )
end
sample_numbering(chain) = (;)
