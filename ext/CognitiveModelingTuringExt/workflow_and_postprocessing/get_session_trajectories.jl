### THE TRAJECTORIES AND THE PARAMETERS OF FITTED SESSIONS, READ FROM A CHAIN ###

#Makes a copy of the model whose sessions save what the save histories select, reruns it on every sample of the chain, and makes a chain of the trajectories the sessions saved
function CognitiveModeling.get_session_trajectories(
    model::DynamicPPL.Model,
    chain::Turing.FlexiChains.FlexiChain;
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
)
    saving_model = with_session_info(
        model,
        info -> with_save_history(info, cognitive_save_history, environment_save_history),
        "get_session_trajectories",
    )
    returned_values = returned(saving_model, chain)
    numbering = sample_numbering(chain)
    session_info = info_of(saving_model)
    if session_info isa SessionInfo
        return get_session_trajectories(returned_values; numbering...)
    else
        return get_session_trajectories(returned_values, session_info; numbering...)
    end
end

#Reruns a full model, several sessions or a single session on every sample of the chain, and makes a chain of the parameters each session was given
function CognitiveModeling.get_session_parameters(
    model::DynamicPPL.Model, chain::Turing.FlexiChains.FlexiChain
)
    #A model of one's own has no session info to name the parameters by, so what it returns is given instead
    if !is_session_model(model)
        throw(
            ArgumentError(
                "get_session_parameters(model, chain) takes a model made by full_model, multiple_sessions_model or session_model, but this one is made by $(model.f). For a model of one's own, give get_session_parameters what returned(model, chain) gives instead.",
            ),
        )
    end
    returned_values = returned(model, chain)
    numbering = sample_numbering(chain)
    session_info = info_of(model)
    if session_info isa SessionInfo
        return get_session_parameters(returned_values; numbering...)
    else
        return get_session_parameters(returned_values, session_info; numbering...)
    end
end

#A chain of another kind is an error, which says what the functions take
function CognitiveModeling.get_session_trajectories(
    model::DynamicPPL.Model, chain; keywords...
)
    return throw_not_flexichain("get_session_trajectories(model, chain)", chain)
end
function CognitiveModeling.get_session_parameters(model::DynamicPPL.Model, chain)
    return throw_not_flexichain("get_session_parameters(model, chain)", chain)
end

"""
    throw_not_flexichain(call, chain)

Throw an `ArgumentError` which says that `call`, a function of a model and a chain, takes a
`FlexiChain`, and the type of `chain`, which isn't one. Used by
`get_session_trajectories`, `get_session_parameters` and `predict_session_trajectories`.
"""
function throw_not_flexichain(call::AbstractString, chain)
    return throw(
        ArgumentError(
            "$call takes a FlexiChain, but the chain is of type $(nameof(typeof(chain)))."
        ),
    )
end

"""
    is_session_model(model)

Return whether `model` is a Turing.jl model made by `turing_full_model`,
`turing_multiple_behavioral_sessions` or `turing_behavioral_session`. Used by
`get_session_parameters`, `decondition_sessions` and `condition_sessions`.
"""
function is_session_model(model::DynamicPPL.Model)
    return model.f === CognitiveModeling.turing_full_model ||
           model.f === CognitiveModeling.turing_multiple_behavioral_sessions ||
           model.f === CognitiveModeling.turing_behavioral_session
end

### HELPERS FOR READING AND SIMULATING FITTED SESSIONS ###

"""
    with_session_info(model, change, function_name)

Return a copy of `model`, a Turing.jl model made by `turing_full_model`,
`turing_multiple_behavioral_sessions` or `turing_behavioral_session`, whose session info is
what `change` makes of its own: a
`SessionInfo` for a single session, and a `MultipleSessionsInfo` for several. Its other
arguments, its keyword arguments and whether it is marked thread-safe stay as they are. Any
other model is an error, which names `function_name`, the function that was called. Used by
`get_session_trajectories`, `predict_session_trajectories`, `decondition_sessions` and
`condition_sessions`.
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
    elseif model.f === CognitiveModeling.turing_full_model
        #The sessions of a full model run their environment when its parameters are sampled, as when the model was made
        new_args = merge(
            args,
            (;
                sessions_info=include_environments_if_estimated(
                    model.defaults.environment_parameters_model, change(args.sessions_info)
                ),
            ),
        )
    else
        throw(
            ArgumentError(
                "$function_name(model, chain; cognitive_save_history) takes a model made by full_model, multiple_sessions_model or session_model, but this one is made by $(model.f). For a model of one's own, give get_session_trajectories what returned(model, chain) gives instead.",
            ),
        )
    end
    return DynamicPPL.Model{Threaded,missings}(
        model.f, new_args, model.defaults, model.context
    )
end

"""
    info_of(model)

Return the session info that `model` runs, a model made by `turing_full_model`,
`turing_multiple_behavioral_sessions` or `turing_behavioral_session`, or by
`with_session_info`: the `SessionInfo` of a single session, or the `MultipleSessionsInfo`
of several. Used by `get_session_trajectories`, `get_session_parameters` and
`condition_sessions`.
"""
function info_of(model::DynamicPPL.Model)
    if model.f === CognitiveModeling.turing_behavioral_session
        return model.args.session_info
    else
        return model.args.sessions_info
    end
end

"""
    sample_numbering(chain)

Return the numbers of the iterations and the chains of `chain`, a `FlexiChain`, as keyword
arguments, so that the chain of the trajectories or the parameters is numbered the same way.
Used by `get_session_trajectories`, `get_session_parameters` and
`predict_session_trajectories`.
"""
function sample_numbering(chain::Turing.FlexiChains.FlexiChain)
    return (;
        iter_indices=Turing.FlexiChains.iter_indices(chain),
        chain_indices=Turing.FlexiChains.chain_indices(chain),
    )
end
