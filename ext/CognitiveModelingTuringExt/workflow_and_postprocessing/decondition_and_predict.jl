### TURING.JL'S OWN FUNCTIONS FOR PREDICTIONS, ON THE MODELS OF THE PACKAGE ###

#The Turing.jl models that the package makes around sessions, those that is_session_model recognizes, whose session info with_session_info can change
const SessionModel = DynamicPPL.Model{
    <:Union{
        typeof(CognitiveModeling.turing_full_model),
        typeof(CognitiveModeling.turing_multiple_behavioral_sessions),
        typeof(CognitiveModeling.turing_behavioral_session),
        typeof(CognitiveModeling.turing_catch_invalid_parameters),
    },
}

#Turing.jl's decondition leaves the arguments of a model as they are, and the models of the package take their data as arguments, so their own method deconditions their sessions too
"""
    decondition(model)

Return a copy of `model`, a Turing.jl model made by [`full_model`](@ref),
[`multiple_sessions_model`](@ref) or [`session_model`](@ref) with `TuringBackend()`, or such
a model inside [`catch_invalid_parameters`](@ref), whose sessions are deconditioned with
[`decondition_sessions`](@ref), and which is conditioned on nothing that `condition` gave it.
Its data is then simulated instead of fitted to: `predict` simulates it with the parameters
of each sample of a chain, and `sample` with `Prior()` simulates it from the prior.
"""
function DynamicPPL.decondition(model::SessionModel)
    with_deconditioned_sessions = with_session_info(
        model, decondition_sessions, "decondition"
    )
    #What condition gave the model is removed by Turing.jl's own method, as for any model
    return invoke(
        DynamicPPL.decondition, Tuple{DynamicPPL.Model}, with_deconditioned_sessions
    )
end
