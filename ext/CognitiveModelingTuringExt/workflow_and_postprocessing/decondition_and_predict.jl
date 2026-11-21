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

#Turing.jl's predict keeps the value that the chain has for a variable, so the actions that a fit imputed would stay as they were, so the method for the models of the package leaves the variables within the timesteps out of the chain first, as predict_session_trajectories does. That also keeps FlexiChains.jl from rebuilding a sample whose sessions have variables at different timesteps, which fails, as without_timestep_variables says
"""
    predict([rng,] model, chain; include_all=true)

Simulate `model`, a Turing.jl model made by [`full_model`](@ref),
[`multiple_sessions_model`](@ref) or [`session_model`](@ref) with `TuringBackend()`, or such
a model inside [`catch_invalid_parameters`](@ref), for every sample of `chain`, a
`FlexiChain`, and return a `FlexiChain` of the variables it simulated, beside the variables
of `chain` when `include_all` is `true`, as Turing.jl's own `predict` does. The parameters of
each sample, and any start that a model samples, are taken from `chain`, and every variable
within the timesteps of the sessions is simulated again, also an action or an observation
that was imputed in the fit, as data is in a predictive check. With a model from
`decondition(model)`, every action is simulated, and every observation too when the sessions
have an environment.
"""
function DynamicPPL.predict(
    rng::AbstractRNG,
    model::SessionModel,
    chain::Turing.FlexiChains.FlexiChain{<:VarName};
    include_all::Bool=true,
)
    #The statistics of the sampler are kept, which Turing.jl's own predict returns too
    chain_without_timestep_variables = merge(
        Turing.FlexiChains.subset_extras(chain), without_timestep_variables(chain)
    )
    return invoke(
        DynamicPPL.predict,
        Tuple{AbstractRNG,DynamicPPL.Model,typeof(chain_without_timestep_variables)},
        rng,
        model,
        chain_without_timestep_variables;
        include_all,
    )
end
