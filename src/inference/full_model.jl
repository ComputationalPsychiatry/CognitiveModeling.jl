## The full model of an inference backend ##

"""
    AbstractInferenceBackend

Supertype of the markers of the inference backends, which [`full_model`](@ref) takes to
make the full model of one, such as [`TuringBackend`](@ref).
"""
abstract type AbstractInferenceBackend end

"""
    full_model(backend::AbstractInferenceBackend, cognitive_parameters_model,
               sessions_info::MultipleSessionsInfo;
               environment_parameters_model=NoParameters(),
               downstream_model=NoDownstreamModel(), fixed_cognitive_parameters=(;),
               fixed_environment_parameters=(;), keywords...)

Create the full model of the inference backend that `backend` marks, which fits a cognitive
model to the sessions in `sessions_info`: a parameter model gives or samples the parameters
of every session, `cognitive_parameters_model` for the cognitive model and
`environment_parameters_model` for the environment model, the sessions run with them, and
`downstream_model` can use the trajectories they saved. Every backend takes these arguments
and keywords, which [`prepare_full_model`](@ref) prepares, and its page describes them and
the keywords of its own. The package of the backend must be loaded.
"""
function full_model end

#Without a method for the backend and these arguments, the package of the backend isn't loaded, or the arguments are of other kinds
function full_model(backend::AbstractInferenceBackend, arguments...; keywords...)
    return throw(
        ArgumentError(
            "There is no full_model method for $(nameof(typeof(backend)))() with arguments of the types $(map(typeof, arguments)). Load the package of the inference backend, such as Turing.jl for TuringBackend(), and give a parameter model and a MultipleSessionsInfo.",
        ),
    )
end
