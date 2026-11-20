## The models that every inference backend makes ##

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

"""
    session_model(backend::AbstractInferenceBackend, session_info::SessionInfo,
                  cognitive_parameters, environment_parameters=(;))

Create the model of one session in the inference backend that `backend` marks: the
cognitive model in `session_info` goes through the timesteps of its data with
`cognitive_parameters`, and the actions in the data are scored against the action
distributions it gives. A session with an environment model takes `environment_parameters`
too. This can be used to evaluate or sample a session with given parameters, and as a part
of a model written in the backend. [`full_model`](@ref) estimates the parameters of several
sessions instead. The package of the backend must be loaded, and its page describes the
model.
"""
function session_model end

#Without a method for the backend and these arguments, the package of the backend isn't loaded, or the arguments are of other kinds
function session_model(backend::AbstractInferenceBackend, arguments...)
    return throw(
        ArgumentError(
            "There is no session_model method for $(nameof(typeof(backend)))() with arguments of the types $(map(typeof, arguments)). Load the package of the inference backend, such as Turing.jl for TuringBackend(), and give a SessionInfo and the parameters of its models.",
        ),
    )
end

"""
    multiple_sessions_model(backend::AbstractInferenceBackend,
                            sessions_info::MultipleSessionsInfo, cognitive_parameters,
                            environment_parameters=(;); keywords...)

Create the model of several sessions in the inference backend that `backend` marks, in
which each session in `sessions_info` runs as [`session_model`](@ref) with its own
parameters. `cognitive_parameters` is a `NamedTuple` with one vector for each parameter,
with one value for every session in the order of `sessions_info`, and
`environment_parameters` is in the same form, for sessions with an environment model. This
can be used as a part of a model written in the backend which gives the sessions their
parameters, as [`full_model`](@ref) does with parameter models. The package of the backend
must be loaded, and its page describes the model and the keywords of its own.
"""
function multiple_sessions_model end

#Without a method for the backend and these arguments, the package of the backend isn't loaded, or the arguments are of other kinds
function multiple_sessions_model(
    backend::AbstractInferenceBackend, arguments...; keywords...
)
    return throw(
        ArgumentError(
            "There is no multiple_sessions_model method for $(nameof(typeof(backend)))() with arguments of the types $(map(typeof, arguments)). Load the package of the inference backend, such as Turing.jl for TuringBackend(), and give a MultipleSessionsInfo and the parameters of its sessions.",
        ),
    )
end

"""
    catch_invalid_parameters(backend::AbstractInferenceBackend, model)

Create a model of the inference backend that `backend` marks which runs `model`, a model of
the backend, and rejects the parameters when a cognitive model or an environment model
throws an [`InvalidParametersError`](@ref), so that a sampler goes on instead of stopping.
The package of the backend must be loaded, and its page describes how the parameters are
rejected.
"""
function catch_invalid_parameters end

#Without a method for the backend and this argument, the package of the backend isn't loaded, or the model is of another kind
function catch_invalid_parameters(backend::AbstractInferenceBackend, arguments...)
    return throw(
        ArgumentError(
            "There is no catch_invalid_parameters method for $(nameof(typeof(backend)))() with arguments of the types $(map(typeof, arguments)). Load the package of the inference backend, such as Turing.jl for TuringBackend(), and give a model of the backend.",
        ),
    )
end
