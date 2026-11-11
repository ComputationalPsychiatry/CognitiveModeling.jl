### PARAMETERS MODELS, WHICH GIVE OR SAMPLE THE PARAMETERS OF EVERY SESSION ###
# The same kinds serve the parameters of the cognitive model and those of the environment model. Each returns one vector
# for each parameter, with a value for every session, in the order of the sessions.

#No parameters, so that each takes the value the model gives it by default
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    ::NoParameters, sessions_info::MultipleSessionsInfo
)
    return (;)
end

#The same given values for every session
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::FixedSessionParameters, sessions_info::MultipleSessionsInfo
)
    return map(value -> fill(value, n_sessions(sessions_info)), parameters_model.parameters)
end

#Given values for each session, found by the session's identifier
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::FixedParametersBySession, sessions_info::MultipleSessionsInfo
)
    return parameters_by_session(
        parameters_model.parameters_by_session, sessions_info.session_ids
    )
end

#Each parameter sampled once, and given to every session
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::IdenticalSessionParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_sample_identical_parameters(
            parameters_model.priors, n_sessions(sessions_info)
        ),
        false,
    )
    return parameters
end

#Each parameter sampled for every session from the same prior
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::IndependentSessionParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_sample_independent_parameters(
            parameters_model.priors, n_sessions(sessions_info)
        ),
        false,
    )
    return parameters
end

"""
    turing_sample_identical_parameters(priors, n_sessions)

Turing.jl model which samples one parameter at a time from `priors`, each once, with
[`turing_sample_shared_parameter`](@ref) under the parameter's name, and returns them in a
`NamedTuple`, each as a vector with that value for each of `n_sessions` sessions. Used by
[`IdenticalSessionParameters`](@ref).
"""
Turing.@model function turing_sample_identical_parameters(
    priors::NamedTuple{names}, n_sessions
) where {names}
    name = first(names)
    value ~ to_submodel(
        prefix(turing_sample_shared_parameter(first(priors)), VarName{name}()), false
    )
    rest ~
    to_submodel(turing_sample_identical_parameters(Base.tail(priors), n_sessions), false)
    return merge(NamedTuple{(name,)}((fill(value, n_sessions),)), rest)
end

Turing.@model function turing_sample_identical_parameters(
    priors::NamedTuple{()}, n_sessions
)
    return (;)
end

"""
    turing_sample_shared_parameter(prior)

Turing.jl model which samples the value of one parameter from `prior`, as the variable
`shared`, so that under the parameter's prefix it is named like `learning_rate.shared`. Used
by [`turing_sample_identical_parameters`](@ref).
"""
Turing.@model function turing_sample_shared_parameter(prior)
    shared ~ prior
    return shared
end

"""
    turing_sample_independent_parameters(priors, n_sessions)

Turing.jl model which samples one parameter at a time from `priors`, each for every session
with [`turing_sample_parameter_across_sessions`](@ref) under the parameter's name, and
returns them in a `NamedTuple`. Used by [`IndependentSessionParameters`](@ref).
"""
Turing.@model function turing_sample_independent_parameters(
    priors::NamedTuple{names}, n_sessions
) where {names}
    name = first(names)
    sessions ~ to_submodel(
        prefix(
            turing_sample_parameter_across_sessions(filldist(first(priors), n_sessions)),
            VarName{name}(),
        ),
        false,
    )
    rest ~
    to_submodel(turing_sample_independent_parameters(Base.tail(priors), n_sessions), false)
    return merge(NamedTuple{(name,)}((sessions,)), rest)
end

Turing.@model function turing_sample_independent_parameters(
    priors::NamedTuple{()}, n_sessions
)
    return (;)
end

"""
    turing_sample_parameter_across_sessions(distribution)

Turing.jl model which samples the values of one parameter for every session from
`distribution`, a distribution over a vector, as the variable `session`, so that under the
parameter's prefix it is named like `learning_rate.session`. Used by
[`turing_sample_independent_parameters`](@ref).
"""
Turing.@model function turing_sample_parameter_across_sessions(distribution)
    session ~ distribution
    return session
end
