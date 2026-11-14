### EACH PARAMETER SAMPLED ONCE, FOR EVERY SESSION ###

#Each parameter sampled once, and given to every session
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::IdenticalSessionParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_identical_parameters(parameters_model.priors, n_sessions(sessions_info)),
        false,
    )
    return parameters
end

"""
    turing_identical_parameters(priors, n_sessions)

Turing.jl model which samples one parameter at a time from `priors`, each once, with
[`turing_identical_parameter`](@ref) under the parameter's name, and returns them in a
`NamedTuple`, each as a vector with that value for each of `n_sessions` sessions. Used by
[`IdenticalSessionParameters`](@ref).
"""
Turing.@model function turing_identical_parameters(
    priors::NamedTuple{names}, n_sessions
) where {names}
    name = first(names)
    value ~
    to_submodel(prefix(turing_identical_parameter(first(priors)), VarName{name}()), false)
    rest ~ to_submodel(turing_identical_parameters(Base.tail(priors), n_sessions), false)
    return merge(NamedTuple{(name,)}((fill(value, n_sessions),)), rest)
end

Turing.@model function turing_identical_parameters(priors::NamedTuple{()}, n_sessions)
    return (;)
end

"""
    turing_identical_parameter(prior)

Turing.jl model which samples the value of one parameter from `prior`, as the variable
`shared`, so that under the parameter's prefix it is named like `learning_rate.shared`. Used
by [`turing_identical_parameters`](@ref).
"""
Turing.@model function turing_identical_parameter(prior)
    shared ~ prior
    return shared
end
