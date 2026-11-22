### EACH PARAMETER SAMPLED FOR EACH SESSION ###

#Each parameter sampled for every session from the same prior
Turing.@model function CognitiveModeling.turing_parameters_model(
    parameters_model::IndependentParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_independent_parameters(parameters_model.priors, n_sessions(sessions_info)),
        false,
    )
    return parameters
end

"""
    turing_independent_parameters(priors, n_sessions)

Turing.jl model which samples one parameter at a time from `priors`, each for every session
with [`turing_independent_parameter`](@ref) under the parameter's name, and returns them in
a `NamedTuple`. Used by [`IndependentParameters`](@ref).
"""
Turing.@model function turing_independent_parameters(
    priors::NamedTuple{names}, n_sessions
) where {names}
    name = first(names)
    sessions ~ to_submodel(
        prefix(
            turing_independent_parameter(filldist(first(priors), n_sessions)),
            VarName{name}(),
        ),
        false,
    )
    rest ~ to_submodel(turing_independent_parameters(Base.tail(priors), n_sessions), false)
    return merge(NamedTuple{(name,)}((sessions,)), rest)
end

Turing.@model function turing_independent_parameters(priors::NamedTuple{()}, n_sessions)
    return (;)
end

"""
    turing_independent_parameter(distribution)

Turing.jl model which samples the values of one parameter for every session from
`distribution`, a distribution over an array with the value of each session along its last
dimension, as the variable `sessions`, so that under the parameter's prefix it is named like
`learning_rate.sessions`, and returns the value of each session with
[`values_of_sessions`](@ref). Used by [`turing_independent_parameters`](@ref).
"""
Turing.@model function turing_independent_parameter(distribution)
    sessions ~ distribution
    return values_of_sessions(sessions)
end
