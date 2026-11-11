## Each parameter sampled once, for every session ##

"""
    IdenticalSessionParameters(priors)

The session parameter model of a full model which samples each parameter once, from its
prior, and gives every session that value. `priors` is a `NamedTuple` with the name of each
parameter that is fitted and the distribution it is sampled from, such as
`(; learning_rate = Beta(2, 2))`. The priors are kept in the field `priors`.
"""
struct IdenticalSessionParameters{Tpriors<:NamedTuple} <: AbstractSessionParametersModel
    priors::Tpriors
end

#It gives the parameters it has priors for
function parameter_names(parameters_model::IdenticalSessionParameters)
    return keys(parameters_model.priors)
end
