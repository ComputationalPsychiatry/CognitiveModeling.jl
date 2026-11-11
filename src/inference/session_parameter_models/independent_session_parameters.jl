## Each parameter sampled for each session ##

"""
    IndependentSessionParameters(priors)

The session parameter model of a full model which samples each parameter for every session
from the same prior, with nothing shared between the sessions. `priors` is a `NamedTuple`
with the name of each parameter that is fitted and the distribution it is sampled from, such
as `(; learning_rate = Beta(2, 2), inverse_temperature = LogNormal(0, 1))`. The priors are
kept in the field `priors`.
"""
struct IndependentSessionParameters{Tpriors<:NamedTuple} <: AbstractSessionParametersModel
    priors::Tpriors
end

#It gives the parameters it has priors for
function parameter_names(parameters_model::IndependentSessionParameters)
    return keys(parameters_model.priors)
end
