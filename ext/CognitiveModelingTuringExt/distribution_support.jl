### WHAT THE DISTRIBUTIONS OF DISTRIBUTIONS.JL CAN GIVE ###

#The values that a univariate distribution can give are from its minimum to its maximum, and discrete for a discrete distribution, so that a prior which reaches outside the domain of its parameter is an error when a full model is prepared
function CognitiveModeling.distribution_support(distribution::UnivariateDistribution)
    return (;
        lower=minimum(distribution),
        upper=maximum(distribution),
        discrete=distribution isa DiscreteUnivariateDistribution,
    )
end
