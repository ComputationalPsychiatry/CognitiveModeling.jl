### THE VALUE OF EACH SESSION ###

#A parameter whose value is an array, such as a vector of probabilities from a Dirichlet prior, is sampled for every session at once, as an array with one more dimension, which has the sessions along it. Each session gets its own array, copied rather than viewed: a model which reads its parameter on every timestep runs faster on a vector of its own than on a view, which makes up for the copy many times over, and not every distribution takes a view, as Distributions.jl's Categorical doesn't
"""
    values_of_sessions(values)

Return the value of each session in `values`: a vector with a value for every session as it
is, and a matrix or an array of three dimensions with the value of each session along its
last dimension as a vector with the vector or the matrix of each session. Used by
[`turing_independent_parameter`](@ref) and [`turing_hierarchical_grouped_parameter`](@ref).
"""
values_of_sessions(values::AbstractVector) = values
function values_of_sessions(values::AbstractMatrix)
    return [values[:, session] for session in axes(values, 2)]
end
function values_of_sessions(values::AbstractArray{<:Any,3})
    return [values[:, :, session] for session in axes(values, 3)]
end
