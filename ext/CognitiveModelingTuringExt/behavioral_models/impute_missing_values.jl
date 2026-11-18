### IMPUTING THE VALUES THAT ARE MISSING IN THE DATA ###

"""
    turing_impute_missing_values(data, distributions)

Turing.jl model which fills in the outputs of one model on one timestep: each value that is
[`MissingValue`](@ref) in `data` is imputed as a latent variable named after it, such as
`choice.imputed`, with the distribution the model gives for it in `distributions`, and the
rest are taken from the data. It goes through the names of `data` one at a time and imputes
each value with [`turing_impute_value`](@ref), and returns the rest of the data as it is
once nothing in it is missing. For an environment it runs under the prefix `environment`,
so that an observation and an action with the same name get a latent variable each. Used by
[`turing_timestep_one_perspective`](@ref).
"""
Turing.@model function turing_impute_missing_values(
    data::NamedTuple{names}, distributions
) where {names}

    #The data as it is when nothing in it is missing
    if !has_missing_value(data)
        return data
    end

    #The first value, imputed under its name when it is missing, and then the rest of the data
    name = first(names)
    value ~ to_submodel(
        prefix(
            turing_impute_value(first(data), getproperty(distributions, name)),
            VarName{name}(),
        ),
        false,
    )
    rest ~ to_submodel(turing_impute_missing_values(Base.tail(data), distributions), false)
    return NamedTuple{names}((value, Tuple(rest)...))
end

Turing.@model function turing_impute_missing_values(data::NamedTuple{()}, distributions)
    return (;)
end

"""
    turing_impute_value(data_value, distribution)

Turing.jl model for one value of the data: a value that is there is taken as it is, and so
is any marker but [`MissingValue`](@ref), which is a latent variable named `imputed` with
`distribution`, the value of a [`Deterministic`](@ref), or the marker itself when the model
gives [`NoValue`](@ref) on that timestep. Used by [`turing_impute_missing_values`](@ref).
"""
Turing.@model function turing_impute_value(data_value, distribution)
    return data_value
end

#A missing value is a latent, with the distribution the model gives for it
Turing.@model function turing_impute_value(::MissingValue, distribution)
    imputed ~ distribution
    return imputed
end

#A missing value that the model gives without randomness is that value, with nothing to sample
Turing.@model function turing_impute_value(::MissingValue, deterministic::Deterministic)
    return deterministic.value
end

#A missing value where the model takes no action or gives no observation on this timestep stays as the model gives it
Turing.@model function turing_impute_value(::MissingValue, no_value::AbstractNoValue)
    return no_value
end
