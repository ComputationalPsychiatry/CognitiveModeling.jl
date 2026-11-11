## Several session parameter models, each giving some of the parameters ##

"""
    CompositeSessionParameters(parts...)

The session parameter model of a full model which gives some of the parameters with each of
`parts`, which are session parameter models, such as a
[`GroupedHierarchicalSessionParameters`](@ref) for the learning rate, an
[`IndependentSessionParameters`](@ref) for the inverse temperature and a
[`FixedSessionParameters`](@ref) for a bias. What the parts give is merged, and the
variables of each part are named as that part names them. No two parts can give the same
parameter: this is checked when it is made, for the parts whose
[`CognitiveModeling.parameter_names`](@ref) are known, and when it runs, for the others. It
samples when any part does. The parts are kept in the field `parts`, as a tuple.

# Examples
```julia
using Turing

session_parameter_model = CompositeSessionParameters(
    IndependentSessionParameters((; learning_rate = Beta(2, 2))),
    FixedSessionParameters((; inverse_temperature = 2.0)),
)
```
"""
struct CompositeSessionParameters{Tparts<:Tuple} <: AbstractSessionParametersModel
    parts::Tparts

    #Checks that no two of the parts whose parameters are known give the same parameter
    function CompositeSessionParameters(parts::AbstractSessionParametersModel...)
        check_parts_give_different_parameters(parts)
        return new{typeof(parts)}(parts)
    end
end

#It gives what its parts give, when every part tells what it gives, and it estimates when any part does
function parameter_names(parameters_model::CompositeSessionParameters)
    names_of_parts = map(parameter_names, parameters_model.parts)
    any(isnothing, names_of_parts) && return nothing
    return Tuple(Iterators.flatten(names_of_parts))
end

function estimates_parameters(parameters_model::CompositeSessionParameters)
    return any(estimates_parameters, parameters_model.parts)
end

#Each part is prepared for the sessions
function prepare_parameters_model(
    parameters_model::CompositeSessionParameters, sessions_info::MultipleSessionsInfo
)
    return CompositeSessionParameters(
        map(
            part -> prepare_parameters_model(part, sessions_info), parameters_model.parts
        )...,
    )
end

"""
    check_parts_give_different_parameters(parts)

Throw an informative error if two of `parts`, the parts of a
[`CompositeSessionParameters`](@ref), give the same parameter, for the parts whose
[`parameter_names`](@ref) are known. Used by the constructor of
`CompositeSessionParameters`.
"""
function check_parts_give_different_parameters(parts::Tuple)
    given_by = Dict{Symbol,Any}()
    for part in parts
        names = parameter_names(part)
        names === nothing && continue
        for name in names
            if haskey(given_by, name)
                throw(
                    ArgumentError(
                        "Two parts of CompositeSessionParameters give the parameter $name, of the types $(nameof(typeof(given_by[name]))) and $(nameof(typeof(part))). Each parameter must come from one part.",
                    ),
                )
            end
            given_by[name] = part
        end
    end
    return nothing
end

"""
    merge_parameters_of_parts(parameters, other_parameters)

Return the parameters that two parts of a [`CompositeSessionParameters`](@ref) give, each a
`NamedTuple` with one vector for each parameter, merged into one, after checking that they
give different parameters, which matters for the parts whose names weren't known when the
model was made. Used by the models of `CompositeSessionParameters` in the inference
backends.
"""
function merge_parameters_of_parts(parameters::NamedTuple, other_parameters::NamedTuple)
    shared_names = Tuple(
        name for name in keys(parameters) if name in keys(other_parameters)
    )
    if !isempty(shared_names)
        throw(
            ArgumentError(
                "Two parts of CompositeSessionParameters give the parameters $shared_names. Each parameter must come from one part.",
            ),
        )
    end
    return merge(parameters, other_parameters)
end
