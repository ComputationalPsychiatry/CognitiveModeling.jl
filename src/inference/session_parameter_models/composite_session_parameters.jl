## Several session parameter models, each giving some of the parameters ##

"""
    CompositeSessionParameters(parts...; named_parts...)

The session parameter model of a full model which gives some of the parameters with each of
its parts, which are session parameter models, such as a
[`GroupedHierarchicalSessionParameters`](@ref) for the learning rate, an
[`IndependentSessionParameters`](@ref) for the inverse temperature and a
[`FixedSessionParameters`](@ref) for a bias. What the parts give is merged. The variables
of a part in `parts` are named as that part names them, and those of a part given by
keyword in `named_parts` are named under its keyword, such as `noise.learning_rate.session`
for a part given as `noise`. No two parts can give the same parameter: this is checked when
it is made, for the parts whose [`CognitiveModeling.parameter_names`](@ref) are known, and
when it runs, for the others. It samples when any part does. The parts are kept in the
field `parts`, as a tuple, with those in `named_parts` last.

# Examples
```julia
using Turing

session_parameter_model = CompositeSessionParameters(
    IndependentSessionParameters((; learning_rate = Beta(2, 2))),
    FixedSessionParameters((; inverse_temperature = 2.0)),
)

#The variables of the part given as noise are named under noise, as noise.bias.session
session_parameter_model = CompositeSessionParameters(
    IndependentSessionParameters((; learning_rate = Beta(2, 2)));
    noise = IndependentSessionParameters((; bias = Normal(0, 1))),
)
```
"""
struct CompositeSessionParameters{Tparts<:Tuple,Tpart_names} <:
       AbstractSessionParametersModel
    parts::Tparts

    #Takes the name of each part, or nothing for a part without one, in the type, and checks that no two of the parts whose parameters are known give the same parameter
    function CompositeSessionParameters{Tparts,Tpart_names}(
        parts::Tparts
    ) where {Tparts<:Tuple,Tpart_names}
        check_parts_give_different_parameters(parts)
        return new{Tparts,Tpart_names}(parts)
    end
end

#Takes the parts without a name first, and those with a name as keyword arguments, which must be session parameter models too
function CompositeSessionParameters(
    parts::AbstractSessionParametersModel...; named_parts...
)
    named_parts = NamedTuple(named_parts)
    for (name, part) in pairs(named_parts)
        if !(part isa AbstractSessionParametersModel)
            throw(
                ArgumentError(
                    "The part $name of CompositeSessionParameters is of type $(typeof(part)), but every part must be a session parameter model, a subtype of AbstractSessionParametersModel.",
                ),
            )
        end
    end
    all_parts = (parts..., values(named_parts)...)
    part_names = (map(_ -> nothing, parts)..., keys(named_parts)...)
    return CompositeSessionParameters{typeof(all_parts),part_names}(all_parts)
end

"""
    as_session_parameters_model(parameters_model)

Return `parameters_model` as one session parameter model: a tuple of session parameter
models as a [`CompositeSessionParameters`](@ref) of them, and a `NamedTuple` of them as a
`CompositeSessionParameters` with each part under its name, which its variables are then
named under. Anything else is returned as it is, for [`prepare_full_model`](@ref) to check.
A tuple or a `NamedTuple` with something other than a session parameter model in it is an
error. Used by `prepare_full_model` and by the functions of the inference backends that
make a full model, such as `turing_full_model`.
"""
as_session_parameters_model(parameters_model) = parameters_model
function as_session_parameters_model(parts::Tuple)
    for (index, part) in enumerate(parts)
        if !(part isa AbstractSessionParametersModel)
            throw(
                ArgumentError(
                    "A session parameter model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it was given a tuple whose part $index has type $(typeof(part)). A tuple of session parameter models is made into a CompositeSessionParameters of them.",
                ),
            )
        end
    end
    return CompositeSessionParameters(parts...)
end
function as_session_parameters_model(named_parts::NamedTuple)
    for (name, part) in pairs(named_parts)
        if !(part isa AbstractSessionParametersModel)
            throw(
                ArgumentError(
                    "A session parameter model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it was given a NamedTuple whose $name has type $(typeof(part)). A NamedTuple of session parameter models is made into a CompositeSessionParameters of them, each under its name.",
                ),
            )
        end
    end
    return CompositeSessionParameters(; named_parts...)
end

"""
    part_names(parameters_model)

Return the name of each part of `parameters_model`, a [`CompositeSessionParameters`](@ref),
in a tuple in the order of its parts, with `nothing` for a part given without a name. Used
by the models of `CompositeSessionParameters` in the inference backends, which name the
variables of a part under its name, and by [`prepare_parameters_model`](@ref) and
[`add_part_without_name`](@ref), which keep the names.
"""
function part_names(
    ::CompositeSessionParameters{Tparts,Tpart_names}
) where {Tparts,Tpart_names}
    return Tpart_names
end

"""
    add_part_without_name(parameters_model, part)

Return `parameters_model` with `part` added to it as a part without a name: a
[`CompositeSessionParameters`](@ref) with its own parts, which keep their names, and then
`part`, or a new `CompositeSessionParameters` of `parameters_model` and `part` for any other
session parameter model. Used by [`with_fixed_parameters`](@ref).
"""
function add_part_without_name(parameters_model::CompositeSessionParameters, part)
    parts = (parameters_model.parts..., part)
    return CompositeSessionParameters{
        typeof(parts),(part_names(parameters_model)..., nothing)
    }(
        parts
    )
end
function add_part_without_name(parameters_model, part)
    return CompositeSessionParameters(parameters_model, part)
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

#Each part is prepared for the sessions, and keeps its name
function prepare_parameters_model(
    parameters_model::CompositeSessionParameters, sessions_info::MultipleSessionsInfo
)
    prepared_parts = map(
        part -> prepare_parameters_model(part, sessions_info), parameters_model.parts
    )
    return CompositeSessionParameters{typeof(prepared_parts),part_names(parameters_model)}(
        prepared_parts
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
