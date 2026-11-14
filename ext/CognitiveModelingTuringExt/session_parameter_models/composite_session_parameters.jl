### SEVERAL SESSION PARAMETER MODELS, EACH GIVING SOME OF THE PARAMETERS ###

#Each part gives some of the parameters, under its name if it has one, and they are merged
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::CompositeSessionParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_composite_parameters(
            parameters_model.parts, Val(part_names(parameters_model)), sessions_info
        ),
        false,
    )
    return parameters
end

"""
    turing_composite_parameters(parts, Val(part_names), sessions_info)

Turing.jl model which runs the session parameter models in `parts`, a tuple, one at a time,
each with its variables under its name in `part_names`, or as it names them itself when its
name is `nothing`, and returns the parameters they give merged into one `NamedTuple`, which
is an error when two give the same parameter. Used by [`CompositeSessionParameters`](@ref).
"""
Turing.@model function turing_composite_parameters(
    parts::Tuple, ::Val{Tpart_names}, sessions_info::MultipleSessionsInfo
) where {Tpart_names}
    parameters ~ to_submodel(
        turing_part_model(first(parts), Val(first(Tpart_names)), sessions_info), false
    )
    other_parameters ~ to_submodel(
        turing_composite_parameters(
            Base.tail(parts), Val(Base.tail(Tpart_names)), sessions_info
        ),
        false,
    )
    return merge_parameters_of_parts(parameters, other_parameters)
end

Turing.@model function turing_composite_parameters(
    parts::Tuple{}, ::Val{()}, sessions_info::MultipleSessionsInfo
)
    return (;)
end

"""
    turing_part_model(part, Val(name), sessions_info)

Return the Turing.jl model of `part`, a part of a [`CompositeSessionParameters`](@ref), with
its variables under `name`, or as the part names them itself when `name` is `nothing`. Used
by [`turing_composite_parameters`](@ref).
"""
function turing_part_model(part, ::Val{nothing}, sessions_info::MultipleSessionsInfo)
    return CognitiveModeling.turing_session_parameters_model(part, sessions_info)
end
function turing_part_model(
    part, ::Val{name}, sessions_info::MultipleSessionsInfo
) where {name}
    return prefix(
        CognitiveModeling.turing_session_parameters_model(part, sessions_info),
        VarName{name}(),
    )
end
