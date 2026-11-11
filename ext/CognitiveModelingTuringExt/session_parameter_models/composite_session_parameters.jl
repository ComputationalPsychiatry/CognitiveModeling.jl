### SEVERAL SESSION PARAMETER MODELS, EACH GIVING SOME OF THE PARAMETERS ###

#Each part gives some of the parameters, which are merged
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::CompositeSessionParameters, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        turing_sample_parameters_of_parts(parameters_model.parts, sessions_info), false
    )
    return parameters
end

"""
    turing_sample_parameters_of_parts(parts, sessions_info)

Turing.jl model which runs the session parameter models in `parts`, a tuple, one at a time,
each as it names its own variables, and returns the parameters they give merged into one
`NamedTuple`, which is an error when two give the same parameter. Used by
[`CompositeSessionParameters`](@ref).
"""
Turing.@model function turing_sample_parameters_of_parts(
    parts::Tuple, sessions_info::MultipleSessionsInfo
)
    parameters ~ to_submodel(
        CognitiveModeling.turing_session_parameters_model(first(parts), sessions_info),
        false,
    )
    other_parameters ~
    to_submodel(turing_sample_parameters_of_parts(Base.tail(parts), sessions_info), false)
    return merge_parameters_of_parts(parameters, other_parameters)
end

Turing.@model function turing_sample_parameters_of_parts(
    parts::Tuple{}, sessions_info::MultipleSessionsInfo
)
    return (;)
end
