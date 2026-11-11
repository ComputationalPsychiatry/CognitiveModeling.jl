## Given values for each session ##

"""
    FixedParametersBySession(parameters_by_session)

The session parameter model of a full model which gives each session values of its own, with
nothing sampled. `parameters_by_session` holds the values of each session, either in a
`NamedTuple` with an entry for each session, named by the session's name as
[`CognitiveModeling.session_name`](@ref) gives it, such as
`(; S1 = (; scale = 2.0), S2 = (; scale = 3.0))`, or in a dictionary from each session's
identifier to its values, such as `Dict("S1" => (; scale = 2.0), "S2" => (; scale = 3.0))`.
An identifier from several session columns is the `NamedTuple` of their values, so its
entry is named like `id_S1__block_2`, and its key is like `(; id = "S1", block = 2)`. Every
session must have values for the same parameters. The values can also be given in the order
of the sessions, as a `NamedTuple` with one vector for each parameter, such as
`(; scale = [2.0, 3.0])`, which is what a full model makes of the other two forms once,
before it is made, so that no session is looked up when it runs. The values are kept in the
field `parameters_by_session`.
"""
struct FixedParametersBySession{Tparameters_by_session<:Union{NamedTuple,AbstractDict}} <:
       AbstractSessionParametersModel
    parameters_by_session::Tparameters_by_session
end

#It gives the parameters of its vectors, or those that its first session has values for, and estimates nothing
function parameter_names(parameters_model::FixedParametersBySession)
    values_by_session = parameters_model.parameters_by_session
    isempty(values_by_session) && return ()
    has_vector_for_each_parameter(values_by_session) && return keys(values_by_session)
    return keys(first(values(values_by_session)))
end
estimates_parameters(::FixedParametersBySession) = false

#The values of every session are looked up once, when a full model is prepared, and kept as one vector for each parameter
function prepare_parameters_model(
    parameters_model::FixedParametersBySession, sessions_info::MultipleSessionsInfo
)
    return FixedParametersBySession(
        parameters_by_session(
            parameters_model.parameters_by_session, sessions_info.session_ids
        ),
    )
end

"""
    has_vector_for_each_parameter(values_by_session)

Return whether `values_by_session`, as a [`FixedParametersBySession`](@ref) holds it, is a
`NamedTuple` with one vector for each parameter, with the values in the order of the
sessions, and not the values of each session under its name or its identifier. Used by
[`parameter_names`](@ref) and [`parameters_by_session`](@ref).
"""
has_vector_for_each_parameter(values_by_session) = false
function has_vector_for_each_parameter(values_by_session::NamedTuple)
    return all(values -> values isa AbstractVector, values_by_session)
end

"""
    parameters_by_session(values_by_session, session_ids)

Return the values of each session in `values_by_session`, as
[`FixedParametersBySession`](@ref) holds them, as one vector for each parameter with a value
for every session, in the order of `session_ids`, with the values of each session found by
[`values_of_session`](@ref). A session with other parameters than the first is an error.
Values that are already one vector for each parameter are returned as they are, after
checking that each has a value for every session. Used by
[`prepare_parameters_model`](@ref), and by the models of `FixedParametersBySession` in the
inference backends.
"""
function parameters_by_session(values_by_session, session_ids)

    #Values that are already in the order of the sessions only need a value for each session
    if has_vector_for_each_parameter(values_by_session)
        for (name, values) in pairs(values_by_session)
            if length(values) != length(session_ids)
                throw(
                    DimensionMismatch(
                        "FixedParametersBySession has $(length(values)) values of $name, but there are $(length(session_ids)) sessions. A vector must have one value for every session, in the order of the sessions.",
                    ),
                )
            end
        end
        return values_by_session
    end

    #Otherwise the values of each session are found by its name or its identifier
    values_of_each = map(
        session_id -> values_of_session(values_by_session, session_id), session_ids
    )
    names = keys(first(values_of_each))
    for (session_id, values) in zip(session_ids, values_of_each)
        if keys(values) != names
            throw(
                ArgumentError(
                    "FixedParametersBySession gives the session $(repr(session_id)) the parameters $(keys(values)), and the first session the parameters $names. Every session must have values for the same parameters.",
                ),
            )
        end
    end
    return NamedTuple{names}(
        map(name -> [getproperty(values, name) for values in values_of_each], names)
    )
end

"""
    values_of_session(values_by_session, session_id)

Return the values of the session with the identifier `session_id` in `values_by_session`:
the entry named by the session's name, as [`session_name`](@ref) gives it, in a
`NamedTuple`, and the value stored under the identifier itself in a dictionary. A session
without values is an error. Used by [`parameters_by_session`](@ref).
"""
function values_of_session(values_by_session::NamedTuple, session_id)
    name = session_name(session_id)
    if !haskey(values_by_session, name)
        throw(
            ArgumentError(
                "FixedParametersBySession has no values for the session $name. It has values for $(keys(values_by_session)).",
            ),
        )
    end
    return getproperty(values_by_session, name)
end

function values_of_session(values_by_session::AbstractDict, session_id)
    if !haskey(values_by_session, session_id)
        throw(
            ArgumentError(
                "FixedParametersBySession has no values for the session $(repr(session_id)). It has values for the sessions $(collect(keys(values_by_session))).",
            ),
        )
    end
    return values_by_session[session_id]
end
