## Given parameters for each session ##

"""
    FixedParametersBySession(parameters_by_session)

The session parameter model of a full model which gives each session parameters of its own,
with nothing sampled. `parameters_by_session` holds the parameters of each session in one
of three forms: by the session's symbol name, in a `NamedTuple` with an entry for each
session, such as `(; S1 = (; scale = 2.0), S2 = (; scale = 3.0))`; by its identifier, in a
dictionary, such as `Dict("S1" => (; scale = 2.0), "S2" => (; scale = 3.0))`; or in the
order of the sessions, as a `NamedTuple` with one vector for each parameter, such as
`(; scale = [2.0, 3.0])`. A session with several session columns has an identifier like
`(; id = "S1", block = 2)` and a symbol name like `id_S1__block_2`, as
[`CognitiveModeling.session_id_to_symbol`](@ref) gives it. Every session must have the same
parameters. A full model makes the first two forms into the third once, before it is made.
The parameters are kept in the field `parameters_by_session`.
"""
struct FixedParametersBySession{Tparameters_by_session<:Union{NamedTuple,AbstractDict}} <:
       AbstractSessionParametersModel
    parameters_by_session::Tparameters_by_session
end

#It gives the parameters of its vectors, or those of its first session, and estimates nothing
function parameter_names(parameters_model::FixedParametersBySession)
    parameters_by_session = parameters_model.parameters_by_session
    isempty(parameters_by_session) && return ()
    has_vector_for_each_parameter(parameters_by_session) &&
        return keys(parameters_by_session)
    return keys(first(values(parameters_by_session)))
end
estimates_parameters(::FixedParametersBySession) = false

#The parameters of every session are looked up once, when a full model is prepared, and kept as one vector for each parameter
function prepare_parameters_model(
    parameters_model::FixedParametersBySession, sessions_info::MultipleSessionsInfo
)
    return FixedParametersBySession(
        collect_parameters_in_session_order(
            parameters_model.parameters_by_session, sessions_info.session_ids
        ),
    )
end

"""
    has_vector_for_each_parameter(parameters_by_session)

Return whether `parameters_by_session`, as a [`FixedParametersBySession`](@ref) holds it,
gives the parameters in the order of the sessions, as a `NamedTuple` with one vector for
each parameter, and not by the sessions' symbol names or identifiers. Used by
[`parameter_names`](@ref) and [`collect_parameters_in_session_order`](@ref).
"""
has_vector_for_each_parameter(parameters_by_session) = false
function has_vector_for_each_parameter(parameters_by_session::NamedTuple)
    return all(values -> values isa AbstractVector, parameters_by_session)
end

"""
    collect_parameters_in_session_order(parameters_by_session, session_ids)

Return the parameters of each session in `parameters_by_session`, as
[`FixedParametersBySession`](@ref) holds them, as one vector for each parameter with a value
for every session, in the order of `session_ids`, with the parameters of each session found
by [`parameters_of_session`](@ref). A session with other parameters than the first is an
error. Parameters that are already in the order of the sessions are returned as they are,
after checking that each vector has a value for every session. Used by
[`prepare_parameters_model`](@ref), and by the models of `FixedParametersBySession` in the
inference backends.
"""
function collect_parameters_in_session_order(parameters_by_session, session_ids)

    #Parameters that are already in the order of the sessions only need a value for each session
    if has_vector_for_each_parameter(parameters_by_session)
        for (name, values) in pairs(parameters_by_session)
            if length(values) != length(session_ids)
                throw(
                    DimensionMismatch(
                        "FixedParametersBySession has $(length(values)) values of $name, but there are $(length(session_ids)) sessions. A vector must have one value for every session, in the order of the sessions.",
                    ),
                )
            end
        end
        return parameters_by_session
    end

    #Otherwise the parameters of each session are found by its symbol name or its identifier
    parameters_of_each = map(
        session_id -> parameters_of_session(parameters_by_session, session_id), session_ids
    )
    names = keys(first(parameters_of_each))
    for (session_id, parameters) in zip(session_ids, parameters_of_each)
        if keys(parameters) != names
            throw(
                ArgumentError(
                    "FixedParametersBySession gives the session $(repr(session_id)) the parameters $(keys(parameters)), and the first session the parameters $names. Every session must have the same parameters.",
                ),
            )
        end
    end
    return NamedTuple{names}(
        map(
            name -> [getproperty(parameters, name) for parameters in parameters_of_each],
            names,
        ),
    )
end

"""
    parameters_of_session(parameters_by_session, session_id)

Return the parameters of the session with the identifier `session_id` in
`parameters_by_session`: the entry under the session's symbol name, as
[`session_id_to_symbol`](@ref) gives it, in a `NamedTuple`, and the entry under the
identifier itself in a dictionary. A session without parameters is an error. Used by
[`collect_parameters_in_session_order`](@ref).
"""
function parameters_of_session(parameters_by_session::NamedTuple, session_id)
    symbol_name = session_id_to_symbol(session_id)
    if !haskey(parameters_by_session, symbol_name)
        throw(
            ArgumentError(
                "FixedParametersBySession has no parameters for the session $symbol_name. It has parameters for $(keys(parameters_by_session)).",
            ),
        )
    end
    return getproperty(parameters_by_session, symbol_name)
end

function parameters_of_session(parameters_by_session::AbstractDict, session_id)
    if !haskey(parameters_by_session, session_id)
        throw(
            ArgumentError(
                "FixedParametersBySession has no parameters for the session $(repr(session_id)). It has parameters for the sessions $(collect(keys(parameters_by_session))).",
            ),
        )
    end
    return parameters_by_session[session_id]
end
