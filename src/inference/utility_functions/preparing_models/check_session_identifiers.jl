## Checking the identifiers of the sessions ##

"""
    check_unique_session_symbol_names(session_ids)

Throw an informative error if the identifiers of two sessions give the same symbol name, as
[`session_id_to_symbol`](@ref) gives it: when they are the same identifier, or give the same
text, such as `1` and `"1"`. Used by [`MultipleSessionsInfo`](@ref).
"""
function check_unique_session_symbol_names(session_ids::AbstractVector)
    symbol_names = map(session_id_to_symbol, session_ids)
    allunique(symbol_names) && return nothing
    for (i, symbol_name) in enumerate(symbol_names)
        first_i = findfirst(==(symbol_name), symbol_names)
        first_i == i && continue
        first_id = session_ids[first_i]
        session_id = session_ids[i]
        if isequal(first_id, session_id)
            throw(
                ArgumentError(
                    "The session identifier $(repr(session_id)) is given for more than one session. Every session must have an identifier of its own.",
                ),
            )
        end
        throw(
            ArgumentError(
                "The session identifiers $(repr(first_id)) and $(repr(session_id)) both give the symbol name $symbol_name, which the chains of a fit and FixedParametersBySession find a session by. The identifier of every session must give a symbol name of its own.",
            ),
        )
    end
    return nothing
end
