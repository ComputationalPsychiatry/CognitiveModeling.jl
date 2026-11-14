## Functions that users call to read the sessions ##

"""
    n_sessions(sessions_info)

Return how many sessions a [`MultipleSessionsInfo`](@ref) has, which a session parameter
model gives the parameters of. Used by the session parameter models of the inference
backends, such as those of [`IndependentSessionParameters`](@ref).
"""
n_sessions(sessions_info::MultipleSessionsInfo) = length(sessions_info.sessions)

"""
    session_index(sessions_info, session_id)

Return the index `i` of the session with the identifier `session_id` in a
[`MultipleSessionsInfo`](@ref), which is how the models of several sessions number it: its
parameters are element `i` of each vector a session parameter model gives, and its saved
trajectories are element `i` of what the sessions return. The identifier is the value in the
session column, or a `NamedTuple` of the values when there are several, as in
`sessions_info.session_ids`. Used by users reading the results of a fit, not by the
models.
"""
function session_index(sessions_info::MultipleSessionsInfo, session_id)
    i = findfirst(==(session_id), sessions_info.session_ids)
    i === nothing && throw(
        ArgumentError(
            "There is no session with the identifier $(repr(session_id)). The identifiers are $(sessions_info.session_ids).",
        ),
    )
    return i
end

"""
    session_id_to_symbol(session_id)

Return the symbol name of a session, a `Symbol` made from its identifier: the value of a
single session column, as in `S1`, and the name and the value of each of several columns,
joined by `_` within a column and by `__` between columns, as in `id_S1__block_2`. The
chains of a fit name the entries of a session after it, such as `S1.learning_rate`, and
[`FixedSessionParametersBySession`](@ref) can find the values of a session by it. After
`using CognitiveModeling: session_id_to_symbol`, the learning rate of a session is
`chain[@varname(\$(session_id_to_symbol(session_id)).learning_rate)]`.

# Examples
```jldoctest
julia> CognitiveModeling.session_id_to_symbol("S1")
:S1

julia> CognitiveModeling.session_id_to_symbol((; id = "S1", block = 2))
:id_S1__block_2
```
"""
function session_id_to_symbol(session_id)

    #Asserted, since the compiler can't infer that Symbol gives a Symbol for an identifier of any type
    return Symbol(session_id)::Symbol
end
function session_id_to_symbol(session_id::NamedTuple)
    return Symbol(
        join((string(name, "_", value) for (name, value) in pairs(session_id)), "__")
    )
end

"""
    session_column_values(sessions_info, column)

Return the values of `column` in the table that the sessions in `sessions_info` were split
from, as one vector for each session, of its rows in the table. It is an error when the
sessions were given already prepared, without a table, and when the table has no such
column. Useful for a session parameter model that reads a column, such as one to group the
sessions by, or the covariate of a regression. Used by
[`read_one_value_of_each_session`](@ref).
"""
function session_column_values(sessions_info::MultipleSessionsInfo, column::Symbol)
    table = sessions_info.table
    if table === nothing
        throw(
            ArgumentError(
                "The sessions were given already prepared, without the table they came from, so they have no column $column. Split them from a table with MultipleSessionsInfo, or give it the table and the rows of each session as table and session_rows.",
            ),
        )
    end
    if !haskey(table, column)
        throw(
            ArgumentError(
                "The table that the sessions were split from has no column $column. Its columns are $(keys(table)).",
            ),
        )
    end
    values = getproperty(table, column)
    return [values[rows] for rows in sessions_info.session_rows]
end
