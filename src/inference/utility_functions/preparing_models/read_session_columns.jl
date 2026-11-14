## Reading the columns that a session parameter model uses ##

"""
    read_one_value_of_each_session(sessions_info, column, reader)

Return the value of `column` in every session of `sessions_info`, as a vector in the order
of the sessions, from the table that the sessions were split from. A session with more than
one value in the column, or with a missing one, is an error, which starts with `reader`, a
description of what reads the column, such as
`"GroupedHierarchicalSessionParameters groups learning_rate by"`. Used by
[`read_level_of_each_session`](@ref) and by the continuous predictors of a
[`HierarchicalRegressionSessionParameters`](@ref).
"""
function read_one_value_of_each_session(
    sessions_info::MultipleSessionsInfo, column::Symbol, reader::String
)
    values_of_sessions = session_column_values(sessions_info, column)
    return map(eachindex(values_of_sessions)) do i
        values_of_session = unique(values_of_sessions[i])
        if length(values_of_session) != 1 || ismissing(only(values_of_session))
            throw(
                ArgumentError(
                    "$reader the column $column, which has the values $(values_of_session) in the session $(repr(sessions_info.session_ids[i])). Each session must have one value of every column that a session parameter model reads.",
                ),
            )
        end
        return only(values_of_session)
    end
end

"""
    read_level_of_each_session(sessions_info, columns, reader)

Return the level of every session of `sessions_info` in the columns in `columns`, a tuple of
names, as a vector in the order of the sessions: its value of a single column, or a
`NamedTuple` of its values of several. Each value is read with
[`read_one_value_of_each_session`](@ref), whose errors start with `reader`. Used by
[`group_sessions`](@ref) and by the categorical predictors of a
[`HierarchicalRegressionSessionParameters`](@ref).
"""
function read_level_of_each_session(
    sessions_info::MultipleSessionsInfo, columns::Tuple{Vararg{Symbol}}, reader::String
)

    #The value of each column in each session, which must be one value
    values_of_columns = map(
        column -> read_one_value_of_each_session(sessions_info, column, reader), columns
    )

    #The level of a session is its value of a single column, or the NamedTuple of its values of several
    return map(eachindex(sessions_info.sessions)) do i
        values = map(values_of_column -> values_of_column[i], values_of_columns)
        return length(columns) == 1 ? only(values) : NamedTuple{columns}(values)
    end
end

"""
    group_sessions(sessions_info, group_by, reader)

Return the groups of the sessions in `sessions_info` by the values of the columns in
`group_by`, a tuple of names: a `NamedTuple` with the group of every session,
`group_of_each_session`, the number of groups, `n_groups`, and the symbol name of each
group, `group_symbol_names`, made from its values as [`session_id_to_symbol`](@ref) makes
the symbol name of a session from its identifier. Groups are numbered in the order of their
first session, and with no columns every session is in one group. A session with more than
one value, or a missing value, in one of the columns is an error, which starts with
`reader`, as for [`read_level_of_each_session`](@ref). Used by
[`prepare_parameters_model`](@ref), for the groups of a
[`GroupedHierarchicalSessionParameters`](@ref) and for the varying effects of a
[`HierarchicalRegressionSessionParameters`](@ref).
"""
function group_sessions(sessions_info::MultipleSessionsInfo, ::Tuple{}, ::String)
    return (;
        group_of_each_session=fill(1, n_sessions(sessions_info)),
        n_groups=1,
        group_symbol_names=[:all_sessions],
    )
end

function group_sessions(
    sessions_info::MultipleSessionsInfo, group_by::Tuple{Vararg{Symbol}}, reader::String
)

    #The group of a session is its level in the columns: its value of a single column, or the NamedTuple of its values of several
    group_of_sessions = read_level_of_each_session(sessions_info, group_by, reader)

    #Groups are numbered in the order of their first session
    groups = unique(group_of_sessions)
    return (;
        group_of_each_session=[
            findfirst(isequal(group), groups) for group in group_of_sessions
        ],
        n_groups=length(groups),
        group_symbol_names=map(session_id_to_symbol, groups),
    )
end
