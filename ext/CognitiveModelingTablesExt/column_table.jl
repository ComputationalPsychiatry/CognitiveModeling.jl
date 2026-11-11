"""
    column_table(table)

Return `table`, a table that Tables.jl can read, such as a `DataFrame` or a `CSV.File`, as a
`NamedTuple` with one vector for each column, which `Tables.columntable` gives without
copying the columns of a `DataFrame`. Something that Tables.jl can't read as a table is an
error from Tables.jl, which says why. Used by `CognitiveModeling.column_table`, when
`MultipleSessionsInfo` splits a table that isn't a `NamedTuple` of columns.
"""
column_table(table) = Tables.columntable(table)
