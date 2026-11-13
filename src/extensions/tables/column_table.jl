## Tables of other types than a NamedTuple, read through the Tables.jl extension ##

"""
    column_table(table)

Return `table` as a `NamedTuple` with one vector for each column. A `NamedTuple` is returned
as it is, and any other table is read with Tables.jl, through the package's extension for
it, which loads with Tables.jl, as it does with DataFrames.jl, CSV.jl and the other packages
of tables, since they load Tables.jl. Without it, a table that isn't a `NamedTuple` is an
error. Used by [`MultipleSessionsInfo`](@ref) when it splits a table.
"""
column_table(table::NamedTuple) = table
function column_table(table)
    tables_extension = Base.get_extension(@__MODULE__, :CognitiveModelingTablesExt)
    if tables_extension === nothing
        throw(
            ArgumentError(
                "The table has type $(typeof(table)), which is read with Tables.jl. Load Tables.jl, or a package which loads it, such as DataFrames.jl, or give the table as a NamedTuple with one vector for each column.",
            ),
        )
    end
    return tables_extension.column_table(table)
end
