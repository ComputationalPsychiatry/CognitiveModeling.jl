#A cognitive model with one observation and one action, whose sessions are split from tables
struct ChoosesFromTable <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ChoosesFromTable)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end

#A table given as rows, which Tables.jl reads as it reads a DataFrame, as a stand-in for one
const ROWS_TABLE = [
    (; id="S1", outcome=1.0, choice=true),
    (; id="S1", outcome=0.0, choice=false),
    (; id="S2", outcome=1.0, choice=missing),
    (; id="S2", outcome=0.0, choice=true),
]

#The sessions of a table with the columns of ROWS_TABLE
function sessions_of_table(table)
    return MultipleSessionsInfo(
        ChoosesFromTable(),
        table;
        observation_columns=:outcome,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    )
end

@testset "column_table" begin
    @testset "A table that Tables.jl reads gives the same sessions as the NamedTuple of its columns" begin
        from_rows = sessions_of_table(ROWS_TABLE)
        from_columns = sessions_of_table(Tables.columntable(ROWS_TABLE))
        @test from_rows.session_ids == from_columns.session_ids == ["S1", "S2"]
        @test all(
            isequal(row_session.observations, column_session.observations) &&
                isequal(row_session.actions, column_session.actions) for
            (row_session, column_session) in zip(from_rows.sessions, from_columns.sessions)
        )
        @test isequal(from_rows.sessions[2].actions, (; choice=[MissingValue(), true]))
    end

    @testset "A NamedTuple of columns is used as it is" begin
        columns = Tables.columntable(ROWS_TABLE)
        @test CognitiveModeling.column_table(columns) === columns
    end

    @testset "Something that Tables.jl can't read as a table is an error from Tables.jl, which says why" begin
        @test_throws "doesn't satisfy the Tables.jl `AbstractRow` interface" sessions_of_table([
            1, 2
        ])
    end
end
