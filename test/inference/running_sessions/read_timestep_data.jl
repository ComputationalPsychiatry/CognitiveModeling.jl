@testset "Reading the data of one timestep" begin
    @testset "A row has the value of each column at one timestep, as one of two NamedTuples when a column holds two types" begin
        @test CognitiveModeling.select_row((; reward=[1.0, 0.0], cue=[1, 2]), 2) ===
            (; reward=0.0, cue=2)
        @test @inferred(CognitiveModeling.select_row((; reward=[1.0, 0.0]), 1)) ===
            (; reward=1.0)
        @test CognitiveModeling.select_row((;), 1) === (;)
        #A column that holds two types gives a NamedTuple with a concrete field, of one of two types, which a call is split over
        column = Vector{Union{MissingValue,Float64}}([1.0, MissingValue()])
        @test CognitiveModeling.select_row((; choice=column), 1) === (; choice=1.0)
        @test CognitiveModeling.select_row((; choice=column), 2) ===
            (; choice=MissingValue())
        row_of(t) = CognitiveModeling.select_row((; choice=column, cue=[1, 2]), t)
        @test Base.return_types(row_of, Tuple{Int})[1] === Union{
            NamedTuple{(:choice, :cue),Tuple{MissingValue,Int}},
            NamedTuple{(:choice, :cue),Tuple{Float64,Int}},
        }
        #Written out, such a NamedTuple has no definite type
        @test !isconcretetype(Base.return_types(t -> (; choice=column[t]), Tuple{Int})[1])
    end

    @testset "An entry is one of two NamedTuples when its value can have two types" begin
        @test CognitiveModeling.build_named_entry(Val(:choice), 0.5) === (; choice=0.5)
        @test CognitiveModeling.build_named_entry(Val(:choice), NoValue()) ===
            (; choice=NoValue())
        value_of(t) = CognitiveModeling.build_named_entry(
            Val(:choice), Vector{Union{NoValue,Float64}}([0.5, NoValue()])[t]
        )
        @test Base.return_types(value_of, Tuple{Int})[1] === Union{
            NamedTuple{(:choice,),Tuple{NoValue}},NamedTuple{(:choice,),Tuple{Float64}}
        }
    end

    @testset "has_missing_value is decided at compile time when no entry can be missing" begin
        @test CognitiveModeling.has_missing_value((; choice=MissingValue(), bet=1.0))
        @test !CognitiveModeling.has_missing_value((; choice=true, bet=1.0)) &&
            !CognitiveModeling.has_missing_value((;))
        @test CognitiveModeling.has_missing_value(
            CognitiveModeling.select_row(
                (; choice=Vector{Union{MissingValue,Float64}}([1.0, MissingValue()])), 2
            ),
        )
        folds(f, types...) = all(
            statement ->
                statement isa Core.ReturnNode ||
                statement === nothing ||
                (statement isa Expr && statement.head === :code_coverage_effect),
            first(only(code_typed(f, types))).code,
        )
        @test folds(CognitiveModeling.has_missing_value, typeof((; choice=true, bet=1.0)))
    end
end
