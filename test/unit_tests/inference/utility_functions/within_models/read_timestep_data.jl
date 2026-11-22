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
        #Three columns of one type each, also views such as DynamicPPL gives the parameters as, give a NamedTuple of a definite type
        three_columns = (; reward=[1.0, 0.0], cue=[1, 2], bet=view([0.5, 0.25, 0.75], 2:3))
        @test @inferred(CognitiveModeling.select_row(three_columns, 2)) ===
            (; reward=0.0, cue=2, bet=0.75)
        five_columns = (; a=[1.0], b=[2], c=[3.0], d=[true], e=view([0.0, 5.0], 2:2))
        @test @inferred(CognitiveModeling.select_row(five_columns, 1)) ===
            (; a=1.0, b=2, c=3.0, d=true, e=5.0)
        #A column that holds two types gives one of two NamedTuples also when it comes after two columns or more, and reading it allocates nothing
        four_columns = (; reward=[1.0, 0.0], cue=[1, 2], choice=column, bet=[0.5, 0.25])
        third_of_four(t) = CognitiveModeling.select_row(four_columns, t)
        @test Base.return_types(third_of_four, Tuple{Int})[1] === Union{
            NamedTuple{
                (:reward, :cue, :choice, :bet),Tuple{Float64,Int,MissingValue,Float64}
            },
            NamedTuple{(:reward, :cue, :choice, :bet),Tuple{Float64,Int,Float64,Float64}},
        }
        @test third_of_four(2) === (; reward=0.0, cue=2, choice=MissingValue(), bet=0.25)
        @test (@allocated third_of_four(2)) == 0
        #Arguments of types that aren't known give a NamedTuple
        @test all(
            type -> type <: NamedTuple,
            Base.return_types(CognitiveModeling.select_row, Tuple{Any,Any}),
        )
    end

    @testset "The values of a row are read as a tuple, one value in front of the next" begin
        @test CognitiveModeling.read_row_values(([1.0, 0.0], [1, 2]), 2) === (0.0, 2)
        @test CognitiveModeling.read_row_values((), 1) === ()
        @test CognitiveModeling.prepend_value(1.0, (2, true)) === (1.0, 2, true)
        @test CognitiveModeling.prepend_value(MissingValue(), ()) === (MissingValue(),)
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
        @test CognitiveModeling.has_missing_value((; choice=SimulatedValue(), bet=1.0))
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
