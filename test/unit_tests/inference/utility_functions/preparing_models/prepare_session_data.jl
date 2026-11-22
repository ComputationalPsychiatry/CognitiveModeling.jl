@testset "Preparing the data of a session" begin
    actions = (;
        choice=[true, ImputedMissingValue(), false, NoValue(), true],
        bet=Any[
            ImputedMissingValue(), ImputedMissingValue(), 1.0, 2.0, ImputedMissingValue()
        ],
    )

    @testset "The values to condition on of each name are kept in order, with the type of the column without the markers" begin
        values_to_condition_on = CognitiveModeling.collect_values_to_condition_on(
            CognitiveModeling.mark_missing_values(actions, (missing,), true)
        )
        @test values_to_condition_on.choice == [true, false, true] &&
            values_to_condition_on.choice isa Vector{Bool}
        @test values_to_condition_on.bet == [1.0, 2.0] &&
            values_to_condition_on.bet isa Vector{Float64}
        @test CognitiveModeling.collect_values_to_condition_on((; choice=[true, false])) ==
            (; choice=[true, false])
        #A name without values to condition on is kept as an empty vector of the type other sessions have, so that the sessions share a type
        never = CognitiveModeling.collect_values_to_condition_on((;
            choice=Vector{Union{ImputedMissingValue,Bool}}([ImputedMissingValue()]),
            skipped=Vector{Union{NoValue,Float64}}([NoValue(), NoValue()]),
        ))
        @test never.choice == Bool[] && never.choice isa Vector{Bool}
        @test never.skipped == Float64[] && never.skipped isa Vector{Float64}
        #A name whose column can only hold the markers is left out, by its type
        @test CognitiveModeling.collect_values_to_condition_on((;
            choice=[true], never=[NoValue(), NoValue()], skipped=[ImputedMissingValue()]
        )) == (; choice=[true])
        @test CognitiveModeling.collect_values_to_condition_on((;)) === (;)
        #The same for observations, where nothing happening is NoValue
        observed = CognitiveModeling.collect_values_to_condition_on((;
            reward=Vector{Union{Float64,NoValue,ImputedMissingValue}}([
                1.0, NoValue(), ImputedMissingValue(), 0.0
            ]),
        ))
        @test observed.reward == [1.0, 0.0] && observed.reward isa Vector{Float64}
    end

    @testset "Without imputation, missing values become the marker for one that isn't imputed, also those marked in the data already" begin
        marked = CognitiveModeling.mark_missing_values(actions, (missing,), false)
        @test marked.choice == [true, UnimputedMissingValue(), false, NoValue(), true]
        @test marked.bet == [
            UnimputedMissingValue(),
            UnimputedMissingValue(),
            1.0,
            2.0,
            UnimputedMissingValue(),
        ]
        @test CognitiveModeling.collect_values_to_condition_on(marked) ==
            CognitiveModeling.collect_values_to_condition_on(actions)
        #The columns of the data aren't changed, and a column that can't hold a missing value keeps its type
        @test actions.choice[2] === ImputedMissingValue()
        @test CognitiveModeling.mark_missing_values(
            (; choice=[true, false]), (missing,), false
        ).choice isa Vector{Bool}
        @test CognitiveModeling.mark_missing_values((;), (missing,), false) === (;)
        #A column that can hold a missing value gets the marker in its type whether it holds one or not, and loses the type of the value that was replaced
        @test marked.choice isa Vector{Union{UnimputedMissingValue,NoValue,Bool}}
        @test CognitiveModeling.mark_missing_values(
            (; choice=Vector{Union{ImputedMissingValue,Bool}}([true, false])),
            (missing,),
            false,
        ).choice isa Vector{Union{UnimputedMissingValue,Bool}}
        #With imputation, an ImputedMissingValue in the data stays what it is
        @test CognitiveModeling.mark_missing_values(actions, (missing,), true).choice ==
            [true, ImputedMissingValue(), false, NoValue(), true]
        @test CognitiveModeling.missing_value_marker(true) === ImputedMissingValue() &&
            CognitiveModeling.missing_value_marker(false) === UnimputedMissingValue()
    end

    @testset "The values which stand for a missing one are replaced by its marker, by the element type of the column" begin
        marked = CognitiveModeling.mark_missing_values(
            (; reward=[1.0, missing, 0.0], cue=[1, 2, 3]), (missing,), true
        )
        @test isequal(marked.reward, [1.0, ImputedMissingValue(), 0.0]) &&
            marked.reward isa Vector{Union{ImputedMissingValue,Float64}}
        @test marked.cue == [1, 2, 3] && marked.cue isa Vector{Int}
        #A column whose element type allows the value gets the marker in its type whether it holds the value or not
        @test CognitiveModeling.mark_missing_values(
            (; reward=Vector{Union{Missing,Float64}}([1.0, 0.0])), (missing,), true
        ).reward isa Vector{Union{ImputedMissingValue,Float64}}
        #Several values can stand for a missing one
        @test CognitiveModeling.mark_missing_values(
            [1, missing, -1], (missing, -1), true
        ) == [1, ImputedMissingValue(), ImputedMissingValue()]
        #Values are compared with isequal, so a value of another type which is equal is replaced too, and NaN is equal to itself
        @test CognitiveModeling.mark_missing_values([0.5, -1.0], (-1,), true) ==
            [0.5, ImputedMissingValue()]
        @test CognitiveModeling.mark_missing_values([0.5, 0.7], (-1,), true) isa
            Vector{Float64}
        @test CognitiveModeling.mark_missing_values([0.5, NaN], (NaN,), false) ==
            [0.5, UnimputedMissingValue()]
        #Without values to replace, every value is kept, missing too
        @test isequal(
            CognitiveModeling.mark_missing_values([1.0, missing], (), true), [1.0, missing]
        )
        #Anything that isn't a vector, or a NamedTuple of them, is returned as it is, for check_data_fits to report
        @test CognitiveModeling.mark_missing_values("data", (missing,), true) == "data"
        #A column with the element type Any is narrowed to the types it holds
        @test CognitiveModeling.mark_missing_values(
            (; choice=[true, ImputedMissingValue(), true]), (missing,), true
        ).choice isa Vector{Union{ImputedMissingValue,Bool}}
        @test CognitiveModeling.mark_missing_values(
            (; choice=Any[true, missing, true]), (missing,), false
        ).choice isa Vector{Union{UnimputedMissingValue,Bool}}
    end

    @testset "A function among the values that stand for a missing one marks each value it returns true for, and the arrays of the values kept are narrowed" begin
        has_missing_element(value) = value isa AbstractVector && any(ismissing, value)
        #Every vector of ratings can hold a missing value by its type, and the first holds one
        ratings = Vector{Union{Missing,Float64}}[[0.1, missing, 0.2], [0.3, 0.4, 0.5]]
        marked = CognitiveModeling.mark_missing_values(
            ratings, (missing, has_missing_element), true
        )
        @test marked == [ImputedMissingValue(), [0.3, 0.4, 0.5]]
        @test marked isa Vector{Union{ImputedMissingValue,Vector{Float64}}}
        @test CognitiveModeling.mark_missing_values(
            ratings, (missing, has_missing_element), false
        ) == [UnimputedMissingValue(), [0.3, 0.4, 0.5]]
        #A single function is taken too, and the data isn't changed
        @test CognitiveModeling.mark_missing_values(
            ratings, CognitiveModeling.as_tuple(has_missing_element), true
        ) == marked
        @test isequal(ratings[1], [0.1, missing, 0.2]) &&
            ratings[2] isa Vector{Union{Missing,Float64}}
        #Without the function, the vector with a missing element is kept, and the other is narrowed
        kept = CognitiveModeling.mark_missing_values(ratings, (missing,), true)
        @test isequal(kept, ratings) && kept[2] isa Vector{Float64}
        #A function on numbers, beside a value compared with isequal, and a function which returns missing counts as false
        @test CognitiveModeling.mark_missing_values(
            [0.5, -0.2, 0.7, -1.0], (value -> value < 0 && value > -1, -1), true
        ) == [0.5, ImputedMissingValue(), 0.7, ImputedMissingValue()]
        @test CognitiveModeling.mark_missing_values([0.5, -0.2], (<(0),), true) isa
            Vector{Union{ImputedMissingValue,Float64}}
        @test isequal(
            CognitiveModeling.mark_missing_values([1.0, missing], (<(0),), true),
            [1.0, missing],
        )
        #A function that finds nothing missing adds no marker to the type
        @test CognitiveModeling.mark_missing_values([0.5, 0.7], (<(0),), true) isa
            Vector{Float64}
        #Tuples and NamedTuples are narrowed in their arrays too
        @test CognitiveModeling.mark_missing_values(
            [(; ratings=Union{Missing,Float64}[0.1, 0.2], choice=1)], (missing,), true
        )[1].ratings isa Vector{Float64}
        #Marking the rows of a session again, as each session of a table is, keeps the element type of the whole column
        @test CognitiveModeling.mark_missing_values(marked[2:2], (), true) isa
            Vector{Union{ImputedMissingValue,Vector{Float64}}}
        responses = Union{ImputedMissingValue,@NamedTuple{choice::Int,rt::Float64}}[(;
            choice=2, rt=0.4
        )]
        @test CognitiveModeling.mark_missing_values(responses, (), true) === responses
        @test CognitiveModeling.is_missing_data_value([missing], (has_missing_element,)) &&
            !CognitiveModeling.is_missing_data_value([1.0], (has_missing_element,))
        @test CognitiveModeling.matches_missing_value(<(0), -1) &&
            !CognitiveModeling.matches_missing_value(<(0), missing) &&
            CognitiveModeling.matches_missing_value(-1, -1.0)
        #An array whose parts are narrow already is kept as it is
        numbers = [1.0, 2.0]
        @test CognitiveModeling.narrow_parts(numbers) === numbers
        vectors = [[1.0], [2.0]]
        @test CognitiveModeling.narrow_parts(vectors) === vectors
        @test CognitiveModeling.narrow_parts([Union{Missing,Float64}[1.0]]) isa
            Vector{Vector{Float64}}
    end

    @testset "A value is missing when it is the marker itself, or equal to one of the values that stand for a missing one" begin
        @test CognitiveModeling.is_missing_data_value(ImputedMissingValue(), ()) &&
            CognitiveModeling.is_missing_data_value(-1.0, (missing, -1)) &&
            !CognitiveModeling.is_missing_data_value(1.0, (missing, -1))
        @test CognitiveModeling.column_can_hold([1.0, 2.0], 0.5) &&
            CognitiveModeling.column_can_hold([1.0, -1.0], -1) &&
            !CognitiveModeling.column_can_hold([1.0, 2.0], -1)
    end

    @testset "A column is narrowed when its element type is abstract and not a Union" begin
        @test CognitiveModeling.narrow_element_type([true, ImputedMissingValue(), true]) isa
            Vector{Union{ImputedMissingValue,Bool}}
        @test CognitiveModeling.narrow_element_type(Real[1, 2.0]) isa
            Vector{Union{Float64,Int}}
        @test CognitiveModeling.narrow_element_type(Any[1.0, 2.0]) isa Vector{Float64}
        #A concrete element type or a Union is kept, since a Union says what the column can hold whether it does or not
        column = Vector{Union{ImputedMissingValue,Bool}}([true, false])
        @test CognitiveModeling.narrow_element_type(column) === column
        @test CognitiveModeling.narrow_element_type([1.0, 2.0]) isa Vector{Float64}
    end
end
