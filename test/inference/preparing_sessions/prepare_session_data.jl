@testset "Preparing the data of a session" begin
    actions = (;
        choice=[true, MissingValue(), false, NoValue(), true],
        bet=Any[MissingValue(), MissingValue(), 1.0, 2.0, MissingValue()],
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
            choice=Vector{Union{MissingValue,Bool}}([MissingValue()]),
            skipped=Vector{Union{NoValue,Float64}}([NoValue(), NoValue()]),
        ))
        @test never.choice == Bool[] && never.choice isa Vector{Bool}
        @test never.skipped == Float64[] && never.skipped isa Vector{Float64}
        #A name whose column can only hold the markers is left out, by its type
        @test CognitiveModeling.collect_values_to_condition_on((;
            choice=[true], never=[NoValue(), NoValue()], skipped=[MissingValue()]
        )) == (; choice=[true])
        @test CognitiveModeling.collect_values_to_condition_on((;)) === (;)
        #The same for observations, where nothing happening is NoValue
        observed = CognitiveModeling.collect_values_to_condition_on((;
            reward=Vector{Union{Float64,NoValue,MissingValue}}([
                1.0, NoValue(), MissingValue(), 0.0
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
        @test actions.choice[2] === MissingValue()
        @test CognitiveModeling.mark_missing_values(
            (; choice=[true, false]), (missing,), false
        ).choice isa Vector{Bool}
        @test CognitiveModeling.mark_missing_values((;), (missing,), false) === (;)
        #A column that can hold a missing value gets the marker in its type whether it holds one or not, and loses the type of the value that was replaced
        @test marked.choice isa Vector{Union{UnimputedMissingValue,NoValue,Bool}}
        @test CognitiveModeling.mark_missing_values(
            (; choice=Vector{Union{MissingValue,Bool}}([true, false])), (missing,), false
        ).choice isa Vector{Union{UnimputedMissingValue,Bool}}
        #With imputation, a MissingValue in the data stays what it is
        @test CognitiveModeling.mark_missing_values(actions, (missing,), true).choice ==
            [true, MissingValue(), false, NoValue(), true]
        @test CognitiveModeling.missing_value_marker(true) === MissingValue() &&
            CognitiveModeling.missing_value_marker(false) === UnimputedMissingValue()
    end

    @testset "The values which stand for a missing one are replaced by its marker, by the element type of the column" begin
        marked = CognitiveModeling.mark_missing_values(
            (; reward=[1.0, missing, 0.0], cue=[1, 2, 3]), (missing,), true
        )
        @test isequal(marked.reward, [1.0, MissingValue(), 0.0]) &&
            marked.reward isa Vector{Union{MissingValue,Float64}}
        @test marked.cue == [1, 2, 3] && marked.cue isa Vector{Int}
        #A column whose element type allows the value gets the marker in its type whether it holds the value or not
        @test CognitiveModeling.mark_missing_values(
            (; reward=Vector{Union{Missing,Float64}}([1.0, 0.0])), (missing,), true
        ).reward isa Vector{Union{MissingValue,Float64}}
        #Several values can stand for a missing one
        @test CognitiveModeling.mark_missing_values(
            [1, missing, -1], (missing, -1), true
        ) == [1, MissingValue(), MissingValue()]
        #Values are compared with isequal, so a value of another type which is equal is replaced too, and NaN is equal to itself
        @test CognitiveModeling.mark_missing_values([0.5, -1.0], (-1,), true) ==
            [0.5, MissingValue()]
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
            (; choice=[true, MissingValue(), true]), (missing,), true
        ).choice isa Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.mark_missing_values(
            (; choice=Any[true, missing, true]), (missing,), false
        ).choice isa Vector{Union{UnimputedMissingValue,Bool}}
    end

    @testset "A value is missing when it is the marker itself, or equal to one of the values that stand for a missing one" begin
        @test CognitiveModeling.is_missing_data_value(MissingValue(), ()) &&
            CognitiveModeling.is_missing_data_value(-1.0, (missing, -1)) &&
            !CognitiveModeling.is_missing_data_value(1.0, (missing, -1))
        @test CognitiveModeling.column_can_hold([1.0, 2.0], 0.5) &&
            CognitiveModeling.column_can_hold([1.0, -1.0], -1) &&
            !CognitiveModeling.column_can_hold([1.0, 2.0], -1)
    end

    @testset "A column is narrowed when its element type is abstract and not a Union" begin
        @test CognitiveModeling.narrow_element_type([true, MissingValue(), true]) isa
            Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.narrow_element_type(Real[1, 2.0]) isa
            Vector{Union{Float64,Int}}
        @test CognitiveModeling.narrow_element_type(Any[1.0, 2.0]) isa Vector{Float64}
        #A concrete element type or a Union is kept, since a Union says what the column can hold whether it does or not
        column = Vector{Union{MissingValue,Bool}}([true, false])
        @test CognitiveModeling.narrow_element_type(column) === column
        @test CognitiveModeling.narrow_element_type([1.0, 2.0]) isa Vector{Float64}
    end
end
