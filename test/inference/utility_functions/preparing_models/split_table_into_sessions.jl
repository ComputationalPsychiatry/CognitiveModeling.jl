@testset "Splitting a table into sessions" begin
    table = (;
        id=["S1", "S1", "S2"], block=[1, 2, 1], outcome=[1.0, 0.0, 1.0], dt=[0.5, 1.0, 1.0]
    )

    @testset "The columns of a kind map the model's names to the table's" begin
        #A pair goes from the column's name to the model's, as in a NameMap
        @test CognitiveModeling.column_map(
            :outcome => :reward, SessionCounter(), :observations
        ) === (; reward=:outcome)
        @test CognitiveModeling.column_map(
            (:outcome => :reward, :cue), ObservesRewardAndCue(), :observations
        ) === (; reward=:outcome, cue=:cue)
        @test CognitiveModeling.column_map(:reward, SessionCounter(), :observations) ===
            (; reward=:reward)
        #A single name is the column of the model's only observation or action, whatever it is called
        @test CognitiveModeling.column_map(:outcome, SessionCounter(), :observations) ===
            (; reward=:outcome)
        @test CognitiveModeling.column_map(:response, SessionCounter(), :actions) ===
            (; choice=:response)
        @test CognitiveModeling.column_map(
            (:reward, :cue), ObservesRewardAndCue(), :observations
        ) === (; reward=:reward, cue=:cue)
        @test CognitiveModeling.column_map(
            [:reward, :cue], ObservesRewardAndCue(), :observations
        ) === (; reward=:reward, cue=:cue)
        @test_throws MethodError CognitiveModeling.column_map(
            "reward", SessionCounter(), :observations
        )
        #Each name can only be given once, and an entry is a name or a pair which ends in the model's name
        @test_throws "The columns of the observations give the names (:reward, :reward), but each name can only be given once." CognitiveModeling.column_map(
            (:outcome => :reward, :payoff => :reward), ObservesRewardAndCue(), :observations
        )
        @test_throws "An entry of the columns must end in the name the model uses, such as :outcome => :reward or (:choicecol, :rtcol) => combine => :ssm_action, but it is :outcome => 1." CognitiveModeling.column_map(
            (:outcome => 1,), ObservesRewardAndCue(), :observations
        )
        @test_throws "An entry of the columns is a name, or a pair from a column to the name the model uses, such as :outcome => :reward, but it is \"reward\"." CognitiveModeling.column_map(
            ("reward",), ObservesRewardAndCue(), :observations
        )
    end

    @testset "A value built from several columns is given before the model's name, or without it when the model has only one of the kind" begin
        #A Tuple or a NamedTuple of names, or a Tuple of names and a function, followed by the model's name
        @test CognitiveModeling.column_map(
            (:choicecol, :rtcol) => :choice, SessionCounter(), :actions
        ) === (; choice=(:choicecol, :rtcol))
        @test CognitiveModeling.column_map(
            (; choice=:choicecol, rt=:rtcol) => :choice, SessionCounter(), :actions
        ) === (; choice=(; choice=:choicecol, rt=:rtcol))
        @test CognitiveModeling.column_map(
            (:choicecol, :rtcol) => tuple => :choice, SessionCounter(), :actions
        ) === (; choice=((:choicecol, :rtcol) => tuple))
        @test CognitiveModeling.column_map(
            ((:a, :b) => :reward, :stimulus => :cue), ObservesRewardAndCue(), :observations
        ) === (; reward=(:a, :b), cue=:stimulus)
        #Without the name, for the model's only action: a NamedTuple of names, or names and a function
        @test CognitiveModeling.column_map(
            (; choice=:choicecol, rt=:rtcol), SessionCounter(), :actions
        ) === (; choice=(; choice=:choicecol, rt=:rtcol))
        @test CognitiveModeling.column_map(
            (:choicecol, :rtcol) => tuple, SessionCounter(), :actions
        ) === (; choice=((:choicecol, :rtcol) => tuple))
        #A model with several of the kind needs the name
        @test_throws "observation_columns is (a = :x, b = :y), which builds one value from several columns without the name the model uses for it. That is only possible when the model has only one kind of them in its attributes, but the attributes of a cognitive model of type ObservesRewardAndCue have the observations (:reward, :cue). Give the name after the columns, such as (:choicecol, :rtcol) => :ssm_action." CognitiveModeling.column_map(
            (; a=:x, b=:y), ObservesRewardAndCue(), :observations
        )
        @test_throws "observation_columns is (:x, :y) => tuple, which builds one value from several columns without the name the model uses for it." CognitiveModeling.column_map(
            (:x, :y) => tuple, ObservesRewardAndCue(), :observations
        )
        @test CognitiveModeling.names_its_attribute(:outcome => :reward) &&
            CognitiveModeling.names_its_attribute((:a, :b) => tuple => :reward) &&
            !CognitiveModeling.names_its_attribute((:a, :b) => tuple)
        @test CognitiveModeling.column_entry(:cue) === (:cue => :cue)
    end

    @testset "A column can be built from several, as a NamedTuple, a Tuple or what a function gives" begin
        composite = (;
            choicecol=[true, false, missing], rtcol=[0.5, 0.7, 0.9], cue=[1, 2, 3]
        )
        @test CognitiveModeling.columns_used(:cue) === (:cue,)
        @test CognitiveModeling.columns_used((; choice=:choicecol, rt=:rtcol)) ===
            (:choicecol, :rtcol)
        @test CognitiveModeling.columns_used((:choicecol, :rtcol)) === (:choicecol, :rtcol)
        @test CognitiveModeling.columns_used((:choicecol, :rtcol) => tuple) ===
            (:choicecol, :rtcol)
        @test_throws "A column is specified by its name, a NamedTuple or a Tuple of names" CognitiveModeling.columns_used(
            "choicecol"
        )
        #A missing value in any column used makes the whole value of the row missing
        named = CognitiveModeling.build_column(
            composite, (; choice=:choicecol, rt=:rtcol), (missing,), true
        )
        @test named ==
              [(; choice=true, rt=0.5), (; choice=false, rt=0.7), MissingValue()] &&
            named isa Vector{Union{MissingValue,@NamedTuple{choice::Bool,rt::Float64}}}
        @test CognitiveModeling.build_column(composite, (:rtcol, :cue), (missing,), true) ==
            [(0.5, 1), (0.7, 2), (0.9, 3)]
        @test CognitiveModeling.build_column(
            composite, (:rtcol, :cue) => (rt, cue) -> [rt, cue], (missing,), true
        ) == [[0.5, 1.0], [0.7, 2.0], [0.9, 3.0]]
        @test CognitiveModeling.build_column(composite, :cue, (missing,), true) == [1, 2, 3]
        #Any of the values which stand for a missing one does, and without imputation the whole value is the marker for that
        @test CognitiveModeling.build_column(
            composite, (:choicecol, :cue), (missing, 2), false
        ) == [(true, 1), UnimputedMissingValue(), UnimputedMissingValue()]
        #Without values to replace, the values are combined as they are
        @test isequal(
            CognitiveModeling.build_column(composite, (:choicecol, :cue), (), true),
            [(true, 1), (false, 2), (missing, 3)],
        )
        #The columns of a kind come with the model's names, and the rows of a session are selected from them
        columns = CognitiveModeling.build_columns(
            composite,
            (; ssm_action=(; choice=:choicecol, rt=:rtcol), cue=:cue),
            (missing,),
            true,
        )
        @test CognitiveModeling.select_rows(columns, [1, 3]) ==
            (; ssm_action=[(; choice=true, rt=0.5), MissingValue()], cue=[1, 3])
    end

    @testset "The columns are built for the whole table, and the rows of a session are selected from them" begin
        columns = CognitiveModeling.build_columns(table, (; reward=:outcome), (), true)
        @test CognitiveModeling.select_rows(columns, [1, 2]) == (; reward=[1.0, 0.0])
        #A column with the element type Any is narrowed as a whole, so every session gets the same type
        @test CognitiveModeling.select_rows(
            CognitiveModeling.build_columns(
                (; choice=[true, false, MissingValue()]), (; choice=:choice), (), true
            ),
            [1, 2],
        ).choice isa Vector{Union{MissingValue,Bool}}
        #A value is replaced in the whole column, so a session which doesn't have it gets the marker in its type too
        @test CognitiveModeling.select_rows(
            CognitiveModeling.build_columns(
                (; choice=[1.0, 0.0, -1.0]), (; choice=:choice), (-1,), true
            ),
            [1, 2],
        ).choice isa Vector{Union{MissingValue,Float64}}
        @test CognitiveModeling.build_columns(table, (;), (), true) === (;)
        @test CognitiveModeling.select_rows((;), [1, 2]) === (;)
        @test CognitiveModeling.select_timestep_sizes(table, :dt, [1, 2]) == [0.5, 1.0]
        @test CognitiveModeling.select_timestep_sizes(table, 2, [1, 2]) == 2
    end

    @testset "The table is checked" begin
        @test CognitiveModeling.check_table(table, (:id, :outcome)) === nothing
        @test_throws "The table has no column named reward. Its columns are (:id, :block, :outcome, :dt)." CognitiveModeling.check_table(
            table, (:reward,)
        )
        @test_throws "The table has no columns." CognitiveModeling.check_table((;), ())
        @test_throws "Every column of the table must be a vector" CognitiveModeling.check_table(
            (; id="S1"), (:id,)
        )
        @test_throws "Every column of the table must have the same length, but their lengths are (id = 2, outcome = 3)." CognitiveModeling.check_table(
            (; id=["S1", "S2"], outcome=[1.0, 0.0, 1.0]), (:id,)
        )
    end
end
