@testset "Preparing the data of a session" begin
    actions = (;
        choice=[true, MissingValue(), false, NoAction(), true],
        bet=Any[MissingValue(), MissingValue(), 1.0, 2.0, MissingValue()],
    )

    @testset "The recorded values of each name are kept in order, with the type of the column without the markers" begin
        recorded = CognitiveModeling.recorded_values(
            CognitiveModeling.replace_data_values(actions, (missing => MissingValue(),))
        )
        @test recorded.choice == [true, false, true] && recorded.choice isa Vector{Bool}
        @test recorded.bet == [1.0, 2.0] && recorded.bet isa Vector{Float64}
        @test CognitiveModeling.recorded_values((; choice=[true, false])) ==
            (; choice=[true, false])
        #A name without recorded values is kept as an empty vector of the type other sessions have, so that the sessions share a type
        never = CognitiveModeling.recorded_values((;
            choice=Vector{Union{MissingValue,Bool}}([MissingValue()]),
            skipped=Vector{Union{NoAction,Float64}}([NoAction(), NoAction()]),
        ))
        @test never.choice == Bool[] && never.choice isa Vector{Bool}
        @test never.skipped == Float64[] && never.skipped isa Vector{Float64}
        #A name whose column can only hold the markers is left out, by its type
        @test CognitiveModeling.recorded_values((;
            choice=[true], never=[NoAction(), NoAction()], skipped=[MissingValue()]
        )) == (; choice=[true])
        @test CognitiveModeling.recorded_values((;)) === (;)
        #The same for observations, where nothing happening is NoObservation
        observed = CognitiveModeling.recorded_values((;
            reward=Vector{Union{Float64,NoObservation,MissingValue}}([
                1.0, NoObservation(), MissingValue(), 0.0
            ]),
        ))
        @test observed.reward == [1.0, 0.0] && observed.reward isa Vector{Float64}
        @test CognitiveModeling.is_recorded(1.0) &&
            !CognitiveModeling.is_recorded(NoObservation()) &&
            !CognitiveModeling.is_recorded(MissingValue()) &&
            !CognitiveModeling.is_recorded(UnimputedMissingValue())
    end

    @testset "A marker that is in the data already can be replaced by another, which leaves the recorded values as they are" begin
        unimputed = (MissingValue() => UnimputedMissingValue(),)
        replaced = CognitiveModeling.replace_data_values(actions, unimputed)
        @test replaced.choice == [true, UnimputedMissingValue(), false, NoAction(), true]
        @test replaced.bet == [
            UnimputedMissingValue(),
            UnimputedMissingValue(),
            1.0,
            2.0,
            UnimputedMissingValue(),
        ]
        @test CognitiveModeling.recorded_values(replaced) ==
            CognitiveModeling.recorded_values(actions)
        #The columns of the data aren't changed, and a column that can't hold the replaced value keeps its type
        @test actions.choice[2] === MissingValue()
        @test CognitiveModeling.replace_data_values((; choice=[true, false]), unimputed).choice isa
            Vector{Bool}
        @test CognitiveModeling.replace_data_values((;), unimputed) === (;)
        #A column that can hold the replaced value gets the marker in its type whether it holds one or not, and loses the type of the replaced value, since none is left
        @test replaced.choice isa Vector{Union{UnimputedMissingValue,NoAction,Bool}}
        @test CognitiveModeling.replace_data_values(
            (; choice=Vector{Union{MissingValue,Bool}}([true, false])), unimputed
        ).choice isa Vector{Union{UnimputedMissingValue,Bool}}
    end

    @testset "The values that the replacements name are replaced by their markers, by the element type of the column" begin
        default = (missing => MissingValue(),)
        replaced = CognitiveModeling.replace_data_values(
            (; reward=[1.0, missing, 0.0], cue=[1, 2, 3]), default
        )
        @test isequal(replaced.reward, [1.0, MissingValue(), 0.0]) &&
            replaced.reward isa Vector{Union{MissingValue,Float64}}
        @test replaced.cue == [1, 2, 3] && replaced.cue isa Vector{Int}
        #A column whose element type allows the value gets the marker in its type whether it holds the value or not
        @test CognitiveModeling.replace_data_values(
            (; reward=Vector{Union{Missing,Float64}}([1.0, 0.0])),
            (missing => NoObservation(),),
        ).reward isa Vector{Union{NoObservation,Float64}}
        @test CognitiveModeling.replace_data_values(
            (; reward=[1.0, missing]), (missing => NoObservation(),)
        ).reward == [1.0, NoObservation()]
        #Several values can be replaced, each by its own marker, and a value is replaced by the first pair that is equal to it
        @test CognitiveModeling.replace_data_values(
            [1, missing, -1], (missing => MissingValue(), -1 => NoAction())
        ) == [1, MissingValue(), NoAction()]
        @test CognitiveModeling.replace_data_values(
            [1, -1], (-1 => NoAction(), -1 => MissingValue())
        ) == [1, NoAction()]
        #Values are compared with isequal, so a value of another type which is equal is replaced too, and NaN is equal to itself
        @test CognitiveModeling.replace_data_values([0.5, -1.0], (-1 => NoAction(),)) ==
            [0.5, NoAction()]
        @test CognitiveModeling.replace_data_values([0.5, 0.7], (-1 => NoAction(),)) isa
            Vector{Float64}
        @test CognitiveModeling.replace_data_values(
            [0.5, NaN], (NaN => UnimputedMissingValue(),)
        ) == [0.5, UnimputedMissingValue()]
        #Without replacements every value is kept, missing too
        @test isequal(
            CognitiveModeling.replace_data_values([1.0, missing], ()), [1.0, missing]
        )
        #Anything that isn't a vector, or a NamedTuple of them, is returned as it is, for check_data to report
        @test CognitiveModeling.replace_data_values("data", default) == "data"
        #A column with the element type Any is narrowed to the types it holds
        @test CognitiveModeling.replace_data_values(
            (; choice=[true, MissingValue(), true]), default
        ).choice isa Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.replace_data_values(
            (; choice=Any[true, missing, true]), (missing => NoAction(),)
        ).choice isa Vector{Union{NoAction,Bool}}
    end

    @testset "The replacements are a tuple of pairs, from a single pair, a tuple or a vector of them" begin
        @test CognitiveModeling.replacement_pairs(missing => MissingValue()) ===
            (missing => MissingValue(),)
        @test CognitiveModeling.replacement_pairs(()) === ()
        @test isequal(
            CognitiveModeling.replacement_pairs([
                missing => MissingValue(), -1 => NoAction()
            ]),
            (missing => MissingValue(), -1 => NoAction()),
        )
        @test_throws "The replacements of values in the data must be pairs from a value to the marker it becomes, such as (missing => MissingValue(), -1 => NoAction()), but they are MissingValue()." CognitiveModeling.replacement_pairs(
            MissingValue()
        )
        #What a value is replaced by comes wrapped, so that a replacement by nothing is told from no replacement
        @test CognitiveModeling.replacement_of(missing, (missing => MissingValue(),)) ===
            Some(MissingValue())
        @test CognitiveModeling.replacement_of(1.0, (missing => MissingValue(),)) ===
            nothing
        @test CognitiveModeling.replacement_of(-1, (-1 => nothing,)) === Some(nothing)
        #Of several values, the first pair which is equal to any of them gives the replacement
        @test CognitiveModeling.replacement_of_any(
            (1.0, missing, -1), (-1 => NoAction(), missing => MissingValue())
        ) === Some(NoAction())
        @test CognitiveModeling.replacement_of_any((1.0, 2.0), (-1 => NoAction(),)) ===
            nothing
        #The element type loses a replaced value that is the only one of its type, and gets the type of each marker
        @test CognitiveModeling.replaced_eltype(
            Union{Missing,Float64}, (missing => MissingValue(),)
        ) === Union{MissingValue,Float64}
        @test CognitiveModeling.replaced_eltype(Int, (-1 => NoAction(),)) ===
            Union{NoAction,Int}
        @test CognitiveModeling.column_can_hold([1.0, 2.0], 0.5) &&
            CognitiveModeling.column_can_hold([1.0, -1.0], -1) &&
            !CognitiveModeling.column_can_hold([1.0, 2.0], -1)
    end

    @testset "A column is narrowed when its element type is abstract and not a Union" begin
        @test CognitiveModeling.narrow_column([true, MissingValue(), true]) isa
            Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.narrow_column(Real[1, 2.0]) isa Vector{Union{Float64,Int}}
        @test CognitiveModeling.narrow_column(Any[1.0, 2.0]) isa Vector{Float64}
        #A concrete element type or a Union is kept, since a Union says what the column can hold whether it does or not
        column = Vector{Union{MissingValue,Bool}}([true, false])
        @test CognitiveModeling.narrow_column(column) === column
        @test CognitiveModeling.narrow_column([1.0, 2.0]) isa Vector{Float64}
    end

    @testset "is_recorded" begin
        @test CognitiveModeling.is_recorded(1.0) && CognitiveModeling.is_recorded(true)
        @test !CognitiveModeling.is_recorded(MissingValue()) &&
            !CognitiveModeling.is_recorded(NoAction())
    end

    @testset "A row has the value of each column at one timestep, as one of two NamedTuples when a column holds two types" begin
        @test CognitiveModeling.row((; reward=[1.0, 0.0], cue=[1, 2]), 2) ===
            (; reward=0.0, cue=2)
        @test @inferred(CognitiveModeling.row((; reward=[1.0, 0.0]), 1)) === (; reward=1.0)
        @test CognitiveModeling.row((;), 1) === (;)
        #A column that holds two types gives a NamedTuple with a concrete field, of one of two types, which a call is split over
        column = Vector{Union{MissingValue,Float64}}([1.0, MissingValue()])
        @test CognitiveModeling.row((; choice=column), 1) === (; choice=1.0)
        @test CognitiveModeling.row((; choice=column), 2) === (; choice=MissingValue())
        row_of(t) = CognitiveModeling.row((; choice=column, cue=[1, 2]), t)
        @test Base.return_types(row_of, Tuple{Int})[1] === Union{
            NamedTuple{(:choice, :cue),Tuple{MissingValue,Int}},
            NamedTuple{(:choice, :cue),Tuple{Float64,Int}},
        }
        #Written out, such a NamedTuple has no definite type
        @test !isconcretetype(Base.return_types(t -> (; choice=column[t]), Tuple{Int})[1])
    end

    @testset "An entry is one of two NamedTuples when its value can have two types" begin
        @test CognitiveModeling.entry(Val(:choice), 0.5) === (; choice=0.5)
        @test CognitiveModeling.entry(Val(:choice), NoAction()) === (; choice=NoAction())
        value_of(t) = CognitiveModeling.entry(
            Val(:choice), Vector{Union{NoAction,Float64}}([0.5, NoAction()])[t]
        )
        @test Base.return_types(value_of, Tuple{Int})[1] === Union{
            NamedTuple{(:choice,),Tuple{NoAction}},NamedTuple{(:choice,),Tuple{Float64}}
        }
    end

    @testset "has_missing_value is decided at compile time when no entry can be missing" begin
        @test CognitiveModeling.has_missing_value((; choice=MissingValue(), bet=1.0))
        @test !CognitiveModeling.has_missing_value((; choice=true, bet=1.0)) &&
            !CognitiveModeling.has_missing_value((;))
        @test CognitiveModeling.has_missing_value(
            CognitiveModeling.row(
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

@testset "Splitting a table into sessions" begin
    table = (;
        id=["S1", "S1", "S2"], block=[1, 2, 1], outcome=[1.0, 0.0, 1.0], dt=[0.5, 1.0, 1.0]
    )

    @testset "The columns of a kind map the model's names to the table's" begin
        @test CognitiveModeling.column_map(
            (; reward=:outcome), SessionCounter(), :observations
        ) === (; reward=:outcome)
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
        #A value that is replaced by a marker in any column used makes the whole value of the row that marker
        default = (missing => MissingValue(),)
        named = CognitiveModeling.build_column(
            composite, (; choice=:choicecol, rt=:rtcol), default
        )
        @test named ==
              [(; choice=true, rt=0.5), (; choice=false, rt=0.7), MissingValue()] &&
            named isa Vector{Union{MissingValue,@NamedTuple{choice::Bool,rt::Float64}}}
        @test CognitiveModeling.build_column(composite, (:rtcol, :cue), default) ==
            [(0.5, 1), (0.7, 2), (0.9, 3)]
        @test CognitiveModeling.build_column(
            composite, (:rtcol, :cue) => (rt, cue) -> [rt, cue], default
        ) == [[0.5, 1.0], [0.7, 2.0], [0.9, 3.0]]
        @test CognitiveModeling.build_column(composite, :cue, default) == [1, 2, 3]
        #Each replaced value gives its own marker, from the first pair which is equal to any value of the row
        @test CognitiveModeling.build_column(
            composite, (:choicecol, :cue), (missing => MissingValue(), 2 => NoAction())
        ) == [(true, 1), NoAction(), MissingValue()]
        #Without replacements, the values are combined as they are
        @test isequal(
            CognitiveModeling.build_column(composite, (:choicecol, :cue), ()),
            [(true, 1), (false, 2), (missing, 3)],
        )
        #The columns of a kind come with the model's names, and the rows of a session are selected from them
        columns = CognitiveModeling.build_columns(
            composite, (; ssm_action=(; choice=:choicecol, rt=:rtcol), cue=:cue), default
        )
        @test CognitiveModeling.select_rows(columns, [1, 3]) ==
            (; ssm_action=[(; choice=true, rt=0.5), MissingValue()], cue=[1, 3])
    end

    @testset "Each session is its rows, with its identifier" begin
        @test CognitiveModeling.rows_per_session(table, (:id,)) ==
            ["S1" => [1, 2], "S2" => [3]]
        @test CognitiveModeling.rows_per_session(table, (:id, :block)) == [
            (; id="S1", block=1) => [1],
            (; id="S1", block=2) => [2],
            (; id="S2", block=1) => [3],
        ]
        @test CognitiveModeling.rows_per_session(table, (:block,)) ==
            [1 => [1, 3], 2 => [2]]
    end

    @testset "The columns are built for the whole table, and the rows of a session are selected from them" begin
        columns = CognitiveModeling.build_columns(table, (; reward=:outcome), ())
        @test CognitiveModeling.select_rows(columns, [1, 2]) == (; reward=[1.0, 0.0])
        #A column with the element type Any is narrowed as a whole, so every session gets the same type
        @test CognitiveModeling.select_rows(
            CognitiveModeling.build_columns(
                (; choice=[true, false, MissingValue()]), (; choice=:choice), ()
            ),
            [1, 2],
        ).choice isa Vector{Union{MissingValue,Bool}}
        #A value is replaced in the whole column, so a session which doesn't have it gets the marker in its type too
        @test CognitiveModeling.select_rows(
            CognitiveModeling.build_columns(
                (; choice=[1.0, 0.0, -1.0]), (; choice=:choice), (-1 => NoAction(),)
            ),
            [1, 2],
        ).choice isa Vector{Union{NoAction,Float64}}
        @test CognitiveModeling.build_columns(table, (;), ()) === (;)
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

#Distributions written by hand, which stand in for the ones a model gives, so that the tests need no package of
#distributions. The type parameter lets two of them have different types, as Normal{Float64} and Normal{Float32} do
struct NormalByHand{T<:Real}
    mean::T
    standard_deviation::T
end
struct BernoulliByHand
    probability::Float64
end

#A model for the error messages about scoring, and an environment for the record of one
struct ScoringModel <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ScoringModel)
    return ModelAttributes(;
        parameters=(), states=:expected_reward, inputs=:reward, outputs=:choice
    )
end
struct ScoringEnvironment <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::ScoringEnvironment)
    return ModelAttributes(; parameters=(), states=:level, inputs=:choice, outputs=:reward)
end

@testset "Scoring the recorded values" begin
    distributions = (; choice=NormalByHand(0.0, 1.0), bet=NoAction())

    @testset "An output is recorded on a timestep when the data has a value, and a recorded value where the model gave nothing is an error" begin
        @test CognitiveModeling.recorded_on_timestep(
            (; choice=0.5, bet=NoAction()), distributions, ScoringModel()
        ) === (; choice=true, bet=false)
        @test CognitiveModeling.recorded_on_timestep(
            (; choice=MissingValue(), bet=MissingValue()), distributions, ScoringModel()
        ) === (; choice=false, bet=false)
        @test CognitiveModeling.recorded_on_timestep((;), (;), ScoringModel()) === (;)
        @test_throws "A cognitive model of type ScoringModel gave no output bet on a timestep where the data has 2.0 for it." CognitiveModeling.recorded_on_timestep(
            (; choice=0.5, bet=2.0), distributions, ScoringModel()
        )
        #The same for the observations an environment gives
        @test CognitiveModeling.recorded_on_timestep(
            (; reward=1.0), (; reward=NormalByHand(0.0, 1.0)), ScoringEnvironment()
        ) === (; reward=true)
        @test_throws "An environment model of type ScoringEnvironment gave no output reward on a timestep where the data has 1.0 for it." CognitiveModeling.recorded_on_timestep(
            (; reward=1.0), (; reward=NoObservation()), ScoringEnvironment()
        )
    end

    @testset "What a session saves of a timestep is selected by the save histories, for the cognitive model alone or for both models" begin
        #The values of a timestep have the cognitive model's at the top, and without an environment only an empty state under environment
        timestep_values = (;
            state=(; expected_reward=0.5),
            observation=(; reward=1.0),
            action=(; choice=true),
            action_distribution=(; choice=BernoulliByHand(0.7)),
            environment=(; state=nothing),
        )
        #The SaveHistory is in the cognitive model's own names, so the action distributions are its output distributions
        session_info = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            cognitive_save_history=SaveHistory(;
                output_distributions=:choice, states=SaveEverything()
            ),
        )
        #What is saved is arranged as the session returns it, without an entry for an environment
        saved = CognitiveModeling.saved_session_values(session_info, timestep_values)
        @test saved === (;
            states=(; expected_reward=0.5),
            observations=(;),
            actions=(;),
            action_distributions=(; choice=BernoulliByHand(0.7)),
        )
        #Timestep 0 has BeforeFirstTimestep for every action distribution
        previous = (;
            state=(; expected_reward=0.5),
            observation=(; reward=BeforeFirstTimestep()),
            action=(; choice=BeforeFirstTimestep()),
            environment=(; state=nothing),
        )
        saving_everything = SessionInfo(
            session_info; cognitive_save_history=SaveHistory(SaveEverything())
        )
        initial = CognitiveModeling.initial_session_values(
            Val(true), saving_everything, previous
        )
        @test initial.action_distributions === (; choice=BeforeFirstTimestep()) &&
            initial.actions === (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.initial_session_values(
            Val(false), saving_everything, previous
        ) === nothing
        #The storage of a simulation takes the arrangement without an environment too: it starts with timestep 0 when it is kept, and later timesteps go in by index
        storage = CognitiveModeling.initialize_all_storage(initial, saved, 2)
        @test keys(storage) == (:states, :observations, :actions, :action_distributions)
        @test length(storage.action_distributions) == 3 &&
            storage.action_distributions[1] === (; choice=BeforeFirstTimestep())
        storage = CognitiveModeling.store_all_values!(storage, saved, 3)
        @test storage.action_distributions[3] === (; choice=BernoulliByHand(0.7)) &&
            storage.states[3] === (; expected_reward=0.5)
        @test length(CognitiveModeling.initialize_all_storage(nothing, saved, 2).states) ==
            2
        #What is returned has the vectors as they are stored, with the timestep sizes
        returned_trajectories = CognitiveModeling.saved_trajectories(storage, [1, 1, 1])
        @test keys(returned_trajectories) ==
            (:states, :observations, :actions, :action_distributions, :timestep_sizes)
        @test returned_trajectories.action_distributions === storage.action_distributions
        #With an environment, its record is saved too, as its save history selects it, under environment
        with_environment = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            environment=ScoringEnvironment(),
            cognitive_save_history=SaveHistory(; states=SaveEverything()),
            environment_save_history=SaveHistory(;
                states=:level, outputs=:reward, output_distributions=SaveEverything()
            ),
        )
        #The environment's observation and action are in its own names, which are the same here
        values_with_environment = merge(
            timestep_values,
            (;
                environment=(;
                    state=(; level=0.3),
                    observation=timestep_values.observation,
                    action=timestep_values.action,
                    observation_distribution=(; reward=NormalByHand(0.0, 1.0)),
                ),
            ),
        )
        saved_both = CognitiveModeling.saved_session_values(
            with_environment, values_with_environment
        )
        @test saved_both === (;
            states=(; expected_reward=0.5),
            observations=(;),
            actions=(;),
            action_distributions=(;),
            environment=(;
                states=(; level=0.3),
                observations=(; reward=1.0),
                actions=(;),
                observation_distributions=(; reward=NormalByHand(0.0, 1.0)),
            ),
        )
        initial_both = CognitiveModeling.initial_session_values(
            Val(true),
            with_environment,
            merge(
                previous,
                (;
                    environment=(;
                        state=(; level=0.0),
                        observation=previous.observation,
                        action=previous.action,
                    ),
                ),
            ),
        )
        @test initial_both.environment.observation_distributions ===
              (; reward=BeforeFirstTimestep()) &&
            initial_both.environment.states === (; level=0.0)
        storage_both = CognitiveModeling.initialize_all_storage(initial_both, saved_both, 2)
        storage_both = CognitiveModeling.store_all_values!(storage_both, saved_both, 3)
        returned_both = CognitiveModeling.saved_trajectories(storage_both, [1, 1, 1])
        @test keys(returned_both) == (
            :states,
            :observations,
            :actions,
            :action_distributions,
            :environment,
            :timestep_sizes,
        )
        @test keys(returned_both.environment) ==
            (:states, :observations, :actions, :observation_distributions)
        @test returned_both.environment.states ==
            [(; level=0.0), (; level=0.3), (; level=0.3)]
        #A session whose environment isn't included gets an empty record of it, with the entries in the same order
        with_empty = CognitiveModeling.with_empty_environment_entry(returned_trajectories)
        @test keys(with_empty) == keys(returned_both)
        @test with_empty.environment.states == fill((;), 3) &&
            with_empty.environment.observation_distributions == fill((;), 3)
        #A session that ends early has its vectors shortened in place, with or without an environment, and its timestep sizes cut
        CognitiveModeling.map_saved(vector -> resize!(vector, 2), storage)
        @test length(storage.states) == 2 && length(storage.action_distributions) == 2
        CognitiveModeling.map_saved(vector -> resize!(vector, 2), storage_both)
        @test length(storage_both.states) == 2 &&
            length(storage_both.environment.states) == 2
        @test CognitiveModeling.timestep_sizes_until([BeforeFirstTimestep(), 1, 1], 2) ==
            [BeforeFirstTimestep(), 1]
        @test CognitiveModeling.timestep_sizes_until(Int[], 2) == Int[]
    end

    @testset "The vectors of distributions are typed by the first timestep, and widened when a later one differs" begin
        initial = CognitiveModeling.initialize_distributions((;
            choice=NormalByHand(0.0, 1.0), bet=NoAction()
        ))
        @test initial.choice isa Vector{NormalByHand{Float64}} && isempty(initial.choice)
        @test initial.bet isa Vector{Union{}}
        stored = CognitiveModeling.store_distributions(
            initial,
            (; choice=NormalByHand(0.0, 1.0), bet=NoAction()),
            (; choice=true, bet=false),
        )
        @test stored.choice == [NormalByHand(0.0, 1.0)] && isempty(stored.bet)
        stored = CognitiveModeling.store_distributions(
            stored,
            (; choice=NormalByHand(1.0, 1.0), bet=NormalByHand(2.0, 0.5)),
            (; choice=false, bet=true),
        )
        @test stored.choice == [NormalByHand(0.0, 1.0)] &&
            stored.bet == [NormalByHand(2.0, 0.5)] &&
            stored.bet isa Vector{NormalByHand{Float64}}
        #A distribution of another type widens the vector
        stored = CognitiveModeling.store_distributions(
            stored,
            (; choice=NormalByHand(1.0f0, 1.0f0), bet=NormalByHand(2.0, 0.5)),
            (; choice=true, bet=false),
        )
        @test stored.choice isa
              Vector{Union{NormalByHand{Float64},NormalByHand{Float32}}} &&
            length(stored.choice) == 2
    end
end
