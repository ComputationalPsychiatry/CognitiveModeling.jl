@testset "Preparing the data of a session" begin
    actions = (;
        choice=[true, MissingValue(), false, NoAction(), true],
        bet=Any[MissingValue(), MissingValue(), 1.0, 2.0, MissingValue()],
    )

    @testset "The recorded actions of each name are kept in order, with the type of the column without the markers" begin
        recorded = CognitiveModeling.recorded_actions(
            CognitiveModeling.replace_missing(actions, MissingValue())
        )
        @test recorded.choice == [true, false, true] && recorded.choice isa Vector{Bool}
        @test recorded.bet == [1.0, 2.0] && recorded.bet isa Vector{Float64}
        @test CognitiveModeling.recorded_actions((; choice=[true, false])) ==
            (; choice=[true, false])
        #An action without recorded values is kept as an empty vector of the type other sessions have, so that the sessions share a type
        never = CognitiveModeling.recorded_actions((;
            choice=Vector{Union{MissingValue,Bool}}([MissingValue()]),
            skipped=Vector{Union{NoAction,Float64}}([NoAction(), NoAction()]),
        ))
        @test never.choice == Bool[] && never.choice isa Vector{Bool}
        @test never.skipped == Float64[] && never.skipped isa Vector{Float64}
        #An action whose column can only hold the markers is left out, by its type
        @test CognitiveModeling.recorded_actions((;
            choice=[true], never=[NoAction(), NoAction()], skipped=[MissingValue()]
        )) == (; choice=[true])
        @test CognitiveModeling.recorded_actions((;)) === (;)
    end

    @testset "Missing actions can be replaced by NoAction, which leaves the recorded ones as they are" begin
        replaced = CognitiveModeling.without_missing_actions(actions)
        @test replaced.choice == [true, NoAction(), false, NoAction(), true]
        @test replaced.bet == [NoAction(), NoAction(), 1.0, 2.0, NoAction()]
        @test CognitiveModeling.recorded_actions(replaced) ==
            CognitiveModeling.recorded_actions(actions)
        #The columns of the data aren't changed, and columns that can't hold a missing value keep their type
        @test actions.choice[2] === MissingValue()
        @test CognitiveModeling.without_missing_actions((; choice=[true, false])).choice isa
            Vector{Bool}
        @test CognitiveModeling.without_missing_actions((;)) === (;)
        #A column that can hold a missing value gets NoAction in its type whether it holds one or not, and one with the element type Any is left as it is here
        @test replaced.choice isa Vector{Any} && replaced.bet isa Vector{Any}
        @test CognitiveModeling.without_missing_actions((;
            choice=Vector{Union{MissingValue,Bool}}([true, false])
        )).choice isa Vector{Union{NoAction,Bool}}
        @test CognitiveModeling.without_missing_actions((;
            choice=Vector{Union{MissingValue,Bool}}([true, MissingValue()])
        )).choice isa Vector{Union{NoAction,Bool}}
    end

    @testset "A missing in the data is replaced by the given marker, by the element type of the column" begin
        replaced = CognitiveModeling.replace_missing(
            (; reward=[1.0, missing, 0.0], cue=[1, 2, 3]), MissingValue()
        )
        @test isequal(replaced.reward, [1.0, MissingValue(), 0.0]) &&
            replaced.reward isa Vector{Union{MissingValue,Float64}}
        @test replaced.cue == [1, 2, 3] && replaced.cue isa Vector{Int}
        @test CognitiveModeling.replace_missing(
            (; reward=Vector{Union{Missing,Float64}}([1.0, 0.0])), NoObservation()
        ).reward isa Vector{Union{NoObservation,Float64}}
        @test CognitiveModeling.replace_missing(
            (; reward=[1.0, missing]), NoObservation()
        ).reward == [1.0, NoObservation()]
        #Anything else is returned as it is, for check_data to report
        @test CognitiveModeling.replace_missing([1.0, missing], MissingValue()) ==
            [1.0, MissingValue()]
        @test CognitiveModeling.replace_missing("data", MissingValue()) == "data"
        #A column with the element type Any is narrowed to the types it holds
        @test CognitiveModeling.replace_missing(
            (; choice=[true, MissingValue(), true]), MissingValue()
        ).choice isa Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.replace_missing(
            (; choice=Any[true, missing, true]), NoAction()
        ).choice isa Vector{Union{NoAction,Bool}}
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
        #A row with a missing in any column used is missing
        named = CognitiveModeling.build_column(
            composite, (; choice=:choicecol, rt=:rtcol), 1:3
        )
        @test isequal(
            named, [(; choice=true, rt=0.5), (; choice=false, rt=0.7), missing]
        ) && named isa Vector{Union{Missing,@NamedTuple{choice::Bool,rt::Float64}}}
        @test CognitiveModeling.build_column(composite, (:rtcol, :cue), [1, 2]) ==
            [(0.5, 1), (0.7, 2)]
        @test CognitiveModeling.build_column(
            composite, (:rtcol, :cue) => (rt, cue) -> [rt, cue], [2]
        ) == [[0.7, 2.0]]
        @test CognitiveModeling.build_column(composite, :cue, [3, 1]) == [3, 1]
        #The rows of a session come with the model's names
        @test isequal(
            CognitiveModeling.select_rows(
                composite, (; ssm_action=(; choice=:choicecol, rt=:rtcol), cue=:cue), [1, 3]
            ),
            (; ssm_action=[(; choice=true, rt=0.5), missing], cue=[1, 3]),
        )
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

    @testset "The rows of a session are selected with the model's names" begin
        @test CognitiveModeling.select_rows(table, (; reward=:outcome), [1, 2]) ==
            (; reward=[1.0, 0.0])
        #A column with the element type Any is narrowed as a whole, so every session gets the same type
        @test CognitiveModeling.select_rows(
            (; choice=[true, false, MissingValue()]), (; choice=:choice), [1, 2]
        ).choice isa Vector{Union{MissingValue,Bool}}
        @test CognitiveModeling.select_rows(table, (;), [1, 2]) === (;)
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
        @test_throws "The table must be a NamedTuple with one vector for each column" CognitiveModeling.check_table(
            [1, 2], (:id,)
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

#A model for the error messages about scoring
struct ScoringModel <: AbstractCognitiveModel end

@testset "Scoring the recorded actions" begin
    distributions = (; choice=NormalByHand(0.0, 1.0), bet=NoAction())

    @testset "An action is recorded on a timestep when the data has a value, and a recorded action where the model took none is an error" begin
        @test CognitiveModeling.recorded_on_timestep(
            (; choice=0.5, bet=NoAction()), distributions, ScoringModel()
        ) === (; choice=true, bet=false)
        @test CognitiveModeling.recorded_on_timestep(
            (; choice=MissingValue(), bet=MissingValue()), distributions, ScoringModel()
        ) === (; choice=false, bet=false)
        @test CognitiveModeling.recorded_on_timestep((;), (;), ScoringModel()) === (;)
        @test_throws "The cognitive model of type ScoringModel took no action bet on a timestep where the data has 2.0 for it." CognitiveModeling.recorded_on_timestep(
            (; choice=0.5, bet=2.0), distributions, ScoringModel()
        )
    end

    @testset "What a session saves of a timestep is selected by the SaveHistory, with the action distributions" begin
        outputs = (;
            observation=(; reward=1.0),
            action=(; choice=true),
            action_distribution=(; choice=BernoulliByHand(0.7)),
            cognitive_state=(; expected_reward=0.5),
        )
        saved = CognitiveModeling.saved_session_outputs(
            ScoringModel(),
            outputs,
            SaveHistory(; action_distributions=:choice, states=SaveEverything()),
        )
        @test saved === (;
            observation=(;),
            action=(;),
            action_distribution=(; choice=BernoulliByHand(0.7)),
            state=(; expected_reward=0.5),
        )
        #Timestep 0 has BeforeFirstTimestep for every action distribution
        previous = (;
            observation=(; reward=BeforeFirstTimestep()),
            action=(; choice=BeforeFirstTimestep()),
            cognitive_state=(; expected_reward=0.5),
        )
        CognitiveModeling.attribute_names(::ScoringModel) = ModelAttributes(;
            parameters=(),
            states=:expected_reward,
            observations=:reward,
            actions=:choice,
        )
        initial = CognitiveModeling.initial_session_outputs(
            Val(true), ScoringModel(), previous, SaveHistory(SaveEverything())
        )
        @test initial.action_distribution === (; choice=BeforeFirstTimestep()) &&
            initial.action === (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.initial_session_outputs(
            Val(false), ScoringModel(), previous, SaveHistory(SaveEverything())
        ) === nothing
        #The storage starts with timestep 0 when it is kept, and later timesteps go in by index
        storage = CognitiveModeling.initialize_session_storage(initial, saved, 2)
        @test length(storage.action_distributions) == 3 &&
            storage.action_distributions[1] === (; choice=BeforeFirstTimestep())
        storage = CognitiveModeling.store_session_outputs!(storage, saved, 3)
        @test storage.action_distributions[3] === (; choice=BernoulliByHand(0.7)) &&
            storage.states[3] === (; expected_reward=0.5)
        @test length(
            CognitiveModeling.initialize_session_storage(nothing, saved, 2).states
        ) == 2
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
