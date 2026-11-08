## Models for the tests ##
#Lists its attributes, which is all that preparing a session needs from a model
struct SessionCounter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::SessionCounter)
    return ModelAttributes(;
        parameters=:probability, states=:n_timesteps, observations=:reward, actions=:choice
    )
end

#Lists two observations, so that a single vector or a single column name can't say which is which
struct ObservesRewardAndCue <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ObservesRewardAndCue)
    return ModelAttributes(;
        parameters=:probability,
        states=:n_timesteps,
        observations=(:reward, :cue),
        actions=:choice,
    )
end

#An environment which gives the reward, and one which gives nothing
struct GivesReward <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesReward)
    return ModelAttributes(;
        parameters=:scale, states=(), observations=:reward, actions=:choice
    )
end
struct GivesNoReward <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesNoReward)
    return ModelAttributes(; parameters=(), states=(), observations=(), actions=())
end

@testset "Types for data" begin
    @testset "The marker for a missing value is its own type, apart from those for nothing happening" begin
        @test MissingValue() === MissingValue()
        @test MissingValue() !== NoAction() && MissingValue() !== NoObservation()
        @test !(MissingValue() isa AbstractNoValue)
        @test isbitstype(MissingValue)
    end

    @testset "It can stand in for one entry of the actions and observations" begin
        action = (; choice=MissingValue(), bet=1.0)
        @test action.choice isa MissingValue && action.bet == 1.0
        #Unlike missing, it doesn't pass through arithmetic without an error
        @test_throws MethodError MissingValue() + 1
        @test_throws MethodError MissingValue() * 2.0
    end
end

const COUNTER_REWARDS = (; reward=[1.0, 0.0, 1.0])
const COUNTER_CHOICES = (; choice=[true, false, true])

@testset "SessionInfo" begin
    @testset "It keeps the data, the models and the settings, with their defaults" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        @test session_info.cognitive_model === SessionCounter()
        @test session_info.environment === NoEnvironment()
        @test session_info.observations === COUNTER_REWARDS &&
            session_info.actions === COUNTER_CHOICES
        @test session_info.recorded == (; choice=[true, false, true])
        @test session_info.n_timesteps == 3
        @test session_info.who_starts === EnvironmentStarts()
        @test session_info.timestep_sizes == 1
        @test session_info.saved_timestep_sizes == Int[]
        @test session_info.save_history === SaveHistory()
        @test session_info.previous_observation === (; reward=BeforeFirstTimestep())
        @test session_info.previous_action === (; choice=BeforeFirstTimestep())
        @test !CognitiveModeling.has_environment(session_info)
        @test CognitiveModeling.recorded_action_names(session_info) === Val((:choice,))
    end

    @testset "The settings are kept as given" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesReward(),
            timestep_sizes=[0.5, 1.0, 1.5],
            save_history=SaveHistory(; states=:n_timesteps),
            previous_action=(; choice=false),
        )
        @test session_info.environment === GivesReward() &&
            CognitiveModeling.has_environment(session_info)
        @test session_info.who_starts === AgentStarts()
        @test session_info.timestep_sizes == [0.5, 1.0, 1.5]
        @test isequal(
            session_info.saved_timestep_sizes, [BeforeFirstTimestep(), 0.5, 1.0, 1.5]
        )
        @test session_info.save_history === SaveHistory(; states=:n_timesteps)
        @test session_info.previous_action === (; choice=false)
        @test_throws UndefKeywordError SessionInfo(
            SessionCounter(), COUNTER_REWARDS, COUNTER_CHOICES
        )
        @test_throws MethodError SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            catch_invalid_parameters=true,
        )
    end

    @testset "A copy with another save history has the sizes it returns made again, and everything else the same" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            timestep_sizes=0.5,
        )
        copy = SessionInfo(session_info; save_history=SaveHistory(; states=:n_timesteps))
        @test copy.save_history === SaveHistory(; states=:n_timesteps)
        @test isequal(copy.saved_timestep_sizes, [BeforeFirstTimestep(), 0.5, 0.5, 0.5]) &&
            session_info.saved_timestep_sizes == Float64[]
        @test copy.observations === session_info.observations &&
            copy.actions === session_info.actions &&
            copy.recorded === session_info.recorded
        @test copy.who_starts === EnvironmentStarts() && copy.timestep_sizes == 0.5
    end

    @testset "A missing in the data becomes MissingValue by default, or the marker for nothing happening" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, missing, true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test session_info.observations.reward == [1.0, MissingValue(), 1.0] &&
            session_info.observations.reward isa Vector{Union{MissingValue,Float64}}
        @test session_info.actions.choice == [true, MissingValue(), true] &&
            session_info.actions.choice isa Vector{Union{MissingValue,Bool}}
        @test session_info.recorded == (; choice=[true, true])
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, missing, true]);
            who_starts=EnvironmentStarts(),
            missing_observations=NoObservation(),
            missing_actions=NoAction(),
        )
        @test session_info.observations.reward == [1.0, NoObservation(), 1.0] &&
            session_info.observations.reward isa Vector{Union{NoObservation,Float64}}
        @test session_info.actions.choice == [true, NoAction(), true] &&
            session_info.actions.choice isa Vector{Union{NoAction,Bool}}
        #The markers can also be in the data already, and the original columns aren't changed
        rewards = (; reward=[1.0, missing, 1.0])
        SessionInfo(
            SessionCounter(),
            rewards,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            missing_observations=NoObservation(),
        )
        @test rewards.reward[2] === missing
        @test_throws TypeError SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            missing_actions=NoObservation(),
        )
    end

    @testset "Missing actions that aren't imputed are replaced by NoAction" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            impute_missing_actions=false,
        )
        @test session_info.actions.choice == [true, NoAction(), true] &&
            session_info.actions.choice isa Vector{Union{NoAction,Bool}}
        @test session_info.recorded == (; choice=[true, true])
    end

    @testset "A column written with a marker in it has the element type Any, and is narrowed to the types it holds" begin
        @test [true, MissingValue(), true] isa Vector{Any}
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=Any[1.0, 0.0, 1.0]),
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
        )
        @test session_info.actions.choice isa Vector{Union{MissingValue,Bool}}
        @test session_info.observations.reward isa Vector{Float64}
        @test SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=Real[1, 0.5, 1]);
            who_starts=EnvironmentStarts(),
        ).actions.choice isa Vector{Union{Float64,Int}}
    end

    @testset "The environment runs only when an observation is missing, and is kept either way" begin
        with_missing_reward = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test with_missing_reward.runs_environment
        #With every observation in the data, it has nothing to fill in, also when an action is missing, and there is no warning
        all_rewards = @test_logs SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !all_rewards.runs_environment
        @test all_rewards.environment === GivesReward()
        @test CognitiveModeling.has_environment(all_rewards)
        #Without an environment, nothing runs
        @test !SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        ).runs_environment
        #The copy without the environment is the same session otherwise
        without = CognitiveModeling.without_environment(with_missing_reward)
        @test without.environment === NoEnvironment() && !without.runs_environment
        @test without.observations === with_missing_reward.observations &&
            without.previous_observation === with_missing_reward.previous_observation
    end

    @testset "A single vector is the data of the model's only observation or action, whatever its name" begin
        session_info = SessionInfo(
            SessionCounter(),
            [1.0, 0.0, 1.0],
            [true, false, true];
            who_starts=EnvironmentStarts(),
        )
        @test session_info.observations == COUNTER_REWARDS &&
            session_info.actions == COUNTER_CHOICES
        #A missing in it becomes the marker, as in a NamedTuple
        @test isequal(
            SessionInfo(
                SessionCounter(),
                COUNTER_REWARDS,
                [true, missing, true];
                who_starts=EnvironmentStarts(),
            ).actions.choice,
            [true, MissingValue(), true],
        )
        #NamedTuples with the model's names are the data written as a row for each timestep, which is still an error
        @test_throws "must each be a NamedTuple with one vector for each name" SessionInfo(
            SessionCounter(),
            [(; reward=1.0)],
            [(; choice=true)];
            who_starts=EnvironmentStarts(),
        )
        #A model with several observations needs them by name
        @test_throws "The observations are a single vector, which can only be the data of a model's only one, but a cognitive model of type ObservesRewardAndCue lists the observations (:reward, :cue)." SessionInfo(
            ObservesRewardAndCue(),
            [1.0, 0.0, 1.0],
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "The data, the names and the environment are checked" begin
        @test_throws "The observations in the data are named (:cue,), but attribute_names lists the observations of a cognitive model of type SessionCounter as (:reward,)." SessionInfo(
            SessionCounter(),
            (; cue=[1, 2, 3]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The actions in the data are named (:bet,), but attribute_names lists the actions of a cognitive model of type SessionCounter as (:choice,)." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; bet=[1, 2, 3]);
            who_starts=EnvironmentStarts(),
        )
        @test_throws "Every observation and action in the data must have one value for each timestep, but their lengths are (reward = 3,) and (choice = 2,)." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
        )
        @test_throws "must each be a NamedTuple with one vector for each name" SessionInfo(
            SessionCounter(),
            Dict(:reward => [1.0, 0.0, 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        @test_throws "timestep_sizes must have one size for each timestep, but it has 2 sizes and the data has 3 timesteps." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            timestep_sizes=[1.0, 2.0],
        )
        #Without an environment, no observation can be missing
        @test_throws "The observations have entries that are MissingValue(), but no environment was given to fill them in." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        #The environment must list the same names as the model
        @test_throws "A cognitive model of type SessionCounter lists the observations (:reward,), but an environment model of type GivesNoReward gives ()." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesNoReward(),
        )
        #The data can have more observations than the model lists
        @test SessionInfo(
            SessionCounter(),
            (; reward=[1.0, 0.0, 1.0], cue=[1, 2, 3]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        ).observations.cue == [1, 2, 3]
    end

    @testset "The REPL shows a summary" begin
        @test repr(
            "text/plain",
            SessionInfo(
                SessionCounter(),
                COUNTER_REWARDS,
                COUNTER_CHOICES;
                who_starts=EnvironmentStarts(),
            ),
        ) == """
SessionInfo
Cognitive model: SessionCounter
Environment: none
Timesteps: 3
Observations: reward
Actions: choice
Who starts: environment"""
        @test repr(
            "text/plain",
            SessionInfo(
                SessionCounter(),
                (; reward=[1.0, MissingValue(), 1.0]),
                COUNTER_CHOICES;
                who_starts=AgentStarts(),
                environment=GivesReward(),
            ),
        ) == """
SessionInfo
Cognitive model: SessionCounter
Environment: GivesReward
Timesteps: 3
Observations: reward
Actions: choice
Who starts: agent"""
    end
end

const COUNTER_TABLE = (;
    id=["S1", "S1", "S2", "S2", "S2"],
    block=[1, 1, 1, 1, 2],
    outcome=[1.0, 0.0, 1.0, 1.0, 0.0],
    choice=[true, false, true, missing, false],
    dt=[1.0, 0.5, 1.0, 2.0, 1.0],
)

@testset "MultipleSessionsInfo" begin
    @testset "It splits a table into sessions by the session columns, in the order of first appearance" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test sessions_info.session_ids == ["S1", "S2"]
        @test CognitiveModeling.n_sessions(sessions_info) == 2
        @test sessions_info.sessions[1].observations == (; reward=[1.0, 0.0]) &&
            sessions_info.sessions[1].actions == (; choice=[true, false])
        @test sessions_info.sessions[2].observations == (; reward=[1.0, 1.0, 0.0])
        @test isequal(
            sessions_info.sessions[2].actions.choice, [true, MissingValue(), false]
        )
        #Every session has the same type, since a column which can hold missing gets the marker in its type in every session
        @test sessions_info.sessions[1].actions.choice isa Vector{Union{MissingValue,Bool}}
        @test isconcretetype(eltype(sessions_info.sessions))
        @test !CognitiveModeling.has_environment(sessions_info)
        #Several session columns give a NamedTuple as the identifier
        by_block = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=(:id, :block),
            who_starts=EnvironmentStarts(),
        )
        @test by_block.session_ids ==
            [(; id="S1", block=1), (; id="S2", block=1), (; id="S2", block=2)]
        @test by_block.sessions[3].observations == (; reward=[0.0])
    end

    @testset "A single column name is the column of the model's only observation or action, whatever its name" begin
        table = (;
            id=["S1", "S1", "S2"], outcome=[1.0, 0.0, 1.0], response=[true, false, true]
        )
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome,
            action_columns=:response,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test sessions_info.sessions[1].observations == (; reward=[1.0, 0.0]) &&
            sessions_info.sessions[2].actions == (; choice=[true])
        #A name the model lists as the other kind may mean that the two keyword arguments are swapped
        swapped_table = (; id=["S1", "S1"], reward=[1.0, 0.0], choice=[true, false])
        @test_throws "observation_columns is :choice, which a cognitive model of type SessionCounter lists among its actions, not its observations. Check that observation_columns and action_columns aren't swapped." MultipleSessionsInfo(
            SessionCounter(),
            swapped_table;
            observation_columns=:choice,
            action_columns=:reward,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        #A model with several observations needs the column of each
        @test_throws "A cognitive model of type ObservesRewardAndCue lists the observations (:reward, :cue), so observation_columns must give the column of each by name, such as (reward = :reward, cue = :cue), and not the single name :outcome." MultipleSessionsInfo(
            ObservesRewardAndCue(),
            table;
            observation_columns=:outcome,
            action_columns=:response,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "An attribute can be built from several columns, and the session identifier is looked up" begin
        table = (;
            id=["S1", "S1", "S2"],
            outcome=[1.0, 0.0, 1.0],
            choicecol=[true, false, missing],
            rtcol=[0.5, 0.7, 0.9],
        )
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=(; choice=(; choice=:choicecol, rt=:rtcol)),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test sessions_info.sessions[1].actions.choice ==
            [(; choice=true, rt=0.5), (; choice=false, rt=0.7)]
        @test sessions_info.sessions[2].actions.choice == [MissingValue()] &&
            sessions_info.sessions[2].actions.choice isa
              Vector{Union{MissingValue,@NamedTuple{choice::Bool,rt::Float64}}}
        @test sessions_info.sessions[1].recorded ==
            (; choice=[(; choice=true, rt=0.5), (; choice=false, rt=0.7)])
        #A session without a recorded value has an empty vector of the same type, so that the sessions share a type
        @test sessions_info.sessions[2].recorded.choice == [] &&
            sessions_info.sessions[2].recorded.choice isa
              Vector{@NamedTuple{choice::Bool,rt::Float64}}
        @test MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=(; choice=(:choicecol, :rtcol) => (c, rt) -> (c ? 1 : -1) * rt),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        ).sessions[1].actions.choice == [0.5, -0.7]
        @test_throws "The table has no column named rt." MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=(; choice=(; choice=:choicecol, rt=:rt)),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test CognitiveModeling.session_index(sessions_info, "S2") == 2
        @test_throws "There is no session with the identifier \"S3\". The identifiers are [\"S1\", \"S2\"]." CognitiveModeling.session_index(
            sessions_info, "S3"
        )
    end

    @testset "The settings go to every session, and the sizes of the timesteps can come from a column" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=AgentStarts(),
            timestep_sizes=:dt,
            save_history=SaveHistory(; states=:n_timesteps),
            impute_missing_actions=false,
        )
        @test all(
            session ->
                session.who_starts === AgentStarts() &&
                session.save_history === SaveHistory(; states=:n_timesteps),
            sessions_info.sessions,
        )
        @test sessions_info.sessions[1].timestep_sizes == [1.0, 0.5] &&
            sessions_info.sessions[2].timestep_sizes == [1.0, 2.0, 1.0]
        @test sessions_info.sessions[2].actions.choice == [true, NoAction(), false]
        @test MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=AgentStarts(),
            timestep_sizes=0.5,
        ).sessions[1].timestep_sizes == 0.5
    end

    @testset "An environment runs only in the sessions where an observation is missing" begin
        table = (; COUNTER_TABLE..., outcome=[1.0, 0.0, 1.0, missing, 0.0])
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test CognitiveModeling.has_environment(sessions_info)
        @test !sessions_info.sessions[1].runs_environment &&
            sessions_info.sessions[2].runs_environment
        #The sessions still have one type, and with no observation missing in any, none runs it, with no warning
        @test isconcretetype(eltype(sessions_info.sessions))
        all_rewards = @test_logs MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !any(session -> session.runs_environment, all_rewards.sessions)
        #A copy with another save history for every session
        copy = MultipleSessionsInfo(
            sessions_info; save_history=SaveHistory(; states=:n_timesteps)
        )
        @test all(
            session -> session.save_history === SaveHistory(; states=:n_timesteps),
            copy.sessions,
        ) && copy.session_ids == sessions_info.session_ids
    end

    @testset "The table and its columns are checked" begin
        @test_throws "The table has no column named reward. Its columns are (:id, :block, :outcome, :choice, :dt)." MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The table has no column named time." MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            timestep_sizes=:time,
        )
        @test_throws "Every column of the table must have the same length" MultipleSessionsInfo(
            SessionCounter(),
            (; id=["S1", "S1"], outcome=[1.0, 0.0, 1.0], choice=[true, false]);
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The table must be a NamedTuple with one vector for each column" MultipleSessionsInfo(
            SessionCounter(),
            [1, 2];
            observation_columns=:reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test_throws "Every column of the table must be a vector" MultipleSessionsInfo(
            SessionCounter(),
            (; id="S1", outcome=1.0, choice=true);
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "Prepared sessions can be given directly, and must have one type" begin
        session = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        sessions_info = MultipleSessionsInfo([session, session], ["A", "B"])
        @test sessions_info.sessions == [session, session] &&
            sessions_info.session_ids == ["A", "B"]
        other_type = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The sessions must all have the same type" MultipleSessionsInfo(
            [session, other_type], ["A", "B"]
        )
        @test_throws "There are 2 sessions and 1 session identifiers" MultipleSessionsInfo(
            [session, session], ["A"]
        )
        @test_throws "There must be at least one session." MultipleSessionsInfo(
            typeof(session)[], String[]
        )
    end

    @testset "The REPL shows a summary" begin
        @test repr(
            "text/plain",
            MultipleSessionsInfo(
                SessionCounter(),
                COUNTER_TABLE;
                observation_columns=(; reward=:outcome),
                action_columns=:choice,
                session_columns=:id,
                who_starts=EnvironmentStarts(),
            ),
        ) == """
MultipleSessionsInfo
Sessions: 2
Cognitive model: SessionCounter
Environment: none
Observations: reward
Actions: choice"""
    end
end

@testset "predictive_sessions" begin
    @testset "Every action is missing, and the observations stay those in the data without an environment" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, missing, true]);
            who_starts=AgentStarts(),
            timestep_sizes=[1.0, 2.0, 1.0],
            save_history=SaveHistory(; states=:n_timesteps),
            impute_missing_actions=false,
        )
        predictive = predictive_sessions(session_info)
        @test predictive.actions == (; choice=fill(MissingValue(), 3))
        @test predictive.observations === session_info.observations
        @test isempty(predictive.recorded) && !predictive.runs_environment
        #Everything else is kept
        @test predictive.who_starts === AgentStarts() && predictive.n_timesteps == 3
        @test predictive.timestep_sizes == [1.0, 2.0, 1.0]
        @test predictive.save_history === session_info.save_history
        @test predictive.previous_observation === session_info.previous_observation
    end

    @testset "With an environment, every observation is missing too, so the environment runs" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        predictive = predictive_sessions(session_info)
        @test predictive.observations == (; reward=fill(MissingValue(), 3))
        @test predictive.runs_environment && predictive.environment === GivesReward()
    end

    @testset "Several sessions are each copied, with their identifiers" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        predictive = predictive_sessions(sessions_info)
        @test predictive.session_ids == sessions_info.session_ids
        @test [session.n_timesteps for session in predictive.sessions] == [2, 3]
        @test all(
            session -> all(==(MissingValue()), session.actions.choice), predictive.sessions
        )
    end
end

@testset "The parts of a full model, and how the models run" begin
    @test IndependentSessions((; probability=1)) isa AbstractPopulationModel
    @test IndependentSessions((; probability=1)).priors === (; probability=1)
    @test_throws MethodError IndependentSessions([1, 2])
    @test NoDownstreamModel() isa AbstractDownstreamModel
    @test isbitstype(NoDownstreamModel) && isbitstype(NoEnvironment)
    @test isbitstype(NoParallelisation) && isbitstype(ParallelSessions)
end
