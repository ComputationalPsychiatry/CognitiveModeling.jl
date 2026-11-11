## Models for the tests ##
#Lists its attributes, which is all that preparing a session needs from a model
struct SessionCounter <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::SessionCounter)
    return ModelAttributes(;
        parameters=:probability, states=:n_timesteps, inputs=:reward, outputs=:choice
    )
end

#Lists two observations, so that a single vector or a single column name can't say which is which
struct ObservesRewardAndCue <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ObservesRewardAndCue)
    return ModelAttributes(;
        parameters=:probability,
        states=:n_timesteps,
        inputs=(:reward, :cue),
        outputs=:choice,
    )
end

#An environment which gives the reward, and one which gives nothing. The choice it receives is its input
struct GivesReward <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesReward)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end
struct GivesNoReward <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesNoReward)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=())
end

#An environment which names the choice and the reward otherwise, and gives a cue which the counter doesn't observe
struct GivesPayoffAndCue <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesPayoffAndCue)
    return ModelAttributes(;
        parameters=:scale, states=(), inputs=:pull, outputs=(:payoff, :cue)
    )
end

#A session parameter model of one's own, without any of the optional methods
struct ParametersModelOfOwn <: AbstractSessionParametersModel end

@testset "Types for data" begin
    @testset "The marker for a missing value is its own type, apart from the one for nothing happening" begin
        @test MissingValue() === MissingValue()
        @test MissingValue() !== NoValue()
        @test !(MissingValue() isa AbstractNoValue)
        @test isbitstype(MissingValue)
    end

    @testset "The marker for a missing value that isn't imputed is passed on like those for nothing happening" begin
        @test UnimputedMissingValue() === UnimputedMissingValue()
        @test UnimputedMissingValue() isa AbstractNoValue
        @test UnimputedMissingValue() !== MissingValue()
        @test isbitstype(UnimputedMissingValue)
        @test_throws MethodError UnimputedMissingValue() + 1
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
        @test session_info.recorded_actions == (; choice=[true, false, true])
        @test session_info.recorded_observations == (; reward=[1.0, 0.0, 1.0])
        @test session_info.n_timesteps == 3
        @test session_info.who_starts === EnvironmentStarts()
        @test session_info.timestep_sizes == 1
        @test session_info.saved_timestep_sizes == Int[]
        @test session_info.cognitive_save_history === SaveHistory()
        @test session_info.environment_save_history === SaveHistory()
        @test session_info.check_stopping_criterion === nothing
        @test session_info.first_timestep_to_check_stopping == 4
        @test session_info.previous_observation === (; reward=BeforeFirstTimestep())
        @test session_info.previous_action === (; choice=BeforeFirstTimestep())
        @test !CognitiveModeling.has_environment(session_info)
        @test CognitiveModeling.recorded_action_names(session_info) === Val((:choice,))
        #Without an environment, no observation is scored, so none is named for scoring
        @test CognitiveModeling.recorded_observation_names(session_info) === Val(())
    end

    @testset "A save history can only select names that the models list" begin
        @test_throws "The SaveHistory selects the states (:surprise,), but attribute_names lists the states of a cognitive model of type SessionCounter" SessionInfo(
            SessionCounter(),
            (; reward=[1.0, 0.0, 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            cognitive_save_history=SaveHistory(; states=:surprise),
        )
        @test_throws "The SaveHistory selects the outputs (:cue,), but attribute_names lists the outputs of an environment model of type GivesReward" SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesReward(),
            environment_save_history=SaveHistory(; outputs=:cue),
        )
    end

    @testset "With maps of names, the environment's data and start are in its own names" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 0.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesPayoffAndCue(),
            observation_map=NameMap(:payoff => :reward),
            action_map=NameMap(:choice => :pull),
        )
        @test session_info.observation_map === NameMap(:payoff => :reward)
        @test session_info.action_map === NameMap(:choice => :pull)
        #The data stays in the cognitive model's names
        @test keys(session_info.observations) == (:reward,)
        #The environment's data has its own names, and the cue, which isn't passed on, is imputed on every timestep
        @test isequal(
            session_info.environment_observations.payoff, [1.0, MissingValue(), 0.0]
        )
        @test session_info.environment_observations.cue == fill(MissingValue(), 3)
        @test session_info.previous_environment_observation ===
            (; payoff=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
        #The recorded observations are scored in the environment's names
        @test keys(session_info.recorded_observations) == (:payoff,)
        @test session_info.recorded_observations.payoff == [1.0, 0.0]
        #A copy prepared again keeps the maps
        @test SessionInfo(
            session_info; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        ).observation_map === NameMap(:payoff => :reward)
        @test prepare_for_prediction(session_info).action_map === NameMap(:choice => :pull)
    end

    @testset "Maps of names must fit the models, and need an environment" begin
        @test_throws "The observation_map gives the observations (:feedback,), but a cognitive model of type SessionCounter lists (:reward,) as its inputs." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesPayoffAndCue(),
            observation_map=NameMap(:payoff => :feedback),
            action_map=NameMap(:choice => :pull),
        )
        @test_throws "A cognitive model of type SessionCounter lists the observations (:reward,) as its inputs, but an environment model of type GivesPayoffAndCue gives (:payoff, :cue)." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesPayoffAndCue(),
        )
        @test_throws "A session without an environment takes no observation_map or action_map" SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            observation_map=NameMap(:reward),
        )
    end

    @testset "The settings are kept as given" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesReward(),
            timestep_sizes=[0.5, 1.0, 1.5],
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            environment_save_history=SaveHistory(; outputs=:reward),
            check_stopping_criterion=false,
            previous_action=(; choice=false),
        )
        @test session_info.environment === GivesReward() &&
            CognitiveModeling.has_environment(session_info)
        @test session_info.who_starts === AgentStarts()
        @test session_info.timestep_sizes == [0.5, 1.0, 1.5]
        @test isequal(
            session_info.saved_timestep_sizes, [BeforeFirstTimestep(), 0.5, 1.0, 1.5]
        )
        @test session_info.cognitive_save_history === SaveHistory(; states=:n_timesteps)
        @test session_info.environment_save_history === SaveHistory(; outputs=:reward)
        @test session_info.check_stopping_criterion === false
        @test session_info.first_timestep_to_check_stopping == 4
        @test session_info.previous_action === (; choice=false)
        #The recorded observations leave out the missing one, and are named for scoring with an environment
        @test session_info.recorded_observations == (; reward=[1.0, 1.0])
        @test CognitiveModeling.recorded_observation_names(session_info) === Val((:reward,))
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

    @testset "A copy with other save histories has the sizes it returns made again, and everything else the same" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            timestep_sizes=0.5,
            check_stopping_criterion=false,
        )
        copy = SessionInfo(
            session_info; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        )
        @test copy.cognitive_save_history === SaveHistory(; states=:n_timesteps)
        @test copy.environment_save_history === SaveHistory()
        @test isequal(copy.saved_timestep_sizes, [BeforeFirstTimestep(), 0.5, 0.5, 0.5]) &&
            session_info.saved_timestep_sizes == Float64[]
        @test copy.observations === session_info.observations &&
            copy.actions === session_info.actions &&
            copy.recorded_actions == session_info.recorded_actions
        @test copy.who_starts === EnvironmentStarts() && copy.timestep_sizes == 0.5
        @test copy.check_stopping_criterion === false &&
            copy.first_timestep_to_check_stopping == 4
        #Saving the environment's history makes a session with one run its environment
        with_environment = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !with_environment.includes_environment
        saving = SessionInfo(
            with_environment; environment_save_history=SaveHistory(; outputs=:reward)
        )
        @test saving.includes_environment &&
            saving.environment_save_history === SaveHistory(; outputs=:reward) &&
            saving.cognitive_save_history === SaveHistory()
        #Checking the stopping criterion on every timestep makes it run too
        @test SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            check_stopping_criterion=true,
        ).includes_environment
    end

    @testset "The stopping criterion is checked from the timestep after the last one with data" begin
        counter_session(observations, actions; keywords...) = SessionInfo(
            SessionCounter(),
            observations,
            actions;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            keywords...,
        )
        #With data on every timestep, the check starts after the last one, and with none it starts at the first
        @test counter_session(COUNTER_REWARDS, COUNTER_CHOICES).first_timestep_to_check_stopping ==
            4
        @test counter_session(
            (; reward=fill(MissingValue(), 3)), (; choice=fill(MissingValue(), 3))
        ).first_timestep_to_check_stopping == 1
        #The last timestep with data in either kind counts, and a marker for nothing happening is data
        @test counter_session(
            (; reward=[1.0, MissingValue(), MissingValue()]),
            (; choice=[true, false, MissingValue()]),
        ).first_timestep_to_check_stopping == 3
        @test counter_session(
            (; reward=[1.0, MissingValue(), NoValue()]),
            (; choice=[true, MissingValue(), MissingValue()]),
        ).first_timestep_to_check_stopping == 4
        #true checks from the first timestep, and false never
        @test counter_session(
            COUNTER_REWARDS, COUNTER_CHOICES; check_stopping_criterion=true
        ).first_timestep_to_check_stopping == 1
        @test counter_session(
            COUNTER_REWARDS, COUNTER_CHOICES; check_stopping_criterion=false
        ).first_timestep_to_check_stopping == 4
        @test CognitiveModeling.last_timestep_with_data((;)) == 0
    end

    @testset "A missing in the data becomes MissingValue by default, and the replacements say which values become which markers" begin
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
        @test session_info.recorded_actions == (; choice=[true, true])
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, missing, true]);
            who_starts=EnvironmentStarts(),
            observation_replacements=(missing => NoValue(),),
            action_replacements=(missing => NoValue(),),
        )
        @test session_info.observations.reward == [1.0, NoValue(), 1.0] &&
            session_info.observations.reward isa Vector{Union{NoValue,Float64}}
        @test session_info.actions.choice == [true, NoValue(), true] &&
            session_info.actions.choice isa Vector{Union{NoValue,Bool}}
        #The markers can also be in the data already, and the original columns aren't changed
        rewards = (; reward=[1.0, missing, 1.0])
        SessionInfo(
            SessionCounter(),
            rewards,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            observation_replacements=(missing => NoValue(),),
        )
        @test rewards.reward[2] === missing
        #A single pair can be given on its own, several values can each become a marker, and anything that isn't pairs is an error
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, -1.0]),
            (; choice=[1, -1, missing]);
            who_starts=EnvironmentStarts(),
            observation_replacements=(missing => UnimputedMissingValue(), -1 => NoValue()),
            action_replacements=-1 => NoValue(),
        )
        @test session_info.observations.reward == [1.0, UnimputedMissingValue(), NoValue()]
        @test isequal(session_info.actions.choice, [1, NoValue(), missing])
        @test_throws "The replacements of values in the data must be pairs from a value to the marker it becomes, such as (missing => MissingValue(), -1 => NoValue()), but they are NoValue()." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            action_replacements=NoValue(),
        )
    end

    @testset "Missing values that aren't imputed are passed on, and need no environment" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            observation_replacements=(missing => UnimputedMissingValue(),),
            action_replacements=(MissingValue() => UnimputedMissingValue(),),
        )
        @test session_info.observations.reward == [1.0, UnimputedMissingValue(), 1.0] &&
            session_info.observations.reward isa
              Vector{Union{UnimputedMissingValue,Float64}}
        @test session_info.actions.choice == [true, UnimputedMissingValue(), true] &&
            session_info.actions.choice isa Vector{Union{UnimputedMissingValue,Bool}}
        @test session_info.recorded_actions == (; choice=[true, true])
        #Such a timestep has data, so the stopping criterion is checked after it
        @test session_info.first_timestep_to_check_stopping == 4
        #A copy with other save histories, and one for prediction, replace nothing more in the data, also when missing was kept as a value
        kept = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            observation_replacements=(),
        )
        @test isequal(kept.observations.reward, [1.0, missing, 1.0])
        @test isequal(
            SessionInfo(kept; cognitive_save_history=SaveHistory(; states=:n_timesteps)).observations.reward,
            [1.0, missing, 1.0],
        )
        @test isequal(prepare_for_prediction(kept).observations.reward, [1.0, missing, 1.0])
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

    @testset "The environment runs only when an observation or an action is missing, and is kept either way" begin
        with_missing_reward = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test with_missing_reward.includes_environment
        #A missing action makes it run too, since the observations it scores inform the action
        @test SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        ).includes_environment
        #A missing action that isn't imputed leaves nothing for the observations to inform, and neither does a missing observation that isn't
        @test !SessionInfo(
            SessionCounter(),
            (; reward=[1.0, UnimputedMissingValue(), 1.0]),
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            action_replacements=(MissingValue() => UnimputedMissingValue(),),
        ).includes_environment
        #With nothing missing, it doesn't run, and there is no warning
        all_recorded = @test_logs SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !all_recorded.includes_environment
        @test all_recorded.environment === GivesReward()
        @test CognitiveModeling.has_environment(all_recorded)
        #Without an environment, nothing runs
        @test !SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        ).includes_environment
        #The copy without the environment is the same session otherwise
        without = CognitiveModeling.without_environment(with_missing_reward)
        @test without.environment === NoEnvironment() && !without.includes_environment
        @test without.observations === with_missing_reward.observations &&
            without.previous_observation === with_missing_reward.previous_observation
    end

    @testset "Whether the environment is included can be told, and what needs it is then an error without it" begin
        #By default the setting is nothing, and the data decides
        recorded = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test recorded.include_environment === nothing && !recorded.includes_environment
        #It can be included when nothing needs it
        included = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=true,
        )
        @test included.include_environment === true && included.includes_environment
        #And left out when an action is missing, which is then imputed without the observations informing it
        left_out = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, MissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=false,
        )
        @test left_out.include_environment === false && !left_out.includes_environment
        #The setting is kept in a copy with other save histories, and a copy for prediction then keeps its observations
        @test SessionInfo(included; cognitive_save_history=SaveHistory(SaveEverything())).includes_environment
        predictive = prepare_for_prediction(left_out)
        @test predictive.observations === left_out.observations &&
            !predictive.includes_environment
        #What needs the environment is an error without it
        @test_throws "include_environment = false leaves the environment out of the session, but the data has observations that are MissingValue(), which only the environment can impute." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=false,
        )
        @test_throws "include_environment = false leaves the environment out of the session, but environment_save_history saves its history." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=false,
            environment_save_history=SaveHistory(; outputs=:reward),
        )
        @test_throws "include_environment = false leaves the environment out of the session, but check_stopping_criterion = true checks its stopping criterion on every timestep." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=false,
            check_stopping_criterion=true,
        )
        @test_throws "include_environment = true needs an environment model, but the session has none." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            include_environment=true,
        )
        #A full model which estimates the environment's parameters includes it in every session, which a session that leaves it out can't be
        @test CognitiveModeling.including_environment(recorded).includes_environment
        @test_throws "The environment's parameters are estimated, which every session must include its environment for, but a session was prepared with include_environment = false." CognitiveModeling.including_environment(
            left_out
        )
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
        @test_throws "The observations are a single vector, which can only be their data when the model has only one of them in its attributes, but a cognitive model of type ObservesRewardAndCue has the observations (:reward, :cue)." SessionInfo(
            ObservesRewardAndCue(),
            [1.0, 0.0, 1.0],
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "The data, the names and the environment are checked" begin
        @test_throws "The observations in the data are named (:cue,), but attribute_names lists the observations of a cognitive model of type SessionCounter, its inputs, as (:reward,). The data must have every observation the cognitive model lists." SessionInfo(
            SessionCounter(),
            (; cue=[1, 2, 3]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The actions in the data are named (:bet,), but attribute_names lists the actions of a cognitive model of type SessionCounter, its outputs, as (:choice,)." SessionInfo(
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
        @test_throws "The observations have entries that are MissingValue(), but no environment was given to impute them. Give an environment model, or have the model receive them unimputed, with observation_replacements = (missing => UnimputedMissingValue(),)." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, MissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        #The environment must list the same names as the model
        @test_throws "A cognitive model of type SessionCounter lists the observations (:reward,) as its inputs, but an environment model of type GivesNoReward gives (). Without a map, the two must list the same observations." SessionInfo(
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
        @test sessions_info.sessions[1].recorded_actions ==
            (; choice=[(; choice=true, rt=0.5), (; choice=false, rt=0.7)])
        #A session without a recorded value has an empty vector of the same type, so that the sessions share a type
        @test sessions_info.sessions[2].recorded_actions.choice == [] &&
            sessions_info.sessions[2].recorded_actions.choice isa
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

    @testset "The replacements are made in the whole table, also in the columns an attribute is built from" begin
        table = (;
            id=["S1", "S1", "S2", "S2"],
            outcome=[1.0, missing, 1.0, 0.0],
            choicecol=[1, 0, -1, 1],
            rtcol=[0.5, 0.7, 0.9, missing],
        )
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=(; choice=(; choice=:choicecol, rt=:rtcol)),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            observation_replacements=(missing => UnimputedMissingValue(),),
            action_replacements=(missing => MissingValue(), -1 => NoValue()),
        )
        #A missing observation that isn't imputed needs no environment, and every session has the marker in its type
        @test sessions_info.sessions[1].observations.reward ==
            [1.0, UnimputedMissingValue()]
        @test sessions_info.sessions[2].observations.reward isa
            Vector{Union{UnimputedMissingValue,Float64}}
        #A replaced value in any column that an attribute is built from makes the whole value its marker
        @test sessions_info.sessions[1].actions.choice ==
            [(; choice=1, rt=0.5), (; choice=0, rt=0.7)]
        @test sessions_info.sessions[2].actions.choice == [NoValue(), MissingValue()]
        @test isconcretetype(eltype(sessions_info.sessions))
        #A value which is equal to a pair's without having its type is replaced too, and every session then has the marker in its type
        with_code = MultipleSessionsInfo(
            SessionCounter(),
            (; id=["S1", "S2"], outcome=[1.0, -1.0], choice=[true, false]);
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            observation_replacements=-1 => NoValue(),
        )
        @test with_code.sessions[2].observations.reward == [NoValue()] &&
            with_code.sessions[1].observations.reward isa Vector{Union{NoValue,Float64}}
        #Without replacements, missing stays in the data as a value
        kept = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=(; reward=:outcome),
            action_columns=:choicecol,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            observation_replacements=(),
        )
        @test isequal(kept.sessions[1].observations.reward, [1.0, missing])
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
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            action_replacements=(missing => UnimputedMissingValue(),),
        )
        @test all(
            session ->
                session.who_starts === AgentStarts() &&
                session.cognitive_save_history === SaveHistory(; states=:n_timesteps),
            sessions_info.sessions,
        )
        @test sessions_info.sessions[1].timestep_sizes == [1.0, 0.5] &&
            sessions_info.sessions[2].timestep_sizes == [1.0, 2.0, 1.0]
        @test sessions_info.sessions[2].actions.choice ==
            [true, UnimputedMissingValue(), false]
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

    @testset "An environment runs only in the sessions where an observation or an action is missing" begin
        #S2 misses a choice in COUNTER_TABLE, and S1 nothing
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test CognitiveModeling.has_environment(sessions_info)
        @test !sessions_info.sessions[1].includes_environment &&
            sessions_info.sessions[2].includes_environment
        #The sessions still have one type, and with nothing missing in any, none runs it, with no warning
        @test isconcretetype(eltype(sessions_info.sessions))
        all_recorded = @test_logs MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            action_replacements=(missing => UnimputedMissingValue(),),
        )
        @test !any(session -> session.includes_environment, all_recorded.sessions)
        #A copy with another save history for every session
        copy = MultipleSessionsInfo(
            sessions_info; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        )
        @test all(
            session ->
                session.cognitive_save_history === SaveHistory(; states=:n_timesteps),
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
        @test_throws "Every column of the table must be a vector" MultipleSessionsInfo(
            SessionCounter(),
            (; id="S1", outcome=1.0, choice=true);
            observation_columns=(; reward=:outcome),
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "The table is kept, with the rows of each session, also in copies, and prepared sessions have none unless it is given" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test sessions_info.table === COUNTER_TABLE
        @test sessions_info.session_rows == [[1, 2], [3, 4, 5]]
        #The copies with other save histories and for prediction keep both
        for copy in (
            MultipleSessionsInfo(
                sessions_info; cognitive_save_history=SaveHistory(SaveEverything())
            ),
            prepare_for_prediction(sessions_info),
        )
            @test copy.table === COUNTER_TABLE
            @test copy.session_rows == [[1, 2], [3, 4, 5]]
        end
        #The values of a column are read for each session
        @test CognitiveModeling.session_column_values(sessions_info, :block) ==
            [[1, 1], [1, 1, 2]]
        @test_throws "The table that the sessions were split from has no column age. Its columns are (:id, :block, :outcome, :choice, :dt)." CognitiveModeling.session_column_values(
            sessions_info, :age
        )
        #Sessions that are given already prepared have no table, unless it is given with the rows of each
        prepared = MultipleSessionsInfo(sessions_info.sessions, sessions_info.session_ids)
        @test prepared.table === nothing && prepared.session_rows === nothing
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column block." CognitiveModeling.session_column_values(
            prepared, :block
        )
        with_table = MultipleSessionsInfo(
            sessions_info.sessions,
            sessions_info.session_ids;
            table=COUNTER_TABLE,
            session_rows=[1:2, 3:5],
        )
        @test CognitiveModeling.session_column_values(with_table, :block) ==
            [[1, 1], [1, 1, 2]]
        @test_throws "A table must come with the rows of each session in it, as session_rows, and the rows with the table." MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids; table=COUNTER_TABLE
        )
        @test_throws "There are 2 sessions and rows for 1, but there must be rows for each session." MultipleSessionsInfo(
            sessions_info.sessions,
            sessions_info.session_ids;
            table=COUNTER_TABLE,
            session_rows=[1:5],
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
        #Every session needs a name of its own, which its identifier gives
        @test_throws "The session identifier \"A\" is given for more than one session." MultipleSessionsInfo(
            [session, session], ["A", "A"]
        )
        @test_throws "The session identifiers 1 and \"1\" both give the session name 1," MultipleSessionsInfo(
            [session, session], Any[1, "1"]
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

@testset "prepare_for_prediction" begin
    @testset "Every action is missing, and the observations stay those in the data without an environment" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, missing, true]);
            who_starts=AgentStarts(),
            timestep_sizes=[1.0, 2.0, 1.0],
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            action_replacements=(missing => UnimputedMissingValue(),),
        )
        predictive = prepare_for_prediction(session_info)
        @test predictive.actions == (; choice=fill(MissingValue(), 3))
        @test predictive.observations === session_info.observations
        @test isempty(predictive.recorded_actions) && !predictive.includes_environment
        #The observations are still data, so the stopping criterion is checked after them, which is never
        @test predictive.first_timestep_to_check_stopping == 4
        #Everything else is kept
        @test predictive.who_starts === AgentStarts() && predictive.n_timesteps == 3
        @test predictive.timestep_sizes == [1.0, 2.0, 1.0]
        @test predictive.cognitive_save_history === session_info.cognitive_save_history
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
        predictive = prepare_for_prediction(session_info)
        @test predictive.observations == (; reward=fill(MissingValue(), 3))
        @test predictive.includes_environment && predictive.environment === GivesReward()
        #With nothing recorded, the stopping criterion is checked from the first timestep
        @test predictive.first_timestep_to_check_stopping == 1
    end

    @testset "The copy checks the stopping criterion as it is told, whatever the session was prepared with" begin
        never_checked = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            check_stopping_criterion=false,
        )
        @test never_checked.first_timestep_to_check_stopping == 4
        #By default as a simulation does, from the first timestep, since the copy has no data
        @test prepare_for_prediction(never_checked).first_timestep_to_check_stopping == 1
        @test prepare_for_prediction(never_checked).check_stopping_criterion === nothing
        #Or never, when it is told so
        @test prepare_for_prediction(never_checked; check_stopping_criterion=false).first_timestep_to_check_stopping ==
            4
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
        predictive = prepare_for_prediction(sessions_info)
        @test predictive.session_ids == sessions_info.session_ids
        @test [session.n_timesteps for session in predictive.sessions] == [2, 3]
        @test all(
            session -> all(==(MissingValue()), session.actions.choice), predictive.sessions
        )
        #A keyword is passed on to every session
        @test all(
            session -> session.check_stopping_criterion === false,
            prepare_for_prediction(sessions_info; check_stopping_criterion=false).sessions,
        )
    end
end

@testset "The parts of a full model, and how the models run" begin
    @test NoDownstreamModel() isa AbstractDownstreamModel
    @test isbitstype(NoDownstreamModel) && isbitstype(NoEnvironment)
    @test isbitstype(NoParallelization) && isbitstype(ParallelSessions)

    @testset "The name of a session is its identifier, or the name and the value of each of several columns" begin
        @test CognitiveModeling.session_name("S1") === :S1
        @test CognitiveModeling.session_name(2) === Symbol("2")
        @test CognitiveModeling.session_name((; id="S1", block=2)) === :id_S1__block_2
    end

    @testset "With sampled environment parameters, every session with an environment runs it" begin
        #S1 has nothing missing, so it doesn't run its environment until the parameters are sampled
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !sessions_info.sessions[1].includes_environment
        @test CognitiveModeling.sessions_for_environment_parameters(
            NoParameters(), sessions_info
        ) === sessions_info
        @test CognitiveModeling.sessions_for_environment_parameters(
            FixedSessionParameters((; scale=2.0)), sessions_info
        ) === sessions_info
        for sampling in (
            IdenticalSessionParameters((; scale=1)),
            IndependentSessionParameters((; scale=1)),
            ParametersModelOfOwn(),
        )
            sampled = CognitiveModeling.sessions_for_environment_parameters(
                sampling, sessions_info
            )
            @test all(session -> session.includes_environment, sampled.sessions)
            @test sampled.session_ids == sessions_info.session_ids
            @test sampled.sessions[1].observations ===
                sessions_info.sessions[1].observations
        end
        #Sessions without an environment still have none to run
        without = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test !any(
            session -> session.includes_environment,
            CognitiveModeling.sessions_for_environment_parameters(
                IdenticalSessionParameters((; scale=1)), without
            ).sessions,
        )
    end
end
