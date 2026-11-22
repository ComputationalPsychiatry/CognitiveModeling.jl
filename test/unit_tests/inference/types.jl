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

#A parameter model of one's own, without any of the optional methods
struct ParametersModelOfOwn <: AbstractParametersModel end

#A model whose reward, its input, is between 0 and 1, and whose choice, its output, is 1 or 2
struct ChoosesWithinDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ChoosesWithinDomains)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
function CognitiveModeling.input_domains(::ChoosesWithinDomains)
    return (; reward=Domain(; lower=0, upper=1))
end
function CognitiveModeling.output_domains(::ChoosesWithinDomains)
    return (; choice=Domain(; possible_values=(1, 2)))
end

#Models whose domains of inputs or outputs aren't right: for an input it doesn't have, and not Domains
struct DomainForAnotherInput <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::DomainForAnotherInput)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
CognitiveModeling.input_domains(::DomainForAnotherInput) = (; cue=Domain(; lower=0))
struct OutputDomainsNotDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::OutputDomainsNotDomains)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
CognitiveModeling.output_domains(::OutputDomainsNotDomains) = (; choice=(1, 2))

@testset "Types for data" begin
    @testset "The marker for a missing value is its own type, apart from the one for nothing happening" begin
        @test ImputedMissingValue() === ImputedMissingValue()
        @test ImputedMissingValue() !== NoValue()
        @test !(ImputedMissingValue() isa AbstractNoValue)
        @test isbitstype(ImputedMissingValue)
    end

    @testset "The marker for a missing value that isn't imputed is passed on like those for nothing happening" begin
        @test UnimputedMissingValue() === UnimputedMissingValue()
        @test UnimputedMissingValue() isa AbstractNoValue
        @test UnimputedMissingValue() !== ImputedMissingValue()
        @test isbitstype(UnimputedMissingValue)
        @test_throws MethodError UnimputedMissingValue() + 1
    end

    @testset "The marker for a value to simulate is its own type, apart from the one for a missing value" begin
        @test SimulatedMissingValue() === SimulatedMissingValue()
        @test SimulatedMissingValue() !== ImputedMissingValue()
        @test !(SimulatedMissingValue() isa AbstractNoValue)
        @test isbitstype(SimulatedMissingValue)
        @test_throws MethodError SimulatedMissingValue() + 1
    end

    @testset "It can stand in for one entry of the actions and observations" begin
        action = (; choice=ImputedMissingValue(), bet=1.0)
        @test action.choice isa ImputedMissingValue && action.bet == 1.0
        #Unlike missing, it doesn't pass through arithmetic without an error
        @test_throws MethodError ImputedMissingValue() + 1
        @test_throws MethodError ImputedMissingValue() * 2.0
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
        @test session_info.actions_to_condition_on == (; choice=[true, false, true])
        @test session_info.observations_to_condition_on == (; reward=[1.0, 0.0, 1.0])
        @test session_info.n_timesteps == 3
        @test session_info.who_starts === EnvironmentStarts()
        @test session_info.timestep_sizes == 1
        @test session_info.saved_timestep_sizes == Int[]
        @test session_info.cognitive_save_history === SaveHistory()
        @test session_info.environment_save_history === SaveHistory()
        @test session_info.check_stopping_criterion === nothing
        @test session_info.first_timestep_to_check_stopping == 4
        @test session_info.check_parameter_domains === true
        @test session_info.previous_observation === (; reward=BeforeFirstTimestep())
        @test session_info.previous_action === (; choice=BeforeFirstTimestep())
        @test !CognitiveModeling.has_environment(session_info)
        @test CognitiveModeling.action_names_to_condition_on(session_info) ===
            Val((:choice,))
        #Without an environment, no observation is scored, so none is named for scoring
        @test CognitiveModeling.observation_names_to_condition_on(session_info) === Val(())
    end

    @testset "A save history can only select names that the models list" begin
        @test_throws "The SaveHistory selects the states (:surprise,), but the states in the attributes of a cognitive model of type SessionCounter" SessionInfo(
            SessionCounter(),
            (; reward=[1.0, 0.0, 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            cognitive_save_history=SaveHistory(; states=:surprise),
        )
        @test_throws "The SaveHistory selects the outputs (:cue,), but the outputs in the attributes of an environment model of type GivesReward" SessionInfo(
            SessionCounter(),
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesReward(),
            environment_save_history=SaveHistory(; outputs=:cue),
        )
    end

    @testset "With maps of names, the environment's data and start are in its own names" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, ImputedMissingValue(), 0.0]),
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
            session_info.environment_observations.payoff, [1.0, ImputedMissingValue(), 0.0]
        )
        @test session_info.environment_observations.cue == fill(ImputedMissingValue(), 3)
        @test session_info.previous_environment_observation ===
            (; payoff=BeforeFirstTimestep(), cue=BeforeFirstTimestep())
        #The observations to condition on are in the environment's names
        @test keys(session_info.observations_to_condition_on) == (:payoff,)
        @test session_info.observations_to_condition_on.payoff == [1.0, 0.0]
        #A copy prepared again keeps the maps
        @test SessionInfo(
            session_info; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        ).observation_map === NameMap(:payoff => :reward)
        @test decondition_sessions(session_info).action_map === NameMap(:choice => :pull)
        #In the deconditioned copy, the cue is simulated like the observations, also in a copy of the copy, which is
        #prepared again
        deconditioned = decondition_sessions(session_info)
        @test deconditioned.environment_observations.cue == fill(SimulatedMissingValue(), 3)
        @test SessionInfo(
            deconditioned; cognitive_save_history=SaveHistory(; states=:n_timesteps)
        ).environment_observations.cue == fill(SimulatedMissingValue(), 3)
    end

    @testset "Maps of names must fit the models, and need an environment" begin
        @test_throws "The observation_map gives the observations (:feedback,), but the attributes of a cognitive model of type SessionCounter have (:reward,) as inputs." SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesPayoffAndCue(),
            observation_map=NameMap(:payoff => :feedback),
            action_map=NameMap(:choice => :pull),
        )
        @test_throws "The attributes of a cognitive model of type SessionCounter have the observations (:reward,) as inputs, but an environment model of type GivesPayoffAndCue gives (:payoff, :cue)." SessionInfo(
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
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=AgentStarts(),
            environment=GivesReward(),
            timestep_sizes=[0.5, 1.0, 1.5],
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            environment_save_history=SaveHistory(; outputs=:reward),
            check_stopping_criterion=false,
            check_parameter_domains=false,
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
        @test session_info.check_parameter_domains === false
        @test session_info.previous_action === (; choice=false)
        #The observations to condition on leave out the missing one, and are named for scoring with an environment
        @test session_info.observations_to_condition_on == (; reward=[1.0, 1.0])
        @test CognitiveModeling.observation_names_to_condition_on(session_info) ===
            Val((:reward,))
        @test_throws UndefKeywordError SessionInfo(
            SessionCounter(), COUNTER_REWARDS, COUNTER_CHOICES
        )
        @test_throws MethodError SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            parallelization=SerialSessions(),
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
            check_parameter_domains=false,
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
            copy.actions_to_condition_on == session_info.actions_to_condition_on
        @test copy.who_starts === EnvironmentStarts() && copy.timestep_sizes == 0.5
        @test copy.check_stopping_criterion === false &&
            copy.first_timestep_to_check_stopping == 4
        @test copy.check_parameter_domains === false
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
            (; reward=fill(ImputedMissingValue(), 3)),
            (; choice=fill(ImputedMissingValue(), 3)),
        ).first_timestep_to_check_stopping == 1
        #The last timestep with data in either kind counts, and a marker for nothing happening is data
        @test counter_session(
            (; reward=[1.0, ImputedMissingValue(), ImputedMissingValue()]),
            (; choice=[true, false, ImputedMissingValue()]),
        ).first_timestep_to_check_stopping == 3
        @test counter_session(
            (; reward=[1.0, ImputedMissingValue(), NoValue()]),
            (; choice=[true, ImputedMissingValue(), ImputedMissingValue()]),
        ).first_timestep_to_check_stopping == 4
        #A value to simulate isn't data either
        @test counter_session(
            (; reward=[1.0, SimulatedMissingValue(), SimulatedMissingValue()]),
            (; choice=[true, false, SimulatedMissingValue()]),
        ).first_timestep_to_check_stopping == 3
        #true checks from the first timestep, and false never
        @test counter_session(
            COUNTER_REWARDS, COUNTER_CHOICES; check_stopping_criterion=true
        ).first_timestep_to_check_stopping == 1
        @test counter_session(
            COUNTER_REWARDS, COUNTER_CHOICES; check_stopping_criterion=false
        ).first_timestep_to_check_stopping == 4
        @test CognitiveModeling.last_timestep_with_data((;)) == 0
    end

    @testset "A value to simulate is sampled as a missing one is: it isn't conditioned on, and the environment runs for it" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, SimulatedMissingValue(), 1.0]),
            (; choice=[true, SimulatedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test session_info.includes_environment
        @test session_info.actions_to_condition_on.choice == [true, true]
        @test session_info.observations_to_condition_on.reward == [1.0, 1.0]
        #Without an environment, a value to simulate among the observations is an error, as a missing one is
        @test_throws "The observations have entries that are ImputedMissingValue() or SimulatedMissingValue(), but no environment was given to impute or simulate them." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, SimulatedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "A missing in the data becomes ImputedMissingValue by default" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, missing, true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test session_info.observations.reward == [1.0, ImputedMissingValue(), 1.0] &&
            session_info.observations.reward isa
              Vector{Union{ImputedMissingValue,Float64}}
        @test session_info.actions.choice == [true, ImputedMissingValue(), true] &&
            session_info.actions.choice isa Vector{Union{ImputedMissingValue,Bool}}
        @test session_info.actions_to_condition_on == (; choice=[true, true])
        #The markers can also be in the data already, and the original columns aren't changed
        rewards = (; reward=[1.0, missing, 1.0])
        SessionInfo(
            SessionCounter(),
            rewards,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            impute_missing_observations=false,
        )
        @test rewards.reward[2] === missing
    end

    @testset "Missing values that aren't imputed are passed on, and need no environment" begin
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
            impute_missing_observations=false,
            impute_missing_actions=false,
        )
        @test session_info.observations.reward == [1.0, UnimputedMissingValue(), 1.0] &&
            session_info.observations.reward isa
              Vector{Union{UnimputedMissingValue,Float64}}
        @test session_info.actions.choice == [true, UnimputedMissingValue(), true] &&
            session_info.actions.choice isa Vector{Union{UnimputedMissingValue,Bool}}
        @test session_info.actions_to_condition_on == (; choice=[true, true])
        #Such a timestep has data, so the stopping criterion is checked after it
        @test session_info.first_timestep_to_check_stopping == 4
        #A copy with other save histories, and one for prediction, replace nothing more in the data, also when missing was kept as a value
        kept = SessionInfo(
            SessionCounter(),
            (; reward=[1.0, missing, 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            missing_observation_values=(),
        )
        @test isequal(kept.observations.reward, [1.0, missing, 1.0])
        @test isequal(
            SessionInfo(kept; cognitive_save_history=SaveHistory(; states=:n_timesteps)).observations.reward,
            [1.0, missing, 1.0],
        )
        @test isequal(decondition_sessions(kept).observations.reward, [1.0, missing, 1.0])
    end

    @testset "A column written with a marker in it has the element type Any, and is narrowed to the types it holds" begin
        @test [true, ImputedMissingValue(), true] isa Vector{Any}
        session_info = SessionInfo(
            SessionCounter(),
            (; reward=Any[1.0, 0.0, 1.0]),
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
        )
        @test session_info.actions.choice isa Vector{Union{ImputedMissingValue,Bool}}
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
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test with_missing_reward.includes_environment
        #A missing action makes it run too, since the observations it scores inform the action
        @test SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        ).includes_environment
        #A missing action that isn't imputed leaves nothing for the observations to inform, and neither does a missing observation that isn't
        @test !SessionInfo(
            SessionCounter(),
            (; reward=[1.0, UnimputedMissingValue(), 1.0]),
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            impute_missing_actions=false,
        ).includes_environment
        #With nothing missing, it doesn't run, and there is no warning
        nothing_missing = @test_logs SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test !nothing_missing.includes_environment
        @test nothing_missing.environment === GivesReward()
        @test CognitiveModeling.has_environment(nothing_missing)
        #Without an environment, nothing runs
        @test !SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        ).includes_environment
        #The copy without the environment is the same session otherwise
        without = CognitiveModeling.remake_session_without_environment(with_missing_reward)
        @test without.environment === NoEnvironment() && !without.includes_environment
        @test without.observations === with_missing_reward.observations &&
            without.previous_observation === with_missing_reward.previous_observation
    end

    @testset "Whether the environment is included can be told, and what needs it is then an error without it" begin
        #By default the setting is nothing, and the data decides
        nothing_missing = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        )
        @test nothing_missing.include_environment === nothing &&
            !nothing_missing.includes_environment
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
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            include_environment=false,
        )
        @test left_out.include_environment === false && !left_out.includes_environment
        #The setting is kept in a copy with other save histories, and a copy for prediction then keeps its observations
        @test SessionInfo(included; cognitive_save_history=SaveHistory(SaveEverything())).includes_environment
        predictive = decondition_sessions(left_out)
        @test predictive.observations === left_out.observations &&
            !predictive.includes_environment
        #What needs the environment is an error without it
        @test_throws "include_environment = false leaves the environment out of the session, but the data has observations that are ImputedMissingValue() or SimulatedMissingValue(), which only the environment can impute or simulate." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
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
        @test CognitiveModeling.remake_session_including_environment(nothing_missing).includes_environment
        @test_throws "The environment's parameters are estimated, which every session must include its environment for, but a session was prepared with include_environment = false." CognitiveModeling.remake_session_including_environment(
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
            [true, ImputedMissingValue(), true],
        )
        #NamedTuples with the model's names are the data written as a row for each timestep, which is still an error
        @test_throws "must each be a NamedTuple with one vector for each name" SessionInfo(
            SessionCounter(),
            [(; reward=1.0)],
            [(; choice=true)];
            who_starts=EnvironmentStarts(),
        )
        #A model with several observations needs them by name
        @test_throws "The observations are a single vector, which can only be their data when the model has only one kind of them in its attributes, but a cognitive model of type ObservesRewardAndCue has the observations (:reward, :cue)." SessionInfo(
            ObservesRewardAndCue(),
            [1.0, 0.0, 1.0],
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "The observations and the actions are checked against the domains of the model's inputs and outputs, apart from markers" begin
        @test SessionInfo(
            ChoosesWithinDomains(), [0.5, 1.0], [1, 2]; who_starts=EnvironmentStarts()
        ).actions == (; choice=[1, 2])
        @test_throws ArgumentError(
            "The action choice at timestep 2 is 3, which isn't one of its possible values, (1, 2).",
        ) SessionInfo(
            ChoosesWithinDomains(), [0.5, 1.0], [1, 3]; who_starts=EnvironmentStarts()
        )
        @test_throws ArgumentError(
            "The observation reward at timestep 1 is 1.5, which is outside its domain, from 0 to 1.",
        ) SessionInfo(
            ChoosesWithinDomains(), [1.5, 1.0], [1, 2]; who_starts=EnvironmentStarts()
        )
        #A missing value, and a marker for nothing happening, aren't checked
        @test SessionInfo(
            ChoosesWithinDomains(),
            [0.5, missing],
            [NoValue(), ImputedMissingValue()];
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
        ) isa SessionInfo
        #The domains must be Domains under the names of the model's inputs and outputs
        @test_throws ArgumentError(
            "input_domains of a cognitive model of type DomainForAnotherInput has domains for (:cue,), but the inputs in its attributes are (:reward,).",
        ) SessionInfo(
            DomainForAnotherInput(), [0.5, 1.0], [1, 2]; who_starts=EnvironmentStarts()
        )
        @test_throws "output_domains of a cognitive model of type OutputDomainsNotDomains returns (choice = (1, 2),), but it must return a NamedTuple with a Domain under the name of each of its outputs that has a domain" SessionInfo(
            OutputDomainsNotDomains(), [0.5, 1.0], [1, 2]; who_starts=EnvironmentStarts()
        )
    end

    @testset "The data, the names and the environment are checked" begin
        @test_throws "The observations in the data are named (:cue,), but the observations in the attributes of a cognitive model of type SessionCounter, its inputs, are (:reward,). The data must have every observation in the cognitive model's attributes." SessionInfo(
            SessionCounter(),
            (; cue=[1, 2, 3]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The actions in the data are named (:bet,), but the actions in the attributes of a cognitive model of type SessionCounter, its outputs, are (:choice,)." SessionInfo(
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
        @test_throws "The observations have entries that are ImputedMissingValue() or SimulatedMissingValue(), but no environment was given to impute or simulate them. Give an environment model, or have the model receive the missing ones as UnimputedMissingValue(), with impute_missing_observations = false." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        )
        #The environment must have the same names in its attributes as the model
        @test_throws "The attributes of a cognitive model of type SessionCounter have the observations (:reward,) as inputs, but an environment model of type GivesNoReward gives (). Without a map, the two must have the same observations in their attributes." SessionInfo(
            SessionCounter(),
            (; reward=[1.0, ImputedMissingValue(), 1.0]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
            environment=GivesNoReward(),
        )
        #The data can have more observations than those in the model's attributes
        @test SessionInfo(
            SessionCounter(),
            (; reward=[1.0, 0.0, 1.0], cue=[1, 2, 3]),
            COUNTER_CHOICES;
            who_starts=EnvironmentStarts(),
        ).observations.cue == [1, 2, 3]
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
            observation_columns=:outcome => :reward,
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
            sessions_info.sessions[2].actions.choice, [true, ImputedMissingValue(), false]
        )
        #Every session has the same type, since a column which can hold missing gets the marker in its type in every session
        @test sessions_info.sessions[1].actions.choice isa
            Vector{Union{ImputedMissingValue,Bool}}
        @test isconcretetype(eltype(sessions_info.sessions))
        @test !CognitiveModeling.has_environment(sessions_info)
        #Several session columns give a NamedTuple as the data identifier
        by_block = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome => :reward,
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
        #A name of the other kind in the model's attributes may mean that the two keyword arguments are swapped
        swapped_table = (; id=["S1", "S1"], reward=[1.0, 0.0], choice=[true, false])
        @test_throws "observation_columns is :choice, which is among the actions in the attributes of a cognitive model of type SessionCounter, not its observations. Check that observation_columns and action_columns aren't swapped." MultipleSessionsInfo(
            SessionCounter(),
            swapped_table;
            observation_columns=:choice,
            action_columns=:reward,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        #A model with several observations needs the column of each
        @test_throws "The attributes of a cognitive model of type ObservesRewardAndCue have the observations (:reward, :cue), so observation_columns must give the column of each, such as (:reward, :cue) when the columns have those names, or with a pair for a column of another name, as in :outcome => :reward, and not the single name :outcome." MultipleSessionsInfo(
            ObservesRewardAndCue(),
            table;
            observation_columns=:outcome,
            action_columns=:response,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "An attribute can be built from several columns" begin
        table = (;
            id=["S1", "S1", "S2"],
            outcome=[1.0, 0.0, 1.0],
            choicecol=[true, false, missing],
            rtcol=[0.5, 0.7, 0.9],
        )
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome => :reward,
            action_columns=(; choice=:choicecol, rt=:rtcol) => :choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test sessions_info.sessions[1].actions.choice ==
            [(; choice=true, rt=0.5), (; choice=false, rt=0.7)]
        @test sessions_info.sessions[2].actions.choice == [ImputedMissingValue()] &&
            sessions_info.sessions[2].actions.choice isa
              Vector{Union{ImputedMissingValue,@NamedTuple{choice::Bool,rt::Float64}}}
        @test sessions_info.sessions[1].actions_to_condition_on ==
            (; choice=[(; choice=true, rt=0.5), (; choice=false, rt=0.7)])
        #A session without a value to condition on has an empty vector of the same type, so that the sessions share a type
        @test sessions_info.sessions[2].actions_to_condition_on.choice == [] &&
            sessions_info.sessions[2].actions_to_condition_on.choice isa
              Vector{@NamedTuple{choice::Bool,rt::Float64}}
        @test MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome => :reward,
            action_columns=(:choicecol, :rtcol) => ((c, rt) -> (c ? 1 : -1) * rt) => :choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        ).sessions[1].actions.choice == [0.5, -0.7]
        @test_throws "The table has no column named rt." MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome => :reward,
            action_columns=(; choice=:choicecol, rt=:rt),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
    end

    @testset "Missing values are marked in the whole table, also in the columns an attribute is built from" begin
        table = (;
            id=["S1", "S1", "S2", "S2"],
            outcome=[1.0, missing, 1.0, 0.0],
            choicecol=[1, 0, -1, 1],
            rtcol=[0.5, 0.7, 0.9, missing],
        )
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome => :reward,
            action_columns=(; choice=:choicecol, rt=:rtcol) => :choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            impute_missing_observations=false,
            missing_action_values=(missing, -1),
        )
        #A missing observation that isn't imputed needs no environment, and every session has the marker in its type
        @test sessions_info.sessions[1].observations.reward ==
            [1.0, UnimputedMissingValue()]
        @test sessions_info.sessions[2].observations.reward isa
            Vector{Union{UnimputedMissingValue,Float64}}
        #A missing value in any column that an attribute is built from makes the whole value missing
        @test sessions_info.sessions[1].actions.choice ==
            [(; choice=1, rt=0.5), (; choice=0, rt=0.7)]
        @test sessions_info.sessions[2].actions.choice ==
            [ImputedMissingValue(), ImputedMissingValue()]
        @test isconcretetype(eltype(sessions_info.sessions))
        #A value which is equal to one of them without having its type is marked too, and every session then has the marker in its type
        with_code = MultipleSessionsInfo(
            SessionCounter(),
            (; id=["S1", "S2"], outcome=[1.0, -1.0], choice=[true, false]);
            observation_columns=:outcome => :reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            missing_observation_values=(-1,),
            impute_missing_observations=false,
        )
        @test with_code.sessions[2].observations.reward == [UnimputedMissingValue()] &&
            with_code.sessions[1].observations.reward isa
              Vector{Union{UnimputedMissingValue,Float64}}
        #Without values that stand for a missing one, missing stays in the data as a value
        kept = MultipleSessionsInfo(
            SessionCounter(),
            table;
            observation_columns=:outcome => :reward,
            action_columns=:choicecol,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            missing_observation_values=(),
        )
        @test isequal(kept.sessions[1].observations.reward, [1.0, missing])
    end

    @testset "The settings go to every session, and the sizes of the timesteps can come from a column" begin
        sessions_info = MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome => :reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=AgentStarts(),
            timestep_sizes=:dt,
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
            impute_missing_actions=false,
            check_parameter_domains=false,
        )
        @test all(
            session ->
                session.who_starts === AgentStarts() &&
                session.cognitive_save_history === SaveHistory(; states=:n_timesteps) &&
                session.check_parameter_domains === false,
            sessions_info.sessions,
        )
        #A copy of the sessions with other save histories keeps the setting
        @test all(
            session -> session.check_parameter_domains === false,
            MultipleSessionsInfo(sessions_info; cognitive_save_history=SaveHistory()).sessions,
        )
        @test sessions_info.sessions[1].timestep_sizes == [1.0, 0.5] &&
            sessions_info.sessions[2].timestep_sizes == [1.0, 2.0, 1.0]
        @test sessions_info.sessions[2].actions.choice ==
            [true, UnimputedMissingValue(), false]
        @test MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome => :reward,
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
            observation_columns=:outcome => :reward,
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
        nothing_missing = @test_logs MultipleSessionsInfo(
            SessionCounter(),
            COUNTER_TABLE;
            observation_columns=:outcome => :reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            environment=GivesReward(),
            impute_missing_actions=false,
        )
        @test !any(session -> session.includes_environment, nothing_missing.sessions)
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

    @testset "The values in a table are checked against the domains of the model's inputs and outputs, and an error names the row" begin
        table = (;
            id=["A", "A", "B", "B"], reward=[0.5, 0.2, 0.9, 0.1], choice=[1, 2, 2, 5]
        )
        @test_throws ArgumentError(
            "The action choice in row 4 of the table is 5, which isn't one of its possible values, (1, 2).",
        ) MultipleSessionsInfo(
            ChoosesWithinDomains(),
            table;
            observation_columns=:reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        within_domains = merge(table, (; choice=[1, 2, 2, 1]))
        @test MultipleSessionsInfo(
            ChoosesWithinDomains(),
            within_domains;
            observation_columns=:reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        ).sessions[2].actions == (; choice=[2, 1])
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
            observation_columns=:outcome => :reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
            timestep_sizes=:time,
        )
        @test_throws "Every column of the table must have the same length" MultipleSessionsInfo(
            SessionCounter(),
            (; id=["S1", "S1"], outcome=[1.0, 0.0, 1.0], choice=[true, false]);
            observation_columns=:outcome => :reward,
            action_columns=:choice,
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test_throws "Every column of the table must be a vector" MultipleSessionsInfo(
            SessionCounter(),
            (; id="S1", outcome=1.0, choice=true);
            observation_columns=:outcome => :reward,
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
        #The field of the table has the type of the table itself, which Reactant.jl needs to compile a function with the sessions in it
        @test fieldtype(typeof(sessions_info), :table) === typeof(COUNTER_TABLE)
        #The copies with other save histories and for prediction keep both
        for copy in (
            MultipleSessionsInfo(
                sessions_info; cognitive_save_history=SaveHistory(SaveEverything())
            ),
            decondition_sessions(sessions_info),
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
        @test fieldtype(typeof(prepared), :table) === Nothing
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
            (; choice=[true, ImputedMissingValue(), true]);
            who_starts=EnvironmentStarts(),
        )
        @test_throws "The sessions must all have the same type" MultipleSessionsInfo(
            [session, other_type], ["A", "B"]
        )
        @test_throws "There are 2 sessions and 1 data identifiers" MultipleSessionsInfo(
            [session, session], ["A"]
        )
        @test_throws "There must be at least one session." MultipleSessionsInfo(
            typeof(session)[], String[]
        )
        #Every session needs a name of its own, which its data identifier gives
        @test_throws "The data identifier \"A\" is given for more than one session." MultipleSessionsInfo(
            [session, session], ["A", "A"]
        )
        @test_throws "The data identifiers 1 and \"1\" both give the symbol name 1," MultipleSessionsInfo(
            [session, session], Any[1, "1"]
        )
    end
end

@testset "The parts of a full model, and how the models run" begin
    @test NoDownstreamModel() isa AbstractDownstreamModel
    @test isbitstype(NoDownstreamModel) && isbitstype(NoEnvironment)
    @test isbitstype(SerialSessions) && isbitstype(ParallelSessions)
end
