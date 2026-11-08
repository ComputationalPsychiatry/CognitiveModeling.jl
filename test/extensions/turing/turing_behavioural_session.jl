## Models and environments for the sessions ##
#A Rescorla-Wagner model with a softmax choice, which doesn't learn when there is no observation
struct SessionRescorlaWagner <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::SessionRescorlaWagner, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_model(model::SessionRescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    r = input.observation.reward
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + input.parameters.learning_rate * (r - V₋₁)
    return ModelOutput(
        (; choice=BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))),
        (; expected_reward=Vₜ),
    )
end
function CognitiveModeling.attribute_names(model::SessionRescorlaWagner)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        observations=:reward,
        actions=:choice,
    )
end
function CognitiveModeling.default_parameters(model::SessionRescorlaWagner)
    return (; inverse_temperature=2.0)
end

#Two continuous actions, so that missing ones can be imputed
struct TwoContinuousActions <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TwoContinuousActions, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_model(model::TwoContinuousActions, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput(
        (; choice=Normal(Vₜ, 1.0), bet=Normal(2Vₜ, 0.5)), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::TwoContinuousActions)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=:expected_reward,
        observations=:reward,
        actions=(:choice, :bet),
    )
end

#Acts only on every second timestep, and saves the timestep size
struct ActsEverySecondTimestep <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::ActsEverySecondTimestep, parameters)
    return (; n_timesteps=0, timestep_size=0)
end
function CognitiveModeling.apply_model(model::ActsEverySecondTimestep, input::ModelInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    choice = iseven(n_timesteps) ? Bernoulli(input.parameters.probability) : NoAction()
    return ModelOutput((; choice), (; n_timesteps, timestep_size=input.timestep_size))
end
function CognitiveModeling.attribute_names(model::ActsEverySecondTimestep)
    return ModelAttributes(;
        parameters=:probability,
        states=(:n_timesteps, :timestep_size),
        observations=(),
        actions=:choice,
    )
end

#Samples noise within each timestep in Turing
struct NoisySessionModel <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::NoisySessionModel, parameters)
    return (; expected_reward=0.5)
end
Turing.@model function CognitiveModeling.turing_apply_model(
    model::NoisySessionModel, input::ModelInput
)
    noise ~ Normal(0, input.parameters.noise_scale)
    Vₜ =
        input.previous_state.expected_reward +
        0.5 * (input.observation.reward - input.previous_state.expected_reward) +
        noise
    return ModelOutput((; choice=Normal(Vₜ, 1.0)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::NoisySessionModel)
    return ModelAttributes(;
        parameters=:noise_scale,
        states=:expected_reward,
        observations=:reward,
        actions=:choice,
    )
end

#Repeats its previous choice with a probability of 0.9, and chooses at random when there is none
struct RepeatsPreviousChoice <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RepeatsPreviousChoice, parameters) = (;)
function CognitiveModeling.apply_model(model::RepeatsPreviousChoice, input::ModelInput)
    previous = input.previous_action.choice
    return ModelOutput((; choice=Bernoulli(
        if previous isa AbstractNoValue
            0.5
        elseif previous
            0.9
        else
            0.1
        end,
    )))
end
function CognitiveModeling.attribute_names(model::RepeatsPreviousChoice)
    return ModelAttributes(; parameters=(), states=(), observations=(), actions=:choice)
end

#Keeps who started as its state, to check what the model is told
struct RecordsWhoStarts <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::RecordsWhoStarts, parameters)
    return (; who_starts=UnknownWhoStarts())
end
function CognitiveModeling.apply_model(model::RecordsWhoStarts, input::ModelInput)
    return ModelOutput((; choice=Deterministic(true)), (; who_starts=input.who_starts))
end
function CognitiveModeling.attribute_names(model::RecordsWhoStarts)
    return ModelAttributes(;
        parameters=(), states=:who_starts, observations=:reward, actions=:choice
    )
end

#Throws the error for invalid parameters when the learning rate is above 1
struct RejectsHighLearningRate <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::RejectsHighLearningRate, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.apply_model(model::RejectsHighLearningRate, input::ModelInput)
    input.parameters.learning_rate > 1 &&
        throw(InvalidParametersError("The learning rate must be at most 1."))
    Vₜ =
        input.previous_state.expected_reward +
        input.parameters.learning_rate *
        (input.observation.reward - input.previous_state.expected_reward)
    return ModelOutput((; choice=Normal(Vₜ, 1.0)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::RejectsHighLearningRate)
    return ModelAttributes(;
        parameters=:learning_rate,
        states=:expected_reward,
        observations=:reward,
        actions=:choice,
    )
end

#Starts its expected reward from a parameter, so that the start is fitted with the rest
struct StartsFromParameter <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::StartsFromParameter, parameters)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.apply_model(model::StartsFromParameter, input::ModelInput)
    Vₜ =
        input.previous_state.expected_reward +
        input.parameters.learning_rate *
        (input.observation.reward - input.previous_state.expected_reward)
    return ModelOutput(
        (; choice=BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))),
        (; expected_reward=Vₜ),
    )
end
function CognitiveModeling.attribute_names(model::StartsFromParameter)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature, :initial_value),
        states=:expected_reward,
        observations=:reward,
        actions=:choice,
    )
end
function CognitiveModeling.default_parameters(model::StartsFromParameter)
    return (; inverse_temperature=2.0, initial_value=0.5)
end

#Samples where its expected reward starts in Turing, around a parameter
struct SamplesItsStart <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::SamplesItsStart, parameters)
    return (; expected_reward=parameters.initial_value)
end
Turing.@model function CognitiveModeling.turing_initialize_state(
    model::SamplesItsStart, parameters
)
    start_noise ~ Normal(0, 0.1)
    return (; expected_reward=parameters.initial_value + start_noise)
end
function CognitiveModeling.apply_model(model::SamplesItsStart, input::ModelInput)
    Vₜ =
        input.previous_state.expected_reward +
        input.parameters.learning_rate *
        (input.observation.reward - input.previous_state.expected_reward)
    return ModelOutput((; choice=Normal(Vₜ, 1.0)), (; expected_reward=Vₜ))
end
function CognitiveModeling.attribute_names(model::SamplesItsStart)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value),
        states=:expected_reward,
        observations=:reward,
        actions=:choice,
    )
end

#Gives the previous choice as the reward, scaled, so that filled in observations can be checked
struct RewardsPreviousChoice <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RewardsPreviousChoice, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.apply_model(
    environment::RewardsPreviousChoice, input::EnvironmentInput
)
    reward = input.action.choice isa AbstractNoValue ? 0.5 : float(input.action.choice)
    return EnvironmentOutput(
        (; reward=Deterministic(reward * input.parameters.scale)),
        (; n_timesteps=input.previous_state.n_timesteps + 1),
    )
end
function CognitiveModeling.attribute_names(environment::RewardsPreviousChoice)
    return ModelAttributes(;
        parameters=:scale, states=:n_timesteps, observations=:reward, actions=:choice
    )
end
CognitiveModeling.default_parameters(environment::RewardsPreviousChoice) = (; scale=1.0)

#Gives no observations at all
struct GivesNothing <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::GivesNothing, parameters) = (;)
function CognitiveModeling.apply_model(environment::GivesNothing, input::EnvironmentInput)
    return EnvironmentOutput((;))
end
function CognitiveModeling.attribute_names(environment::GivesNothing)
    return ModelAttributes(; parameters=(), states=(), observations=(), actions=())
end

#Gives a distribution to sample its reward from
struct NoisyRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::NoisyRewards, parameters) = (;)
function CognitiveModeling.apply_model(environment::NoisyRewards, input::EnvironmentInput)
    return EnvironmentOutput((; reward=Normal(0.5, 1.0)))
end
function CognitiveModeling.attribute_names(environment::NoisyRewards)
    return ModelAttributes(;
        parameters=(), states=(), observations=:reward, actions=:choice
    )
end

#Samples a drift of its state within each timestep in Turing, and gives the state as a reward which isn't random
struct DriftingRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::DriftingRewards, parameters) = (; level=0.5)
Turing.@model function CognitiveModeling.turing_apply_model(
    environment::DriftingRewards, input::EnvironmentInput
)
    drift ~ Normal(0, 1)
    level = input.previous_state.level + drift
    return EnvironmentOutput((; reward=Deterministic(level)), (; level))
end
function CognitiveModeling.attribute_names(environment::DriftingRewards)
    return ModelAttributes(;
        parameters=(), states=:level, observations=:reward, actions=:choice
    )
end

#Gives no reward before the first choice, and the previous choice afterwards
struct RewardsAfterFirstChoice <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsAfterFirstChoice, parameters) = (;)
function CognitiveModeling.apply_model(
    environment::RewardsAfterFirstChoice, input::EnvironmentInput
)
    input.action.choice isa AbstractNoValue &&
        return EnvironmentOutput((; reward=NoObservation()))
    return EnvironmentOutput((; reward=Deterministic(float(input.action.choice))))
end
function CognitiveModeling.attribute_names(environment::RewardsAfterFirstChoice)
    return ModelAttributes(;
        parameters=(), states=(), observations=:reward, actions=:choice
    )
end

#The log density of the recorded choices of a Rescorla-Wagner session, and the saved states, by hand
function session_by_hand(rewards, choices, learning_rate, inverse_temperature, who_starts)
    V = 0.5
    log_density = 0.0
    states = Float64[]
    for t in eachindex(rewards)
        #When the agent starts, it acts on the observation of the timestep before, and on nothing on the first timestep
        if who_starts isa AgentStarts && t > 1
            V = V + learning_rate * (rewards[t - 1] - V)
        elseif who_starts isa EnvironmentStarts
            V = V + learning_rate * (rewards[t] - V)
        end
        choices[t] isa MissingValue || (
            log_density += logpdf(
                BernoulliLogit(inverse_temperature * (V - 0.5)), choices[t]
            )
        )
        push!(states, V)
    end
    return log_density, states
end

const SESSION_REWARDS = (; reward=[1.0, 0.0, 1.0, 1.0, 0.0])
const SESSION_CHOICES = (; choice=[true, false, true, true, false])
const SESSION_PARAMETERS = (; learning_rate=0.3, inverse_temperature=2.0)

#Shortens the calls below: prepares the session and makes the Turing model of it, with the environment parameters when there are any
function session_for_test(
    model=SessionRescorlaWagner(),
    parameters=SESSION_PARAMETERS,
    observations=SESSION_REWARDS,
    actions=SESSION_CHOICES;
    who_starts=EnvironmentStarts(),
    environment_parameters=nothing,
    catch_invalid_parameters=false,
    keywords...,
)
    session_info = SessionInfo(model, observations, actions; who_starts, keywords...)
    session = if environment_parameters === nothing
        turing_behavioural_session(session_info, parameters)
    else
        turing_behavioural_session(session_info, parameters, environment_parameters)
    end
    return catch_invalid_parameters ? turing_catch_invalid_parameters(session) : session
end

@testset "turing_behavioural_session" begin
    @testset "The session starts with BeforeFirstTimestep for the previous action, and for the observation when the agent starts" begin
        #RepeatsPreviousChoice chooses at random when there is no previous action, and repeats its choice with 0.9 otherwise
        @test logjoint(
            session_for_test(
                RepeatsPreviousChoice(), (;), (;), (; choice=[true, true, false])
            ),
            (;),
        ) ≈ log(0.5) + log(0.9) + log(0.1)
        #A previous action of its own replaces the marker
        @test logjoint(
            session_for_test(
                RepeatsPreviousChoice(),
                (;),
                (;),
                (; choice=[true, true, false]);
                previous_action=(; choice=true),
            ),
            (;),
        ) ≈ 2 * log(0.9) + log(0.1)
    end

    @testset "The log density is the sum over the recorded actions, with the agent or the environment starting" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            model = session_for_test(;
                who_starts, save_history=SaveHistory(SaveEverything())
            )
            expected, states = session_by_hand(
                SESSION_REWARDS.reward, SESSION_CHOICES.choice, 0.3, 2.0, who_starts
            )
            @test logjoint(model, (;)) ≈ expected
            @test loglikelihood(model, (;)) ≈ expected
            #There are no latents, and what is returned is what happened, after timestep 0 with where the session started
            @test isempty(keys(rand(Xoshiro(1), model)))
            history = returned(model, (;))
            @test [state.expected_reward for state in history.states] ≈ [0.5; states]
            @test history.actions == [
                (; choice=BeforeFirstTimestep()),
                ((; choice) for choice in SESSION_CHOICES.choice)...,
            ]
            @test history.observations == [
                (; reward=BeforeFirstTimestep()),
                ((; reward) for reward in SESSION_REWARDS.reward)...,
            ]
        end
    end

    @testset "Default parameters are filled in, and the names and the data are checked" begin
        @test logjoint(
            session_for_test(SessionRescorlaWagner(), (; learning_rate=0.3)), (;)
        ) ≈ logjoint(session_for_test(), (;))
        #The parameters are only known when the model is evaluated, so they are checked then
        @test_throws "The parameters are named (:inverse_temperature, :learning_rate, :noise), but attribute_names lists the parameters of a model of type SessionRescorlaWagner" logjoint(
            session_for_test(SessionRescorlaWagner(), (; learning_rate=0.3, noise=1.0)), (;)
        )
        #The data is checked when the session is prepared
        @test_throws "The observations in the data are named (:cue,), but attribute_names lists the observations of a cognitive model of type SessionRescorlaWagner as (:reward,). The data must have every observation the cognitive model lists." session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; cue=[1.0, 0.0]),
            (; choice=[true, false]),
        )
        @test_throws "The actions in the data are named (:bet,), but attribute_names lists the actions of a cognitive model of type SessionRescorlaWagner as (:choice,)." session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            SESSION_REWARDS,
            (; bet=SESSION_CHOICES.choice),
        )
        @test_throws "Every observation and action in the data must have one value for each timestep, but their lengths are (reward = 2,) and (choice = 5,)." session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=[1.0, 0.0]),
            SESSION_CHOICES,
        )
        @test_throws "must each be a NamedTuple with one vector for each name" session_for_test(
            SessionRescorlaWagner(), SESSION_PARAMETERS, [(; reward=1.0)], [(; choice=true)]
        )
        @test_throws "timestep_sizes must have one size for each timestep, but it has 2 sizes and the data has 5 timesteps." session_for_test(;
            timestep_sizes=[1.0, 2.0]
        )
        @test_throws "The previous action is (bet = true,), but it must be a NamedTuple with the actions that attribute_names lists for a model of type SessionRescorlaWagner" logjoint(
            session_for_test(; previous_action=(; bet=true)), (;)
        )
        #The data can have more observations than the model lists
        @test logjoint(
            session_for_test(
                SessionRescorlaWagner(),
                SESSION_PARAMETERS,
                (; reward=SESSION_REWARDS.reward, cue=1:5),
            ),
            (;),
        ) ≈ logjoint(session_for_test(), (;))
        #Environment parameters can only be given when there is an environment
        @test_throws "Environment parameters were given, (scale = 1.0,), but the session has no environment." logjoint(
            session_for_test(; environment_parameters=(; scale=1.0)), (;)
        )
    end

    @testset "The model is told who starts" begin
        #The data has no recorded actions, since a Deterministic action can't be scored yet
        for who_starts in (EnvironmentStarts(), AgentStarts())
            model = session_for_test(
                RecordsWhoStarts(),
                (;),
                (; reward=[1.0, 0.0]),
                (; choice=[NoAction(), NoAction()]);
                who_starts,
                save_history=SaveHistory(; states=:who_starts),
            )
            @test returned(model, (;)).states[end] === (; who_starts)
        end
    end

    @testset "The states where the session starts come from initialize_state with the parameters, so a start which is a parameter is fitted" begin
        #StartsFromParameter has initial_value as a parameter with a default of 0.5, which SessionRescorlaWagner starts from
        @test logjoint(
            session_for_test(StartsFromParameter(), (; learning_rate=0.3)), (;)
        ) ≈ logjoint(session_for_test(), (;))
        model = session_for_test(
            StartsFromParameter(),
            (; learning_rate=0.3, initial_value=0.2);
            save_history=SaveHistory(; states=:expected_reward),
        )
        _, states = session_by_hand(
            SESSION_REWARDS.reward, SESSION_CHOICES.choice, 0.3, 2.0, EnvironmentStarts()
        )
        @test returned(model, (;)).states[1] == (; expected_reward=0.2)
        @test returned(model, (;)).states[2].expected_reward ≈ 0.2 + 0.3 * (1.0 - 0.2)
        #A model which samples its start in turing_initialize_state has the variable under initial_cognitive_state
        model = session_for_test(
            SamplesItsStart(),
            (; learning_rate=0.3, initial_value=0.5),
            (; reward=[1.0, 0.0]),
            (; choice=[0.2, 0.4]);
            save_history=SaveHistory(; states=:expected_reward),
        )
        @test collect(keys(rand(Xoshiro(1), model))) ==
            [@varname(initial_cognitive_state.start_noise)]
        given = (; initial_cognitive_state=(; start_noise=0.1))
        @test returned(model, given).states[1] == (; expected_reward=0.6)
        V₁ = 0.6 + 0.3 * (1.0 - 0.6)
        V₂ = V₁ + 0.3 * (0.0 - V₁)
        @test logjoint(model, given) ≈
            logpdf(Normal(0, 0.1), 0.1) +
              logpdf(Normal(V₁, 1.0), 0.2) +
              logpdf(Normal(V₂, 1.0), 0.4)
    end

    @testset "A missing in the data becomes MissingValue, or the marker for nothing happening, and a column keeps one type" begin
        session_info = SessionInfo(
            SessionRescorlaWagner(),
            (; reward=[1.0, missing, 0.0]),
            (; choice=[true, missing, false]);
            who_starts=EnvironmentStarts(),
            missing_observations=NoObservation(),
        )
        @test session_info.observations.reward == [1.0, NoObservation(), 0.0] &&
            session_info.observations.reward isa Vector{Union{NoObservation,Float64}}
        @test session_info.actions.choice == [true, MissingValue(), false] &&
            session_info.actions.choice isa Vector{Union{MissingValue,Bool}}
        @test session_info.recorded == (; choice=[true, false])
        #The model doesn't learn from the observation that wasn't made, and the missing choice is imputed
        model = turing_behavioural_session(session_info, SESSION_PARAMETERS)
        @test collect(keys(@test_logs match_mode = :any rand(Xoshiro(1), model))) ==
            [@varname(timestep[2].choice.imputed)]
        #A column whose element type allows missing gets the marker in its type even without a missing, so that sessions of one table have one type
        @test SessionInfo(
            SessionRescorlaWagner(),
            SESSION_REWARDS,
            (; choice=Vector{Union{Missing,Bool}}(SESSION_CHOICES.choice));
            who_starts=EnvironmentStarts(),
        ).actions.choice isa Vector{Union{MissingValue,Bool}}
        @test SessionInfo(
            SessionRescorlaWagner(),
            SESSION_REWARDS,
            (; choice=Vector{Union{Missing,Bool}}(SESSION_CHOICES.choice));
            who_starts=EnvironmentStarts(),
            missing_actions=NoAction(),
        ).actions.choice isa Vector{Union{NoAction,Bool}}
        #Without an environment, a missing observation can't be MissingValue
        @test_throws "The observations have entries that are MissingValue(), but no environment was given to fill them in." SessionInfo(
            SessionRescorlaWagner(),
            (; reward=[1.0, missing]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
        )
        @test SessionInfo(
            SessionRescorlaWagner(),
            (; reward=[1.0, missing]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            missing_observations=NoObservation(),
        ).observations.reward[2] === NoObservation()
    end

    @testset "save_history selects what is returned, and nothing by default" begin
        #Timestep 0 and the five timesteps
        history = returned(session_for_test(), (;))
        @test history.observations == fill((;), 6) &&
            history.actions == fill((;), 6) &&
            history.states == fill((;), 6)
        history = returned(
            session_for_test(;
                save_history=SaveHistory(; states=:expected_reward, actions=:choice)
            ),
            (;),
        )
        @test history.observations == fill((;), 6)
        @test history.actions == [
            (; choice=BeforeFirstTimestep()),
            ((; choice) for choice in SESSION_CHOICES.choice)...,
        ]
        @test history.states[1:2] == [(; expected_reward=0.5), (; expected_reward=0.65)]
        #Without timestep 0, element t is timestep t
        history = returned(
            session_for_test(;
                save_history=SaveHistory(;
                    states=:expected_reward, actions=:choice, save_initial_values=false
                ),
            ),
            (;),
        )
        @test history.actions == [(; choice) for choice in SESSION_CHOICES.choice]
        @test history.states[1] == (; expected_reward=0.65) && length(history.states) == 5
        @test_throws FieldError returned(
            session_for_test(
                TwoContinuousActions(),
                (; learning_rate=0.3),
                SESSION_REWARDS,
                (; choice=ones(5), bet=ones(5));
                save_history=SaveHistory(; observations=:cue),
            ),
            (;),
        )
    end

    @testset "The sizes of the timesteps are returned when anything is saved, and are empty otherwise" begin
        #They start with BeforeFirstTimestep for timestep 0
        @test isequal(
            returned(
                session_for_test(; save_history=SaveHistory(; states=:expected_reward)), (;)
            ).timestep_sizes,
            [BeforeFirstTimestep(); fill(1, 5)],
        )
        @test isequal(
            returned(
                session_for_test(;
                    timestep_sizes=0.5, save_history=SaveHistory(; actions=SaveEverything())
                ),
                (;),
            ).timestep_sizes,
            [BeforeFirstTimestep(); fill(0.5, 5)],
        )
        sizes = [0.5, 1.0, 1.5, 2.0, 2.5]
        @test isequal(
            returned(
                session_for_test(;
                    timestep_sizes=sizes, save_history=SaveHistory(SaveEverything())
                ),
                (;),
            ).timestep_sizes,
            [BeforeFirstTimestep(); sizes],
        )
        @test returned(
            session_for_test(;
                timestep_sizes=sizes,
                save_history=SaveHistory(SaveEverything(); save_initial_values=false),
            ),
            (;),
        ).timestep_sizes == sizes
        #Nothing is saved by default
        @test returned(session_for_test(), (;)).timestep_sizes == Int[]
        @test returned(session_for_test(; timestep_sizes=sizes), (;)).timestep_sizes ==
            Float64[]
        @test keys(returned(session_for_test(), (;))) ==
            (:observations, :actions, :action_distributions, :states, :timestep_sizes)
        #The action distributions are saved when selected, with BeforeFirstTimestep at timestep 0
        distributions = returned(
            session_for_test(; save_history=SaveHistory(; action_distributions=:choice)),
            (;),
        ).action_distributions
        @test distributions[1] === (; choice=BeforeFirstTimestep()) &&
            length(distributions) == 6
        @test distributions[2].choice ≈ BernoulliLogit(2.0 * (0.65 - 0.5))
        @test returned(session_for_test(), (;)).action_distributions == fill((;), 6)
    end

    @testset "Missing actions are latents named after their timestep, with the model's distributions" begin
        actions = (;
            choice=[0.2, MissingValue(), 0.4, MissingValue()],
            bet=[1.0, 1.5, MissingValue(), 0.8],
        )
        model = session_for_test(
            TwoContinuousActions(),
            (; learning_rate=0.3),
            (; reward=[1.0, 0.0, 1.0, 1.0]),
            actions;
            save_history=SaveHistory(; actions=SaveEverything()),
        )
        #One latent for each missing action
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([
            @varname(timestep[2].choice.imputed),
            @varname(timestep[3].bet.imputed),
            @varname(timestep[4].choice.imputed)
        ])
        imputed = [0.1, -0.2, 0.3]
        given = Dict(
            @varname(timestep[2].choice.imputed) => imputed[1],
            @varname(timestep[3].bet.imputed) => imputed[2],
            @varname(timestep[4].choice.imputed) => imputed[3],
        )
        V = 0.5
        expected_prior = 0.0
        expected_likelihood = 0.0
        filled = []
        for t in 1:4
            V = V + 0.3 * ([1.0, 0.0, 1.0, 1.0][t] - V)
            choice = if actions.choice[t] isa MissingValue
                (t == 2 ? imputed[1] : imputed[3])
            else
                actions.choice[t]
            end
            bet = actions.bet[t] isa MissingValue ? imputed[2] : actions.bet[t]
            #An imputed action is a latent, so its density is prior, and a recorded one is likelihood
            if actions.choice[t] isa MissingValue
                expected_prior += logpdf(Normal(V, 1.0), choice)
            else
                expected_likelihood += logpdf(Normal(V, 1.0), choice)
            end
            if actions.bet[t] isa MissingValue
                expected_prior += logpdf(Normal(2V, 0.5), bet)
            else
                expected_likelihood += logpdf(Normal(2V, 0.5), bet)
            end
            push!(filled, (; choice, bet))
        end
        @test (@test_logs match_mode = :any logjoint(model, given)) ≈
            expected_prior + expected_likelihood
        @test (@test_logs match_mode = :any logprior(model, given)) ≈ expected_prior
        @test (@test_logs match_mode = :any loglikelihood(model, given)) ≈
            expected_likelihood
        #The returned actions have the imputed values, after timestep 0
        @test (@test_logs match_mode = :any returned(model, given)).actions ==
            [(; choice=BeforeFirstTimestep(), bet=BeforeFirstTimestep()); filled]
        #A single missing action keeps the types of the rest known
        single = session_for_test(
            TwoContinuousActions(),
            (; learning_rate=0.3),
            (; reward=[1.0, 0.0]),
            (; choice=[0.2, MissingValue()], bet=[1.0, 1.5]),
        )
        @test collect(keys(@test_logs match_mode = :any rand(Xoshiro(1), single))) ==
            [@varname(timestep[2].choice.imputed)]
    end

    @testset "Missing actions that aren't imputed are passed on as NoAction, with nothing sampled" begin
        rewards = (; reward=[1.0, 0.0, 1.0, 1.0])
        actions = (;
            choice=[0.2, MissingValue(), 0.4, MissingValue()],
            bet=[1.0, 1.5, MissingValue(), 0.8],
        )
        model = session_for_test(
            TwoContinuousActions(),
            (; learning_rate=0.3),
            rewards,
            actions;
            impute_missing_actions=false,
            save_history=SaveHistory(; actions=SaveEverything()),
        )
        @test isempty(keys(rand(Xoshiro(1), model)))
        #Only the recorded actions are scored
        V = 0.5
        expected = 0.0
        for t in 1:4
            V = V + 0.3 * (rewards.reward[t] - V)
            actions.choice[t] isa MissingValue ||
                (expected += logpdf(Normal(V, 1.0), actions.choice[t]))
            actions.bet[t] isa MissingValue ||
                (expected += logpdf(Normal(2V, 0.5), actions.bet[t]))
        end
        @test logjoint(model, (;)) ≈ expected
        @test logprior(model, (;)) == 0
        #The same as data which has NoAction there
        with_no_action = (;
            choice=[0.2, NoAction(), 0.4, NoAction()], bet=[1.0, 1.5, NoAction(), 0.8]
        )
        @test logjoint(model, (;)) ≈ logjoint(
            session_for_test(
                TwoContinuousActions(), (; learning_rate=0.3), rewards, with_no_action
            ),
            (;),
        )
        #The returned actions have NoAction for them
        @test returned(model, (;)).actions[2:end] == [
            (; choice=0.2, bet=1.0),
            (; choice=NoAction(), bet=1.5),
            (; choice=0.4, bet=NoAction()),
            (; choice=NoAction(), bet=0.8),
        ]

        #The model receives NoAction as its previous action after a missing one, so here it chooses at random again
        choices = (; choice=[true, MissingValue(), true])
        @test logjoint(
            session_for_test(
                RepeatsPreviousChoice(), (;), (;), choices; impute_missing_actions=false
            ),
            (;),
        ) ≈ 2 * log(0.5)
        #When it is imputed, the model receives the imputed action
        imputed = Dict(@varname(timestep[2].choice.imputed) => true)
        @test (@test_logs match_mode = :any logjoint(
            session_for_test(RepeatsPreviousChoice(), (;), (;), choices), imputed
        )) ≈ log(0.5) + 2 * log(0.9)

        #The environment receives NoAction too: RewardsPreviousChoice gives 0.5 for it, and the previous choice otherwise
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=fill(MissingValue(), 3)),
            choices;
            impute_missing_actions=false,
            environment=RewardsPreviousChoice(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        @test returned(model, (;)).observations[2:end] ==
            [(; reward=0.5), (; reward=1.0), (; reward=0.5)]
    end

    @testset "A missing observation is filled in by the environment, which runs on every timestep" begin
        #RewardsPreviousChoice gives 0.5 before the first choice and then the previous choice
        full = (; reward=[0.5, 1.0, 0.0, 1.0, 1.0])
        with_missing = (; reward=[0.5, MissingValue(), 0.0, MissingValue(), 1.0])
        reference = session_for_test(
            SessionRescorlaWagner(), SESSION_PARAMETERS, full, SESSION_CHOICES
        )
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            with_missing,
            SESSION_CHOICES;
            environment=RewardsPreviousChoice(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        @test logjoint(model, (;)) ≈ logjoint(reference, (;))
        @test returned(model, (;)).observations[2:end] ==
            [(; reward) for reward in full.reward]
        #The environment's defaults are filled in and its names checked when the model is evaluated
        @test_throws "attribute_names lists the parameters of an environment model of type RewardsPreviousChoice" logjoint(
            session_for_test(
                SessionRescorlaWagner(),
                SESSION_PARAMETERS,
                with_missing,
                SESSION_CHOICES;
                environment=RewardsPreviousChoice(),
                environment_parameters=(; size=1.0),
            ),
            (;),
        )
        #Its parameters scale what it gives
        scaled = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            with_missing,
            SESSION_CHOICES;
            environment=RewardsPreviousChoice(),
            environment_parameters=(; scale=2.0),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        @test returned(scaled, (;)).observations[2:end] == [
            (; reward=0.5), (; reward=2.0), (; reward=0.0), (; reward=2.0), (; reward=1.0)
        ]
        @test_throws "A cognitive model of type SessionRescorlaWagner lists the observations (:reward,), but an environment model of type GivesNothing gives (). Without a map, the two must list the same observations." session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            SESSION_REWARDS,
            SESSION_CHOICES;
            environment=GivesNothing(),
        )
        #When the agent starts, the environment responds to the action of the same timestep
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=fill(MissingValue(), 5)),
            SESSION_CHOICES;
            who_starts=AgentStarts(),
            environment=RewardsPreviousChoice(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        @test returned(model, (;)).observations[2:end] ==
            [(; reward=float(choice)) for choice in SESSION_CHOICES.choice]
        #Without an environment, the data can't have missing observations
        @test_throws "The observations have entries that are MissingValue(), but no environment was given to fill them in." session_for_test(
            SessionRescorlaWagner(), SESSION_PARAMETERS, with_missing, SESSION_CHOICES
        )
        #An observation that the environment doesn't give stays NoObservation
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=fill(MissingValue(), 3)),
            (; choice=[true, false, true]);
            environment=RewardsAfterFirstChoice(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        @test returned(model, (;)).observations[2:end] ==
            [(; reward=NoObservation()), (; reward=1.0), (; reward=0.0)]
    end

    @testset "A missing observation is a latent when the environment gives a distribution for it" begin
        #Each one is named after its timestep and the observation, under the environment, and there is none where the data has the observation
        choices = (; choice=[true, false, true])
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=[MissingValue(), 1.0, MissingValue()]),
            choices;
            environment=NoisyRewards(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([
            @varname(timestep[1].environment.reward.imputed),
            @varname(timestep[3].environment.reward.imputed)
        ])
        #The model observes their values
        rewards = Dict(
            @varname(timestep[1].environment.reward.imputed) => 0.7,
            @varname(timestep[3].environment.reward.imputed) => -0.4,
        )
        @test (@test_logs match_mode = :any returned(model, rewards)).observations[2:end] ==
            [(; reward=0.7), (; reward=1.0), (; reward=-0.4)]
        #They count in the log density with the environment's distribution, beside the choices given those rewards
        recorded = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=[0.7, 1.0, -0.4]),
            choices,
        )
        @test (@test_logs match_mode = :any logjoint(model, rewards)) ≈
            logjoint(recorded, (;)) +
              logpdf(Normal(0.5, 1.0), 0.7) +
              logpdf(Normal(0.5, 1.0), -0.4)
        #With the data complete, the environment doesn't run, with no warning, so the session is the same as one without it
        model = @test_logs session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=[0.7, 1.0, -0.4]),
            choices;
            environment=NoisyRewards(),
        )
        @test isempty(keys(rand(Xoshiro(1), model)))
        @test logjoint(model, (;)) ≈ logjoint(recorded, (;))
    end

    @testset "An environment which samples within a timestep has its variables under the timestep" begin
        model = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=fill(MissingValue(), 2)),
            (; choice=[true, false]);
            environment=DriftingRewards(),
            save_history=SaveHistory(; observations=SaveEverything()),
        )
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([
            @varname(timestep[1].environment.drift), @varname(timestep[2].environment.drift)
        ])
        #What it gives from them is what the model observes
        drifts = Dict(
            @varname(timestep[1].environment.drift) => 0.25,
            @varname(timestep[2].environment.drift) => -0.5,
        )
        @test (@test_logs match_mode = :any returned(model, drifts)).observations[2:end] ==
            [(; reward=0.75), (; reward=0.25)]
        #With every reward in the data, it doesn't run, so there are no drifts, and the session is the same as one without it
        complete = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            environment=DriftingRewards(),
        )
        @test isempty(keys(rand(Xoshiro(1), complete)))
        @test logjoint(complete, (;)) ≈ logjoint(
            session_for_test(
                SessionRescorlaWagner(),
                SESSION_PARAMETERS,
                (; reward=[1.0, 0.0]),
                (; choice=[true, false]),
            ),
            (;),
        )
    end

    @testset "A model which samples within a timestep has a variable for each timestep" begin
        model = session_for_test(
            NoisySessionModel(),
            (; noise_scale=1.0),
            (; reward=[1.0, 0.0, 1.0]),
            (; choice=[0.2, 0.4, 0.6]);
            save_history=SaveHistory(; states=:expected_reward),
        )
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([
            @varname(timestep[1].noise),
            @varname(timestep[2].noise),
            @varname(timestep[3].noise)
        ])
        noises = Dict(@varname(timestep[t].noise) => 0.1 * t for t in 1:3)
        V = 0.5
        expected = 0.0
        for t in 1:3
            V = V + 0.5 * ([1.0, 0.0, 1.0][t] - V) + 0.1 * t
            expected +=
                logpdf(Normal(0, 1), 0.1 * t) + logpdf(Normal(V, 1.0), [0.2, 0.4, 0.6][t])
        end
        @test (@test_logs match_mode = :any logjoint(model, noises)) ≈ expected
        @test (@test_logs match_mode = :any returned(model, noises)).states[end].expected_reward ≈
            V
    end

    @testset "NoAction in the data or from the model isn't scored, and both get the timestep sizes" begin
        #The model acts on even timesteps, and the data has NoAction on the odd ones
        actions = (; choice=[NoAction(), true, NoAction(), false])
        model = session_for_test(
            ActsEverySecondTimestep(),
            (; probability=0.7),
            (;),
            actions;
            timestep_sizes=[0.5, 1.0, 1.5, 2.0],
            save_history=SaveHistory(SaveEverything()),
        )
        @test logjoint(model, (;)) ≈
            logpdf(Bernoulli(0.7), true) + logpdf(Bernoulli(0.7), false)
        history = returned(model, (;))
        @test history.actions[2:end] == [
            (; choice=NoAction()), (; choice=true), (; choice=NoAction()), (; choice=false)
        ]
        @test [state.timestep_size for state in history.states[2:end]] == [0.5, 1.0, 1.5, 2.0]
        @test history.observations == fill((;), 5)
        #The data can also have NoAction where the model acts, which isn't scored either
        @test logjoint(
            session_for_test(
                ActsEverySecondTimestep(),
                (; probability=0.7),
                (;),
                (; choice=[NoAction(), NoAction(), NoAction(), false]),
            ),
            (;),
        ) ≈ logpdf(Bernoulli(0.7), false)
        #An action that is never recorded in a session scores nothing, also when the model never took it
        @test logjoint(
            session_for_test(
                ActsEverySecondTimestep(),
                (; probability=0.7),
                (;),
                (; choice=[NoAction(), NoAction(), NoAction(), NoAction()]),
            ),
            (;),
        ) == 0
        @test logjoint(
            session_for_test(
                ActsEverySecondTimestep(), (; probability=0.7), (;), (; choice=[NoAction()])
            ),
            (;),
        ) == 0
        #A recorded action where the model took none is an error
        @test_throws "The cognitive model of type ActsEverySecondTimestep took no action choice on a timestep where the data has true for it." logjoint(
            session_for_test(
                ActsEverySecondTimestep(),
                (; probability=0.7),
                (;),
                (; choice=[true, true, false, false]),
            ),
            (;),
        )
        #The sizes of the timesteps are returned with what is saved
        @test isequal(history.timestep_sizes, [BeforeFirstTimestep(), 0.5, 1.0, 1.5, 2.0])
        #With a single size, every timestep gets it
        @test [
            state.timestep_size for state in returned(
                session_for_test(
                    ActsEverySecondTimestep(),
                    (; probability=0.7),
                    (;),
                    actions;
                    timestep_sizes=0.25,
                    save_history=SaveHistory(; states=:timestep_size),
                ),
                (;),
            ).states[2:end]
        ] == fill(0.25, 4)
    end

    @testset "An InvalidParametersError stops the evaluation, unless turing_catch_invalid_parameters makes it give a log density of -Inf" begin
        actions = (; choice=[0.2, 0.4, 0.6, 0.8, 1.0])
        @test_throws "The learning rate must be at most 1." logjoint(
            session_for_test(
                RejectsHighLearningRate(), (; learning_rate=1.5), SESSION_REWARDS, actions
            ),
            (;),
        )
        caught = session_for_test(
            RejectsHighLearningRate(),
            (; learning_rate=1.5),
            SESSION_REWARDS,
            actions;
            catch_invalid_parameters=true,
        )
        @test logjoint(caught, (;)) == -Inf
        @test returned(caught, (;)) === nothing
        #Valid parameters are unaffected
        valid = session_for_test(
            RejectsHighLearningRate(),
            (; learning_rate=0.3),
            SESSION_REWARDS,
            actions;
            catch_invalid_parameters=true,
            save_history=SaveHistory(; states=:expected_reward),
        )
        @test logjoint(valid, (;)) ≈ logjoint(
            session_for_test(
                RejectsHighLearningRate(), (; learning_rate=0.3), SESSION_REWARDS, actions
            ),
            (;),
        )
        @test returned(valid, (;)).states[2] == (; expected_reward=0.65)
        #Other errors still get through
        @test_throws "took no action" logjoint(
            session_for_test(
                ActsEverySecondTimestep(),
                (; probability=0.7),
                (;),
                (; choice=[true, true]);
                catch_invalid_parameters=true,
            ),
            (;),
        )
    end

    @testset "Evaluating a session only allocates the distributions it keeps, so each timestep is compiled with its types known" begin
        session_with(n_timesteps) = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=repeat([1.0, 0.0], n_timesteps ÷ 2)),
            (; choice=repeat([true, false], n_timesteps ÷ 2)),
        )
        short_session = session_with(10_000)
        long_session = session_with(20_000)
        logjoint(short_session, (;))
        logjoint(long_session, (;))
        #Doubling the timesteps only adds the vector of the distributions of the recorded actions, about 23 bytes per timestep.
        #When the names of the recorded actions reached the timesteps as a plain tuple, it was about 1500 bytes, and each timestep was a hundred times slower
        extra_bytes =
            @allocated(logjoint(long_session, (;))) -
            @allocated(logjoint(short_session, (;)))
        @test extra_bytes <= 40 * 10_000
        #The same holds for a column which can hold a MissingValue, since the rows and the imputed values have the type of their column
        union_session_with(n_timesteps) = session_for_test(
            SessionRescorlaWagner(),
            SESSION_PARAMETERS,
            (; reward=repeat([1.0, 0.0], n_timesteps ÷ 2)),
            (; choice=Vector{Union{Missing,Bool}}(repeat([true, false], n_timesteps ÷ 2))),
        )
        short_union = union_session_with(10_000)
        long_union = union_session_with(20_000)
        logjoint(short_union, (;))
        logjoint(long_union, (;))
        @test @allocated(logjoint(long_union, (;))) -
              @allocated(logjoint(short_union, (;))) <= 40 * 10_000
    end

    @testset "It can be sampled as a submodel, and returned gives the states for each sample, also an initial state which is fitted" begin
        session_info = SessionInfo(
            StartsFromParameter(),
            SESSION_REWARDS,
            SESSION_CHOICES;
            who_starts=EnvironmentStarts(),
            save_history=SaveHistory(; states=:expected_reward),
        )
        Turing.@model function fit_one_session(session_info)
            learning_rate ~ Beta(2, 2)
            inverse_temperature ~ LogNormal(0, 0.5)
            initial_value ~ Beta(2, 2)
            session ~ to_submodel(
                prefix(
                    turing_behavioural_session(
                        session_info, (; learning_rate, inverse_temperature, initial_value)
                    ),
                    @varname(session)
                ),
                false,
            )
            return session
        end
        model = fit_one_session(session_info)
        chain = sample(Xoshiro(1), model, NUTS(), 100; progress=false)
        @test size(chain, 1) == 100
        @test Set(Turing.FlexiChains.parameters(chain)) == Set([
            @varname(learning_rate), @varname(inverse_temperature), @varname(initial_value)
        ])
        trajectories = returned(model, chain)
        @test length(trajectories) == 100
        #Timestep 0 has the initial state of each sample
        @test length(trajectories[1].states) == 6
        initial_value = chain[@varname(initial_value)][1]
        @test trajectories[1].states[1] == (; expected_reward=initial_value)
        @test trajectories[1].states[2].expected_reward ≈
            initial_value + chain[@varname(learning_rate)][1] * (1.0 - initial_value)
    end
end
