## Models and environments for the sessions ##
#A Rescorla-Wagner model with a softmax choice, which doesn't learn when there is no observation
struct SessionRescorlaWagner <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::SessionRescorlaWagner, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::SessionRescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    r = input.observation.reward
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + input.parameters.learning_rate * (r - V₋₁)
    return ModelOutput((; choice = BernoulliLogit(input.parameters.inverse_temperature * (Vₜ - 0.5))), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::SessionRescorlaWagner) = ModelAttributes(; parameters = (:learning_rate, :inverse_temperature), states = :expected_reward, observations = :reward, actions = :choice)
CognitiveModeling.default_parameters(model::SessionRescorlaWagner) = (; inverse_temperature = 2.0)

#Two continuous actions, so that missing ones can be imputed
struct TwoContinuousActions <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TwoContinuousActions, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::TwoContinuousActions, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Normal(Vₜ, 1.0), bet = Normal(2Vₜ, 0.5)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::TwoContinuousActions) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = (:choice, :bet))

#Acts only on every second timestep, and saves the timestep size
struct ActsEverySecondTimestep <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ActsEverySecondTimestep, parameters) = (; n_timesteps = 0, timestep_size = 0)
function CognitiveModeling.apply_model(model::ActsEverySecondTimestep, input::ModelInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    choice = iseven(n_timesteps) ? Bernoulli(input.parameters.probability) : NoAction()
    return ModelOutput((; choice), (; n_timesteps, timestep_size = input.timestep_size))
end
CognitiveModeling.attribute_names(model::ActsEverySecondTimestep) = ModelAttributes(; parameters = :probability, states = (:n_timesteps, :timestep_size), observations = (), actions = :choice)

#Samples noise within each timestep in Turing
struct NoisySessionModel <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::NoisySessionModel, parameters) = (; expected_reward = 0.5)
Turing.@model function CognitiveModeling.turing_apply_model(model::NoisySessionModel, input::ModelInput)
    noise ~ Normal(0, input.parameters.noise_scale)
    Vₜ = input.previous_state.expected_reward + 0.5 * (input.observation.reward - input.previous_state.expected_reward) + noise
    return ModelOutput((; choice = Normal(Vₜ, 1.0)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::NoisySessionModel) = ModelAttributes(; parameters = :noise_scale, states = :expected_reward, observations = :reward, actions = :choice)

#Repeats its previous choice with a probability of 0.9, and chooses at random when there is none
struct RepeatsPreviousChoice <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RepeatsPreviousChoice, parameters) = (;)
function CognitiveModeling.apply_model(model::RepeatsPreviousChoice, input::ModelInput)
    previous = input.previous_action.choice
    return ModelOutput((; choice = Bernoulli(previous isa AbstractNoValue ? 0.5 : previous ? 0.9 : 0.1)))
end
CognitiveModeling.attribute_names(model::RepeatsPreviousChoice) = ModelAttributes(; parameters = (), states = (), observations = (), actions = :choice)

#Throws the error for invalid parameters when the learning rate is above 1
struct RejectsHighLearningRate <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RejectsHighLearningRate, parameters) = (; expected_reward = 0.5)
function CognitiveModeling.apply_model(model::RejectsHighLearningRate, input::ModelInput)
    input.parameters.learning_rate > 1 && throw(InvalidParametersError("The learning rate must be at most 1."))
    Vₜ = input.previous_state.expected_reward + input.parameters.learning_rate * (input.observation.reward - input.previous_state.expected_reward)
    return ModelOutput((; choice = Normal(Vₜ, 1.0)), (; expected_reward = Vₜ))
end
CognitiveModeling.attribute_names(model::RejectsHighLearningRate) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)

#Gives the previous choice as the reward, scaled, so that filled in observations can be checked
struct RewardsPreviousChoice <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::RewardsPreviousChoice, parameters) = (; n_timesteps = 0)
function CognitiveModeling.apply_model(environment::RewardsPreviousChoice, input::EnvironmentInput)
    reward = input.action.choice isa AbstractNoValue ? 0.5 : float(input.action.choice)
    return EnvironmentOutput((; reward = reward * input.parameters.scale), (; n_timesteps = input.previous_state.n_timesteps + 1))
end
CognitiveModeling.attribute_names(environment::RewardsPreviousChoice) = ModelAttributes(; parameters = :scale, states = :n_timesteps, observations = :reward, actions = :choice)
CognitiveModeling.default_parameters(environment::RewardsPreviousChoice) = (; scale = 1.0)

#Gives no observations at all
struct GivesNothing <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::GivesNothing, parameters) = (;)
CognitiveModeling.apply_model(environment::GivesNothing, input::EnvironmentInput) = EnvironmentOutput((;))
CognitiveModeling.attribute_names(environment::GivesNothing) = ModelAttributes(; parameters = (), states = (), observations = (), actions = ())

#Samples its reward in Turing
struct NoisyRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::NoisyRewards, parameters) = (;)
Turing.@model function CognitiveModeling.turing_apply_model(environment::NoisyRewards, input::EnvironmentInput)
    reward ~ Normal(0.5, 1.0)
    return EnvironmentOutput((; reward))
end
CognitiveModeling.attribute_names(environment::NoisyRewards) = ModelAttributes(; parameters = (), states = (), observations = :reward, actions = :choice)

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
        choices[t] isa MissingAction || (log_density += logpdf(BernoulliLogit(inverse_temperature * (V - 0.5)), choices[t]))
        push!(states, V)
    end
    return log_density, states
end

const SESSION_REWARDS = (; reward = [1.0, 0.0, 1.0, 1.0, 0.0])
const SESSION_CHOICES = (; choice = [true, false, true, true, false])
const SESSION_PARAMETERS = (; learning_rate = 0.3, inverse_temperature = 2.0)

#Shortens the calls below
function session_for_test(model = SessionRescorlaWagner(), parameters = SESSION_PARAMETERS, observations = SESSION_REWARDS, actions = SESSION_CHOICES; who_starts = EnvironmentStarts(), keywords...)
    return turing_single_session(model, parameters, observations, actions; who_starts, keywords...)
end

@testset "turing_single_session" begin
    @testset "The session starts with BeforeFirstTimestep for the previous action, and for the observation when the agent starts" begin
        #RepeatsPreviousChoice chooses at random when there is no previous action, and repeats its choice with 0.9 otherwise
        @test logjoint(session_for_test(RepeatsPreviousChoice(), (;), (;), (; choice = [true, true, false])), (;)) ≈ log(0.5) + log(0.9) + log(0.1)
        #A previous action of its own replaces the marker
        @test logjoint(session_for_test(RepeatsPreviousChoice(), (;), (;), (; choice = [true, true, false]); previous_action = (; choice = true)), (;)) ≈ 2 * log(0.9) + log(0.1)
    end

    @testset "The log density is the sum over the recorded actions, with the agent or the environment starting" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            model = session_for_test(; who_starts, save_history = SaveHistory(SaveEverything()))
            expected, states = session_by_hand(SESSION_REWARDS.reward, SESSION_CHOICES.choice, 0.3, 2.0, who_starts)
            @test logjoint(model, (;)) ≈ expected
            @test loglikelihood(model, (;)) ≈ expected
            #There are no latents, and what is returned is what happened
            @test isempty(keys(rand(Xoshiro(1), model)))
            history = returned(model, (;))
            @test [state.expected_reward for state in history.states] ≈ states
            @test history.actions == [(; choice) for choice in SESSION_CHOICES.choice]
            @test history.observations == [(; reward) for reward in SESSION_REWARDS.reward]
        end
    end

    @testset "Default parameters are filled in, and the names and the data are checked" begin
        @test logjoint(session_for_test(SessionRescorlaWagner(), (; learning_rate = 0.3)), (;)) ≈ logjoint(session_for_test(), (;))
        @test_throws "The parameters are named (:inverse_temperature, :learning_rate, :noise), but attribute_names lists the parameters of a model of type SessionRescorlaWagner" session_for_test(SessionRescorlaWagner(), (; learning_rate = 0.3, noise = 1.0))
        @test_throws "The observations in the data are named (:cue,), but attribute_names lists the observations of a cognitive model of type SessionRescorlaWagner as (:reward,). The data must have every observation the cognitive model lists." session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; cue = [1.0, 0.0]), (; choice = [true, false]))
        @test_throws "The actions in the data are named (:bet,), but attribute_names lists the actions of a cognitive model of type SessionRescorlaWagner as (:choice,)." session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, SESSION_REWARDS, (; bet = SESSION_CHOICES.choice))
        @test_throws "Every observation and action in the data must have one value for each timestep, but their lengths are (reward = 2,) and (choice = 5,)." session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; reward = [1.0, 0.0]), SESSION_CHOICES)
        @test_throws "must each be a NamedTuple with one vector for each name" session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, [(; reward = 1.0)], [(; choice = true)])
        @test_throws "timestep_sizes must have one size for each timestep, but it has 2 sizes and the data has 5 timesteps." session_for_test(; timestep_sizes = [1.0, 2.0])
        @test_throws "The state is named (:value,), but attribute_names lists the states of a model of type SessionRescorlaWagner" session_for_test(; previous_cognitive_state = (; value = 0.5))
        #The data can have more observations than the model lists
        @test logjoint(session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; reward = SESSION_REWARDS.reward, cue = 1:5)), (;)) ≈ logjoint(session_for_test(), (;))
    end

    @testset "save_history selects what is returned, and nothing by default" begin
        history = returned(session_for_test(), (;))
        @test history.observations == fill((;), 5) && history.actions == fill((;), 5) && history.states == fill((;), 5)
        history = returned(session_for_test(; save_history = SaveHistory(; states = :expected_reward, actions = :choice)), (;))
        @test history.observations == fill((;), 5)
        @test history.actions == [(; choice) for choice in SESSION_CHOICES.choice]
        @test history.states[1] == (; expected_reward = 0.65)
        @test_throws FieldError returned(session_for_test(TwoContinuousActions(), (; learning_rate = 0.3), SESSION_REWARDS, (; choice = ones(5), bet = ones(5)); save_history = SaveHistory(; observations = :cue)), (;))
    end

    @testset "The sizes of the timesteps are returned when anything is saved, and are empty otherwise" begin
        @test returned(session_for_test(; save_history = SaveHistory(; states = :expected_reward)), (;)).timestep_sizes == fill(1, 5)
        @test returned(session_for_test(; timestep_sizes = 0.5, save_history = SaveHistory(; actions = SaveEverything())), (;)).timestep_sizes == fill(0.5, 5)
        sizes = [0.5, 1.0, 1.5, 2.0, 2.5]
        @test returned(session_for_test(; timestep_sizes = sizes, save_history = SaveHistory(SaveEverything())), (;)).timestep_sizes === sizes
        #Nothing is saved by default
        @test returned(session_for_test(), (;)).timestep_sizes == Int[]
        @test returned(session_for_test(; timestep_sizes = sizes), (;)).timestep_sizes == Float64[]
        @test keys(returned(session_for_test(), (;))) == (:observations, :actions, :states, :timestep_sizes)
    end

    @testset "Missing actions are latents named after their timestep, with the model's distributions" begin
        actions = (; choice = [0.2, MissingAction(), 0.4, MissingAction()], bet = [1.0, 1.5, MissingAction(), 0.8])
        model = session_for_test(TwoContinuousActions(), (; learning_rate = 0.3), (; reward = [1.0, 0.0, 1.0, 1.0]), actions; save_history = SaveHistory(; actions = SaveEverything()))
        #One latent for each missing action
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([@varname(timestep[2].choice.imputed), @varname(timestep[3].bet.imputed), @varname(timestep[4].choice.imputed)])
        imputed = [0.1, -0.2, 0.3]
        given = Dict(@varname(timestep[2].choice.imputed) => imputed[1], @varname(timestep[3].bet.imputed) => imputed[2], @varname(timestep[4].choice.imputed) => imputed[3])
        V = 0.5
        expected_prior = 0.0
        expected_likelihood = 0.0
        filled = []
        for t in 1:4
            V = V + 0.3 * ([1.0, 0.0, 1.0, 1.0][t] - V)
            choice = actions.choice[t] isa MissingAction ? (t == 2 ? imputed[1] : imputed[3]) : actions.choice[t]
            bet = actions.bet[t] isa MissingAction ? imputed[2] : actions.bet[t]
            #An imputed action is a latent, so its density is prior, and a recorded one is likelihood
            if actions.choice[t] isa MissingAction
                expected_prior += logpdf(Normal(V, 1.0), choice)
            else
                expected_likelihood += logpdf(Normal(V, 1.0), choice)
            end
            if actions.bet[t] isa MissingAction
                expected_prior += logpdf(Normal(2V, 0.5), bet)
            else
                expected_likelihood += logpdf(Normal(2V, 0.5), bet)
            end
            push!(filled, (; choice, bet))
        end
        @test (@test_logs match_mode = :any logjoint(model, given)) ≈ expected_prior + expected_likelihood
        @test (@test_logs match_mode = :any logprior(model, given)) ≈ expected_prior
        @test (@test_logs match_mode = :any loglikelihood(model, given)) ≈ expected_likelihood
        #The returned actions have the imputed values
        @test (@test_logs match_mode = :any returned(model, given)).actions == filled
        #A single missing action keeps the types of the rest known
        single = session_for_test(TwoContinuousActions(), (; learning_rate = 0.3), (; reward = [1.0, 0.0]), (; choice = [0.2, MissingAction()], bet = [1.0, 1.5]))
        @test collect(keys(@test_logs match_mode = :any rand(Xoshiro(1), single))) == [@varname(timestep[2].choice.imputed)]
    end

    @testset "Missing actions that aren't imputed are passed on as NoAction, with nothing sampled" begin
        rewards = (; reward = [1.0, 0.0, 1.0, 1.0])
        actions = (; choice = [0.2, MissingAction(), 0.4, MissingAction()], bet = [1.0, 1.5, MissingAction(), 0.8])
        model = session_for_test(TwoContinuousActions(), (; learning_rate = 0.3), rewards, actions; impute_missing_actions = false, save_history = SaveHistory(; actions = SaveEverything()))
        @test isempty(keys(rand(Xoshiro(1), model)))
        #Only the recorded actions are scored
        V = 0.5
        expected = 0.0
        for t in 1:4
            V = V + 0.3 * (rewards.reward[t] - V)
            actions.choice[t] isa MissingAction || (expected += logpdf(Normal(V, 1.0), actions.choice[t]))
            actions.bet[t] isa MissingAction || (expected += logpdf(Normal(2V, 0.5), actions.bet[t]))
        end
        @test logjoint(model, (;)) ≈ expected
        @test logprior(model, (;)) == 0
        #The same as data which has NoAction there
        with_no_action = (; choice = [0.2, NoAction(), 0.4, NoAction()], bet = [1.0, 1.5, NoAction(), 0.8])
        @test logjoint(model, (;)) ≈ logjoint(session_for_test(TwoContinuousActions(), (; learning_rate = 0.3), rewards, with_no_action), (;))
        #The returned actions have NoAction for them
        @test returned(model, (;)).actions == [(; choice = 0.2, bet = 1.0), (; choice = NoAction(), bet = 1.5), (; choice = 0.4, bet = NoAction()), (; choice = NoAction(), bet = 0.8)]

        #The model receives NoAction as its previous action after a missing one, so here it chooses at random again
        choices = (; choice = [true, MissingAction(), true])
        @test logjoint(session_for_test(RepeatsPreviousChoice(), (;), (;), choices; impute_missing_actions = false), (;)) ≈ 2 * log(0.5)
        #When it is imputed, the model receives the imputed action
        imputed = Dict(@varname(timestep[2].choice.imputed) => true)
        @test (@test_logs match_mode = :any logjoint(session_for_test(RepeatsPreviousChoice(), (;), (;), choices), imputed)) ≈ log(0.5) + 2 * log(0.9)

        #The environment receives NoAction too: RewardsPreviousChoice gives 0.5 for it, and the previous choice otherwise
        model = session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; reward = fill(MissingObservation(), 3)), choices; impute_missing_actions = false, environment = RewardsPreviousChoice(), save_history = SaveHistory(; observations = SaveEverything()))
        @test returned(model, (;)).observations == [(; reward = 0.5), (; reward = 1.0), (; reward = 0.5)]
    end

    @testset "A missing observation is filled in by the environment, which runs on every timestep" begin
        #RewardsPreviousChoice gives 0.5 before the first choice and then the previous choice
        full = (; reward = [0.5, 1.0, 0.0, 1.0, 1.0])
        with_missing = (; reward = [0.5, MissingObservation(), 0.0, MissingObservation(), 1.0])
        reference = session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, full, SESSION_CHOICES)
        model = session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, with_missing, SESSION_CHOICES; environment = RewardsPreviousChoice(), save_history = SaveHistory(; observations = SaveEverything()))
        @test logjoint(model, (;)) ≈ logjoint(reference, (;))
        @test returned(model, (;)).observations == [(; reward) for reward in full.reward]
        #The environment's defaults are filled in and its names checked
        @test_throws "attribute_names lists the parameters of an environment model of type RewardsPreviousChoice" session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, with_missing, SESSION_CHOICES; environment = RewardsPreviousChoice(), environment_parameters = (; size = 1.0))
        @test_throws "A cognitive model of type SessionRescorlaWagner lists the observations (:reward,), but an environment model of type GivesNothing gives (). Without a map, the two must list the same observations." session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, SESSION_REWARDS, SESSION_CHOICES; environment = GivesNothing())
        #When the agent starts, the environment responds to the action of the same timestep
        model = session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; reward = fill(MissingObservation(), 5)), SESSION_CHOICES; who_starts = AgentStarts(), environment = RewardsPreviousChoice(), save_history = SaveHistory(; observations = SaveEverything()))
        @test returned(model, (;)).observations == [(; reward = float(choice)) for choice in SESSION_CHOICES.choice]
        #Without an environment, the data can't have missing observations
        @test_throws "The observations have entries that are MissingObservation(), but no environment was given to fill them in." session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, with_missing, SESSION_CHOICES)
        #A random environment has its variables under the timestep, and its values are used
        model = session_for_test(SessionRescorlaWagner(), SESSION_PARAMETERS, (; reward = fill(MissingObservation(), 2)), (; choice = [true, false]); environment = NoisyRewards(), save_history = SaveHistory(; observations = SaveEverything()))
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([@varname(timestep[1].environment.reward), @varname(timestep[2].environment.reward)])
        rewards = Dict(@varname(timestep[1].environment.reward) => 0.7, @varname(timestep[2].environment.reward) => -0.4)
        @test (@test_logs match_mode = :any returned(model, rewards)).observations == [(; reward = 0.7), (; reward = -0.4)]
    end

    @testset "A model which samples within a timestep has a variable for each timestep" begin
        model = session_for_test(NoisySessionModel(), (; noise_scale = 1.0), (; reward = [1.0, 0.0, 1.0]), (; choice = [0.2, 0.4, 0.6]); save_history = SaveHistory(; states = :expected_reward))
        values = @test_logs match_mode = :any rand(Xoshiro(1), model)
        @test Set(keys(values)) == Set([@varname(timestep[1].noise), @varname(timestep[2].noise), @varname(timestep[3].noise)])
        noises = Dict(@varname(timestep[t].noise) => 0.1 * t for t in 1:3)
        V = 0.5
        expected = 0.0
        for t in 1:3
            V = V + 0.5 * ([1.0, 0.0, 1.0][t] - V) + 0.1 * t
            expected += logpdf(Normal(0, 1), 0.1 * t) + logpdf(Normal(V, 1.0), [0.2, 0.4, 0.6][t])
        end
        @test (@test_logs match_mode = :any logjoint(model, noises)) ≈ expected
        @test (@test_logs match_mode = :any returned(model, noises)).states[end].expected_reward ≈ V
    end

    @testset "NoAction in the data or from the model isn't scored, and both get the timestep sizes" begin
        #The model acts on even timesteps, and the data has NoAction on the odd ones
        actions = (; choice = [NoAction(), true, NoAction(), false])
        model = session_for_test(ActsEverySecondTimestep(), (; probability = 0.7), (;), actions; timestep_sizes = [0.5, 1.0, 1.5, 2.0], save_history = SaveHistory(SaveEverything()))
        @test logjoint(model, (;)) ≈ logpdf(Bernoulli(0.7), true) + logpdf(Bernoulli(0.7), false)
        history = returned(model, (;))
        @test history.actions == [(; choice = NoAction()), (; choice = true), (; choice = NoAction()), (; choice = false)]
        @test [state.timestep_size for state in history.states] == [0.5, 1.0, 1.5, 2.0]
        @test history.observations == fill((;), 4)
        #The data can also have NoAction where the model acts, which isn't scored either
        @test logjoint(session_for_test(ActsEverySecondTimestep(), (; probability = 0.7), (;), (; choice = [NoAction(), NoAction(), NoAction(), false])), (;)) ≈ logpdf(Bernoulli(0.7), false)
        #A recorded action where the model took none is an error
        @test_throws "The cognitive model of type ActsEverySecondTimestep took no action choice on a timestep where the data has true for it." logjoint(session_for_test(ActsEverySecondTimestep(), (; probability = 0.7), (;), (; choice = [true, true, false, false])), (;))
        #The sizes of the timesteps are returned with what is saved
        @test history.timestep_sizes == [0.5, 1.0, 1.5, 2.0]
        #With a single size, every timestep gets it
        @test [state.timestep_size for state in returned(session_for_test(ActsEverySecondTimestep(), (; probability = 0.7), (;), actions; timestep_sizes = 0.25, save_history = SaveHistory(; states = :timestep_size)), (;)).states] == fill(0.25, 4)
    end

    @testset "An InvalidParametersError stops the evaluation, unless it is caught to give a log density of -Inf" begin
        actions = (; choice = [0.2, 0.4, 0.6, 0.8, 1.0])
        @test_throws "The learning rate must be at most 1." logjoint(session_for_test(RejectsHighLearningRate(), (; learning_rate = 1.5), SESSION_REWARDS, actions), (;))
        caught = session_for_test(RejectsHighLearningRate(), (; learning_rate = 1.5), SESSION_REWARDS, actions; catch_invalid_parameters = true)
        @test logjoint(caught, (;)) == -Inf
        @test returned(caught, (;)) === nothing
        #Valid parameters are unaffected
        valid = session_for_test(RejectsHighLearningRate(), (; learning_rate = 0.3), SESSION_REWARDS, actions; catch_invalid_parameters = true, save_history = SaveHistory(; states = :expected_reward))
        @test logjoint(valid, (;)) ≈ logjoint(session_for_test(RejectsHighLearningRate(), (; learning_rate = 0.3), SESSION_REWARDS, actions), (;))
        @test returned(valid, (;)).states[1] == (; expected_reward = 0.65)
        #Other errors still get through
        @test_throws "took no action" logjoint(session_for_test(ActsEverySecondTimestep(), (; probability = 0.7), (;), (; choice = [true, true]); catch_invalid_parameters = true), (;))
    end

    @testset "It can be sampled as a submodel, and returned gives the states for each sample" begin
        Turing.@model function fit_one_session(observations, actions)
            learning_rate ~ Beta(2, 2)
            inverse_temperature ~ LogNormal(0, 0.5)
            session ~ to_submodel(prefix(turing_single_session(SessionRescorlaWagner(), (; learning_rate, inverse_temperature), observations, actions; who_starts = EnvironmentStarts(), save_history = SaveHistory(; states = :expected_reward)), @varname(session)), false)
            return session
        end
        model = fit_one_session(SESSION_REWARDS, SESSION_CHOICES)
        chain = sample(Xoshiro(1), model, NUTS(), 100; progress = false)
        @test size(chain, 1) == 100
        @test Set(Turing.FlexiChains.parameters(chain)) == Set([@varname(learning_rate), @varname(inverse_temperature)])
        trajectories = returned(model, chain)
        @test length(trajectories) == 100
        @test length(trajectories[1].states) == 5
        @test trajectories[1].states[1] == (; expected_reward = 0.5 + chain[@varname(learning_rate)][1] * 0.5)
    end
end
