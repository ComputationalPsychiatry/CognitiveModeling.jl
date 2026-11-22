using JET: JET
using Random: AbstractRNG, Xoshiro

## Models and environments for the tests ##
#The tests use the example models that the test suites share. The models below are for what no example model should
#have, such as errors in how a model is defined, or for a feature that no example model has

#An action made by hand, so that the tests don't need a distribution library
struct ChooseWithProbability{T}
    probability::T
end
Base.rand(rng::AbstractRNG, action::ChooseWithProbability) = rand(rng) < action.probability

#An observation made by hand in the same way: a reward of 1.0 with a probability, and of 0.0 otherwise
struct ZeroOrOne{T}
    probability::T
end
Base.rand(rng::AbstractRNG, reward::ZeroOrOne) = float(rand(rng) < reward.probability)

#Has two named actions, and receives a reward which it does nothing with
struct TwoChoices <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::TwoChoices, parameters) = (;)
function CognitiveModeling.timestep_model(model::TwoChoices, context::ModelContext)
    return ModelOutput((;
        first_choice=ChooseWithProbability(0.0), second_choice=ChooseWithProbability(1.0)
    ))
end
function CognitiveModeling.attribute_names(model::TwoChoices)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:reward, outputs=(:first_choice, :second_choice)
    )
end

#Changes a Vector in its state in place, and prepares its state for saving with the function it is given
struct LearnsInPlace{Tprepare_state} <: AbstractCognitiveModel
    prepare_state::Tprepare_state
end
function CognitiveModeling.prepare_state_for_saving(model::LearnsInPlace, state)
    return model.prepare_state(state)
end
CognitiveModeling.initialize_state(model::LearnsInPlace, parameters) = (; values=[0.5, 0.5])
function CognitiveModeling.timestep_model(model::LearnsInPlace, context::ModelContext)
    values = context.previous_state.values
    values[1] += context.parameters.learning_rate * (context.input.reward - values[1])
    return ModelOutput((; choice=ChooseWithProbability(0.5)), context.previous_state)
end
function CognitiveModeling.attribute_names(model::LearnsInPlace)
    return ModelAttributes(;
        parameters=:learning_rate, states=:values, inputs=:reward, outputs=:choice
    )
end

#An environment which keeps who started as its state, to check what the environment is told
struct EnvironmentRecordsWhoStarts <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::EnvironmentRecordsWhoStarts, parameters
)
    return (; who_starts=UnknownWhoStarts())
end
function CognitiveModeling.timestep_model(
    environment::EnvironmentRecordsWhoStarts, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)), (; who_starts=context.who_starts))
end
function CognitiveModeling.attribute_names(environment::EnvironmentRecordsWhoStarts)
    return ModelAttributes(;
        parameters=(), states=:who_starts, inputs=:choice, outputs=:reward
    )
end

#Keeps who started as its state, to check what the model is told
struct RecordsWhoStarts <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::RecordsWhoStarts, parameters)
    return (; who_starts=UnknownWhoStarts())
end
function CognitiveModeling.timestep_model(model::RecordsWhoStarts, context::ModelContext)
    return ModelOutput((; choice=Deterministic(true)), (; who_starts=context.who_starts))
end
function CognitiveModeling.attribute_names(model::RecordsWhoStarts)
    return ModelAttributes(;
        parameters=(), states=:who_starts, inputs=:reward, outputs=:choice
    )
end

#Has an action that is never taken, and a NamedTuple of actions
struct ChoosesInGroups <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ChoosesInGroups, parameters) = (;)
function CognitiveModeling.timestep_model(model::ChoosesInGroups, context::ModelContext)
    return ModelOutput((;
        choice=ChooseWithProbability(1.0),
        skipped=NoValue(),
        movement=(; left=ChooseWithProbability(0.0), right=ChooseWithProbability(1.0)),
    ))
end
function CognitiveModeling.attribute_names(model::ChoosesInGroups)
    return ModelAttributes(;
        parameters=(), states=(), inputs=(), outputs=(:choice, :skipped, :movement)
    )
end

#Has default parameters and names, and an initial state which depends on a parameter
struct HasDefaults <: AbstractCognitiveModel end
function CognitiveModeling.default_parameters(model::HasDefaults)
    return (; learning_rate=0.1, initial_value=0.5)
end
function CognitiveModeling.initialize_state(model::HasDefaults, parameters)
    return (; expected_reward=parameters.initial_value)
end
function CognitiveModeling.timestep_model(model::HasDefaults, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    return ModelOutput(
        (; choice=ChooseWithProbability(V₋₁)),
        (;
            expected_reward=V₋₁ +
                            context.parameters.learning_rate * (context.input.reward - V₋₁)
        ),
    )
end
function CognitiveModeling.attribute_names(model::HasDefaults)
    return ModelAttributes(;
        parameters=(:learning_rate, :initial_value),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

#Has no names, so it can't be simulated
struct WithoutNames <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::WithoutNames, parameters) = (;)
function CognitiveModeling.timestep_model(model::WithoutNames, context::ModelContext)
    return ModelOutput((; choice=ChooseWithProbability(0.5)))
end

#Has default parameters, and reads the agent's choice
struct RewardsWithDefaults <: AbstractEnvironmentModel end
function CognitiveModeling.default_parameters(environment::RewardsWithDefaults)
    return (; reward_size=1.0, bias=0.0)
end
function CognitiveModeling.initialize_state(environment::RewardsWithDefaults, parameters)
    return (; last_choice=NoValue())
end
function CognitiveModeling.timestep_model(
    environment::RewardsWithDefaults, context::ModelContext
)
    reward = if context.input.choice isa AbstractNoValue
        context.parameters.bias
    else
        context.parameters.bias + context.parameters.reward_size * context.input.choice
    end
    return ModelOutput(
        (; reward=Deterministic(reward)), (; last_choice=context.input.choice)
    )
end
function CognitiveModeling.attribute_names(environment::RewardsWithDefaults)
    return ModelAttributes(;
        parameters=(:reward_size, :bias),
        states=:last_choice,
        inputs=:choice,
        outputs=:reward,
    )
end

#Gives only a cue, which a model which learns from rewards can't use
struct OnlyCues <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::OnlyCues, parameters) = (;)
function CognitiveModeling.timestep_model(environment::OnlyCues, context::ModelContext)
    return ModelOutput((; cue=Deterministic(2)))
end
function CognitiveModeling.attribute_names(environment::OnlyCues)
    return ModelAttributes(; parameters=(), states=(), inputs=(), outputs=:cue)
end

#Has no names, so it can't be simulated
struct EnvironmentWithoutNames <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::EnvironmentWithoutNames, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::EnvironmentWithoutNames, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)))
end

#Changes a Vector in its state in place, and copies its state when it's saved
struct CountsInPlace <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::CountsInPlace, parameters) = (; counts=[0])
function CognitiveModeling.timestep_model(environment::CountsInPlace, context::ModelContext)
    context.previous_state.counts[1] += 1
    return ModelOutput((; reward=Deterministic(1.0)), context.previous_state)
end
function CognitiveModeling.prepare_state_for_saving(environment::CountsInPlace, state)
    return deepcopy(state)
end
function CognitiveModeling.attribute_names(environment::CountsInPlace)
    return ModelAttributes(; parameters=(), states=:counts, inputs=:choice, outputs=:reward)
end

#Saves the timestep size it gets in its state, and does nothing with the one it observes. The state can't share the
#name of the observation, since a model uses each name for one attribute
struct SavesTimestepSize <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::SavesTimestepSize, parameters)
    return (; received_timestep_size=0)
end
function CognitiveModeling.timestep_model(model::SavesTimestepSize, context::ModelContext)
    return ModelOutput(
        (; choice=ChooseWithProbability(0.5)),
        (; received_timestep_size=context.timestep_size),
    )
end
function CognitiveModeling.attribute_names(model::SavesTimestepSize)
    return ModelAttributes(;
        parameters=(),
        states=:received_timestep_size,
        inputs=:timestep_size,
        outputs=:choice,
    )
end

#Gives the timestep size it gets as its observation
struct ObservesTimestepSize <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesTimestepSize, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::ObservesTimestepSize, context::ModelContext
)
    return ModelOutput((; timestep_size=Deterministic(context.timestep_size)))
end
function CognitiveModeling.attribute_names(environment::ObservesTimestepSize)
    return ModelAttributes(;
        parameters=(), states=(), inputs=:choice, outputs=:timestep_size
    )
end

#Returns a tuple instead of a ModelOutput
struct ReturnsTuple <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ReturnsTuple, parameters) = (;)
function CognitiveModeling.timestep_model(model::ReturnsTuple, context::ModelContext)
    return (ChooseWithProbability(0.5), (;))
end
function CognitiveModeling.attribute_names(model::ReturnsTuple)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end

#Returns the observation instead of a ModelOutput
struct ReturnsObservation <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ReturnsObservation, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::ReturnsObservation, context::ModelContext
)
    return (; reward=1.0)
end
function CognitiveModeling.attribute_names(environment::ReturnsObservation)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end

#Have a method for the initial state, but none for a timestep
struct OnlyInitializesState <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::OnlyInitializesState, parameters) = (;)
function CognitiveModeling.attribute_names(model::OnlyInitializesState)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end
struct EnvironmentOnlyInitializesState <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::EnvironmentOnlyInitializesState, parameters
)
    return (;)
end
function CognitiveModeling.attribute_names(environment::EnvironmentOnlyInitializesState)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end

#Have a method for a timestep, but none for the initial state, so they must be given a state
struct NeedsGivenState <: AbstractCognitiveModel end
function CognitiveModeling.timestep_model(model::NeedsGivenState, context::ModelContext)
    return ModelOutput((; choice=ChooseWithProbability(0.5)), context.previous_state)
end
function CognitiveModeling.attribute_names(model::NeedsGivenState)
    return ModelAttributes(; parameters=(), states=:value, inputs=:reward, outputs=:choice)
end
struct EnvironmentNeedsGivenState <: AbstractEnvironmentModel end
function CognitiveModeling.timestep_model(
    environment::EnvironmentNeedsGivenState, context::ModelContext
)
    return ModelOutput((; reward=Deterministic(1.0)), context.previous_state)
end
function CognitiveModeling.attribute_names(environment::EnvironmentNeedsGivenState)
    return ModelAttributes(; parameters=(), states=:value, inputs=:choice, outputs=:reward)
end

#Has an observation that is never made, and a NamedTuple of observations, and takes no actions
struct ObservesInGroups <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::ObservesInGroups, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::ObservesInGroups, context::ModelContext
)
    return ModelOutput((;
        reward=ZeroOrOne(1.0),
        skipped=NoValue(),
        cues=(; left=ZeroOrOne(0.0), right=Deterministic(2)),
    ))
end
function CognitiveModeling.attribute_names(environment::ObservesInGroups)
    return ModelAttributes(;
        parameters=(), states=(), inputs=(), outputs=(:reward, :skipped, :cues)
    )
end

#Keeps the reward it gave on the timestep before in its state, as it receives it
struct RemembersReward <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::RemembersReward, parameters)
    return (; previous_reward=BeforeFirstTimestep())
end
function CognitiveModeling.timestep_model(
    environment::RemembersReward, context::ModelContext
)
    return ModelOutput(
        (; reward=ZeroOrOne(0.5)), (; previous_reward=context.previous_output.reward)
    )
end
function CognitiveModeling.attribute_names(environment::RemembersReward)
    return ModelAttributes(;
        parameters=(), states=:previous_reward, inputs=:choice, outputs=:reward
    )
end

#Counts its timesteps, gives a reward of 1 on each, and sets parameters_valid to false on its third
struct InvalidOnThirdTimestep <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::InvalidOnThirdTimestep, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::InvalidOnThirdTimestep, context::ModelContext
)
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; reward=Deterministic(1.0)), (; n_timesteps); parameters_valid=n_timesteps < 3
    )
end
function CognitiveModeling.attribute_names(environment::InvalidOnThirdTimestep)
    return ModelAttributes(;
        parameters=(), states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

#Give a value itself where something to sample from is needed
struct GivesUnwrapped{T} <: AbstractEnvironmentModel
    reward::T
end
CognitiveModeling.initialize_state(environment::GivesUnwrapped, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::GivesUnwrapped, context::ModelContext
)
    return ModelOutput((; reward=environment.reward))
end
function CognitiveModeling.attribute_names(environment::GivesUnwrapped)
    return ModelAttributes(; parameters=(), states=(), inputs=:choice, outputs=:reward)
end
struct ChoosesUnwrapped{T} <: AbstractCognitiveModel
    choice::T
end
CognitiveModeling.initialize_state(model::ChoosesUnwrapped, parameters) = (;)
function CognitiveModeling.timestep_model(model::ChoosesUnwrapped, context::ModelContext)
    return ModelOutput((; choice=model.choice))
end
function CognitiveModeling.attribute_names(model::ChoosesUnwrapped)
    return ModelAttributes(; parameters=(), states=(), inputs=:reward, outputs=:choice)
end

#A learning rate of a half: the parameters of WithoutOutputs and LearnsInPlace, and of WithDefaultParameters with its
#default
const LEARNING = (; learning_rate=0.5)
const REWARDS = (; reward_probability=0.8)

#The parameters of the baseline, whose learning rate of a half takes each expected reward halfway to the reward
const BASELINE_PARAMETERS = (; learning_rate=0.5, inverse_temperature=2.0)

#The parameters of DependsOnPreviousOutput, which counts its timesteps beside its expected reward
const DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS = (; BASELINE_PARAMETERS..., stickiness=1.0)

#The probability with which the baseline chooses true with BASELINE_PARAMETERS, from its expected reward
baseline_choice_probability(expected_reward) = inv(1 + exp(-2.0 * (expected_reward - 0.5)))

#The reward for a choice from BaselineEnvironment and those like it, with REWARDS and the means of their distributions
reward_for_choice(choice) = 0.8 * (1 + choice) / 2

#The maps between the names of the baseline and those of OtherNamesEnvironment
const PAYOFF_AS_REWARD = NameMap(:payoff => :reward)
const CHOICE_AS_LEVER = NameMap(:choice => :lever)

#As many names as a large model has, for checking that comparing them costs nothing
const MANY_NAMES = Tuple(Symbol(:state_, i) for i in 1:65)
const MANY_NAMES_REVERSED = reverse(MANY_NAMES)

#Whether a call compiles to code which only returns, with no calls or branches left. With code coverage on, as in CI, each line
#of code leaves a statement for the coverage, and a statement that was removed leaves nothing in its place, which both are fine
function folds(f, argument_types...)
    statements = only(code_typed(f, argument_types))[1].code
    return all(
        statement ->
            statement isa Core.ReturnNode ||
            statement === nothing ||
            Meta.isexpr(statement, :code_coverage_effect),
        statements,
    )
end

#Saves everything but timestep 0, so that element t of what a simulation returns is timestep t
const EVERY_TIMESTEP = SaveHistory(SaveEverything(); save_initial_timestep_values=false)

#Shortens the calls below
function simulate_for_test(
    model,
    environment;
    who_starts=EnvironmentStarts(),
    n_steps=50,
    cognitive_parameters=BASELINE_PARAMETERS,
    environment_parameters=REWARDS,
    rng=Xoshiro(1),
    cognitive_save_history=EVERY_TIMESTEP,
    environment_save_history=EVERY_TIMESTEP,
    keywords...,
)
    return simulate_session(
        model,
        environment;
        who_starts,
        n_steps,
        cognitive_parameters,
        environment_parameters,
        rng,
        cognitive_save_history,
        environment_save_history,
        keywords...,
    )
end

#The eight vectors a simulation returns: the cognitive model's four, and the environment's four
function saved_vectors(simulation)
    return (
        simulation.observations,
        simulation.actions,
        simulation.states,
        simulation.action_distributions,
        simulation.environment.states,
        simulation.environment.observations,
        simulation.environment.actions,
        simulation.environment.observation_distributions,
    )
end

#What a simulation returns of the cognitive model, without the environment's part, for comparing simulations with different environments
function cognitive_values(simulation)
    return (;
        simulation.observations,
        simulation.actions,
        simulation.states,
        simulation.action_distributions,
        simulation.timestep_sizes,
    )
end

#An environment whose reward probability has a domain, and which gives the reward 1.0 on every timestep
struct RewardsWithinDomain <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(::RewardsWithinDomain, parameters) = (;)
function CognitiveModeling.timestep_model(::RewardsWithinDomain, context::ModelContext)
    return ModelOutput((; reward=Deterministic(1.0)))
end
function CognitiveModeling.attribute_names(::RewardsWithinDomain)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end
function CognitiveModeling.parameter_domains(::RewardsWithinDomain)
    return (; reward_probability=Domain(; lower=0, upper=1),)
end

@testset "A parameter outside its domain is an error before anything runs" begin
    @testset "A single timestep checks its parameters, unless it is told not to" begin
        @test simulate_timestep(
            WithDomains(ByHand()), (; reward=1.0); parameters=BASELINE_PARAMETERS
        )[2] == (; expected_reward=0.75)
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{ByHand} is 1.5, which is outside its domain, from 0 to 1.",
        ) simulate_timestep(
            WithDomains(ByHand()),
            (; reward=1.0);
            parameters=(; learning_rate=1.5, inverse_temperature=2.0),
        )
        @test simulate_timestep(
            WithDomains(ByHand()),
            (; reward=1.0);
            parameters=(; learning_rate=1.5, inverse_temperature=2.0),
            check_parameter_domains=false,
        )[2] == (; expected_reward=1.25)
    end

    @testset "A whole simulation checks the parameters of both models, unless it is told not to" begin
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{ByHand} is 1.5, which is outside its domain, from 0 to 1.",
        ) simulate_session(
            WithDomains(ByHand()),
            RewardsWithinDomain();
            who_starts=EnvironmentStarts(),
            n_steps=2,
            cognitive_parameters=(; learning_rate=1.5, inverse_temperature=2.0),
            environment_parameters=(; reward_probability=0.5),
        )
        @test_throws ParameterDomainError(
            "The parameter reward_probability of an environment model of type RewardsWithinDomain is -0.5, which is outside its domain, from 0 to 1.",
        ) simulate_session(
            WithDomains(ByHand()),
            RewardsWithinDomain();
            who_starts=EnvironmentStarts(),
            n_steps=2,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=(; reward_probability=-0.5),
        )
        #With rewards of 1, a learning rate of 1.5 takes the expected reward past 1, and back
        simulation = simulate_session(
            WithDomains(ByHand()),
            RewardsWithinDomain();
            who_starts=EnvironmentStarts(),
            n_steps=2,
            cognitive_parameters=(; learning_rate=1.5, inverse_temperature=2.0),
            environment_parameters=(; reward_probability=-0.5),
            cognitive_save_history=EVERY_TIMESTEP,
            check_parameter_domains=false,
        )
        @test simulation.states == [(; expected_reward=1.25), (; expected_reward=0.875)]
    end

    @testset "An instantiated model checks its parameters when it is created, unless it is told not to" begin
        @test InstantiatedModel(
            WithDomains(ByHand()), (; learning_rate=1, inverse_temperature=2.0)
        ).parameters == (; learning_rate=1, inverse_temperature=2.0)
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type WithDomains{ByHand} is -0.1, which is outside its domain, from 0 to 1.",
        ) InstantiatedModel(
            WithDomains(ByHand()), (; learning_rate=-0.1, inverse_temperature=2.0)
        )
        @test_throws ParameterDomainError InstantiatedModel(
            WithDomains(ByHand()),
            (; learning_rate=-0.1, inverse_temperature=2.0),
            (; expected_reward=0.0),
            (; choice=BeforeFirstTimestep()),
        )
        agent = InstantiatedModel(
            WithDomains(ByHand()),
            (; learning_rate=-0.1, inverse_temperature=2.0);
            check_parameter_domains=false,
        )
        @test agent.parameters == (; learning_rate=-0.1, inverse_temperature=2.0)
        simulate_timestep!(agent, (; reward=1.0))
        @test agent.state == (; expected_reward=0.45)
        @test InstantiatedModel(
            WithDomains(ByHand()),
            (; learning_rate=-0.1, inverse_temperature=2.0),
            (; expected_reward=0.0),
            (; choice=BeforeFirstTimestep());
            check_parameter_domains=false,
        ).parameters == (; learning_rate=-0.1, inverse_temperature=2.0)
    end
end

@testset "A parameter that is a vector is simulated with, and checked against its domain" begin
    #EveryFeatureWithoutSampling has a learning rate for each of three options, a vector with a domain, and chooses an
    #option in proportion to its expected reward. It is given the rewards of the options only
    every_feature_parameters =
        pair_named("EveryFeatureWithoutSampling").cognitive_parameters
    with_learning_rates(learning_rates) =
        merge(every_feature_parameters, (; learning_rates))
    rewards_of_options(rewards) = (; reward=NoValue(), rewards, feedback=NoValue())

    @testset "The model chooses with the probabilities that the vector gives" begin
        #Rewards of 1 take the expected rewards to 0.6, 0.65 and 0.75, for probabilities of 0.3, 0.325 and 0.375
        rng = Xoshiro(1)
        options = [
            simulate_timestep(
                EveryFeatureWithoutSampling(ByHand()),
                rewards_of_options([1.0, 1.0, 1.0]);
                parameters=with_learning_rates([0.2, 0.3, 0.5]),
                rng,
            )[1].option for _ in 1:3000
        ]
        @test all(in(1:3), options)
        shares = [count(==(option), options) / 3000 for option in 1:3]
        @test all(abs.(shares .- [0.3, 0.325, 0.375]) .< 0.03)
        #A view of a vector is taken as the vector is. Learning rates of 1 and rewards of 0 leave the first two options
        #no chance
        output, _ = simulate_timestep(
            EveryFeatureWithoutSampling(ByHand()),
            rewards_of_options([0.0, 0.0, 1.0]);
            parameters=with_learning_rates(view([1.0, 1.0, 0.5, 9.0], 1:3)),
        )
        @test output.option == 3
    end

    @testset "A vector with an element outside the domain, or a number where the domain takes a vector, is an error, unless the model is told not to check" begin
        @test_throws ParameterDomainError(
            "The parameter learning_rates of a cognitive model of type EveryFeatureWithoutSampling{ByHand} has learning_rates[3] = -0.2, which is outside its domain, from 0 to 1.",
        ) InstantiatedModel(
            EveryFeatureWithoutSampling(ByHand()), with_learning_rates([0.5, 0.7, -0.2])
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rates of a cognitive model of type EveryFeatureWithoutSampling{ByHand} is 0.5, of type Float64, but its domain takes values of type AbstractVector{Real}.",
        ) simulate_timestep(
            EveryFeatureWithoutSampling(ByHand()),
            rewards_of_options([1.0, 1.0, 1.0]);
            parameters=with_learning_rates(0.5),
        )
        agent = InstantiatedModel(
            EveryFeatureWithoutSampling(ByHand()),
            with_learning_rates([0.5, 0.7, -0.2]);
            check_parameter_domains=false,
        )
        @test agent.parameters.learning_rates == [0.5, 0.7, -0.2]
    end
end

#An observation made by hand which is a vector: a payoff of each option, which is the size of the option with a probability, and zero otherwise
struct PayoffsWithProbability{Tsizes,Tprobability}
    sizes::Tsizes
    probability::Tprobability
end
function Base.rand(rng::AbstractRNG, payoffs::PayoffsWithProbability)
    return [rand(rng) < payoffs.probability ? size : zero(size) for size in payoffs.sizes]
end

#Pays each option its size with a probability, a vector, and gives the number of the trial, a number. It counts how often each option was chosen, a vector, beside the number of trials, a number. Its sizes of payoffs are a vector, which grows from the first option to the last
#No example model has a constraint on a parameter, which its sizes of payoffs have
struct PaysEachOption <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::PaysEachOption, parameters)
    return (; n_choices=zeros(Int, length(parameters.payoff_sizes)), n_trials=0)
end
function CognitiveModeling.timestep_model(
    environment::PaysEachOption, context::ModelContext
)
    (; n_choices, n_trials) = context.previous_state
    choice = context.input.choice
    if !(choice isa AbstractNoValue)
        n_choices = [n + (option == choice) for (option, n) in enumerate(n_choices)]
    end
    n_trials += 1
    payoffs = PayoffsWithProbability(
        context.parameters.payoff_sizes, context.parameters.payoff_probability
    )
    return ModelOutput((; payoffs, trial=Deterministic(n_trials)), (; n_choices, n_trials))
end
function CognitiveModeling.attribute_names(environment::PaysEachOption)
    return ModelAttributes(;
        parameters=(:payoff_sizes, :payoff_probability),
        states=(:n_choices, :n_trials),
        inputs=(:ratings, :choice),
        outputs=(:payoffs, :trial),
    )
end
function CognitiveModeling.parameter_domains(environment::PaysEachOption)
    return (;
        payoff_sizes=Domain(; type=AbstractVector{Real}, lower=0, constraint=issorted),
        payoff_probability=Domain(; lower=0, upper=1),
    )
end

#Updates expected rewards by hand, as VectorAttributes does, with the rewards it receives
function expected_rewards_by_hand(learning_rates, received_rewards)
    expected_rewards = fill(0.5, length(learning_rates))
    for rewards in received_rewards
        expected_rewards =
            expected_rewards .+ learning_rates .* (rewards .- expected_rewards)
    end
    return expected_rewards
end

@testset "Observations, actions, states and parameters that are vectors, beside ones that are numbers" begin
    #VectorAttributes learns an expected reward for each of three options, chooses one, a number, and rates each, a
    #vector. VectorAttributesEnvironment rewards each option, with a bonus for the one chosen, exactly with its means
    vector_parameters = (; learning_rates=[0.2, 0.5, 0.9], noise=0.1)
    vector_environment_parameters = (; payoffs=[0.4, 0.6, 0.8], noise=0.1)
    function simulate_vectors(who_starts; rng=Xoshiro(1))
        return simulate_session(
            VectorAttributes(ByHand()),
            VectorAttributesEnvironment(Means());
            who_starts,
            n_steps=6,
            cognitive_parameters=vector_parameters,
            environment_parameters=vector_environment_parameters,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
    end

    @testset "Each timestep's vectors are saved, and each state follows from what the models received, whoever starts" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            simulation = simulate_vectors(who_starts)
            @test length(simulation.states) == length(simulation.environment.states) == 7
            @test eltype(simulation.observations) == Union{
                @NamedTuple{rewards::BeforeFirstTimestep},
                @NamedTuple{rewards::Vector{Float64}},
            }
            for t in 2:7
                #The environment receives the choice of the same timestep when the agent starts, and of the timestep before
                #when the environment starts, which at timestep 1 is none
                received_choice = if who_starts isa AgentStarts
                    simulation.actions[t].choice
                else
                    simulation.actions[t - 1].choice
                end
                #The rewards are a vector, each option's payoff with a bonus of 0.1 for the option chosen
                bonuses = [option == received_choice ? 0.1 : 0.0 for option in 1:3]
                @test simulation.observations[t].rewards == [0.4, 0.6, 0.8] .+ bonuses
                #The environment counts its timesteps, a number
                @test simulation.environment.states[t] == (; n_timesteps=t - 1)
                #The agent receives the observation of the same timestep when the environment starts, and of the timestep before
                #when the agent starts, which at timestep 1 is none
                received = if who_starts isa EnvironmentStarts
                    [observation.rewards for observation in simulation.observations[2:t]]
                else
                    [observation.rewards for observation in simulation.observations[2:(t - 1)]]
                end
                (; expected_rewards) = simulation.states[t]
                @test expected_rewards ≈ expected_rewards_by_hand([0.2, 0.5, 0.9], received)
                #The ratings are a vector around the expected rewards, and the choice a number, with probabilities in
                #proportion to them
                @test simulation.action_distributions[t].ratings.means == expected_rewards
                @test length(simulation.actions[t].ratings) == 3
                @test simulation.action_distributions[t].choice.probabilities ==
                    expected_rewards ./ sum(expected_rewards)
                @test simulation.actions[t].choice in 1:3
                #The environment receives the ratings with the choice
                @test simulation.environment.actions[t].ratings ==
                    simulation.actions[t].ratings
            end
            #The same seed gives the same vectors. The distributions are left out, since a struct with a vector in it is only
            #equal to itself
            same_seed = simulate_vectors(who_starts)
            other_seed = simulate_vectors(who_starts; rng=Xoshiro(2))
            for name in (:states, :observations, :actions)
                @test getproperty(same_seed, name) == getproperty(simulation, name)
                @test getproperty(other_seed, name) != getproperty(simulation, name)
            end
            @test same_seed.environment.states == simulation.environment.states
        end
    end

    @testset "An agent simulated on one observation at a time takes and gives vectors" begin
        #Learning rates of 1 and rewards of 0 leave the first two options no chance, so that the third is chosen
        agent = InstantiatedModel(
            VectorAttributes(ByHand()), (; learning_rates=[1.0, 1.0, 0.9], noise=0.1)
        )
        action = simulate_timestep!(agent, (; rewards=[0.0, 0.0, 3.0]); rng=Xoshiro(2))
        @test length(action.ratings) == 3 && action.choice == 3
        @test agent.state.expected_rewards ≈ [0.0, 0.0, 2.75]
        simulate_timestep!(agent, (; rewards=[0.0, 2.0, 3.0]); rng=Xoshiro(2))
        @test agent.state == (;
            expected_rewards=[0.0, 0.0, 2.75] .+
                             [1.0, 1.0, 0.9] .* ([0.0, 2.0, 3.0] .- [0.0, 0.0, 2.75])
        )
    end

    @testset "Fixed observations that are vectors are given in order, and the simulation stops after the last" begin
        fixed = FixedObservationsEnvironment((; rewards=[[1.0, 0.0, 3.0], [0.0, 2.0, 3.0]]))
        simulation = simulate_session(
            VectorAttributes(ByHand()),
            fixed;
            who_starts=EnvironmentStarts(),
            n_steps=5,
            cognitive_parameters=vector_parameters,
            environment_parameters=(;),
            action_map=NameMap(),
            rng=Xoshiro(1),
            cognitive_save_history=SaveHistory(SaveEverything()),
        )
        @test [state.expected_rewards for state in simulation.states] ≈ [
            [0.5, 0.5, 0.5],
            expected_rewards_by_hand([0.2, 0.5, 0.9], [[1.0, 0.0, 3.0]]),
            expected_rewards_by_hand([0.2, 0.5, 0.9], [[1.0, 0.0, 3.0], [0.0, 2.0, 3.0]]),
        ]
        @test simulation.observations[3] == (; rewards=[0.0, 2.0, 3.0])
    end

    @testset "The vector parameters of both models are checked against their domains, with the bounds of each element and the constraint of the whole" begin
        #EveryFeatureWithoutSampling has bounds for each of its learning rates, and PaysEachOption a constraint on its
        #sizes of payoffs, which VectorAttributes receives as rewards
        every_feature = pair_named("EveryFeatureWithoutSampling")
        @test_throws ParameterDomainError(
            "The parameter learning_rates of a cognitive model of type EveryFeatureWithoutSampling{ByHand} has learning_rates[2] = 1.5, which is outside its domain, from 0 to 1.",
        ) simulate_session(
            EveryFeatureWithoutSampling(ByHand()),
            EveryFeatureEnvironment(ByHand());
            who_starts=EnvironmentStarts(),
            n_steps=6,
            cognitive_parameters=merge(
                every_feature.cognitive_parameters, (; learning_rates=[0.2, 1.5, 0.9])
            ),
            every_feature.environment_parameters,
        )
        @test_throws ParameterDomainError(
            "The parameter payoff_sizes of an environment model of type PaysEachOption is [3.0, 1.0, 2.0], which doesn't satisfy the constraint issorted of its domain.",
        ) simulate_session(
            VectorAttributes(ByHand()),
            PaysEachOption();
            who_starts=AgentStarts(),
            n_steps=6,
            cognitive_parameters=vector_parameters,
            environment_parameters=(; payoff_sizes=[3.0, 1.0, 2.0], payoff_probability=0.7),
            observation_map=NameMap(:payoffs => :rewards),
        )
    end
end

@testset "simulate_session" begin
    expected_rewards(cognitive_states) =
        [state.expected_reward for state in cognitive_states]

    @testset "Who starts decides what the model learns from" begin
        fixed_rewards = FixedObservationsEnvironment((; reward=[1.0, 0.0, 1.0]))
        #Learns from the reward of the same timestep
        simulation = simulate_for_test(
            Baseline(ByHand()),
            fixed_rewards;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test expected_rewards(simulation.states) == [0.75, 0.375, 0.6875]
        @test simulation.environment.states[end] == (; n_observations_given=3)
        #Learns from the reward of the timestep before, and from nothing on the first
        simulation = simulate_for_test(
            Baseline(ByHand()),
            fixed_rewards;
            who_starts=AgentStarts(),
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test expected_rewards(simulation.states) == [0.5, 0.75, 0.375]
        @test simulation.environment.states[end] == (; n_observations_given=3)
    end

    @testset "The environment responds to the agent's actions" begin
        #The environment sees the action of the same timestep, and rewards it exactly with the means of its distributions
        (; observations, actions) = simulate_for_test(
            Baseline(ByHand()),
            NoOutputBeforeFirstInputEnvironment(Means());
            who_starts=AgentStarts(),
        )
        @test [observation.reward for observation in observations] == [reward_for_choice(action.choice) for action in actions]
        #The environment sees the action of the timestep before, and nothing on the first
        (; observations, actions) = simulate_for_test(
            Baseline(ByHand()),
            NoOutputBeforeFirstInputEnvironment(Means());
            who_starts=EnvironmentStarts(),
        )
        @test observations[1] === (; reward=NoValue())
        @test [observation.reward for observation in observations[2:end]] == [reward_for_choice(action.choice) for action in actions[1:(end - 1)]]
    end

    @testset "The same seed gives the same simulation, for the model and the environment" begin
        for who_starts in (EnvironmentStarts(), AgentStarts()),
            (model, cognitive_parameters) in (
                (Baseline(ByHand()), BASELINE_PARAMETERS),
                (
                    SamplingWithinTimestep(ByHand()),
                    (; BASELINE_PARAMETERS..., update_noise=0.1),
                ),
            )

            first_simulation = simulate_for_test(
                model,
                BaselineEnvironment(ByHand());
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(3),
            )
            second_simulation = simulate_for_test(
                model,
                BaselineEnvironment(ByHand());
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(3),
            )
            other_seed = simulate_for_test(
                model,
                BaselineEnvironment(ByHand());
                who_starts,
                cognitive_parameters,
                rng=Xoshiro(4),
            )
            @test first_simulation == second_simulation
            @test first_simulation != other_seed
        end
    end

    @testset "It returns the trajectories in a SessionTrajectories, with the environment's under environment" begin
        simulation = simulate_for_test(
            Baseline(ByHand()), BaselineEnvironment(ByHand()); n_steps=3
        )
        @test simulation isa SessionTrajectories
        @test propertynames(simulation) == (
            :states,
            :observations,
            :actions,
            :action_distributions,
            :environment,
            :timestep_sizes,
            :parameters_valid,
            :n_timesteps,
            :last_timestep,
        )
        #A model that never sets parameters_valid gives valid parameters on every timestep
        @test simulation.parameters_valid === true &&
            simulation.n_timesteps == 3 &&
            simulation.last_timestep === nothing
        @test keys(simulation.environment) ==
            (:states, :observations, :actions, :observation_distributions)
        #The names can be destructured
        (; observations, actions, states, environment, timestep_sizes) = simulation
        @test observations === simulation.observations &&
            actions === simulation.actions &&
            states === simulation.states &&
            environment === simulation.environment &&
            timestep_sizes === simulation.timestep_sizes
        @test all(vector -> length(vector) == 3, saved_vectors(simulation))
    end

    @testset "The return type is known and the loop doesn't allocate" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate_session(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            who_starts,
            n_steps,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa SessionTrajectories
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                who_starts, 10
            )
            run_simulation(who_starts, 10)
            #Doubling the timesteps only adds the storage: for the cognitive model, 8 bytes each for an action distribution and a
            #state, and 1 each for an observation and an action, and a byte each for the observation, the action and the action
            #distribution to say which of two types it has, since timestep 0 has BeforeFirstTimestep. The environment's
            #observations, actions and observation distributions take the same again, and its state 8 bytes
            extra_bytes =
                @allocated(run_simulation(who_starts, 20_000)) -
                @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 48 * 10_000
        end
    end

    @testset "Outputs whose types change are all kept" begin
        #An observation that is sometimes missing: OutputNotEveryTimestepEnvironment gives a reward on every second timestep
        (; observations, states) = simulate_for_test(
            Baseline(ByHand()), OutputNotEveryTimestepEnvironment(ByHand())
        )
        @test count(observation -> observation.reward isa NoValue, observations) == 25
        @test expected_rewards(states)[1] == expected_rewards(states)[2]
        #A first observation that is missing
        (; observations) = simulate_for_test(
            Baseline(ByHand()), NoOutputBeforeFirstInputEnvironment(ByHand())
        )
        @test observations[1] === (; reward=NoValue())
        @test observations[2].reward isa Bool
        #A state that changes from an integer to a Float64
        (; states) = simulate_for_test(
            StateChangesType(ByHand()),
            FixedObservationsEnvironment((; reward=[0.0, 0.0, 0.0]));
            who_starts=AgentStarts(),
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test expected_rewards(states) == [1, 0.5, 0.25]
    end

    @testset "Models and environments can leave things out" begin
        #A model without observations, with an environment without a state: a map passes none of the observations on
        simulation = simulate_for_test(
            WithoutInputs(ByHand()),
            WithoutStatesEnvironment(ByHand());
            observation_map=NameMap(),
        )
        @test eltype(simulation.observations) === typeof((;))
        @test eltype(simulation.environment.states) === typeof((;))
        #The environment still gave the observations the model didn't receive
        @test eltype(simulation.environment.observations) === @NamedTuple{reward::Bool}
        #A model without a state
        simulation = simulate_for_test(
            WithoutStates(ByHand()),
            BaselineEnvironment(ByHand());
            cognitive_parameters=(; inverse_temperature=2.0),
        )
        @test eltype(simulation.states) === typeof((;))
        #A model without actions, with an environment which takes none
        simulation = simulate_for_test(
            WithoutOutputs(),
            WithoutInputsEnvironment(ByHand());
            cognitive_parameters=LEARNING,
        )
        @test eltype(simulation.actions) === typeof((;))
        @test eltype(simulation.states) <: NamedTuple
        @test eltype(simulation.environment.actions) === typeof((;))
    end

    @testset "A single action keeps the name the model gave it" begin
        (; actions) = simulate_for_test(Baseline(ByHand()), BaselineEnvironment(ByHand()))
        @test actions isa Vector{@NamedTuple{choice::Bool}}
    end

    @testset "Each of several named actions is sampled" begin
        #ManyAttributes chooses, rates its confidence and reports the last reward. With the means of the distributions, the
        #first are a probability of a half, the expected reward of a half with a bonus of a quarter, and a reward of a half
        (; actions, action_distributions) = simulate_for_test(
            ManyAttributes(Means()),
            ManyAttributesEnvironment(Means());
            cognitive_parameters=(; BASELINE_PARAMETERS..., noise=0.2),
            environment_parameters=(; REWARDS..., noise=0.2, bonus_size=0.5),
        )
        @test actions[1] === (; choice=0.5, confidence=0.75, report=0.5)
        #Each action is the value of what it was sampled from
        @test actions == [
            map(distribution -> distribution.value, distributions) for
            distributions in action_distributions
        ]
    end

    @testset "NoValue and a NamedTuple inside the actions are sampled on their own" begin
        (; actions) = simulate_for_test(
            ChoosesInGroups(),
            WithoutInputsEnvironment(ByHand());
            cognitive_parameters=(;),
            observation_map=NameMap(),
            action_map=NameMap(),
        )
        @test all(
            action ->
                action ===
                (; choice=true, skipped=NoValue(), movement=(; left=false, right=true)),
            actions,
        )
        @test simulate_timestep!(InstantiatedModel(ChoosesInGroups(), (;)), (;)) ===
            (; choice=true, skipped=NoValue(), movement=(; left=false, right=true))
    end

    @testset "Each observation is sampled from what the environment gives, and a Deterministic is its value" begin
        #BaselineEnvironment gives something to sample the reward from
        (; observations) = simulate_for_test(
            Baseline(ByHand()), BaselineEnvironment(ByHand())
        )
        @test observations isa Vector{@NamedTuple{reward::Bool}}
        @test Set(observation.reward for observation in observations) == Set([false, true])
        #WithoutInputsEnvironment rewards every timestep with a reward probability of 1, and takes no actions
        (; observations) = simulate_for_test(
            Baseline(ByHand()),
            WithoutInputsEnvironment(ByHand());
            environment_parameters=(; reward_probability=1.0),
            action_map=NameMap(),
        )
        @test all(observation -> observation === (; reward=true), observations)
        #FixedObservationsEnvironment gives each reward in a Deterministic
        (; observations) = simulate_for_test(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 0.0, 0.25]));
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test observations == [(; reward=1.0), (; reward=0.0), (; reward=0.25)]
        #One that is sampled and one that isn't, in the same observation, which the environment's history has both of as its outputs
        #OtherNamesEnvironment samples its payoff, and gives the number of the timestep in a Deterministic
        environment = InstantiatedModel(
            OtherNamesEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=50,
            rng=Xoshiro(1),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        @test [
            observation.timestep_number for
            observation in environment.history.outputs[2:end]
        ] == 1:50
        @test Set(
            observation.payoff for observation in environment.history.outputs[2:end]
        ) == Set([false, true])
    end

    @testset "NoValue and a NamedTuple inside the observations are sampled on their own" begin
        environment = InstantiatedModel(
            ObservesInGroups(), (;); save_history=SaveHistory(SaveEverything())
        )
        simulate_session!(
            InstantiatedModel(WithoutOutputs(), LEARNING),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            observation_map=NameMap(:reward),
        )
        sampled = (; reward=1.0, skipped=NoValue(), cues=(; left=0.0, right=2))
        @test all(
            observation -> observation === sampled, environment.history.outputs[2:end]
        )
        @test environment.previous_output === sampled
    end

    @testset "A number, a string or a symbol that isn't wrapped gives an informative error" begin
        for value in (1.0, 1, true, "left", :left)
            @test_throws "A model gave $(repr(value)), of type $(typeof(value)), as what to sample an output from, but it is a value. An output that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))." simulate_for_test(
                Baseline(ByHand()), GivesUnwrapped(value); environment_parameters=(;)
            )
            @test_throws "A model gave $(repr(value)), of type $(typeof(value)), as what to sample an output from, but it is a value. An output that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))." simulate_for_test(
                ChoosesUnwrapped(value),
                BaselineEnvironment(ByHand());
                cognitive_parameters=(;),
            )
            @test_throws ArgumentError simulate_timestep!(
                InstantiatedModel(ChoosesUnwrapped(value), (;)), (; reward=1.0)
            )
        end
        #Wrapped, the same values are what is observed and done
        (; observations, actions) = simulate_for_test(
            ChoosesUnwrapped(Deterministic(:left)),
            GivesUnwrapped(Deterministic(1));
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
        )
        @test observations == fill((; reward=1), 2) && actions == fill((; choice=:left), 2)
        #A collection is still something to sample from, so one that isn't wrapped gives one of its elements
        (; observations) = simulate_for_test(
            ChoosesUnwrapped(Deterministic(:left)),
            GivesUnwrapped([2.0, 2.0]);
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
        )
        @test observations == fill((; reward=2.0), 2)
    end

    @testset "The environment receives the observation that was sampled, on the next timestep" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            simulation = simulate_for_test(
                Baseline(ByHand()),
                RemembersReward();
                who_starts,
                environment_parameters=(;),
            )
            environment_states = simulation.environment.states
            @test environment_states[1] === (; previous_reward=BeforeFirstTimestep())
            @test [state.previous_reward for state in environment_states[2:end]] == [
                observation.reward for observation in simulation.observations[1:(end - 1)]
            ]
        end
    end

    @testset "cognitive_save_history and environment_save_history select what is saved of each kind" begin
        #DependsOnPreviousOutput counts its timesteps beside its expected reward
        run_with(; keywords...) = simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 0.0, 1.0]));
            n_steps=3,
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            environment_parameters=(;),
            action_map=NameMap(),
            keywords...,
        )
        #The helper saves everything, which selecting everything also does
        everything = run_with()
        @test run_with(;
            cognitive_save_history=SaveHistory(
                SaveEverything(); save_initial_timestep_values=false
            ),
            environment_save_history=SaveHistory(
                SaveEverything(); save_initial_timestep_values=false
            ),
        ) == everything
        @test everything.states == [
            (; expected_reward=0.75, n_timesteps=1),
            (; expected_reward=0.375, n_timesteps=2),
            (; expected_reward=0.6875, n_timesteps=3),
        ]
        #Names keep those entries, and a kind that isn't given keeps nothing
        simulation = run_with(;
            cognitive_save_history=SaveHistory(;
                states=:n_timesteps, save_initial_timestep_values=false
            ),
        )
        @test simulation.observations == fill((;), 3) &&
            eltype(simulation.observations) === typeof((;))
        @test simulation.actions == fill((;), 3)
        @test simulation.states == [(; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        @test simulation.environment == everything.environment
        #Several names keep the entries in the order given
        (; states) = run_with(;
            cognitive_save_history=SaveHistory(;
                states=(:n_timesteps, :expected_reward), save_initial_timestep_values=false
            ),
        )
        @test states[1] === (; n_timesteps=1, expected_reward=0.75)
        #The environment's selection only affects what is returned under environment, in its own names: its outputs are the
        #observations and its inputs the actions, which BaselineEnvironment receives
        with_actions(; keywords...) = simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            n_steps=3,
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            keywords...,
        )
        everything_with_actions = with_actions()
        simulation = with_actions(;
            environment_save_history=SaveHistory(;
                outputs=:reward, inputs=:choice, save_initial_timestep_values=false
            ),
        )
        @test cognitive_values(simulation) == cognitive_values(everything_with_actions)
        @test simulation.environment.states == fill((;), 3)
        @test simulation.environment.observations ==
            everything_with_actions.environment.observations
        @test simulation.environment.actions == everything_with_actions.environment.actions
        @test simulation.environment.observation_distributions == fill((;), 3)
        #Nothing at all gives empty NamedTuples of everything, followed by the size of the timesteps
        nothing_saved = run_with(;
            cognitive_save_history=SaveHistory(; save_initial_timestep_values=false),
            environment_save_history=SaveHistory(; save_initial_timestep_values=false),
        )
        @test all(==(fill((;), 3)), saved_vectors(nothing_saved))
        @test nothing_saved.timestep_sizes === 1
    end

    @testset "Nothing is saved by default, for timestep 0 and each timestep after it" begin
        @test all(
            ==(fill((;), 4)),
            saved_vectors(
                simulate_session(
                    Baseline(ByHand()),
                    BaselineEnvironment(ByHand());
                    who_starts=EnvironmentStarts(),
                    n_steps=3,
                    cognitive_parameters=BASELINE_PARAMETERS,
                    environment_parameters=REWARDS,
                    rng=Xoshiro(1),
                ),
            ),
        )
    end

    @testset "Timestep 0 is saved by default, with the previous observation and action and the states before the first timestep" begin
        everything = SaveHistory(SaveEverything())
        for who_starts in (EnvironmentStarts(), AgentStarts())
            with_timestep_zero = simulate_for_test(
                DependsOnPreviousOutput(ByHand()),
                BaselineEnvironment(ByHand());
                who_starts,
                n_steps=3,
                cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
                cognitive_save_history=everything,
                environment_save_history=everything,
            )
            @test with_timestep_zero.observations[1] === (; reward=BeforeFirstTimestep())
            @test with_timestep_zero.actions[1] === (; choice=BeforeFirstTimestep())
            @test with_timestep_zero.states[1] === (; expected_reward=0.5, n_timesteps=0)
            @test with_timestep_zero.environment.states[1] === (; n_timesteps=0)
            #The environment's observation and action before the first timestep, and its observation distributions, which are
            #BeforeFirstTimestep since none was given yet
            @test with_timestep_zero.environment.observations[1] ===
                (; reward=BeforeFirstTimestep())
            @test with_timestep_zero.environment.actions[1] ===
                (; choice=BeforeFirstTimestep())
            @test with_timestep_zero.environment.observation_distributions[1] ===
                (; reward=BeforeFirstTimestep())
            #After it come the timesteps, as without it
            without_timestep_zero = simulate_for_test(
                DependsOnPreviousOutput(ByHand()),
                BaselineEnvironment(ByHand());
                who_starts,
                n_steps=3,
                cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            )
            for (with_vector, without_vector) in
                zip(saved_vectors(with_timestep_zero), saved_vectors(without_timestep_zero))
                @test isequal(with_vector[2:end], without_vector)
            end
        end
        #What is given as previous is what timestep 0 has, and when it has the types of the later timesteps, the vectors have them only
        simulation = simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            n_steps=2,
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            previous_observation=(; reward=true),
            previous_action=(; choice=true),
            previous_cognitive_state=(; expected_reward=0.1, n_timesteps=5),
            cognitive_save_history=everything,
            environment_save_history=everything,
        )
        @test (
            simulation.observations[1],
            simulation.actions[1],
            simulation.states[1],
            simulation.environment.states[1],
        ) === (
            (; reward=true),
            (; choice=true),
            (; expected_reward=0.1, n_timesteps=5),
            (; n_timesteps=0),
        )
        @test simulation.observations isa Vector{@NamedTuple{reward::Bool}} &&
            simulation.actions isa Vector{@NamedTuple{choice::Bool}}
        @test simulation.environment.observations[1] === (; reward=true) &&
            simulation.environment.actions[1] === (; choice=true)
        #With maps, the observation and the action have the cognitive model's names, and under environment the environment's
        simulation = simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            n_steps=2,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            cognitive_save_history=everything,
            environment_save_history=everything,
        )
        @test simulation.observations[1] === (; reward=BeforeFirstTimestep()) &&
            simulation.actions[1] === (; choice=BeforeFirstTimestep())
        @test simulation.environment.observations[1] ===
              (; payoff=BeforeFirstTimestep(), timestep_number=BeforeFirstTimestep()) &&
            simulation.environment.actions[1] === (; lever=BeforeFirstTimestep())
        #A selection applies to it too, and cognitive_save_history decides for the environment's vectors as well, so that all of
        #them have the same length
        simulation = simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            n_steps=2,
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            cognitive_save_history=SaveHistory(; states=:n_timesteps),
        )
        @test simulation.states == [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        @test simulation.environment.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        @test all(vector -> length(vector) == 3, saved_vectors(simulation))
        #The model is told who starts, and that nobody is known to when it runs on a single input
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test simulate_for_test(
                RecordsWhoStarts(),
                BaselineEnvironment(ByHand());
                who_starts,
                n_steps=2,
                cognitive_parameters=(;),
                cognitive_save_history=SaveHistory(; states=:who_starts),
            ).states[end] === (; who_starts)
            @test simulate_for_test(
                RecordsWhoStarts(),
                EnvironmentRecordsWhoStarts();
                who_starts,
                n_steps=2,
                cognitive_parameters=(;),
                environment_parameters=(;),
                cognitive_save_history=SaveHistory(; states=:who_starts),
            ).environment.states[end] === (; who_starts)
        end
        @test simulate_timestep(RecordsWhoStarts(), (; reward=1.0); parameters=(;))[2] ===
            (; who_starts=UnknownWhoStarts())
        #The action distributions are saved when selected, with BeforeFirstTimestep for every action at timestep 0
        fixed_rewards = FixedObservationsEnvironment((; reward=[1.0, 0.0, 1.0]))
        (; actions, action_distributions) = simulate_for_test(
            Baseline(ByHand()),
            fixed_rewards;
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
            cognitive_save_history=SaveHistory(;
                outputs=:choice, output_distributions=:choice
            ),
        )
        @test action_distributions[1] === (; choice=BeforeFirstTimestep())
        @test [
            distribution.choice.probability for distribution in action_distributions[2:end]
        ] ≈ baseline_choice_probability.([0.75, 0.375, 0.6875])
        @test length(action_distributions) == length(actions) == 4
        @test simulate_for_test(
            Baseline(ByHand()),
            fixed_rewards;
            n_steps=3,
            environment_parameters=(;),
            action_map=NameMap(),
            cognitive_save_history=SaveHistory(; outputs=:choice),
        ).action_distributions == fill((;), 4)
        @test_throws "The SaveHistory selects the output distributions (:bet,), but the outputs in the attributes of a cognitive model of type Baseline{ByHand} are (:choice,). A history can only select names in the model's attributes." InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(; output_distributions=:bet),
        )
        #A state changed in place is saved before the first timestep changes it
        (; states) = simulate_for_test(
            StateChangedInPlace(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 1.0]));
            n_steps=2,
            environment_parameters=(;),
            action_map=NameMap(),
            cognitive_save_history=everything,
        )
        @test [state.expected_reward[1] for state in states] == [0.5, 0.75, 0.875]
    end

    @testset "The environment's observations, actions and observation distributions are returned as it gave and received them" begin
        everything = SaveHistory(SaveEverything())
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #OtherNamesEnvironment, with the means of its distributions, gives the payoff for each lever exactly
            simulation = simulate_for_test(
                Baseline(ByHand()),
                OtherNamesEnvironment(Means());
                who_starts,
                n_steps=3,
                observation_map=PAYOFF_AS_REWARD,
                action_map=CHOICE_AS_LEVER,
                cognitive_save_history=everything,
                environment_save_history=everything,
            )
            environment = simulation.environment
            #The observations are as the environment gave them, before the observation_map, with the number of the timestep the
            #model didn't receive
            @test all(
                observation -> keys(observation) == (:payoff, :timestep_number),
                environment.observations,
            )
            @test isequal(
                [observation.payoff for observation in environment.observations],
                [observation.reward for observation in simulation.observations],
            )
            @test [
                observation.timestep_number for
                observation in environment.observations[2:end]
            ] == 1:3
            #The actions are as the environment received them, after the action_map
            @test all(action -> keys(action) == (:lever,), environment.actions)
            @test isequal(
                [action.lever for action in environment.actions],
                [action.choice for action in simulation.actions],
            )
            #The observation distributions are what the environment gave its observations to be sampled from
            @test all(
                distribution -> keys(distribution) == (:payoff, :timestep_number),
                environment.observation_distributions,
            )
            @test [
                distribution.timestep_number for
                distribution in environment.observation_distributions[2:end]
            ] == Deterministic.(1:3)
            for (observation, distribution) in zip(
                environment.observations[2:end],
                environment.observation_distributions[2:end],
            )
                @test distribution.payoff === Deterministic(observation.payoff)
            end
            #At timestep 0 every observation distribution is BeforeFirstTimestep
            @test environment.observation_distributions[1] ===
                (; payoff=BeforeFirstTimestep(), timestep_number=BeforeFirstTimestep())
            #The cognitive model's action distributions are what it gave its actions to be sampled from
            @test [
                distribution.choice.probability for
                distribution in simulation.action_distributions[2:end]
            ] == baseline_choice_probability.(expected_rewards(simulation.states[2:end]))
            #When the environment starts, it receives the previous action, so its first payoff is the one for no lever, a half
            if who_starts === EnvironmentStarts()
                @test environment.observation_distributions[2] ===
                    (; payoff=Deterministic(0.5), timestep_number=Deterministic(1))
            else
                @test environment.observation_distributions[2] === (;
                    payoff=Deterministic(reward_for_choice(simulation.actions[2].choice)),
                    timestep_number=Deterministic(1),
                )
            end
        end
        #The environment's selection applies to them, in its own names
        simulation = simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(Means());
            n_steps=3,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            environment_save_history=SaveHistory(;
                outputs=:timestep_number, save_initial_timestep_values=false
            ),
        )
        @test simulation.environment.observations ==
            [(; timestep_number=1), (; timestep_number=2), (; timestep_number=3)]
        @test simulation.environment.actions == fill((;), 3)
        @test simulation.environment.observation_distributions == fill((;), 3)
        @test simulation.environment.states == fill((;), 3)
        simulation = simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(Means());
            n_steps=3,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
            environment_save_history=SaveHistory(;
                inputs=:lever,
                output_distributions=:payoff,
                save_initial_timestep_values=false,
            ),
        )
        @test simulation.environment.observations == fill((;), 3)
        @test [action.lever for action in simulation.environment.actions] == [action.choice for action in simulation.actions]
        @test all(
            distribution -> keys(distribution) == (:payoff,),
            simulation.environment.observation_distributions,
        )
    end

    @testset "An entry that is NoValue is saved like any other, and a model without actions saves empty NamedTuples" begin
        (; observations, actions) = simulate_for_test(
            Baseline(ByHand()),
            NoOutputBeforeFirstInputEnvironment(Means());
            n_steps=3,
            cognitive_save_history=SaveHistory(;
                inputs=:reward, outputs=:choice, save_initial_timestep_values=false
            ),
        )
        @test observations[1] === (; reward=NoValue())
        @test observations[2] == (; reward=reward_for_choice(actions[1].choice))
        for cognitive_save_history in
            (SaveHistory(; outputs=(), save_initial_timestep_values=false), EVERY_TIMESTEP)
            (; actions) = simulate_for_test(
                WithoutOutputs(),
                WithoutInputsEnvironment(ByHand());
                n_steps=3,
                cognitive_parameters=LEARNING,
                cognitive_save_history,
            )
            @test actions == fill((;), 3)
        end
    end

    @testset "The output before the first timestep has every output as BeforeFirstTimestep" begin
        #For a cognitive model, the outputs are its actions
        @test CognitiveModeling.output_before_first_timestep(Baseline(ByHand())) ===
            (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.output_before_first_timestep(WithoutOutputs()) === (;)
        #The names are in the type, so it is inferred
        @test @inferred(
            CognitiveModeling.output_before_first_timestep(ManyAttributes(ByHand()))
        ) === (;
            choice=BeforeFirstTimestep(),
            confidence=BeforeFirstTimestep(),
            report=BeforeFirstTimestep(),
        )
        @test InstantiatedModel(
            ManyAttributes(ByHand()), (; BASELINE_PARAMETERS..., noise=0.2)
        ).previous_output === (;
            choice=BeforeFirstTimestep(),
            confidence=BeforeFirstTimestep(),
            report=BeforeFirstTimestep(),
        )
        #The environment receives it on the first timestep when it starts, and NoOutputBeforeFirstInputEnvironment gives no
        #reward for it
        (; observations) = simulate_for_test(
            Baseline(ByHand()), NoOutputBeforeFirstInputEnvironment(ByHand()); n_steps=2
        )
        @test observations[1] === (; reward=NoValue())
        #For an environment model, the outputs are the observations it gives
        @test CognitiveModeling.output_before_first_timestep(
            BaselineEnvironment(ByHand())
        ) === (; reward=BeforeFirstTimestep())
        @test CognitiveModeling.output_before_first_timestep(ObservesTimestepSize()) ===
            (; timestep_size=BeforeFirstTimestep())
        @test @inferred(
            CognitiveModeling.output_before_first_timestep(OtherNamesEnvironment(ByHand()))
        ) === (; payoff=BeforeFirstTimestep(), timestep_number=BeforeFirstTimestep())
        @test InstantiatedModel(OtherNamesEnvironment(ByHand()), REWARDS).previous_output ===
            (; payoff=BeforeFirstTimestep(), timestep_number=BeforeFirstTimestep())
    end

    @testset "The input before the first timestep has every input as BeforeFirstTimestep" begin
        #For a cognitive model, the inputs are its observations
        @test CognitiveModeling.input_before_first_timestep(Baseline(ByHand())) ===
            (; reward=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(WithoutInputs(ByHand())) === (;)
        #The names are in the type, so it is inferred
        @test @inferred(
            CognitiveModeling.input_before_first_timestep(ManyAttributes(ByHand()))
        ) === (;
            reward=BeforeFirstTimestep(),
            cue=BeforeFirstTimestep(),
            bonus=BeforeFirstTimestep(),
        )
        #The model receives it on the first timestep when it starts, so it learns nothing there
        (; states) = simulate_for_test(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 1.0]));
            who_starts=AgentStarts(),
            n_steps=2,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test expected_rewards(states) == [0.5, 0.75]
        #For an environment model, the inputs are the actions it receives, which it can have none of
        @test CognitiveModeling.input_before_first_timestep(
            BaselineEnvironment(ByHand())
        ) === (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(
            OtherNamesEnvironment(ByHand())
        ) === (; lever=BeforeFirstTimestep())
        @test CognitiveModeling.input_before_first_timestep(
            WithoutInputsEnvironment(ByHand())
        ) === (;)
    end

    @testset "The markers for nothing there share a supertype, and the one for before the first timestep is its own" begin
        @test NoValue() isa AbstractNoValue && BeforeFirstTimestep() isa AbstractNoValue
        @test !(BeforeFirstTimestep() isa NoValue)
        @test !(MissingValue() isa AbstractNoValue)
    end

    @testset "A state isn't prepared for saving when none of it is saved, also at timestep 0" begin
        (; states) = simulate_for_test(
            LearnsInPlace(state -> error("prepared anyway")),
            BaselineEnvironment(ByHand());
            cognitive_parameters=LEARNING,
            cognitive_save_history=SaveHistory(; inputs=SaveEverything()),
        )
        @test states == fill((;), 51)
    end

    @testset "Selecting a name that isn't there, or from something that isn't a NamedTuple, gives an error" begin
        @test_throws "The SaveHistory selects the states (:surprise,), but the states in the attributes of a cognitive model of type DependsOnPreviousOutput{ByHand} are (:expected_reward, :n_timesteps)." simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            cognitive_save_history=SaveHistory(; states=:surprise),
        )
        @test_throws "The SaveHistory selects the inputs (:cue,), but the inputs in the attributes of a cognitive model of type DependsOnPreviousOutput{ByHand} are (:reward,)." simulate_for_test(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            cognitive_save_history=SaveHistory(; inputs=:cue),
        )
        @test_throws "can only select names from a NamedTuple" simulate_for_test(
            LearnsInPlace(state -> state.values),
            BaselineEnvironment(ByHand());
            cognitive_parameters=LEARNING,
            cognitive_save_history=SaveHistory(; states=:values),
        )
    end

    @testset "A selection keeps the return type known and the loop free of allocations" begin
        rng = Xoshiro(1)
        run_simulation(cognitive_save_history, n_steps) = simulate_session(
            DependsOnPreviousOutput(ByHand()),
            BaselineEnvironment(ByHand());
            who_starts=EnvironmentStarts(),
            n_steps,
            cognitive_parameters=DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history,
            environment_save_history=SaveHistory(),
        )
        for cognitive_save_history in (
            SaveHistory(; states=:n_timesteps),
            SaveHistory(; inputs=:reward, states=(:expected_reward, :n_timesteps)),
            SaveHistory(),
        )
            @test @inferred(run_simulation(cognitive_save_history, 10)) isa
                SessionTrajectories
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                cognitive_save_history, 10
            )
        end
        #Saving only the integer state adds 8 bytes for each timestep
        run_simulation(SaveHistory(; states=:n_timesteps), 10)
        extra_bytes =
            @allocated(run_simulation(SaveHistory(; states=:n_timesteps), 20_000)) -
            @allocated(run_simulation(SaveHistory(; states=:n_timesteps), 10_000))
        @test extra_bytes <= 9 * 10_000
    end

    @testset "A method which copies the state keeps the saved values of a state changed in place" begin
        #StateChangedInPlaceSavedWithDeepcopy copies its state with deepcopy, and StateChangedInPlace copies the vector in it
        for model in
            (StateChangedInPlaceSavedWithDeepcopy(ByHand()), StateChangedInPlace(ByHand()))
            (; states) = simulate_for_test(model, BaselineEnvironment(ByHand()))
            @test length(unique(state.expected_reward[1] for state in states)) > 1
            @test allunique(objectid(state.expected_reward) for state in states)
        end
    end

    @testset "A method which does nothing copies nothing" begin
        (; states) = simulate_for_test(
            LearnsInPlace(identity),
            BaselineEnvironment(ByHand());
            cognitive_parameters=LEARNING,
        )
        @test all(state -> state.values === first(states).values, states)
    end

    @testset "An environment can also have a method which copies its state" begin
        simulation = simulate_for_test(
            Baseline(ByHand()), CountsInPlace(); environment_parameters=(;)
        )
        environment_states = simulation.environment.states
        @test [state.counts[1] for state in environment_states] == 1:length(environment_states)
    end

    @testset "What prepare_state_for_saving returns is what is saved" begin
        (; states) = simulate_for_test(
            StateChangedInPlace(ByHand()), BaselineEnvironment(ByHand())
        )
        @test eltype(states) === @NamedTuple{expected_reward::Vector{Float64}}
        @test length(unique(states)) > 1
    end

    @testset "There must be at least one timestep" begin
        @test length(
            simulate_for_test(Baseline(ByHand()), BaselineEnvironment(ByHand()); n_steps=1).actions,
        ) == 1
        @test_throws ArgumentError simulate_for_test(
            Baseline(ByHand()), BaselineEnvironment(ByHand()); n_steps=0
        )
    end

    @testset "A missing timestep method gives an informative error" begin
        @test_throws "There is no timestep_model method for a cognitive model of type OnlyInitializesState with a context of type " simulate_for_test(
            OnlyInitializesState(), BaselineEnvironment(ByHand()); cognitive_parameters=(;)
        )
        @test_throws "Each model must define CognitiveModeling.timestep_model(model::OnlyInitializesState, context::CognitiveModeling.ModelContext)." simulate_for_test(
            OnlyInitializesState(), BaselineEnvironment(ByHand()); cognitive_parameters=(;)
        )
        @test_throws "There is no timestep_model method for an environment model of type EnvironmentOnlyInitializesState" simulate_for_test(
            Baseline(ByHand()),
            EnvironmentOnlyInitializesState();
            environment_parameters=(;),
        )
    end

    @testset "A missing method for the initial state gives an informative error, unless a state is given" begin
        @test_throws "There is no initialize_state method for a cognitive model of type NeedsGivenState. Each model must define CognitiveModeling.initialize_state(model::NeedsGivenState, parameters), which returns (;) if the model has no state." simulate_for_test(
            NeedsGivenState(), BaselineEnvironment(ByHand()); cognitive_parameters=(;)
        )
        @test_throws "There is no initialize_state method for an environment model of type EnvironmentNeedsGivenState" simulate_for_test(
            Baseline(ByHand()), EnvironmentNeedsGivenState(); environment_parameters=(;)
        )
        simulation = simulate_session(
            NeedsGivenState(),
            EnvironmentNeedsGivenState();
            who_starts=EnvironmentStarts(),
            n_steps=2,
            cognitive_parameters=(;),
            environment_parameters=(;),
            previous_cognitive_state=(; value=1.0),
            previous_environment_state=(; value=2.0),
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        #The given states are those of timestep 0
        @test simulation.states == fill((; value=1.0), 3)
        @test simulation.environment.states == fill((; value=2.0), 3)
    end

    @testset "A timestep method which returns the wrong type gives an informative error" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test_throws "The timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput, but it returned a value of type Tuple" simulate_for_test(
                ReturnsTuple(),
                BaselineEnvironment(ByHand());
                who_starts,
                cognitive_parameters=(;),
            )
            @test_throws "The timestep_model method of an environment model of type ReturnsObservation must return a CognitiveModeling.ModelOutput, but it returned a value of type @NamedTuple" simulate_for_test(
                Baseline(ByHand()),
                ReturnsObservation();
                who_starts,
                environment_parameters=(;),
            )
        end
    end

    @testset "The model and the environment both get the size of each timestep" begin
        observed_sizes(observations) =
            [observation.timestep_size for observation in observations]
        received_sizes(states) = [state.received_timestep_size for state in states]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            run_with_sizes(; keywords...) = simulate_session(
                SavesTimestepSize(),
                ObservesTimestepSize();
                who_starts,
                n_steps=3,
                cognitive_parameters=(;),
                environment_parameters=(;),
                rng=Xoshiro(1),
                cognitive_save_history=EVERY_TIMESTEP,
                environment_save_history=EVERY_TIMESTEP,
                keywords...,
            )
            #The default is the integer 1
            (; observations, states) = run_with_sizes()
            @test observed_sizes(observations) == [1, 1, 1] &&
                eltype(observed_sizes(observations)) === Int
            @test received_sizes(states) == [1, 1, 1]
            #A single number is the size of every timestep
            (; observations, states) = run_with_sizes(timestep_sizes=0.5)
            @test observed_sizes(observations) == [0.5, 0.5, 0.5]
            @test received_sizes(states) == [0.5, 0.5, 0.5]
            #A vector has one size for each timestep
            (; observations, states) = run_with_sizes(timestep_sizes=[0.5, 2.0, 0.25])
            @test observed_sizes(observations) == [0.5, 2.0, 0.25]
            @test received_sizes(states) == [0.5, 2.0, 0.25]
            #Any vector can be used, like a range
            (; observations) = run_with_sizes(timestep_sizes=1:3)
            @test observed_sizes(observations) == [1, 2, 3]
            #The type of a timestep size isn't restricted
            (; observations) = run_with_sizes(timestep_sizes=(; fast=0.1, slow=10.0))
            @test observed_sizes(observations) == fill((; fast=0.1, slow=10.0), 3)
            #A vector with the wrong length isn't accepted
            @test_throws "it has 2 sizes and n_steps is 3" run_with_sizes(
                timestep_sizes=[0.5, 2.0]
            )
        end
    end

    @testset "Timestep sizes keep the return type known" begin
        #UsesTimestepSize learns more on a timestep the more time it covers
        rng = Xoshiro(1)
        run_simulation(timestep_sizes) = simulate_session(
            UsesTimestepSize(ByHand()),
            BaselineEnvironment(ByHand());
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        )
        for timestep_sizes in (1, 0.5, [0.5, 2.0, 0.25], 1:3)
            @test @inferred(run_simulation(timestep_sizes)) isa SessionTrajectories
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                timestep_sizes
            )
        end
    end

    @testset "The sizes of the timesteps are returned as they were given" begin
        run_with_sizes(timestep_sizes; keywords...) = simulate_for_test(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            n_steps=3,
            timestep_sizes,
            keywords...,
        ).timestep_sizes
        #One size for every timestep is returned as it is, whatever it is
        @test run_with_sizes(1) === 1
        @test run_with_sizes(0.5) === 0.5
        @test run_with_sizes((; fast=0.1, slow=10.0)) === (; fast=0.1, slow=10.0)
        #A vector has the size of each timestep
        @test run_with_sizes([0.5, 2.0, 0.25]) == [0.5, 2.0, 0.25]
        @test run_with_sizes(1:3) == 1:3
        #They are returned whatever is saved
        @test run_with_sizes(
            0.5;
            cognitive_save_history=SaveHistory(),
            environment_save_history=SaveHistory(),
        ) === 0.5
        #With timestep 0, a vector starts with BeforeFirstTimestep for it, so that it has an element for each element of the others, and one size stays as it is
        @test isequal(
            run_with_sizes(
                [0.5, 2.0, 0.25]; cognitive_save_history=SaveHistory(SaveEverything())
            ),
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25],
        )
        @test run_with_sizes(
            [0.5, 2.0, 0.25]; cognitive_save_history=SaveHistory(SaveEverything())
        ) isa Vector{Union{BeforeFirstTimestep,Float64}}
        @test run_with_sizes(0.5; cognitive_save_history=SaveHistory(SaveEverything())) ===
            0.5
    end

    @testset "An environment's stopping criterion ends the simulation, and n_steps is the most timesteps there can be" begin
        @test CognitiveModeling.stopping_criterion(
            BaselineEnvironment(ByHand()), (; n_timesteps=0), REWARDS
        ) === false
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #WithStoppingCriterionEnvironment ends after the timestep that is its last, a parameter
            run_until_end(n_steps; timestep_sizes=1, keywords...) = simulate_for_test(
                Baseline(ByHand()),
                WithStoppingCriterionEnvironment(ByHand());
                who_starts,
                n_steps,
                timestep_sizes,
                environment_parameters=(; REWARDS..., last_timestep=7),
                keywords...,
            )
            #Everything returned has the timesteps that ran
            simulation = run_until_end(20)
            @test all(vector -> length(vector) == 7, saved_vectors(simulation))
            @test simulation.environment.states[end] == (; n_timesteps=7)
            @test simulation.timestep_sizes === 1
            #The same as a simulation of that many timesteps
            @test run_until_end(20) == run_until_end(7)
            #Fewer timesteps than it takes to end run in full
            @test length(run_until_end(5).observations) == 5
            #A vector of sizes has one for each timestep there can be, and those of the timesteps that ran are returned
            @test run_until_end(20; timestep_sizes=collect(0.5:0.5:10.0)).timestep_sizes ==
                collect(0.5:0.5:3.5)
            #With timestep 0, everything has it too
            with_timestep_zero = run_until_end(
                20;
                timestep_sizes=collect(0.5:0.5:10.0),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )
            @test all(vector -> length(vector) == 8, saved_vectors(with_timestep_zero))
            @test length(with_timestep_zero.timestep_sizes) == 8
            @test with_timestep_zero.environment.states[1] == (; n_timesteps=0) &&
                with_timestep_zero.environment.states[end] == (; n_timesteps=7)
            #The return type is still known
            rng = Xoshiro(1)
            typed_run(n_steps) = simulate_session(
                Baseline(ByHand()),
                WithStoppingCriterionEnvironment(ByHand());
                who_starts,
                n_steps,
                cognitive_parameters=BASELINE_PARAMETERS,
                environment_parameters=(; REWARDS..., last_timestep=7),
                rng,
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )
            @test @inferred(typed_run(20)) isa SessionTrajectories
            JET.@test_opt target_modules = (CognitiveModeling,) typed_run(20)
        end
        #When the environment starts, the model has observed and acted on the last timestep, after which a
        #FixedObservationsEnvironment ends
        simulation = simulate_for_test(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 1.0]));
            n_steps=20,
            environment_parameters=(;),
            action_map=NameMap(),
        )
        @test expected_rewards(simulation.states) == [0.75, 0.875] &&
            length(simulation.actions) == 2
        #An environment that has ended already can't be simulated
        @test_throws "The stopping criterion of an environment model of type WithStoppingCriterionEnvironment{ByHand} is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate_for_test(
            Baseline(ByHand()),
            WithStoppingCriterionEnvironment(ByHand());
            environment_parameters=(; REWARDS..., last_timestep=7),
            previous_environment_state=(; n_timesteps=7),
        )
    end

    @testset "A cognitive model's stopping criterion ends the simulation too, and check_stopping_criterion = false runs every timestep" begin
        @test CognitiveModeling.stopping_criterion(
            Baseline(ByHand()), (; expected_reward=0.5), BASELINE_PARAMETERS
        ) === false
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #WithStoppingCriterion ends once its expected reward is above 0.8. Rewards of 1 take it to 1 - 0.5^(n + 1) after n of
            #them, which it learns from on the same timestep when the environment starts, and on the next when the agent starts
            expected_reward_after(t) =
                1 - 0.5^(who_starts isa EnvironmentStarts ? t + 1 : t)
            run_until_end(; keywords...) = simulate_for_test(
                WithStoppingCriterion(ByHand()),
                FixedObservationsEnvironment((; reward=fill(1.0, 20)));
                who_starts,
                n_steps=20,
                environment_parameters=(;),
                action_map=NameMap(),
                keywords...,
            )
            #The model ends the simulation after the timestep that takes its expected reward to 0.875, the second when the
            #environment starts and the third when the agent starts
            n_timesteps_that_run = who_starts isa EnvironmentStarts ? 2 : 3
            simulation = run_until_end()
            @test all(
                vector -> length(vector) == n_timesteps_that_run, saved_vectors(simulation)
            )
            @test simulation.states[end] == (; expected_reward=0.875)
            #Without the check, every timestep runs
            every_timestep = run_until_end(; check_stopping_criterion=false)
            @test all(vector -> length(vector) == 20, saved_vectors(every_timestep))
            @test expected_rewards(every_timestep.states) == expected_reward_after.(1:20)
            @test every_timestep.timestep_sizes === 1
            #The same goes for an environment which would end the simulation
            environment_ends = simulate_for_test(
                Baseline(ByHand()),
                WithStoppingCriterionEnvironment(ByHand());
                who_starts,
                n_steps=20,
                environment_parameters=(; REWARDS..., last_timestep=7),
                check_stopping_criterion=false,
            )
            @test all(vector -> length(vector) == 20, saved_vectors(environment_ends))
            @test environment_ends.environment.states[end] == (; n_timesteps=20)
            #A model whose criterion is met already can't run a timestep, unless the check is off
            @test_throws "The stopping criterion of a cognitive model of type WithStoppingCriterion{ByHand} is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." run_until_end(
                previous_cognitive_state=(; expected_reward=0.9)
            )
            @test length(
                run_until_end(;
                    previous_cognitive_state=(; expected_reward=0.9),
                    check_stopping_criterion=false,
                ).actions,
            ) == 20
            @test length(
                simulate_for_test(
                    Baseline(ByHand()),
                    WithStoppingCriterionEnvironment(ByHand());
                    who_starts,
                    n_steps=20,
                    environment_parameters=(; REWARDS..., last_timestep=7),
                    previous_environment_state=(; n_timesteps=7),
                    check_stopping_criterion=false,
                ).actions,
            ) == 20
        end
        #Only a Bool is taken
        @test_throws TypeError simulate_for_test(
            Baseline(ByHand()), BaselineEnvironment(ByHand()); check_stopping_criterion=1
        )
    end
end

@testset "simulate_session!" begin
    @testset "Two simulations in a row give the same as one long one" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate_session(
                Baseline(ByHand()),
                NoOutputBeforeFirstInputEnvironment(ByHand());
                who_starts,
                n_steps=60,
                cognitive_parameters=BASELINE_PARAMETERS,
                environment_parameters=REWARDS,
                rng=Xoshiro(9),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )

            agent = InstantiatedModel(
                Baseline(ByHand()),
                BASELINE_PARAMETERS;
                save_history=SaveHistory(SaveEverything()),
            )
            environment = InstantiatedModel(
                NoOutputBeforeFirstInputEnvironment(ByHand()),
                REWARDS;
                save_history=SaveHistory(SaveEverything()),
            )
            rng = Xoshiro(9)
            first_part = simulate_session!(agent, environment; who_starts, n_steps=25, rng)
            second_part = simulate_session!(agent, environment; who_starts, n_steps=35, rng)

            #simulate_session! returns no timestep 0, which the one simulation starts with, and which the histories have
            #from where they started
            for (first_vector, second_vector, whole_vector) in zip(
                saved_vectors(first_part),
                saved_vectors(second_part),
                saved_vectors(one_simulation),
            )
                @test isequal(vcat(first_vector, second_vector), whole_vector[2:end])
            end
            @test isequal(agent.history.inputs, one_simulation.observations)
            @test isequal(agent.history.outputs, one_simulation.actions)
            @test isequal(agent.history.states, one_simulation.states)
            @test isequal(
                agent.history.output_distributions, one_simulation.action_distributions
            )
            @test isequal(environment.history.states, one_simulation.environment.states)
            @test isequal(
                environment.history.outputs, one_simulation.environment.observations
            )
            @test isequal(environment.history.inputs, one_simulation.environment.actions)
            @test isequal(
                environment.history.output_distributions,
                one_simulation.environment.observation_distributions,
            )
            #The size of the timesteps is returned as it was given
            @test first_part.timestep_sizes === 1 &&
                second_part.timestep_sizes === 1 &&
                one_simulation.timestep_sizes === 1
            @test agent.state == one_simulation.states[end]
            @test agent.previous_output == one_simulation.actions[end]
            @test environment.state == one_simulation.environment.states[end]
            @test environment.previous_output == one_simulation.observations[end]
        end
    end

    @testset "Timestep sizes in two simulations in a row give the same as in one long one" begin
        #UsesTimestepSize learns more on a timestep the more time it covers
        sizes = [0.5, 2.0, 0.25, 1.5, 3.0]
        for who_starts in (EnvironmentStarts(), AgentStarts())
            one_simulation = simulate_session(
                UsesTimestepSize(ByHand()),
                BaselineEnvironment(ByHand());
                who_starts,
                n_steps=5,
                timestep_sizes=sizes,
                cognitive_parameters=BASELINE_PARAMETERS,
                environment_parameters=REWARDS,
                rng=Xoshiro(9),
                cognitive_save_history=SaveHistory(SaveEverything()),
                environment_save_history=SaveHistory(SaveEverything()),
            )

            agent = InstantiatedModel(
                UsesTimestepSize(ByHand()),
                BASELINE_PARAMETERS;
                save_history=SaveHistory(SaveEverything()),
            )
            environment = InstantiatedModel(
                BaselineEnvironment(ByHand()),
                REWARDS;
                save_history=SaveHistory(SaveEverything()),
            )
            rng = Xoshiro(9)
            first_part = simulate_session!(
                agent, environment; who_starts, n_steps=2, timestep_sizes=sizes[1:2], rng
            )
            second_part = simulate_session!(
                agent, environment; who_starts, n_steps=3, timestep_sizes=sizes[3:5], rng
            )

            #The vectors, and the sizes of the timesteps that follow them, after timestep 0, which the histories have too
            for (first_vector, second_vector, whole_vector) in zip(
                saved_vectors(first_part),
                saved_vectors(second_part),
                saved_vectors(one_simulation),
            )
                @test isequal(vcat(first_vector, second_vector), whole_vector[2:end])
            end
            @test isequal(
                vcat(first_part.timestep_sizes, second_part.timestep_sizes),
                one_simulation.timestep_sizes[2:end],
            )
            @test isequal(agent.history.timestep_sizes, one_simulation.timestep_sizes)
            @test isequal(environment.history.timestep_sizes, one_simulation.timestep_sizes)
        end
    end

    @testset "Later simulations don't change the outputs of earlier ones" begin
        agent = InstantiatedModel(
            StateChangedInPlaceSavedWithDeepcopy(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        (; states) = simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        last_value = states[end].expected_reward[1]
        simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        @test states[end].expected_reward[1] == last_value
    end

    @testset "simulate_session! continues from the state itself, whatever is saved" begin
        agent = InstantiatedModel(
            StateChangedInPlace(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        first_states = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
        ).states
        @test agent.state.expected_reward[1] == first_states[end].expected_reward[1]
        second_states = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
        ).states
        @test agent.state.expected_reward[1] == second_states[end].expected_reward[1]
    end

    @testset "An agent fills in the default parameters, and starts from them" begin
        #HasDefaults starts from a parameter with a default, which no example model does
        agent = InstantiatedModel(HasDefaults(), (;))
        @test agent.parameters == (; learning_rate=0.1, initial_value=0.5)
        @test agent.state == (; expected_reward=0.5)
        agent = InstantiatedModel(HasDefaults(), (; initial_value=0.9))
        @test agent.parameters == (; learning_rate=0.1, initial_value=0.9)
        @test agent.state == (; expected_reward=0.9)
        @test InstantiatedModel(
            HasDefaults(),
            (; learning_rate=0.3),
            (; expected_reward=0.2),
            (; choice=NoValue()),
        ).parameters == (; learning_rate=0.3, initial_value=0.5)
        #A model without defaults keeps the parameters as they are
        @test InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS).parameters ===
            BASELINE_PARAMETERS
        #The model is kept as it is
        @test InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS).model ===
            Baseline(ByHand())
    end

    @testset "An agent checks the names of the parameters and the state against attribute_names" begin
        #WithDefaultParameters has a default inverse temperature, which is filled in before the names are checked
        @test_throws "The parameters are named (:inverse_temperature, :noise), but the parameters in the attributes of a cognitive model of type WithDefaultParameters{ByHand} are (:learning_rate, :inverse_temperature)" InstantiatedModel(
            WithDefaultParameters(ByHand()), (; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type WithDefaultParameters{ByHand} are (:expected_reward,)" InstantiatedModel(
            WithDefaultParameters(ByHand()), LEARNING, (; value=0.5), (; choice=NoValue())
        )
        @test_throws "The previous output is NoValue(), but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type WithDefaultParameters{ByHand}, (:choice,)." InstantiatedModel(
            WithDefaultParameters(ByHand()), LEARNING, (; expected_reward=0.5), NoValue()
        )
        @test_throws "must be a NamedTuple with the outputs in the attributes of" InstantiatedModel(
            WithDefaultParameters(ByHand()), LEARNING, (; expected_reward=0.5), (; bet=true)
        )
        #The order of the names doesn't matter
        @test InstantiatedModel(
            WithDefaultParameters(ByHand()), (; inverse_temperature=3.0, learning_rate=0.1)
        ).parameters.learning_rate == 0.1
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of a cognitive model of type WithDefaultParameters{ByHand}, (:expected_reward,), also when it is a single object." InstantiatedModel(
            WithDefaultParameters(ByHand()), LEARNING, [0.5], (; choice=NoValue())
        )
        #A model without names can't be simulated or be instantiated
        @test_throws "There is no attribute_names method for a cognitive model of type WithoutNames. Each model must define CognitiveModeling.attribute_names(model::WithoutNames), which returns a ModelAttributes with the names of its parameters, states, inputs and outputs." InstantiatedModel(
            WithoutNames(), (;)
        )
        @test_throws "There is no attribute_names method for a cognitive model of type WithoutNames" simulate_for_test(
            WithoutNames(), BaselineEnvironment(ByHand()); cognitive_parameters=(;)
        )
    end

    @testset "An agent and an environment need a method for the initial state, unless a state is given" begin
        @test_throws "There is no initialize_state method for a cognitive model of type NeedsGivenState" InstantiatedModel(
            NeedsGivenState(), (;)
        )
        @test_throws "There is no initialize_state method for an environment model of type EnvironmentNeedsGivenState" InstantiatedModel(
            EnvironmentNeedsGivenState(), (;)
        )
        @test InstantiatedModel(
            NeedsGivenState(), (;), (; value=1.0), (; choice=NoValue())
        ).state == (; value=1.0)
        @test InstantiatedModel(
            EnvironmentNeedsGivenState(), (;), (; value=2.0), (; reward=NoValue())
        ).state == (; value=2.0)
    end

    @testset "An environment fills in the default parameters, and checks the names like an agent" begin
        environment = InstantiatedModel(RewardsWithDefaults(), (;))
        @test environment.parameters == (; reward_size=1.0, bias=0.0)
        @test InstantiatedModel(RewardsWithDefaults(), (; bias=0.5)).parameters ==
            (; reward_size=1.0, bias=0.5)
        @test InstantiatedModel(
            RewardsWithDefaults(), (; bias=0.5), (; last_choice=true), (; reward=1.0)
        ).parameters == (; reward_size=1.0, bias=0.5)
        @test InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS).parameters ===
            REWARDS
        @test_throws "The parameters are named (:reward_size, :bias, :noise), but the parameters in the attributes of an environment model of type RewardsWithDefaults are (:reward_size, :bias)" InstantiatedModel(
            RewardsWithDefaults(), (; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of an environment model of type RewardsWithDefaults are (:last_choice,)" InstantiatedModel(
            RewardsWithDefaults(), (;), (; value=0.5), (; reward=NoValue())
        )
        @test_throws "The previous output is NoValue(), but it must be a NamedTuple with the outputs in the attributes of an environment model of type RewardsWithDefaults, (:reward,)." InstantiatedModel(
            RewardsWithDefaults(), (;), (; last_choice=NoValue()), NoValue()
        )
        @test_throws "must be a NamedTuple with the outputs in the attributes of" InstantiatedModel(
            RewardsWithDefaults(), (;), (; last_choice=NoValue()), (; cue=1)
        )
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of an environment model of type RewardsWithDefaults, (:last_choice,), also when it is a single object." InstantiatedModel(
            RewardsWithDefaults(), (;), [0.5], (; reward=NoValue())
        )
        #An environment model without names can't be simulated or be instantiated
        @test_throws "There is no attribute_names method for an environment model of type EnvironmentWithoutNames. Each model must define CognitiveModeling.attribute_names(model::EnvironmentWithoutNames)" InstantiatedModel(
            EnvironmentWithoutNames(), (;)
        )
        @test_throws "There is no attribute_names method for an environment model of type EnvironmentWithoutNames" simulate_for_test(
            Baseline(ByHand()), EnvironmentWithoutNames(); environment_parameters=(;)
        )
    end

    @testset "simulate_session fills in the default parameters, and checks the names like an agent" begin
        run_with(model, environment; keywords...) =
            simulate_for_test(model, environment; n_steps=2, keywords...)
        #The defaults are used, also for the initial state, and the given values replace them. HasDefaults starts from a
        #parameter with a default, which no example model does
        (; states) = run_with(
            HasDefaults(), BaselineEnvironment(ByHand()); cognitive_parameters=(;)
        )
        @test states[1].expected_reward != 0.5
        (; states) = run_with(
            HasDefaults(),
            BaselineEnvironment(ByHand());
            cognitive_parameters=(; learning_rate=0.0, initial_value=0.9),
        )
        @test [state.expected_reward for state in states] == [0.9, 0.9]
        simulation = run_with(
            Baseline(ByHand()), RewardsWithDefaults(); environment_parameters=(; bias=0.5)
        )
        @test all(observation -> observation.reward >= 0.5, simulation.observations)
        @test simulation.environment.states[end].last_choice isa Bool
        #The names of the parameters, the states, the previous action and the previous observation are checked
        @test_throws "The parameters are named (:inverse_temperature, :noise), but the parameters in the attributes of a cognitive model of type WithDefaultParameters{ByHand}" run_with(
            WithDefaultParameters(ByHand()),
            BaselineEnvironment(ByHand());
            cognitive_parameters=(; noise=0.1),
        )
        @test_throws "The parameters are named (:reward_probability, :noise), but the parameters in the attributes of an environment model of type BaselineEnvironment{ByHand}" run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            environment_parameters=(; reward_probability=0.8, noise=0.1),
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type Baseline{ByHand}" run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            previous_cognitive_state=(; value=0.5),
        )
        @test_throws "The state is named (:value,), but the states in the attributes of an environment model of type BaselineEnvironment{ByHand}" run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            previous_environment_state=(; value=0),
        )
        @test_throws "The previous output is (bet = true,), but it must be a NamedTuple with the outputs in the attributes of a cognitive model of type Baseline{ByHand}, (:choice,)." run_with(
            Baseline(ByHand()), BaselineEnvironment(ByHand()); previous_action=(; bet=true)
        )
        @test_throws "The previous output is (cue = 1,), but it must be a NamedTuple with the outputs in the attributes of an environment model of type BaselineEnvironment{ByHand}, (:reward,)." run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            previous_observation=(; cue=1),
        )
        @test_throws "must be a NamedTuple with the outputs" run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            previous_observation=NoValue(),
        )
        #The order of the names doesn't matter
        @test run_with(
            Baseline(ByHand()),
            BaselineEnvironment(ByHand());
            cognitive_parameters=(; inverse_temperature=2.0, learning_rate=0.5),
        ) isa SessionTrajectories
        #A state must be a NamedTuple, also when it is a single object
        @test_throws "The state has type Vector{Float64}, but it must be a NamedTuple with the states in the attributes of a cognitive model of type NeedsGivenState, (:value,), also when it is a single object." run_with(
            NeedsGivenState(),
            BaselineEnvironment(ByHand());
            cognitive_parameters=(;),
            previous_cognitive_state=[1.0],
        )
    end

    @testset "The checks of the names are decided at compile time, so they cost nothing" begin
        #Each compiles to code which only returns, however many names there are
        @test folds(
            CognitiveModeling.check_names_fit, Baseline{ByHand}, BaselineEnvironment{ByHand}
        )
        @test folds(
            CognitiveModeling.check_match_attribute_names,
            DependsOnPreviousOutput{ByHand},
            typeof(DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS),
            typeof((; expected_reward=0.5, n_timesteps=0)),
            typeof((; choice=NoValue())),
        )
        @test folds(
            CognitiveModeling.check_match_attribute_names,
            OtherNamesEnvironment{ByHand},
            typeof(REWARDS),
            typeof((; n_timesteps=0)),
            typeof((; payoff=NoValue(), timestep_number=NoValue())),
        )
        @test folds(
            CognitiveModeling.check_input_names,
            Baseline{ByHand},
            typeof((; reward=1.0, cue=2)),
        )
        @test folds(() -> CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED))
        @test CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES_REVERSED)
        @test !CognitiveModeling.same_names(MANY_NAMES, MANY_NAMES[1:64])
        @test CognitiveModeling.has_every_name((:reward,), (:cue, :reward)) &&
            !CognitiveModeling.has_every_name((:reward, :cue), (:reward,))
    end

    @testset "Without maps, the model and the environment must have the same observations and actions in their attributes" begin
        #An environment which gives something else, or more, than the inputs in the model's attributes
        @test_throws "The attributes of a cognitive model of type Baseline{ByHand} have the observations (:reward,) as inputs, but an environment model of type OnlyCues gives (:cue,). Without a map, the two must have the same observations in their attributes. A NameMap as observation_map can select and rename them." simulate_for_test(
            Baseline(ByHand()), OnlyCues(); environment_parameters=(;)
        )
        @test_throws "The attributes of a cognitive model of type Baseline{ByHand} have the observations (:reward,) as inputs, but an environment model of type ManyAttributesEnvironment{ByHand} gives (:reward, :cue, :bonus). Without a map, the two must have the same observations in their attributes." simulate_for_test(
            Baseline(ByHand()),
            ManyAttributesEnvironment(ByHand());
            environment_parameters=(; REWARDS..., noise=0.2, bonus_size=0.5),
        )
        #A model with fewer observations in its attributes than the environment gives
        @test_throws "The attributes of a cognitive model of type WithoutInputs{ByHand} have the observations () as inputs, but an environment model of type BaselineEnvironment{ByHand} gives (:reward,)." simulate_for_test(
            WithoutInputs(ByHand()), BaselineEnvironment(ByHand())
        )
        #A model which gives fewer actions than the inputs in the environment's attributes, or more
        @test_throws "A cognitive model of type WithoutOutputs gives the actions (), but the attributes of an environment model of type BaselineEnvironment{ByHand} have (:choice,) as inputs. Without a map, the two must have the same actions in their attributes. A NameMap as action_map can select and rename them." simulate_for_test(
            WithoutOutputs(), BaselineEnvironment(ByHand()); cognitive_parameters=LEARNING
        )
        @test_throws "A cognitive model of type Baseline{ByHand} gives the actions (:choice,), but the attributes of an environment model of type WithoutInputsEnvironment{ByHand} have () as inputs. Without a map, the two must have the same actions in their attributes." simulate_for_test(
            Baseline(ByHand()), WithoutInputsEnvironment(ByHand())
        )
        #The same names in another order fit, here of three of the actions of EveryFeature, which leaves out its response and
        #its side
        @test CognitiveModeling.check_names_fit(
            EveryFeature(ByHand()),
            EveryFeatureEnvironment(ByHand()),
            nothing,
            NameMap(:option, :rating, :choice),
        ) === nothing
        #simulate_session! checks it too
        @test_throws "Without a map, the two must have the same observations in their attributes." simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            InstantiatedModel(OnlyCues(), (;));
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test_throws "Without a map, the two must have the same actions in their attributes." simulate_session!(
            InstantiatedModel(WithoutOutputs(), LEARNING),
            InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS);
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            InstantiatedModel(OtherNamesEnvironment(ByHand()), REWARDS);
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        ) isa SessionTrajectories
    end

    @testset "Maps of names let a model and an environment with other names be used together" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            #The same simulation as with an environment which uses the model's names, for the cognitive model's part:
            #OtherNamesEnvironment is BaselineEnvironment under other names
            same_names = simulate_for_test(
                Baseline(ByHand()),
                BaselineEnvironment(ByHand());
                who_starts,
                rng=Xoshiro(5),
            )
            mapped = simulate_for_test(
                Baseline(ByHand()),
                OtherNamesEnvironment(ByHand());
                who_starts,
                rng=Xoshiro(5),
                observation_map=PAYOFF_AS_REWARD,
                action_map=CHOICE_AS_LEVER,
            )
            @test isequal(cognitive_values(mapped), cognitive_values(same_names))
            #The observations are returned as the model received them, without the number of the timestep, and the actions as
            #it gave them
            @test keys(mapped.observations[end]) == (:reward,)
            @test keys(mapped.actions[end]) == (:choice,)
            #Under environment, they are as the environment gave and received them
            @test keys(mapped.environment.observations[end]) == (:payoff, :timestep_number)
            @test keys(mapped.environment.actions[end]) == (:lever,)
            @test isequal(
                [observation.payoff for observation in mapped.environment.observations],
                [observation.reward for observation in same_names.environment.observations],
            )
            @test [action.lever for action in mapped.environment.actions] == [action.choice for action in same_names.environment.actions]
        end
        #Without the maps, the two don't fit
        @test_throws "Without a map, the two must have the same observations in their attributes. A NameMap as observation_map can select and rename them." simulate_for_test(
            Baseline(ByHand()), OtherNamesEnvironment(ByHand())
        )
        @test_throws "Without a map, the two must have the same actions in their attributes. A NameMap as action_map can select and rename them." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=PAYOFF_AS_REWARD,
        )
    end

    @testset "A map selects what the receiver gets" begin
        #The model receives only the observations the map takes, and a map without names passes no actions on
        simulation = simulate_for_test(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=[1.0, 0.0], cue=[1, 2]));
            n_steps=2,
            environment_parameters=(;),
            observation_map=NameMap(:reward),
            action_map=NameMap(),
        )
        @test simulation.observations == [(; reward=1.0), (; reward=0.0)]
        @test simulation.environment.observations ==
            [(; reward=1.0, cue=1), (; reward=0.0, cue=2)]
        @test simulation.environment.actions == fill((;), 2)
        #An environment receives only the actions the map takes, here the second of two, which the means of the
        #distributions of NoOutputBeforeFirstInputEnvironment reward exactly
        (; observations, actions) = simulate_for_test(
            TwoChoices(),
            NoOutputBeforeFirstInputEnvironment(Means());
            n_steps=3,
            cognitive_parameters=(;),
            action_map=NameMap(:second_choice => :choice),
        )
        @test [observation.reward for observation in observations] == [NoValue(), 0.8, 0.8]
        @test actions == fill((; first_choice=false, second_choice=true), 3)
        simulation = simulate_for_test(
            TwoChoices(),
            NoOutputBeforeFirstInputEnvironment(Means());
            n_steps=3,
            cognitive_parameters=(;),
            action_map=NameMap(:first_choice => :choice),
        )
        @test [observation.reward for observation in simulation.observations] == [NoValue(), 0.4, 0.4]
        @test simulation.environment.actions == fill((; choice=false), 3)
    end

    @testset "A map must take names that are given, and give exactly the names in the receiver's attributes" begin
        @test_throws "The observation_map takes the observations (:prize,), but an environment model of type OtherNamesEnvironment{ByHand} gives (:payoff, :timestep_number). A map can only take observations the environment gives." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=NameMap(:prize => :reward),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The observation_map gives the observations (:reward, :timestep_number), but the attributes of a cognitive model of type Baseline{ByHand} have (:reward,) as inputs. A map must give exactly the observations in the cognitive model's attributes." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=NameMap(:payoff => :reward, :timestep_number),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The observation_map gives the observations (:timestep_number,), but the attributes of a cognitive model of type Baseline{ByHand} have (:reward,) as inputs." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=NameMap(:timestep_number),
            action_map=CHOICE_AS_LEVER,
        )
        @test_throws "The action_map takes the actions (:press,), but a cognitive model of type Baseline{ByHand} gives (:choice,). A map can only take actions the cognitive model gives." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=PAYOFF_AS_REWARD,
            action_map=NameMap(:press => :lever),
        )
        @test_throws "The action_map gives the actions (:choice,), but the attributes of an environment model of type OtherNamesEnvironment{ByHand} have (:lever,) as inputs. A map must give exactly the actions in the environment's attributes." simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=PAYOFF_AS_REWARD,
            action_map=NameMap(:choice),
        )
        #Only a NameMap is taken
        @test_throws TypeError simulate_for_test(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            observation_map=(:payoff => :reward,),
        )
    end

    @testset "Maps keep the return type known and the loop free of allocations, and their checks cost nothing" begin
        rng = Xoshiro(1)
        run_simulation(who_starts, n_steps) = simulate_session(
            Baseline(ByHand()),
            OtherNamesEnvironment(ByHand());
            who_starts,
            n_steps,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=REWARDS,
            rng,
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        for who_starts in (EnvironmentStarts(), AgentStarts())
            @test @inferred(run_simulation(who_starts, 10)) isa SessionTrajectories
            JET.@test_opt target_modules = (CognitiveModeling,) run_simulation(
                who_starts, 10
            )
            run_simulation(who_starts, 10)
            #As without maps for the cognitive model: 8 bytes each for an action distribution and a state, 1 each for an
            #observation and an action, and a byte each for which type they have. The environment's observations and observation
            #distributions have the number of the timestep too, 16 bytes each with a byte for which type they have, its actions 2
            #bytes and its state 8
            extra_bytes =
                @allocated(run_simulation(who_starts, 20_000)) -
                @allocated(run_simulation(who_starts, 10_000))
            @test extra_bytes <= 70 * 10_000
        end
        @test folds(
            CognitiveModeling.check_names_fit,
            Baseline{ByHand},
            OtherNamesEnvironment{ByHand},
            typeof(PAYOFF_AS_REWARD),
            typeof(CHOICE_AS_LEVER),
        )
    end

    @testset "With maps, each history has the names of its own side" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(
            OtherNamesEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(SaveEverything()),
        )
        simulation = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            rng=Xoshiro(1),
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        )
        #The agent's are the ones returned as observations and actions, under the model's names
        @test agent.history.inputs[2:end] == simulation.observations
        @test agent.history.outputs[2:end] == simulation.actions
        @test length(agent.history.output_distributions) == length(agent.history.outputs) &&
            agent.history.output_distributions[1] === (; choice=BeforeFirstTimestep())
        @test all(observation -> keys(observation) == (:reward,), agent.history.inputs)
        @test all(action -> keys(action) == (:choice,), agent.history.outputs)
        #The environment's are the same values, under its own names, with the number of the timestep the model didn't receive,
        #and are the ones returned under environment
        @test all(
            observation -> keys(observation) == (:payoff, :timestep_number),
            environment.history.outputs,
        )
        @test all(action -> keys(action) == (:lever,), environment.history.inputs)
        @test isequal(
            [observation.payoff for observation in environment.history.outputs],
            [observation.reward for observation in agent.history.inputs],
        )
        @test [action.lever for action in environment.history.inputs[2:end]] == [action.choice for action in simulation.actions]
        @test environment.history.outputs[2:end] == simulation.environment.observations
        @test environment.history.inputs[2:end] == simulation.environment.actions
        @test environment.history.output_distributions[2:end] ==
            simulation.environment.observation_distributions
        @test environment.history.output_distributions[1] ===
            (; payoff=BeforeFirstTimestep(), timestep_number=BeforeFirstTimestep())
        #Where they are left for the next simulation, each side has its own names too
        @test keys(environment.previous_output) == (:payoff, :timestep_number)
        @test keys(agent.previous_output) == (:choice,)
        @test simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=PAYOFF_AS_REWARD,
            action_map=CHOICE_AS_LEVER,
        ) isa SessionTrajectories
        #simulate_session! checks the maps too
        @test_throws "A map must give exactly the observations in the cognitive model's attributes." simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            observation_map=NameMap(:timestep_number),
            action_map=CHOICE_AS_LEVER,
        )
    end

    @testset "It stops when the environment's stopping criterion is met, and the environment must then be reset" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(
            WithStoppingCriterionEnvironment(ByHand()),
            (; REWARDS..., last_timestep=4);
            save_history=SaveHistory(SaveEverything()),
        )
        #Two timesteps run in full, and the next call stops after the two that are left
        @test length(
            simulate_session!(
                agent, environment; who_starts=EnvironmentStarts(), n_steps=2
            ).observations,
        ) == 2
        simulation = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=10,
            timestep_sizes=collect(1.0:10.0),
        )
        @test all(vector -> length(vector) == 2, saved_vectors(simulation))
        @test simulation.timestep_sizes == [1.0, 2.0]
        @test environment.state == (; n_timesteps=4)
        #Both histories have the timesteps that ran, after where they started
        @test length(agent.history.states) == length(agent.history.timestep_sizes) == 5
        @test agent.history.timestep_sizes == [BeforeFirstTimestep(), 1, 1, 1.0, 2.0]
        @test environment.history.timestep_sizes == agent.history.timestep_sizes
        @test length(environment.history.states) == 5
        #It has ended, so it can't be simulated further until it is reset
        @test_throws "The stopping criterion of an environment model of type WithStoppingCriterionEnvironment{ByHand} is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=10
        )
        @test length(agent.history.states) == 5
        #Unless the check is off, in which case it runs on
        @test length(
            simulate_session!(
                agent,
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=3,
                check_stopping_criterion=false,
            ).observations,
        ) == 3
        @test environment.state == (; n_timesteps=7)
        reset!(environment)
        @test length(
            simulate_session!(
                agent, environment; who_starts=EnvironmentStarts(), n_steps=10
            ).observations,
        ) == 4
    end

    @testset "It stops when the agent's stopping criterion is met, and the agent must then be reset" begin
        #WithStoppingCriterion ends once its expected reward is above 0.8, which rewards of 1 take a learning rate of a quarter
        #to on the fourth timestep, at 0.841796875
        agent = InstantiatedModel(
            WithStoppingCriterion(ByHand()),
            (; learning_rate=0.25, inverse_temperature=2.0);
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(
            FixedObservationsEnvironment((; reward=fill(1.0, 10))),
            (;);
            save_history=SaveHistory(SaveEverything()),
        )
        simulate_steps!(n_steps) = simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps,
            action_map=NameMap(),
        )
        #Two timesteps run in full, and the next call stops after the two that are left
        @test length(simulate_steps!(2).actions) == 2
        simulation = simulate_steps!(10)
        @test all(vector -> length(vector) == 2, saved_vectors(simulation))
        @test agent.state == (; expected_reward=0.841796875)
        @test length(agent.history.states) == length(environment.history.states) == 5
        #It has ended, so it can't be simulated further until it is reset
        @test_throws "The stopping criterion of a cognitive model of type WithStoppingCriterion{ByHand} is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!." simulate_steps!(
            10
        )
        reset!(agent)
        @test length(simulate_steps!(10).actions) == 4
    end

    @testset "An error in a timestep method gives the same errors as in simulate_session" begin
        agent = InstantiatedModel(ReturnsTuple(), (;))
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        @test_throws "The timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=1
        )
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(ReturnsObservation(), (;))
        @test_throws "The timestep_model method of an environment model of type ReturnsObservation must return a CognitiveModeling.ModelOutput" simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=1
        )
    end
end

@testset "simulate_timestep!" begin
    @testset "It updates the agent and returns the sampled action" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        action = simulate_timestep!(agent, (; reward=1.0); rng=Xoshiro(1))
        @test action isa @NamedTuple{choice::Bool}
        @test agent.previous_output === action
        @test agent.state == (; expected_reward=0.75)
    end

    @testset "Several in a row give the same as a simulation with the same observations" begin
        rewards = [1.0, 0.0, 1.0, 1.0]
        one_simulation = simulate_session(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=rewards));
            who_starts=EnvironmentStarts(),
            n_steps=4,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=(;),
            action_map=NameMap(),
            rng=Xoshiro(9),
            cognitive_save_history=SaveHistory(SaveEverything()),
        )

        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        rng = Xoshiro(9)
        actions = [simulate_timestep!(agent, (; reward); rng) for reward in rewards]

        #The simulation starts with timestep 0
        @test actions == one_simulation.actions[2:end]
        @test agent.state == one_simulation.states[end]
    end

    @testset "The same seed gives the same action" begin
        action_with_seed(seed) = [
            simulate_timestep!(
                InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
                (; reward=1.0);
                rng=Xoshiro(seed),
            ) for _ in 1:20
        ]
        @test action_with_seed(3) == action_with_seed(3)
        @test action_with_seed(3) != action_with_seed(4)
    end

    @testset "An observation can be NoValue" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        simulate_timestep!(agent, (; reward=NoValue()))
        @test agent.state == (; expected_reward=0.5)
    end

    @testset "The input must be a NamedTuple with the inputs in the model's attributes, and can have more" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        @test_throws "The input is 1.0, but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type Baseline{ByHand}, (:reward,)." simulate_timestep!(
            agent, 1.0
        )
        @test_throws "must be a NamedTuple with the inputs" simulate_timestep!(
            agent, NoValue()
        )
        @test_throws "must be a NamedTuple with the inputs" simulate_timestep!(
            agent, (; cue=1.0)
        )
        @test simulate_timestep!(agent, (; reward=1.0, cue=1.0)) isa NamedTuple
        @test agent.state == (; expected_reward=0.75)
        #A model without inputs takes an empty NamedTuple
        @test simulate_timestep!(
            InstantiatedModel(WithoutInputs(ByHand()), BASELINE_PARAMETERS), (;)
        ) isa NamedTuple
        #The check costs nothing when the types are known
        JET.@test_opt target_modules = (CognitiveModeling,) CognitiveModeling.simulate_model_on_input(
            Baseline(ByHand()),
            (; reward=1.0, cue=1.0),
            (; expected_reward=0.5),
            (; choice=NoValue()),
            BASELINE_PARAMETERS,
            1,
            Xoshiro(1),
            SaveHistory(),
        )
    end

    @testset "An environment model can be run on a single input too, which is an action" begin
        #BaselineEnvironment, with the means of its distributions, gives the reward for a choice exactly
        environment = InstantiatedModel(
            BaselineEnvironment(Means()),
            REWARDS;
            save_history=SaveHistory(SaveEverything()),
        )
        observation = simulate_timestep!(environment, (; choice=true); rng=Xoshiro(1))
        @test observation === (; reward=0.8)
        @test environment.previous_output === observation
        @test environment.state === (; n_timesteps=1)
        @test environment.history.inputs ==
            [(; choice=BeforeFirstTimestep()), (; choice=true)]
        @test environment.history.outputs ==
            [(; reward=BeforeFirstTimestep()), (; reward=0.8)]
        @test environment.history.output_distributions ==
            [(; reward=BeforeFirstTimestep()), (; reward=Deterministic(0.8))]
        @test_throws "The input is (cue = 1,), but it must be a NamedTuple with the inputs in the attributes of an environment model of type BaselineEnvironment{Means}, (:choice,)." simulate_timestep!(
            environment, (; cue=1)
        )
    end

    @testset "The model gets the size of the timestep" begin
        agent = InstantiatedModel(SavesTimestepSize(), (;))
        #The default is the integer 1
        simulate_timestep!(agent, (; timestep_size=NoValue()))
        @test agent.state.received_timestep_size === 1
        simulate_timestep!(agent, (; timestep_size=NoValue()); timestep_size=0.5)
        @test agent.state.received_timestep_size === 0.5
    end

    @testset "It works without actions, and with several actions" begin
        @test simulate_timestep!(
            InstantiatedModel(WithoutOutputs(), LEARNING), (; reward=1.0)
        ) === (;)
        #ManyAttributes chooses, rates its confidence and reports the last reward, which the means of the distributions give
        #exactly
        @test simulate_timestep!(
            InstantiatedModel(
                ManyAttributes(Means()), (; BASELINE_PARAMETERS..., noise=0.2)
            ),
            (; reward=1.0, cue=0.5, bonus=0.0),
        ) === (; choice=inv(1 + exp(-0.5)), confidence=0.75, report=1.0)
    end

    @testset "An error in the timestep method gives the same errors as in simulate_session" begin
        @test_throws "The timestep_model method of a cognitive model of type ReturnsTuple must return a CognitiveModeling.ModelOutput" simulate_timestep!(
            InstantiatedModel(ReturnsTuple(), (;)), (; reward=1.0)
        )
        @test_throws "There is no timestep_model method for a cognitive model of type OnlyInitializesState" simulate_timestep!(
            InstantiatedModel(OnlyInitializesState(), (;)), (; reward=1.0)
        )
    end
end

@testset "simulate_timestep" begin
    @testset "It returns the sampled output and the new state, starting from the initial state by default" begin
        action, cognitive_state = simulate_timestep(
            Baseline(ByHand()),
            (; reward=1.0);
            parameters=BASELINE_PARAMETERS,
            rng=Xoshiro(1),
        )
        @test action isa @NamedTuple{choice::Bool}
        @test cognitive_state == (; expected_reward=0.75)
        #The same as simulate_timestep! of an agent
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        @test simulate_timestep!(agent, (; reward=1.0); rng=Xoshiro(1)) === action
        @test agent.state === cognitive_state
        #An environment model runs on an action in the same way
        @test simulate_timestep(
            BaselineEnvironment(Means()), (; choice=true); parameters=REWARDS
        ) === ((; reward=0.8), (; n_timesteps=1), true)
    end

    @testset "Passing the state and the output on gives the same as a simulation with the same observations" begin
        rewards = [1.0, 0.0, 1.0, 1.0]
        one_simulation = simulate_session(
            Baseline(ByHand()),
            FixedObservationsEnvironment((; reward=rewards));
            who_starts=EnvironmentStarts(),
            n_steps=4,
            cognitive_parameters=BASELINE_PARAMETERS,
            environment_parameters=(;),
            action_map=NameMap(),
            rng=Xoshiro(9),
            cognitive_save_history=SaveHistory(SaveEverything()),
        )

        rng = Xoshiro(9)
        action, cognitive_state = simulate_timestep(
            Baseline(ByHand()), (; reward=rewards[1]); parameters=BASELINE_PARAMETERS, rng
        )
        actions = [action]
        for reward in rewards[2:end]
            action, cognitive_state = simulate_timestep(
                Baseline(ByHand()),
                (; reward);
                parameters=BASELINE_PARAMETERS,
                previous_state=cognitive_state,
                previous_output=action,
                rng,
            )
            push!(actions, action)
        end

        #The simulation starts with timestep 0
        @test actions == one_simulation.actions[2:end]
        @test cognitive_state == one_simulation.states[end]
    end

    @testset "It fills in the default parameters, and checks the names and the input" begin
        #HasDefaults starts from a parameter with a default, which no example model does
        _, cognitive_state = simulate_timestep(
            HasDefaults(),
            (; reward=1.0);
            parameters=(; initial_value=0.9, learning_rate=0.0),
        )
        @test cognitive_state == (; expected_reward=0.9)
        @test simulate_timestep(HasDefaults(), (; reward=1.0); parameters=(;))[2].expected_reward ≈
            0.55
        @test_throws "The parameters are named (:inverse_temperature, :noise), but the parameters in the attributes of a cognitive model of type WithDefaultParameters{ByHand}" simulate_timestep(
            WithDefaultParameters(ByHand()), (; reward=1.0); parameters=(; noise=0.1)
        )
        @test_throws "The state is named (:value,), but the states in the attributes of a cognitive model of type Baseline{ByHand}" simulate_timestep(
            Baseline(ByHand()),
            (; reward=1.0);
            parameters=BASELINE_PARAMETERS,
            previous_state=(; value=0.5),
        )
        @test_throws "The previous output is (bet = true,), but it must be a NamedTuple with the outputs" simulate_timestep(
            Baseline(ByHand()),
            (; reward=1.0);
            parameters=BASELINE_PARAMETERS,
            previous_output=(; bet=true),
        )
        @test_throws "The input is 1.0, but it must be a NamedTuple with the inputs in the attributes of a cognitive model of type Baseline{ByHand}, (:reward,)." simulate_timestep(
            Baseline(ByHand()), 1.0; parameters=BASELINE_PARAMETERS
        )
        #An input can have more than the inputs in the model's attributes, and an observation can be NoValue
        @test simulate_timestep(
            Baseline(ByHand()), (; reward=1.0, cue=2); parameters=BASELINE_PARAMETERS
        )[2] == (; expected_reward=0.75)
        @test simulate_timestep(
            Baseline(ByHand()), (; reward=NoValue()); parameters=BASELINE_PARAMETERS
        )[2] == (; expected_reward=0.5)
    end

    @testset "The model gets the size of the timestep, and the same seed gives the same action" begin
        @test simulate_timestep(
            SavesTimestepSize(), (; timestep_size=NoValue()); parameters=(;)
        )[2] === (; received_timestep_size=1)
        @test simulate_timestep(
            SavesTimestepSize(),
            (; timestep_size=NoValue());
            parameters=(;),
            timestep_size=0.5,
        )[2] === (; received_timestep_size=0.5)
        actions_with_seed(seed) = [
            simulate_timestep(
                Baseline(ByHand()),
                (; reward=1.0);
                parameters=BASELINE_PARAMETERS,
                rng=Xoshiro(seed + i),
            )[1] for i in 1:20
        ]
        @test actions_with_seed(3) == actions_with_seed(3)
        @test actions_with_seed(3) != actions_with_seed(40)
    end

    @testset "The return type is known, and nothing is allocated" begin
        rng = Xoshiro(1)
        simulate_one_timestep(cognitive_state, action) = simulate_timestep(
            Baseline(ByHand()),
            (; reward=1.0);
            parameters=BASELINE_PARAMETERS,
            previous_state=cognitive_state,
            previous_output=action,
            rng,
        )
        @test @inferred(simulate_one_timestep((; expected_reward=0.5), (; choice=true))) isa
            Tuple
        JET.@test_opt target_modules = (CognitiveModeling,) simulate_one_timestep(
            (; expected_reward=0.5), (; choice=true)
        )
        @test @allocated(simulate_one_timestep((; expected_reward=0.5), (; choice=true))) ==
            0
        #The first timestep, from the defaults, is known too
        @test @inferred(
            simulate_timestep(
                Baseline(ByHand()), (; reward=1.0); parameters=BASELINE_PARAMETERS, rng
            )
        ) isa Tuple
    end
end

@testset "The history of an agent" begin
    @testset "It starts where the agent starts" begin
        history = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        ).history
        @test history isa CognitiveModeling.ModelHistory
        @test history.inputs == [(; reward=BeforeFirstTimestep())]
        @test history.outputs == [(; choice=BeforeFirstTimestep())]
        @test history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test history.states == [(; expected_reward=0.5)]
        @test history.timestep_sizes == [BeforeFirstTimestep()]
        #The vectors are untyped, since what is saved can change type between timesteps
        @test history.states isa Vector{Any} && history.timestep_sizes isa Vector{Any}
        #Also if the state and the previous output are given
        history = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS,
            (; expected_reward=0.25),
            (; choice=true);
            save_history=SaveHistory(SaveEverything()),
        ).history
        @test history.inputs == [(; reward=BeforeFirstTimestep())]
        @test history.outputs == [(; choice=true)]
        @test history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test history.states == [(; expected_reward=0.25)]
    end

    @testset "It starts empty when the initial timestep values aren't saved, and then has one element for each timestep" begin
        save_history = SaveHistory(SaveEverything(); save_initial_timestep_values=false)
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS; save_history)
        environment = InstantiatedModel(
            FixedObservationsEnvironment((; reward=[1.0, 0.0])), (;); save_history
        )
        @test isempty(agent.history.states) &&
            isempty(agent.history.timestep_sizes) &&
            isempty(environment.history.states)
        simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            action_map=NameMap(),
        )
        simulate_timestep!(agent, (; reward=1.0); timestep_size=0.5)
        @test agent.history.states == [
            (; expected_reward=0.75), (; expected_reward=0.375), (; expected_reward=0.6875)
        ]
        @test agent.history.inputs == [(; reward=1.0), (; reward=0.0), (; reward=1.0)]
        @test agent.history.timestep_sizes == [1, 1, 0.5]
        @test environment.history.states ==
            [(; n_observations_given=1), (; n_observations_given=2)]
        @test environment.history.outputs == [(; reward=1.0), (; reward=0.0)]
        @test environment.history.output_distributions ==
            [(; reward=Deterministic(1.0)), (; reward=Deterministic(0.0))]
        #reset! starts it empty again
        reset!(agent)
        @test isempty(agent.history.states) && isempty(agent.history.timestep_sizes)
    end

    @testset "Nothing is saved by default, and a SaveHistory selects what is" begin
        @test InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS).save_history ===
            SaveHistory()
        @test InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS,
            (; expected_reward=0.25),
            (; choice=NoValue()),
        ).save_history === SaveHistory()
        @test InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS).save_history ===
            SaveHistory()
        @test InstantiatedModel(
            BaselineEnvironment(ByHand()), REWARDS, (; n_timesteps=0), (; reward=NoValue())
        ).save_history === SaveHistory()
        save_history = SaveHistory(; states=:n_timesteps)
        agent = InstantiatedModel(
            DependsOnPreviousOutput(ByHand()),
            DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS;
            save_history,
        )
        @test agent.save_history === save_history
        @test InstantiatedModel(
            DependsOnPreviousOutput(ByHand()),
            DEPENDS_ON_PREVIOUS_OUTPUT_PARAMETERS,
            (; expected_reward=0.25, n_timesteps=0),
            (; choice=NoValue());
            save_history,
        ).save_history === save_history
        #The history starts with what is selected
        @test agent.history.inputs == [(;)]
        @test agent.history.outputs == [(;)]
        @test agent.history.output_distributions == [(;)]
        @test agent.history.states == [(; n_timesteps=0)]
        #It gets what is selected on a single input and with an environment, which selects its own
        simulate_timestep!(agent, (; reward=1.0))
        environment = InstantiatedModel(
            BaselineEnvironment(ByHand()), REWARDS; save_history=SaveHistory()
        )
        @test environment.save_history === SaveHistory()
        simulation = simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=2
        )
        @test agent.history.inputs == fill((;), 4)
        @test agent.history.outputs == fill((;), 4)
        @test agent.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        @test simulation.states == [(; n_timesteps=2), (; n_timesteps=3)]
        @test all(==([(;), (;)]), saved_vectors(simulation)[5:end])
        #The agent and the environment are still updated in full
        @test agent.state.n_timesteps == 3 && agent.state.expected_reward != 0.5
        @test environment.state == (; n_timesteps=2)
    end

    @testset "A history can only select names in the model's attributes" begin
        @test_throws "The SaveHistory selects the inputs (:reward,), but the inputs in the attributes of a cognitive model of type WithoutInputs{ByHand} are (). A history can only select names in the model's attributes." InstantiatedModel(
            WithoutInputs(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(; inputs=:reward),
        )
        @test_throws "The SaveHistory selects the outputs (:bet,), but the outputs in the attributes of a cognitive model of type Baseline{ByHand} are (:choice,)." InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(; outputs=:bet),
        )
        #For an environment, the inputs are the actions it receives, and the outputs the observations it gives
        @test_throws "The SaveHistory selects the inputs (:choice,), but the inputs in the attributes of an environment model of type WithoutInputsEnvironment{ByHand} are ()." InstantiatedModel(
            WithoutInputsEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(; inputs=:choice),
        )
        @test_throws "The SaveHistory selects the outputs (:cue,), but the outputs in the attributes of an environment model of type BaselineEnvironment{ByHand} are (:reward,)" InstantiatedModel(
            BaselineEnvironment(ByHand()), REWARDS; save_history=SaveHistory(; outputs=:cue)
        )
        @test_throws "The SaveHistory selects the output distributions (:cue,), but the outputs in the attributes of an environment model of type BaselineEnvironment{ByHand} are (:reward,)" InstantiatedModel(
            BaselineEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(; output_distributions=:cue),
        )
        #The states are checked too, since prepare_state_for_saving returns a state with the names in the model's attributes
        @test_throws "The SaveHistory selects the states (:first_value,), but the states in the attributes of a cognitive model of type StateChangedInPlaceSavedWithDeepcopy{ByHand}" InstantiatedModel(
            StateChangedInPlaceSavedWithDeepcopy(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(; states=:first_value),
        )
        #Everything can always be selected, and an input with more than the inputs in the model's attributes is saved in full
        agent = InstantiatedModel(
            WithoutInputs(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        simulate_timestep!(agent, (; cue=1.0))
        @test agent.history.inputs == [(;), (; cue=1.0)]
        #A selected input is kept from an input with more
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(; inputs=:reward),
        )
        simulate_timestep!(agent, (; reward=1.0, cue=1.0))
        @test agent.history.inputs == [(; reward=BeforeFirstTimestep()), (; reward=1.0)]
    end

    @testset "It adds what simulate_session! returns" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            agent = InstantiatedModel(
                Baseline(ByHand()),
                BASELINE_PARAMETERS;
                save_history=SaveHistory(SaveEverything()),
            )
            environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
            rng = Xoshiro(9)
            first_part = simulate_session!(agent, environment; who_starts, n_steps=3, rng)
            second_part = simulate_session!(agent, environment; who_starts, n_steps=2, rng)

            @test agent.history.inputs == vcat(
                [(; reward=BeforeFirstTimestep())],
                first_part.observations,
                second_part.observations,
            )
            @test agent.history.outputs == vcat(
                (; choice=BeforeFirstTimestep()), first_part.actions, second_part.actions
            )
            @test agent.history.output_distributions == vcat(
                (; choice=BeforeFirstTimestep()),
                first_part.action_distributions,
                second_part.action_distributions,
            )
            @test agent.history.states ==
                vcat((; expected_reward=0.5), first_part.states, second_part.states)
        end
    end

    @testset "It adds what happens on a single input" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        first_action = simulate_timestep!(agent, (; reward=1.0))
        second_action = simulate_timestep!(agent, (; reward=NoValue()))

        @test agent.history.inputs ==
            [(; reward=BeforeFirstTimestep()), (; reward=1.0), (; reward=NoValue())]
        @test agent.history.outputs ==
            [(; choice=BeforeFirstTimestep()), first_action, second_action]
        @test agent.history.output_distributions == [
            (; choice=BeforeFirstTimestep()),
            (; choice=ExampleModels.BernoulliByHand(baseline_choice_probability(0.75))),
            (; choice=ExampleModels.BernoulliByHand(baseline_choice_probability(0.75))),
        ]
        @test agent.history.states ==
            [(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.75)]
    end

    @testset "A single input and an environment can be used on the same agent" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        simulate_timestep!(agent, (; reward=1.0))
        simulate_session!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        simulate_timestep!(agent, (; reward=0.0))
        @test length(agent.history.inputs) ==
            length(agent.history.outputs) ==
            length(agent.history.output_distributions) ==
            length(agent.history.states) ==
            6
        @test agent.history.states[end] == agent.state
    end

    @testset "It stays empty by default" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        simulate_timestep!(agent, (; reward=1.0))
        simulation = simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3
        )
        @test isempty(agent.history.inputs)
        @test isempty(agent.history.outputs)
        @test isempty(agent.history.output_distributions)
        @test isempty(agent.history.states)
        #The agent is still updated, and the simulation still returns empty NamedTuples for each timestep
        @test agent.state != (; expected_reward=0.5)
        @test simulation.states == fill((;), 3)
        @test eltype(simulation.environment.states) === typeof((;))
    end

    @testset "It keeps the values of a state changed in place, with a method which copies it" begin
        #StateChangedInPlaceSavedWithDeepcopy copies its state with deepcopy, and StateChangedInPlace copies the vector in it
        for model in
            (StateChangedInPlaceSavedWithDeepcopy(ByHand()), StateChangedInPlace(ByHand()))
            agent = InstantiatedModel(
                model, BASELINE_PARAMETERS; save_history=SaveHistory(SaveEverything())
            )
            environment = InstantiatedModel(
                FixedObservationsEnvironment((; reward=[1.0, 0.0])), (;)
            )
            simulate_session!(
                agent,
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=2,
                action_map=NameMap(),
            )
            simulate_timestep!(agent, (; reward=1.0))
            #The initial state is copied by the same method
            @test [state.expected_reward[1] for state in agent.history.states] == [0.5, 0.75, 0.375, 0.6875]
            @test allunique(
                objectid(state.expected_reward) for state in agent.history.states
            )
        end
    end

    @testset "It saves what the method gives, also of the initial state" begin
        agent = InstantiatedModel(
            LearnsInPlace(state -> (; first_value=state.values[1])),
            LEARNING;
            save_history=SaveHistory(SaveEverything()),
        )
        simulate_timestep!(agent, (; reward=1.0))
        @test agent.history.states == [(; first_value=0.5), (; first_value=0.75)]
    end

    @testset "The initial state is the same object as the agent's, with a method which does nothing" begin
        agent = InstantiatedModel(
            LearnsInPlace(identity), LEARNING; save_history=SaveHistory(SaveEverything())
        )
        @test agent.history.states[1].values === agent.state.values
    end

    @testset "Later simulations don't change it" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        simulate_session!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        states_so_far = copy(agent.history.states)
        simulate_session!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        @test agent.history.states[1:4] == states_so_far
    end

    @testset "It has the size of each timestep, when anything is saved" begin
        agent = InstantiatedModel(
            SavesTimestepSize(), (;); save_history=SaveHistory(; outputs=:choice)
        )
        environment = InstantiatedModel(
            ObservesTimestepSize(),
            (;);
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        #A new history starts with the marker for before the first timestep, as the rest of it
        @test agent.history.timestep_sizes == [BeforeFirstTimestep()] &&
            environment.history.timestep_sizes == [BeforeFirstTimestep()]
        #A vector of sizes, and one size for every timestep
        simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes=[0.5, 2.0, 0.25],
        )
        simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=2, timestep_sizes=4
        )
        @test agent.history.timestep_sizes == [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4]
        @test environment.history.timestep_sizes ==
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4]
        #One entry for each entry of the outputs, which start with the previous output
        @test length(agent.history.outputs) == length(agent.history.timestep_sizes)
        #They are the sizes the environment gave the model as observations, its outputs
        @test [observation.timestep_size for observation in environment.history.outputs] == agent.history.timestep_sizes
        #A single input adds the size of its timestep, which is 1 by default
        simulate_timestep!(agent, (; timestep_size=NoValue()); timestep_size=1.5)
        simulate_timestep!(agent, (; timestep_size=NoValue()))
        @test agent.history.timestep_sizes ==
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25, 4, 4, 1.5, 1]
        #reset! starts from the marker again
        reset!(agent)
        @test agent.history.timestep_sizes == [BeforeFirstTimestep()]
        #Nothing is kept when nothing is saved, on either side
        agent = InstantiatedModel(SavesTimestepSize(), (;))
        environment = InstantiatedModel(
            ObservesTimestepSize(),
            (;);
            save_history=SaveHistory(; outputs=SaveEverything()),
        )
        simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            timestep_sizes=[0.5, 2.0, 0.25],
        )
        simulate_timestep!(agent, (; timestep_size=1.0); timestep_size=1.5)
        @test isempty(agent.history.timestep_sizes)
        @test environment.history.timestep_sizes == [BeforeFirstTimestep(), 0.5, 2.0, 0.25]
    end

    @testset "The environment has a history too, with the same observations and actions as its outputs and inputs" begin
        #BaselineEnvironment, with the means of its distributions, gives the reward for each choice exactly
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(
            BaselineEnvironment(Means()),
            REWARDS;
            save_history=SaveHistory(SaveEverything()),
        )
        @test environment.history isa CognitiveModeling.ModelHistory
        @test environment.history.outputs == [(; reward=BeforeFirstTimestep())]
        #It starts with the actions in its attributes as inputs, which an environment can have none of
        @test environment.history.inputs == [(; choice=BeforeFirstTimestep())]
        @test InstantiatedModel(
            WithoutInputsEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(SaveEverything()),
        ).history.inputs == [(;)]
        @test environment.history.states == [(; n_timesteps=0)]
        #Before the first timestep, it has given nothing to sample the observations from
        @test environment.history.output_distributions == [(; reward=BeforeFirstTimestep())]
        simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=3, rng=Xoshiro(1)
        )
        @test environment.history.outputs == agent.history.inputs
        @test environment.history.inputs[2:end] == agent.history.outputs[2:end]
        @test environment.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2), (; n_timesteps=3)]
        #The observation distributions are what it gave its observations to be sampled from: a reward of a half for no choice,
        #before the first, and then the reward for each choice
        @test environment.history.output_distributions == [
            (; reward=BeforeFirstTimestep()),
            (; reward=Deterministic(0.5)),
            (; reward=Deterministic(reward_for_choice(agent.history.outputs[2].choice))),
            (; reward=Deterministic(reward_for_choice(agent.history.outputs[3].choice))),
        ]
        #Its selection is its own
        environment = InstantiatedModel(
            BaselineEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(; states=:n_timesteps),
        )
        simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test environment.history.outputs == fill((;), 3)
        @test environment.history.inputs == fill((;), 3)
        @test environment.history.output_distributions == fill((;), 3)
        @test environment.history.states ==
            [(; n_timesteps=0), (; n_timesteps=1), (; n_timesteps=2)]
        #It stays empty by default
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
        )
        @test isempty(environment.history.outputs) &&
            isempty(environment.history.inputs) &&
            isempty(environment.history.output_distributions) &&
            isempty(environment.history.states)
    end
end

@testset "reset!" begin
    @testset "An agent goes back to where a new one starts, with a new history" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(BaselineEnvironment(ByHand()), REWARDS)
        first_run = simulate_session!(
            agent, environment; who_starts=EnvironmentStarts(), n_steps=5, rng=Xoshiro(2)
        )
        history = agent.history
        @test reset!(agent) === agent
        @test agent.state === (; expected_reward=0.5)
        @test agent.previous_output === (; choice=BeforeFirstTimestep())
        @test agent.parameters === BASELINE_PARAMETERS
        #The history is a new one, and the one from before is as it was
        @test agent.history !== history
        @test length(history.states) == 6
        @test agent.history.inputs == [(; reward=BeforeFirstTimestep())]
        @test agent.history.outputs == [(; choice=BeforeFirstTimestep())]
        @test agent.history.output_distributions == [(; choice=BeforeFirstTimestep())]
        @test agent.history.states == [(; expected_reward=0.5)]
        #With the environment reset too, the same simulation runs again
        reset!(environment)
        @test isequal(
            simulate_session!(
                agent,
                environment;
                who_starts=EnvironmentStarts(),
                n_steps=5,
                rng=Xoshiro(2),
            ),
            first_run,
        )
    end

    @testset "An agent keeps its parameters and its selection, and gets a new state object" begin
        agent = InstantiatedModel(
            WithDefaultParameters(ByHand()),
            (; learning_rate=0.3);
            save_history=SaveHistory(; states=SaveEverything()),
        )
        parameters = agent.parameters
        simulate_timestep!(agent, (; reward=1.0))
        reset!(agent)
        @test agent.parameters === parameters
        @test agent.save_history === SaveHistory(; states=SaveEverything())
        @test agent.history.inputs == [(;)] && length(agent.history.states) == 1
        #A state that is changed in place is made again
        agent = InstantiatedModel(
            StateChangedInPlaceSavedWithDeepcopy(ByHand()), BASELINE_PARAMETERS
        )
        expected_reward_vector = agent.state.expected_reward
        simulate_timestep!(agent, (; reward=1.0))
        reset!(agent)
        @test agent.state.expected_reward == [0.5]
        @test agent.state.expected_reward !== expected_reward_vector
        #A history that saves nothing stays empty
        @test isempty(agent.history.inputs) &&
            isempty(agent.history.outputs) &&
            isempty(agent.history.states)
    end

    @testset "An agent that was given a state of its own goes back to the initial state" begin
        agent = InstantiatedModel(
            Baseline(ByHand()),
            BASELINE_PARAMETERS,
            (; expected_reward=0.25),
            (; choice=true),
        )
        reset!(agent)
        @test agent.state === (; expected_reward=0.5)
        @test agent.previous_output === (; choice=BeforeFirstTimestep())
    end

    @testset "An environment goes back to where a new one starts, with a new history" begin
        environment = InstantiatedModel(
            FixedObservationsEnvironment((; reward=[1.0, 0.0, 1.0])),
            (;);
            save_history=SaveHistory(SaveEverything()),
        )
        simulate_session!(
            InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            action_map=NameMap(),
        )
        history = environment.history
        @test reset!(environment) === environment
        @test environment.state === (; n_observations_given=0)
        @test environment.previous_output === (; reward=BeforeFirstTimestep())
        @test environment.history !== history
        @test length(history.states) == 3
        @test environment.history.outputs == [(; reward=BeforeFirstTimestep())]
        @test environment.history.inputs == [(;)]
        @test environment.history.output_distributions == [(; reward=BeforeFirstTimestep())]
        @test environment.history.states == [(; n_observations_given=0)]
        #It can then be simulated from the start again
        (; observations) = simulate_session!(
            InstantiatedModel(
                Baseline(ByHand()),
                BASELINE_PARAMETERS;
                save_history=SaveHistory(SaveEverything()),
            ),
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=3,
            action_map=NameMap(),
        )
        @test observations == [(; reward=1.0), (; reward=0.0), (; reward=1.0)]
    end
end

@testset "An instantiated model counts its timesteps" begin
    @testset "Both simulate_session! and simulate_timestep! add to n_timesteps, whatever is saved, and reset! sets it to 0" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(
            BaselineEnvironment(ByHand()),
            REWARDS;
            save_history=SaveHistory(SaveEverything()),
        )
        @test agent.n_timesteps == environment.n_timesteps == 0
        simulate_timestep!(agent, (; reward=1.0))
        @test agent.n_timesteps == 1
        simulate_session!(agent, environment; who_starts=EnvironmentStarts(), n_steps=3)
        @test agent.n_timesteps == 4
        @test environment.n_timesteps == 3
        @test reset!(agent).n_timesteps == 0
        @test environment.n_timesteps == 3
    end

    @testset "Only the timesteps that ran are counted when a stopping criterion ends them" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(
            WithStoppingCriterionEnvironment(ByHand()), (; REWARDS..., last_timestep=4)
        )
        simulate_session!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10)
        @test agent.n_timesteps == environment.n_timesteps == 4
    end
end

@testset "A model that sets parameters_valid to false" begin
    #With a learning rate of 0.95 and a reward on every timestep, the expected reward is 0.975 after the first timestep,
    #and 0.99875 after the second, which is above what WithInvalidParameters takes
    turning_invalid = (; learning_rate=0.95, inverse_temperature=2.0)
    always_rewarded = FixedObservationsEnvironment((; reward=fill(1.0, 5)))
    simulate_turning_invalid(; keywords...) = simulate_for_test(
        WithInvalidParameters(ByHand()),
        always_rewarded;
        n_steps=5,
        cognitive_parameters=turning_invalid,
        environment_parameters=(;),
        action_map=NameMap(),
        keywords...,
    )

    @testset "simulate_session stops before that timestep, returns its states on their own, and warns" begin
        simulation = @test_logs (
            :warn,
            "A cognitive model of type WithInvalidParameters{ByHand} set parameters_valid to false on timestep 2, so the simulation stopped after timestep 1. The states of timestep 2 are under last_timestep in what the simulation returned.",
        ) simulate_turning_invalid()
        @test simulation.parameters_valid === false
        @test simulation.n_timesteps == 1
        #Timestep 1 is kept, in the vectors of both models, which leave timestep 0 out here
        @test simulation.states == [(; expected_reward=0.975)]
        @test length(simulation.actions) == 1 &&
            length(simulation.environment.observations) == 1
        #The last timestep, which turned invalid, has the states of both models
        invalid = simulation.last_timestep
        @test invalid.timestep == 2
        @test invalid.parameters_valid === false &&
            invalid.environment.parameters_valid === true
        @test invalid.previous_state == (; expected_reward=0.975)
        @test invalid.state.expected_reward ≈ 0.99875
        @test keys(invalid.environment) == (:parameters_valid, :previous_state, :state)
        #The same simulation without the warning
        quiet = @test_logs simulate_turning_invalid(; warn_on_invalid_parameters=false)
        @test quiet.n_timesteps == 1 && quiet.last_timestep.timestep == 2
        #A simulation that stays valid has the states of its last timestep, which the trajectories end with, and the same type
        staying_valid = simulate_turning_invalid(;
            cognitive_parameters=(; learning_rate=0.1, inverse_temperature=2.0)
        )
        @test staying_valid.parameters_valid === true && staying_valid.n_timesteps == 5
        @test staying_valid.last_timestep.timestep == 5 &&
            staying_valid.last_timestep.state == staying_valid.states[end]
        @test typeof(staying_valid) === typeof(simulation)
        #The sizes of the timesteps are those of the timesteps that were kept
        with_sizes = simulate_turning_invalid(;
            timestep_sizes=[1, 2, 3, 4, 5], warn_on_invalid_parameters=false
        )
        @test with_sizes.timestep_sizes == [1]
    end

    @testset "A first timestep that turns invalid leaves only timestep 0" begin
        #A learning rate of 1 takes the expected reward to 1 on the first timestep. Without timestep 0, the vectors are
        #empty, with the types of the timesteps
        simulation = simulate_turning_invalid(;
            cognitive_parameters=(; learning_rate=1.0, inverse_temperature=2.0),
            warn_on_invalid_parameters=false,
        )
        @test simulation.n_timesteps == 0 && isempty(simulation.states)
        @test eltype(simulation.states) == @NamedTuple{expected_reward::Float64}
        @test simulation.last_timestep.timestep == 1
        #With timestep 0, it is all that is kept
        with_initial = simulate_turning_invalid(;
            cognitive_parameters=(; learning_rate=1.0, inverse_temperature=2.0),
            cognitive_save_history=SaveHistory(SaveEverything()),
            warn_on_invalid_parameters=false,
        )
        @test with_initial.states == [(; expected_reward=0.5)]
    end

    @testset "The environment model can set it too, whoever starts" begin
        for who_starts in (EnvironmentStarts(), AgentStarts())
            simulation = @test_logs (
                :warn,
                "An environment model of type InvalidOnThirdTimestep set parameters_valid to false on timestep 3, so the simulation stopped after timestep 2. The states of timestep 3 are under last_timestep in what the simulation returned.",
            ) simulate_for_test(
                Baseline(ByHand()),
                InvalidOnThirdTimestep();
                who_starts,
                n_steps=5,
                environment_parameters=(;),
            )
            @test simulation.n_timesteps == 2 &&
                simulation.environment.states[end] == (; n_timesteps=2)
            @test simulation.last_timestep.parameters_valid === true &&
                simulation.last_timestep.environment.parameters_valid === false
        end
    end

    @testset "simulate_session! leaves the agent and the environment after the timestep before" begin
        agent = InstantiatedModel(
            WithInvalidParameters(ByHand()),
            turning_invalid;
            save_history=SaveHistory(SaveEverything()),
        )
        environment = InstantiatedModel(always_rewarded, (;))
        simulation = @test_logs (:warn, r"on timestep 2, so the simulation stopped") simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=5,
            action_map=NameMap(),
        )
        @test simulation.parameters_valid === false && simulation.n_timesteps == 1
        @test agent.state == (; expected_reward=0.975)
        @test agent.n_timesteps == environment.n_timesteps == 1
        @test length(agent.history.states) == 2
        @test simulation.last_timestep.state.expected_reward ≈ 0.99875
        @test_logs simulate_session!(
            agent,
            environment;
            who_starts=EnvironmentStarts(),
            n_steps=2,
            action_map=NameMap(),
            warn_on_invalid_parameters=false,
        )
    end

    @testset "simulate_timestep returns false as its third value, and warns" begin
        output, state, parameters_valid = @test_logs (
            :warn,
            "A cognitive model of type WithInvalidParameters{ByHand} set parameters_valid to false on this timestep.",
        ) simulate_timestep(
            WithInvalidParameters(ByHand()),
            (; reward=1.0);
            parameters=(; learning_rate=1.0, inverse_temperature=2.0),
        )
        @test parameters_valid === false && state == (; expected_reward=1.0)
        @test keys(output) == (:choice,)
        @test simulate_timestep(
            Baseline(ByHand()), (; reward=1.0); parameters=BASELINE_PARAMETERS
        )[3] === true
        quiet = @test_logs simulate_timestep(
            WithInvalidParameters(ByHand()),
            (; reward=1.0);
            parameters=(; learning_rate=1.0, inverse_temperature=2.0),
            warn_on_invalid_parameters=false,
        )
        @test quiet[3] === false
    end

    @testset "simulate_timestep! leaves the instantiated model as it was, and warns" begin
        agent = InstantiatedModel(
            WithInvalidParameters(ByHand()),
            (; learning_rate=1.0, inverse_temperature=2.0);
            save_history=SaveHistory(SaveEverything()),
        )
        @test_logs (
            :warn,
            "A cognitive model of type WithInvalidParameters{ByHand} set parameters_valid to false on this timestep, so the instantiated model was left as it was before the timestep.",
        ) simulate_timestep!(agent, (; reward=1.0))
        @test agent.state == (; expected_reward=0.5) &&
            agent.n_timesteps == 0 &&
            length(agent.history.states) == 1
        @test_logs simulate_timestep!(
            agent, (; reward=1.0); warn_on_invalid_parameters=false
        )
        #A reward of 0 keeps the parameters valid, and the model is updated
        simulate_timestep!(agent, (; reward=0.0))
        @test agent.state == (; expected_reward=0.0) && agent.n_timesteps == 1
    end
end
