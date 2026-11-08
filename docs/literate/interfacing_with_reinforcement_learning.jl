# # Models from ReinforcementLearning.jl as cognitive models
#
# An agent or a learner from ReinforcementLearning.jl can do the learning inside a cognitive
# model. The cognitive model keeps their object in its state, gives it what was observed on
# each timestep, and returns the probabilities of its actions, which a simulation samples
# from and fitting scores the recorded actions against.
#
# This page shows two such cognitive models. It doesn't simulate or fit them yet, and its
# code isn't run when the documentation is built, because ReinforcementLearning.jl is a
# large package to have in the documentation's environment. It was run by hand with
# ReinforcementLearningCore.jl 0.15.

using CognitiveModeling
using Distributions: Categorical
using LogExpFunctions: softmax
using ReinforcementLearningBase: ReinforcementLearningBase as RLBase
using ReinforcementLearningCore: ReinforcementLearningCore as RLCore
using ReinforcementLearningCore: Agent, QBasedPolicy, TDLearner, TabularQApproximator
using ReinforcementLearningCore: EpsilonGreedyExplorer, PreEpisodeStage, PostActStage
using ReinforcementLearningTrajectories: Trajectory, CircularArraySARTSTraces
using ReinforcementLearningTrajectories: DummySampler, InsertSampleRatioController

# Both models list the observations `observation`, `reward` and `terminated`, which are the
# names that [`CommonRLEnvironment`](@ref) gives, so they can be used with an environment
# from ReinforcementLearning.jl as they are. The observation must be the number of the state
# the environment is in, since the learners here keep a table of values.
#
# Both also need the environment to move first on each timestep, with
# [`EnvironmentStarts`](@ref), so that there is an observation to act on. On the first
# timestep of an episode there is no reward yet: the reward is then [`NoObservation`](@ref),
# and the models check for [`AbstractNoValue`](@ref), which covers that.

# ## A whole agent
#
# An agent from ReinforcementLearning.jl reads the state, the reward and the end of the
# episode from an environment object, at the stages of its own run loop. This small
# environment holds what the cognitive model observed on a timestep, for the agent to read.

struct ObservedByAgent <: RLBase.AbstractEnv
    state::Int
    reward::Float64
    terminated::Bool
    n_actions::Int
end

RLBase.state(rl_env::ObservedByAgent) = rl_env.state
RLBase.reward(rl_env::ObservedByAgent) = rl_env.reward
RLBase.is_terminated(rl_env::ObservedByAgent) = rl_env.terminated
RLBase.action_space(rl_env::ObservedByAgent) = Base.OneTo(rl_env.n_actions)

# The cognitive model creates the agent in its initial state, from its parameters: here a
# temporal difference learner with a table of action values, which explores with a
# probability of `epsilon`.

struct TemporalDifferenceAgent <: AbstractCognitiveModel
    n_states::Int
    n_actions::Int
end

function CognitiveModeling.initialize_state(model::TemporalDifferenceAgent, parameters)

    #The learner, with the learning rate and the discount factor from the parameters
    rl_approximator = TabularQApproximator(;
        n_state=model.n_states, n_action=model.n_actions
    )
    rl_learner = TDLearner(
        rl_approximator, :SARS; α=parameters.learning_rate, γ=parameters.discount
    )

    #The explorer, which picks any action with a probability of epsilon
    rl_explorer = EpsilonGreedyExplorer(parameters.epsilon; is_break_tie=true)

    #The agent's memory of the last transition, which the learner learns from
    rl_traces = CircularArraySARTSTraces(;
        capacity=1,
        state=Int64 => (),
        action=Int64 => (),
        reward=Float64 => (),
        terminal=Bool => (),
    )
    rl_trajectory = Trajectory(rl_traces, DummySampler(), InsertSampleRatioController())

    return (; rl_agent=Agent(QBasedPolicy(rl_learner, rl_explorer), rl_trajectory))
end

# On each timestep, the cognitive model takes the agent through the stage of its run loop
# that fits what was observed, and returns the probabilities its policy gives each action.
# The agent is changed in place, so the same object is passed on as the new state.

function CognitiveModeling.apply_model(model::TemporalDifferenceAgent, input::ModelInput)

    ## Unpack input ##
    rl_agent = input.previous_state.rl_agent
    (; observation, reward, terminated) = input.observation

    ## Update state ##
    if reward isa AbstractNoValue
        #An episode starts, so there is nothing to learn from yet
        rl_env = ObservedByAgent(observation, 0.0, false, model.n_actions)
        push!(rl_agent, PreEpisodeStage(), rl_env)
    else
        #Give the agent what followed its previous action, and let it learn from it
        rl_env = ObservedByAgent(observation, reward, terminated, model.n_actions)
        push!(rl_agent, PostActStage(), rl_env, input.previous_action.move)
        RLBase.optimise!(rl_agent, PostActStage())
    end

    ## Compute action probabilities ##
    #The policy gives a distribution over the actions, which the action is sampled from
    Pₜ = RLBase.prob(rl_agent.policy, rl_env)

    return ModelOutput((; move=Pₜ), (; rl_agent))
end

function CognitiveModeling.attribute_names(model::TemporalDifferenceAgent)
    return ModelAttributes(;
        parameters=(:learning_rate, :discount, :epsilon),
        states=:rl_agent,
        observations=(:observation, :reward, :terminated),
        actions=:move,
    )
end

# Since the agent is changed in place, every saved state would be the same object, showing
# the values after the last timestep. This method saves a copy of the table of action values
# on each timestep instead.

function CognitiveModeling.prepare_state_for_saving(model::TemporalDifferenceAgent, state)
    return (; action_values=copy(state.rl_agent.policy.learner.approximator.model))
end

# ## Only the learner
#
# A learner can also be used on its own, with one transition at a time, which needs no
# environment object and no trajectory. The action probabilities are then the cognitive
# model's own: here a softmax over the learner's action values, with an inverse temperature.
# The softmax is the one from LogExpFunctions.jl.

struct TemporalDifferenceLearner <: AbstractCognitiveModel
    n_states::Int
    n_actions::Int
end

function CognitiveModeling.initialize_state(model::TemporalDifferenceLearner, parameters)
    rl_approximator = TabularQApproximator(;
        n_state=model.n_states, n_action=model.n_actions
    )
    rl_learner = TDLearner(
        rl_approximator, :SARS; α=parameters.learning_rate, γ=parameters.discount
    )

    #The learner needs the observation before the action too, so that is kept in the state
    return (; rl_learner, previous_observation=0)
end

function CognitiveModeling.apply_model(model::TemporalDifferenceLearner, input::ModelInput)

    ## Unpack input ##
    (; rl_learner, previous_observation) = input.previous_state
    (; observation, reward, terminated) = input.observation
    β = input.parameters.inverse_temperature

    ## Update state ##
    #Let the learner learn from what followed the previous action, unless an episode starts
    if !(reward isa AbstractNoValue)
        rl_transition = (;
            state=previous_observation,
            next_state=observation,
            action=input.previous_action.move,
            reward,
            terminal=terminated,
        )
        RLBase.optimise!(rl_learner, rl_transition)
    end

    ## Compute action probabilities ##
    #The learner's value of each action where the agent is now
    Qₜ = RLCore.forward(rl_learner, observation)
    #Softmax over the action values
    Pₜ = Categorical(softmax(β .* Qₜ))

    return ModelOutput((; move=Pₜ), (; rl_learner, previous_observation=observation))
end

function CognitiveModeling.attribute_names(model::TemporalDifferenceLearner)
    return ModelAttributes(;
        parameters=(:learning_rate, :discount, :inverse_temperature),
        states=(:rl_learner, :previous_observation),
        observations=(:observation, :reward, :terminated),
        actions=:move,
    )
end

function CognitiveModeling.prepare_state_for_saving(model::TemporalDifferenceLearner, state)
    return (;
        action_values=copy(state.rl_learner.approximator.model), state.previous_observation
    )
end

# ## Fitting
#
# This page doesn't fit the cognitive models yet. When they are fitted with a sampler that
# uses gradients, use Enzyme.jl in reverse mode with runtime activity as the automatic
# differentiation backend, which for Turing.jl is
# `AutoEnzyme(; mode = set_runtime_activity(Reverse), function_annotation = Const)`. It gave
# the right gradient of the log density of a session for both cognitive models here, when
# checked against finite differences with Enzyme.jl 0.13. ForwardDiff.jl and ReverseDiff.jl
# can't differentiate either of them, because the learner keeps its learning rate and its
# discount in `Float64` fields, which the number types of those two don't fit in.
#
# Use the softmax from LogExpFunctions.jl, as above, and not one written by hand. Written
# in one line, a softmax can get a wrong gradient from Enzyme's reverse mode, with no error,
# which the page [Automatic differentiation and Reactant](@ref) describes under its pitfalls.
