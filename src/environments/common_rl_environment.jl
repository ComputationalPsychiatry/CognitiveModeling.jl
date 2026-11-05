"""
    CommonRLEnvironment(env)

An environment model around an environment from CommonRLInterface.jl, which needs
CommonRLInterface.jl to be loaded.

Environments from POMDPs.jl and ReinforcementLearning.jl can be used through it, with
`convert(CommonRLInterface.AbstractEnv, x)`, which POMDPTools.jl defines for a `POMDP` or
`MDP` and ReinforcementLearningBase.jl for its environments.

The environment object is the state: `initialize_state` resets it with `reset!` and returns
it, every timestep changes it in place with `act!` and passes it on as the new state, and
`prepare_state_for_saving` copies it, with `clone` if the environment provides it and with
`deepcopy` otherwise. A method of `prepare_state_for_saving` for a `CommonRLEnvironment`
with a specific environment type can save something lighter. The parameters aren't used,
and neither is the random number generator, since an environment of CommonRLInterface.jl has
its own randomness.

Each observation is a `NamedTuple` `(; observation, reward, terminated)`: what `observe`
gives, the reward that `act!` returned, and whether the episode has ended. Before the first
action, when the action is [`NoAction`](@ref), and on the timestep after an episode ended,
the environment is at its initial observation, with `reset!` in the second case, so the
reward is `missing` and nothing is done with the action. The model must have exactly one
action, which is given to `act!`.
Use [`EnvironmentStarts`](@ref), so that the model observes before it acts.

# Examples
```jldoctest
using CognitiveModeling
using CommonRLInterface
using Random: AbstractRNG

#A two-armed bandit where the first arm always pays, in episodes of two pulls
mutable struct Bandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(env::Bandit) = (env.pulls = 0; nothing)
CommonRLInterface.actions(env::Bandit) = 1:2
CommonRLInterface.observe(env::Bandit) = env.pulls
CommonRLInterface.act!(env::Bandit, arm) = (env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(env::Bandit) = env.pulls == 2

#An action that is always the same
struct Always{T}
    value::T
end
Base.rand(rng::AbstractRNG, action::Always) = action.value

#A model which always pulls the first arm
struct PullsFirstArm <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsFirstArm, parameters) = (;)
CognitiveModeling.apply_model(model::PullsFirstArm, input::ModelInput) = ModelOutput((; arm = Always(1)))
CognitiveModeling.attribute_names(model::PullsFirstArm) = ModelAttributes(; parameters = (), states = (), observations = (:observation, :reward, :terminated), actions = :arm)

observations, actions, _, _ = simulate(
    PullsFirstArm(),
    CommonRLEnvironment(Bandit(0));
    who_starts = EnvironmentStarts(),
    n_steps = 4,
    cognitive_parameters = (;),
    environment_parameters = (;),
    save_history = SaveHistory(SaveEverything()),
)

foreach(println, observations)

# output

(observation = 0, reward = missing, terminated = false)
(observation = 1, reward = 1.0, terminated = false)
(observation = 2, reward = 1.0, terminated = true)
(observation = 0, reward = missing, terminated = false)
```
"""
struct CommonRLEnvironment{Tenv} <: AbstractEnvironmentModel
    env::Tenv
end
