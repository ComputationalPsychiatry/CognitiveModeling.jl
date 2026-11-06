"""
    CommonRLEnvironment(rl_env; action_name)

An environment model around `rl_env`, an environment from CommonRLInterface.jl, which needs
CommonRLInterface.jl to be loaded. `action_name` is the name of the cognitive model's action
that `act!` receives, such as `:arm`.

Environments from POMDPs.jl and ReinforcementLearning.jl can be used through it, with
`convert(CommonRLInterface.AbstractEnv, x)`, which POMDPTools.jl defines for a `POMDP` or
`MDP` and ReinforcementLearningBase.jl for its environments.

`rl_env` itself is the state: `initialize_state` resets it with CommonRLInterface.jl's
`reset!` and returns it, every timestep changes it in place with `act!` and passes it on as
the new state, and `prepare_state_for_saving` copies it, with `clone` if it provides it and
with `deepcopy` otherwise. A method of `prepare_state_for_saving` for a
`CommonRLEnvironment` with a specific type of `rl_env` can save something lighter. The
parameters aren't used, and neither is the random number generator, since an environment of
CommonRLInterface.jl has its own randomness.

Each observation is a `NamedTuple` `(; observation, reward, terminated)`: what `observe`
gives, the reward that `act!` returned, and whether the episode has ended. Before the first
action, when the action is [`BeforeFirstTimestep`](@ref), the environment is at its initial
observation, so the reward is [`NoObservation`](@ref) and nothing is done with the action.
The same goes for a timestep where the model takes no action, [`NoAction`](@ref): the
environment stays where it is. Its [`attribute_names`](@ref) are these three observations,
the one action, no parameters and no states, since `rl_env` isn't a `NamedTuple`. Use
[`EnvironmentStarts`](@ref), so that the model observes before it acts.

A simulation stops by itself when the episode ends, since the
[`stopping_criterion`](@ref) of the environment is `terminated` of `rl_env`, so `n_steps` is
the most timesteps it can take. An episode that
has ended isn't started again, and an action given to it throws an error. A new call of
[`simulate`](@ref) starts a new episode, and so does [`reset!`](@ref) of an
[`Environment`](@ref), since both reset `rl_env` through `initialize_state`.

# Examples
```jldoctest
using CognitiveModeling
using CommonRLInterface
using Random: AbstractRNG

#A two-armed bandit where the first arm always pays, in episodes of two pulls
mutable struct Bandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(rl_env::Bandit) = (rl_env.pulls = 0; nothing)
CommonRLInterface.actions(rl_env::Bandit) = 1:2
CommonRLInterface.observe(rl_env::Bandit) = rl_env.pulls
CommonRLInterface.act!(rl_env::Bandit, arm) = (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(rl_env::Bandit) = rl_env.pulls == 2

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

#The simulation stops when the episode ends, before it has run all of n_steps
observations, actions, _, _, _ = simulate(
    PullsFirstArm(),
    CommonRLEnvironment(Bandit(0); action_name = :arm);
    who_starts = EnvironmentStarts(),
    n_steps = 10,
    cognitive_parameters = (;),
    environment_parameters = (;),
    save_history = SaveHistory(SaveEverything()),
)

foreach(println, observations)

# output

(observation = 0, reward = NoObservation(), terminated = false)
(observation = 1, reward = 1.0, terminated = false)
(observation = 2, reward = 1.0, terminated = true)
```
"""
struct CommonRLEnvironment{Trl_env, action_name} <: AbstractEnvironmentModel
    rl_env::Trl_env
end

#Takes the name of the action by keyword, and keeps it in the type, so that reading the action by name costs nothing
function CommonRLEnvironment(rl_env; action_name::Symbol)
    return CommonRLEnvironment{typeof(rl_env), action_name}(rl_env)
end

#The observations are always the same three, the action is the one named, and rl_env isn't a NamedTuple, so there are no named states
function attribute_names(::CommonRLEnvironment{Trl_env, action_name}) where {Trl_env, action_name}
    return ModelAttributes(; parameters = (), states = (), observations = (:observation, :reward, :terminated), actions = action_name)
end
