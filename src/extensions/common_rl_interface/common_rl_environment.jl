#The methods of the environment, its constructor among them, are in the CommonRLInterface.jl extension, since they need it
"""
    CommonRLEnvironment(rl_env::CommonRLInterface.AbstractEnv; action_name)

An environment model around `rl_env`, an environment from CommonRLInterface.jl, which needs
CommonRLInterface.jl to be loaded. `action_name` is the name of the cognitive model's action
that `act!` receives, such as `:arm`.

Environments from POMDPs.jl and ReinforcementLearning.jl can be used through it.
POMDPTools.jl wraps a `POMDP` as `POMDPCommonRLEnv{Any}(pomdp)` and an `MDP` as
`MDPCommonRLEnv{Any}(mdp)`, which give each observation, or each state of an `MDP`, as the
problem gives it. Its `convert(CommonRLInterface.AbstractEnv, pomdp)` gives the same
environment with each converted to an array, which a cognitive model that reads them as the
problem gives them, such as one with a belief updater of POMDPTools.jl, can't use.
ReinforcementLearningBase.jl defines `convert(CommonRLInterface.AbstractEnv, rl_env)` for
its environments.

The state is `(; rl_env)`: `initialize_state` resets `rl_env` with CommonRLInterface.jl's
`reset!` and returns it in the state, every timestep changes it in place with `act!` and
passes it on in the new state, and `prepare_state_for_saving` copies it, with `clone` if it
provides it and with `deepcopy` otherwise. Every simulation with the same
`CommonRLEnvironment` therefore resets and changes the same `rl_env`, so simulations that
run at the same time, such as on several threads, need one each. A method of
`prepare_state_for_saving` for a `CommonRLEnvironment` with a specific type of `rl_env` can
save something lighter. The parameters aren't used, and neither is the random number
generator, since an environment of CommonRLInterface.jl has its own randomness.

Each observation is a `NamedTuple` `(; observation, reward, terminated)`: what `observe`
gives, copied unless it is plain data such as a number, the reward that `act!` returned, and
whether the episode has ended. Each is given in a [`Deterministic`](@ref), so nothing more
is sampled. Before the first action, when the action is [`BeforeFirstTimestep`](@ref), the
environment is at its initial observation, so the reward is [`NoValue`](@ref) and nothing
is done with the action. The same goes for a timestep where the model takes no action,
[`NoValue`](@ref): the environment stays where it is. Its [`attribute_names`](@ref) are
these three observations, the one action, no parameters and the state `rl_env`. Use
[`EnvironmentStarts`](@ref), so that the model observes before it acts.

A simulation stops by itself when the episode ends, since the
[`stopping_criterion`](@ref) of the environment is `terminated` of `rl_env`, so `n_steps` is
the most timesteps it can take. An episode that
has ended isn't started again, and an action given to it throws an error. A new call of
[`simulate`](@ref) starts a new episode, and so does [`reset!`](@ref) of an
[`InstantiatedModel`](@ref), since both reset `rl_env` through `initialize_state`.

# Examples
```jldoctest
using CognitiveModeling
using CommonRLInterface

#A two-armed bandit where the first arm always pays, in episodes of two pulls
mutable struct Bandit <: CommonRLInterface.AbstractEnv
    pulls::Int
end
CommonRLInterface.reset!(rl_env::Bandit) = (rl_env.pulls = 0; nothing)
CommonRLInterface.actions(rl_env::Bandit) = 1:2
CommonRLInterface.observe(rl_env::Bandit) = rl_env.pulls
CommonRLInterface.act!(rl_env::Bandit, arm) = (rl_env.pulls += 1; arm == 1 ? 1.0 : 0.0)
CommonRLInterface.terminated(rl_env::Bandit) = rl_env.pulls == 2

#A model which always pulls the first arm
struct PullsFirstArm <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::PullsFirstArm, parameters) = (;)
CognitiveModeling.apply_timestep_model(model::PullsFirstArm, context::ModelContext) = ModelOutput((; arm = Deterministic(1)))
CognitiveModeling.attribute_names(model::PullsFirstArm) = ModelAttributes(; parameters = (), states = (), inputs = (:observation, :reward, :terminated), outputs = :arm)

#The simulation stops when the episode ends, before it has run all of n_steps
simulation = simulate(
    PullsFirstArm(),
    CommonRLEnvironment(Bandit(0); action_name = :arm);
    who_starts = EnvironmentStarts(),
    n_steps = 10,
    cognitive_parameters = (;),
    environment_parameters = (;),
    cognitive_save_history = SaveHistory(SaveEverything()),
)

#Timestep 0, before the first timestep, and then what the environment gave on each timestep
foreach(println, simulation.observations)

# output

(observation = BeforeFirstTimestep(), reward = BeforeFirstTimestep(), terminated = BeforeFirstTimestep())
(observation = 0, reward = NoValue(), terminated = false)
(observation = 1, reward = 1.0, terminated = false)
(observation = 2, reward = 1.0, terminated = true)
```
"""
struct CommonRLEnvironment{Trl_env,action_name} <: AbstractEnvironmentModel
    rl_env::Trl_env
end
