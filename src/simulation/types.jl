"""
    AgentHistory(observations, actions, states)

What has happened to a [`CognitiveAgent`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.actions`.

# Arguments
- `observations::Vector{Any}`: the observations, starting with [`NoObservation`](@ref).
- `actions::Vector{Any}`: the actions, starting with the agent's previous action, which is
  usually [`NoAction`](@ref).
- `states::Vector{Any}`: the states of the model, starting with its initial state. Later
  states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
"""
struct AgentHistory
    observations::Vector{Any}
    actions::Vector{Any}
    states::Vector{Any}
end

"""
    CognitiveAgent(model, parameters; save_history=true)
    CognitiveAgent(model, parameters, state, previous_action; save_history=true)

An instantiated cognitive model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

The state defaults to the output of [`initialize_state`](@ref), and the previous action to
[`NoAction`](@ref).

If `save_history` is true, `agent.history` holds an [`AgentHistory`](@ref) with where the
agent started and everything that happens to it. If it is false, the vectors in
`agent.history` stay empty.

See also [`Environment`](@ref).
"""
mutable struct CognitiveAgent{Tmodel <: AbstractCognitiveModel, Tparameters}
    model::Tmodel
    parameters::Tparameters
    #Not typed, because they can change type, for example from NoAction() to a Bool
    state::Any
    previous_action::Any
    history::AgentHistory
    save_history::Bool
end

#Starts the agent from its initial state, with no previous action
function CognitiveAgent(model::AbstractCognitiveModel, parameters; save_history::Bool = true)
    return CognitiveAgent(model, parameters, initialize_state(model, parameters), NoAction(); save_history)
end

#Starts the history where the agent starts, if it is to be saved
function CognitiveAgent(model::AbstractCognitiveModel, parameters, state, previous_action; save_history::Bool = true)
    history = save_history ? AgentHistory(Any[NoObservation()], Any[previous_action], Any[prepare_state_for_saving(model, state)]) : AgentHistory(Any[], Any[], Any[])
    return CognitiveAgent(model, parameters, state, previous_action, history, save_history)
end

"""
    AgentStarts()

Make the agent move first on each timestep of a simulation.

On a timestep, the cognitive model updates and acts on the previous observation, and the
environment then responds to its action. On the first timestep, the cognitive model receives
[`NoObservation`](@ref).

See also [`EnvironmentStarts`](@ref), [`simulate`](@ref).
"""
struct AgentStarts end

"""
    EnvironmentStarts()

Make the environment move first on each timestep of a simulation.

On a timestep, the environment gives an observation in response to the previous action, and
the cognitive model then updates and acts on it. On the first timestep, the environment
receives [`NoAction`](@ref).

See also [`AgentStarts`](@ref), [`simulate`](@ref).
"""
struct EnvironmentStarts end
