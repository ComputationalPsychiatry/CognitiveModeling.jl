"""
    CognitiveAgent(model, parameters)
    CognitiveAgent(model, parameters, state, previous_action)

An instantiated cognitive model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

The state defaults to the output of [`initialize_state`](@ref), and the previous action to
[`NoAction`](@ref).

See also [`Environment`](@ref).
"""
mutable struct CognitiveAgent{Tmodel <: AbstractCognitiveModel, Tparameters}
    model::Tmodel
    parameters::Tparameters
    #Not typed, because they can change type, for example from NoAction() to a Bool
    state::Any
    previous_action::Any
end

#Starts the agent from its initial state, with no previous action
function CognitiveAgent(model::AbstractCognitiveModel, parameters)
    return CognitiveAgent(model, parameters, initialize_state(model, parameters), NoAction())
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
