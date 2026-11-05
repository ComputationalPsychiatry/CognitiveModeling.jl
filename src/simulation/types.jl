"""
    SaveEverything()

Marker for saving every attribute of a kind, used in [`SaveHistory`](@ref).
"""
struct SaveEverything end

"""
    SaveHistory(; observations=(), actions=(), states=())
    SaveHistory(SaveEverything())

Which observations, actions and states a simulation saves, by name.

Each argument is the names of that kind to save, in a `Tuple` of `Symbol`s, a single
`Symbol` or a `Vector` of them, or [`SaveEverything`](@ref) for all of them. A kind that
isn't given saves nothing, so `SaveHistory()` saves nothing, which is the default
everywhere, and `SaveHistory(SaveEverything())` saves everything. The selections are kept
in the type, so that selecting costs nothing in a simulation, and are read as properties,
such as `save_history.states`.

Saving some names of a kind gives a `NamedTuple` with those names on each timestep, which
only works when the value is a `NamedTuple`, and saving none gives an empty one, `(;)`, for
any value. [`NoObservation`](@ref) is saved as it is. For an environment, the states are
its own, and the observations and actions are the same as the agent's, which
[`simulate!`](@ref) adds to the environment's history too, while [`simulate`](@ref) only
uses its states.

See also [`CognitiveAgent`](@ref), [`Environment`](@ref) and [`simulate`](@ref), which take
one.

# Examples
```jldoctest
julia> save_history = SaveHistory(; actions = :choice, states = (:expected_reward, :surprise))
SaveHistory
Observations: none
Actions: choice
States: expected_reward, surprise

julia> save_history.states
(:expected_reward, :surprise)

julia> SaveHistory(SaveEverything()).observations
SaveEverything()
```
"""
struct SaveHistory{Tobservations, Tactions, Tstates} end

#Takes the names by keyword, and puts each selection in the type
function SaveHistory(; observations = (), actions = (), states = ())
    return SaveHistory{as_selection(observations), as_selection(actions), as_selection(states)}()
end

#Saves everything of every kind
function SaveHistory(::SaveEverything)
    return SaveHistory(; observations = SaveEverything(), actions = SaveEverything(), states = SaveEverything())
end

#A selection is a tuple of symbols, or the marker for everything
as_selection(names::Tuple{Vararg{Symbol}}) = names
as_selection(name::Symbol) = (name,)
as_selection(names::AbstractVector{Symbol}) = Tuple(names)
as_selection(::SaveEverything) = SaveEverything()

#Reads the selections in the type as properties
function Base.getproperty(save_history::SaveHistory{Tobservations, Tactions, Tstates}, name::Symbol) where {Tobservations, Tactions, Tstates}
    if name === :observations
        return Tobservations
    elseif name === :actions
        return Tactions
    elseif name === :states
        return Tstates
    else
        throw(FieldError(typeof(save_history), name))
    end
end

Base.propertynames(::SaveHistory) = (:observations, :actions, :states)

#Shows the selections on one line each in the REPL and notebooks
function Base.show(io::IO, ::MIME"text/plain", save_history::SaveHistory)
    print(io, "SaveHistory")
    for (kind, selection) in (("Observations", save_history.observations), ("Actions", save_history.actions), ("States", save_history.states))
        print(io, "\n", kind, ": ")
        if selection isa SaveEverything
            print(io, "everything")
        elseif isempty(selection)
            print(io, "none")
        else
            join(io, selection, ", ")
        end
    end
    return nothing
end

"""
    AgentHistory(observations, actions, states)

What has happened to a [`CognitiveAgent`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.actions`. The agent's
[`SaveHistory`](@ref) selects what is kept of each observation, action and state, and when
it selects nothing at all, the vectors stay empty.

# Arguments
- `observations::Vector{Any}`: the observations, starting with [`NoObservation`](@ref).
- `actions::Vector{Any}`: the actions, starting with the agent's previous action, which by
  default has every action as [`NoAction`](@ref).
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
    CognitiveAgent(model, parameters; save_history=SaveHistory())
    CognitiveAgent(model, parameters, state, previous_action; save_history=SaveHistory())

An instantiated cognitive model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

Parameters that aren't given take the values in [`default_parameters`](@ref). The state
defaults to the output of [`initialize_state`](@ref) with the full parameters, and the
previous action to every action being [`NoAction`](@ref). The model must have a method of
[`attribute_names`](@ref), and the names of the parameters, of the previous action, and of
the state when it is a `NamedTuple`, must be those it lists, or an error is thrown.

`agent.history` holds an [`AgentHistory`](@ref) with where the agent started and everything
that happens to it, as selected by `save_history`, a [`SaveHistory`](@ref). By default
nothing is selected and the vectors in `agent.history` stay empty, and
`SaveHistory(SaveEverything())` saves everything.

See also [`Environment`](@ref).
"""
mutable struct CognitiveAgent{Tmodel <: AbstractCognitiveModel, Tparameters, Tsave_history <: SaveHistory}
    model::Tmodel
    parameters::Tparameters
    #Not typed, because they can change type, for example from NoAction() to a Bool inside the action
    state::Any
    previous_action::Any
    history::AgentHistory
    save_history::Tsave_history
end

#Starts the agent from its initial state, with no action taken yet
function CognitiveAgent(model::AbstractCognitiveModel, parameters; save_history::SaveHistory = SaveHistory())
    parameters = merge(default_parameters(model), parameters)
    return CognitiveAgent(model, parameters, initialize_state(model, parameters), initial_action(model); save_history)
end

#Fills in the default parameters, checks the names, and starts the history where the agent starts, with what is selected of it
function CognitiveAgent(model::AbstractCognitiveModel, parameters, state, previous_action; save_history::SaveHistory = SaveHistory())
    parameters = merge(default_parameters(model), parameters)
    check_names_match(model, parameters, state, previous_action)
    if saves_nothing(save_history)
        history = AgentHistory(Any[], Any[], Any[])
    else
        history = AgentHistory(Any[select_names(NoObservation(), observation_selection(save_history))], Any[select_names(previous_action, action_selection(save_history))], Any[save_state(model, state, state_selection(save_history))])
    end
    return CognitiveAgent(model, parameters, state, previous_action, history, save_history)
end

"""
    EnvironmentHistory(observations, actions, states)

What has happened to an [`Environment`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.states`. The
environment's [`SaveHistory`](@ref) selects what is kept of each observation, action and
state, and when it selects nothing at all, the vectors stay empty. The observations and
actions are the same as in the agent's history.

# Arguments
- `observations::Vector{Any}`: the observations the environment gave, starting with its
  previous observation, which is usually [`NoObservation`](@ref).
- `actions::Vector{Any}`: the actions on each timestep, starting with an empty
  `NamedTuple`, since the environment has received none before the simulation.
- `states::Vector{Any}`: the states of the environment, starting with its initial state.
  Later states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
"""
struct EnvironmentHistory
    observations::Vector{Any}
    actions::Vector{Any}
    states::Vector{Any}
end

"""
    Environment(model, parameters; save_history=SaveHistory())
    Environment(model, parameters, state, previous_observation; save_history=SaveHistory())

An environment model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

The state defaults to the output of `initialize_state`, and the previous observation to
[`NoObservation`](@ref).

`environment.history` holds an [`EnvironmentHistory`](@ref) with where the environment
started and what happens to it, as selected by `save_history`, a [`SaveHistory`](@ref). By
default nothing is selected and the vectors in `environment.history` stay empty, and
`SaveHistory(SaveEverything())` saves everything.

See also [`CognitiveAgent`](@ref).
"""
mutable struct Environment{Tmodel <: AbstractEnvironmentModel, Tparameters, Tsave_history <: SaveHistory}
    model::Tmodel
    parameters::Tparameters
    state::Any
    previous_observation::Any
    history::EnvironmentHistory
    save_history::Tsave_history
end

#Starts the environment from its initial state, with no previous observation
function Environment(model::AbstractEnvironmentModel, parameters; save_history::SaveHistory = SaveHistory())
    return Environment(model, parameters, initialize_state(model, parameters), NoObservation(); save_history)
end

#Starts the history where the environment starts, with what is selected of it
function Environment(model::AbstractEnvironmentModel, parameters, state, previous_observation; save_history::SaveHistory = SaveHistory())
    if saves_nothing(save_history)
        history = EnvironmentHistory(Any[], Any[], Any[])
    else
        history = EnvironmentHistory(Any[select_names(previous_observation, observation_selection(save_history))], Any[(;)], Any[save_state(model, state, state_selection(save_history))])
    end
    return Environment(model, parameters, state, previous_observation, history, save_history)
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
receives an action with every entry [`NoAction`](@ref).

See also [`AgentStarts`](@ref), [`simulate`](@ref).
"""
struct EnvironmentStarts end
