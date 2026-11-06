"""
    NameMap(names...)

Which outputs of one side of a simulation the other side receives, and under which names.

Each argument is `:given => :received`, for an output which the receiver lists under
another name, or a single `Symbol`, for one which keeps its name. Outputs that aren't in the
map aren't passed on, and each name can only be there once. The names are kept in the type,
so that a map costs nothing in a simulation.

[`simulate`](@ref) and [`simulate!`](@ref) take one as `observation_map`, from the
observations the environment model gives to those the cognitive model lists, and one as
`action_map`, from the actions the cognitive model gives to those the environment model
lists. A model and an environment which name things differently can then be used together,
and the map says which outputs the receiver gets: it must give exactly the names the
receiver lists in [`attribute_names`](@ref).

# Examples
```jldoctest
julia> NameMap(:payoff => :reward, :cue)
NameMap(:payoff => :reward, :cue)

julia> NameMap(:payoff => :reward, :cue => :reward)
ERROR: ArgumentError: A NameMap can only give each name once, but it gives (:reward, :reward).
[...]
```
"""
struct NameMap{Tgiven, Treceived} end

#Takes the names one by one, and puts those given and those received in the type
function NameMap(names::Union{Symbol, Pair{Symbol, Symbol}}...)
    given = map(given_name, names)
    received = map(received_name, names)
    allunique(given) || throw(ArgumentError("A NameMap can only take each name once, but it takes $given."))
    allunique(received) || throw(ArgumentError("A NameMap can only give each name once, but it gives $received."))
    return NameMap{given, received}()
end

#A single name is received under the name it is given with
given_name(name::Symbol) = name
given_name(names::Pair{Symbol, Symbol}) = first(names)
received_name(name::Symbol) = name
received_name(names::Pair{Symbol, Symbol}) = last(names)

#Shows the map as it is written
function Base.show(io::IO, ::NameMap{Tgiven, Treceived}) where {Tgiven, Treceived}
    print(io, "NameMap(")
    join(io, (given === received ? repr(given) : string(repr(given), " => ", repr(received)) for (given, received) in zip(Tgiven, Treceived)), ", ")
    print(io, ")")
    return nothing
end

"""
    AgentHistory(observations, actions, states, timestep_sizes)

The history of states, actions and observations for a [`CognitiveAgent`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.actions`. The agent's
[`SaveHistory`](@ref) selects what is kept of each observation, action and state, and when
it selects nothing at all, the vectors stay empty.

# Arguments
- `observations::Vector{Any}`: the observations, starting with one where every observation
  the model lists is [`NoObservation`](@ref).
- `actions::Vector{Any}`: the actions, starting with the agent's previous action, which by
  default has every action as [`NoAction`](@ref).
- `states::Vector{Any}`: the states of the model, starting with its initial state. Later
  states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
- `timestep_sizes::Vector{Any}`: how much time each timestep covered. It has one size for
  each timestep, so one entry fewer than the other vectors, which also have where the agent
  started.
"""
struct AgentHistory
    observations::Vector{Any}
    actions::Vector{Any}
    states::Vector{Any}
    timestep_sizes::Vector{Any}
end

"""
    CognitiveAgent(model, parameters; save_history=SaveHistory())
    CognitiveAgent(model, parameters, state, previous_action; save_history=SaveHistory())

An instantiated cognitive model with its parameters, current state, and previous action, which [`simulate!`](@ref) can
update iteratively.

Parameters that aren't given take the values in [`default_parameters`](@ref), if the model
provides defaults for them. The state defaults to the output of [`initialize_state`](@ref)
with the full parameters, and the previous action to every action being [`NoAction`](@ref).
The model must have a method of [`attribute_names`](@ref), and the names of the parameters,
of the previous action, and of the state when it is a `NamedTuple`, must be those it lists,
or an error is thrown.

`agent.history` holds an [`AgentHistory`](@ref) with where the agent started and everything
that happens to it, as selected by `save_history`, a [`SaveHistory`](@ref). By default
nothing is selected and the vectors in `agent.history` stay empty, and
`SaveHistory(SaveEverything())` saves everything. Observations and actions can only be
selected by name among those the model lists, since the history starts from those names.

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
    parameters = merge_default_parameters(model, parameters)
    return CognitiveAgent(model, parameters, initialize_state(model, parameters), initial_action(model); save_history)
end

#Fills in the default parameters, checks the names, and starts the history where the agent starts, with what is selected of it
function CognitiveAgent(model::AbstractCognitiveModel, parameters, state, previous_action; save_history::SaveHistory = SaveHistory())

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check that the parameters, the state and the previous action have the names the model lists
    check_match_attribute_names(model, parameters, state, previous_action)

    #Check that the observations and actions selected for the history are among those the model lists
    check_savehistory_selection_names(save_history, model)

    #Start the history from where the agent starts
    history = starting_history(model, state, previous_action, save_history)

    return CognitiveAgent(model, parameters, state, previous_action, history, save_history)
end

#The history of an agent which starts in a state and after an action: what is selected of where it starts, or empty vectors when nothing is selected. No timestep has run yet, so there are no timestep sizes
function starting_history(model::AbstractCognitiveModel, state, previous_action, save_history::SaveHistory)
    if saves_nothing(save_history)
        return AgentHistory(Any[], Any[], Any[], Any[])
    end
    return AgentHistory(Any[select_names(initial_observation(model), observation_selection(save_history))], Any[select_names(previous_action, action_selection(save_history))], Any[save_state(model, state, state_selection(save_history))], Any[])
end

"""
    EnvironmentHistory(observations, actions, states, timestep_sizes)

What has happened to an [`Environment`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.states`. The
environment's [`SaveHistory`](@ref) selects what is kept of each observation, action and
state, and when it selects nothing at all, the vectors stay empty. The observations are
those the environment gave and the actions those it received, which are the same as in the
agent's history unless a [`NameMap`](@ref) selects or renames them.

# Arguments
- `observations::Vector{Any}`: the observations the environment gave, starting with its
  previous observation, which by default has every observation as [`NoObservation`](@ref).
- `actions::Vector{Any}`: the actions on each timestep, starting with every action the
  environment lists as [`NoAction`](@ref), since it has received none before the simulation.
- `states::Vector{Any}`: the states of the environment, starting with its initial state.
  Later states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
- `timestep_sizes::Vector{Any}`: how much time each timestep covered. It has one size for
  each timestep, so one entry fewer than the other vectors, which also have where the
  environment started.
"""
struct EnvironmentHistory
    observations::Vector{Any}
    actions::Vector{Any}
    states::Vector{Any}
    timestep_sizes::Vector{Any}
end

"""
    Environment(model, parameters; save_history=SaveHistory())
    Environment(model, parameters, state, previous_observation; save_history=SaveHistory())

An environment model with its parameters and current state, which [`simulate!`](@ref) can
update iteratively.

Parameters that aren't given take the values in [`default_parameters`](@ref), if the
environment model provides defaults for them. The state defaults to the output of
[`initialize_state`](@ref) with the full parameters, and the previous observation to every
observation being [`NoObservation`](@ref). The environment model must have a method of
[`attribute_names`](@ref), and the names of the parameters, of the previous observation, and
of the state when it is a `NamedTuple`, must be those it lists, or an error is thrown.

`environment.history` holds an [`EnvironmentHistory`](@ref) with where the environment
started and what happens to it, as selected by `save_history`, a [`SaveHistory`](@ref). By
default nothing is selected and the vectors in `environment.history` stay empty, and
`SaveHistory(SaveEverything())` saves everything. Observations and actions can only be
selected by name among those the environment model lists, since the history starts from
those names.

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

#Starts the environment from its initial state, with no observation given yet
function Environment(model::AbstractEnvironmentModel, parameters; save_history::SaveHistory = SaveHistory())
    parameters = merge_default_parameters(model, parameters)
    return Environment(model, parameters, initialize_state(model, parameters), initial_observation(model); save_history)
end

#Fills in the default parameters, checks the names, and starts the history where the environment starts, with what is selected of it
function Environment(model::AbstractEnvironmentModel, parameters, state, previous_observation; save_history::SaveHistory = SaveHistory())

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check that the parameters, the state and the previous observation have the names the environment model lists
    check_match_attribute_names(model, parameters, state, previous_observation)

    #Check that the observations and actions selected for the history are among those the environment model lists
    check_savehistory_selection_names(save_history, model)

    #Start the history from where the environment starts
    history = starting_history(model, state, previous_observation, save_history)

    return Environment(model, parameters, state, previous_observation, history, save_history)
end

#The history of an environment which starts in a state and after an observation: what is selected of where it starts, or empty vectors when nothing is selected
function starting_history(model::AbstractEnvironmentModel, state, previous_observation, save_history::SaveHistory)
    if saves_nothing(save_history)
        return EnvironmentHistory(Any[], Any[], Any[], Any[])
    end
    return EnvironmentHistory(Any[select_names(previous_observation, observation_selection(save_history))], Any[select_names(initial_action(model), action_selection(save_history))], Any[save_state(model, state, state_selection(save_history))], Any[])
end

"""
    reset!(agent::CognitiveAgent)
    reset!(environment::Environment)

Put `agent` or `environment` back to where a new one starts, and return it.

The state becomes the output of [`initialize_state`](@ref) with the parameters it has. The
previous action of an agent has every action as [`NoAction`](@ref) again, and the previous
observation of an environment every observation as [`NoObservation`](@ref). The history is
a new one which starts from there, so a history that was read before the reset, as in
`history = agent.history`, is kept as it was. The model, the parameters and the
[`SaveHistory`](@ref) stay the same.

An agent or an environment that was created with a state, a previous action or a previous
observation of its own doesn't return to those. Create a new one for that.

CommonRLInterface.jl and ReinforcementLearning.jl export a `reset!` of their own. When one
of them is loaded with `using` as well, write `CognitiveModeling.reset!`.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
end

agent = CognitiveAgent(RescorlaWagner(), (; learning_rate = 0.5); save_history = SaveHistory(SaveEverything()))

simulate!(agent, (; reward = 1.0))
simulate!(agent, (; reward = 1.0))
reset!(agent)

agent.state, agent.previous_action, agent.history.states

# output

((expected_reward = 0.5,), (choice = NoAction(),), Any[(expected_reward = 0.5,)])
```
"""
function reset! end

#Starts the agent again from its initial state, with no action taken yet and a new history
function reset!(agent::CognitiveAgent)
    agent.state = initialize_state(agent.model, agent.parameters)
    agent.previous_action = initial_action(agent.model)
    agent.history = starting_history(agent.model, agent.state, agent.previous_action, agent.save_history)
    return agent
end

#Starts the environment again from its initial state, with no observation given yet and a new history
function reset!(environment::Environment)
    environment.state = initialize_state(environment.model, environment.parameters)
    environment.previous_observation = initial_observation(environment.model)
    environment.history = starting_history(environment.model, environment.state, environment.previous_observation, environment.save_history)
    return environment
end

"""
    AgentStarts()

Make the agent move first on each timestep of a simulation.

On a timestep, the cognitive model updates and acts on the previous observation, and the
environment then responds to its action. On the first timestep, the cognitive model receives
an observation with every entry [`NoObservation`](@ref).

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
