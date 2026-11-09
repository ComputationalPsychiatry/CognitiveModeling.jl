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
lists. A cognitive model and an environment which name things differently, or where one
gives something the other doesn't list, can then be used together, and the map says which
outputs the receiver gets: it must give exactly the names the receiver lists in
[`attribute_names`](@ref). `NameMap()` passes nothing on, for a receiver which lists
nothing. Without a map, the two must list the same names.

# Examples
```jldoctest
julia> NameMap(:payoff => :reward, :cue)
NameMap(:payoff => :reward, :cue)

julia> NameMap(:payoff => :reward, :cue => :reward)
ERROR: ArgumentError: A NameMap can only give each name once, but it gives (:reward, :reward).
[...]
```
"""
struct NameMap{Tgiven,Treceived} end

#Takes the names one by one, and puts those given and those received in the type
function NameMap(names::Union{Symbol,Pair{Symbol,Symbol}}...)
    given = map(given_name, names)
    received = map(received_name, names)
    allunique(given) ||
        throw(ArgumentError("A NameMap can only take each name once, but it takes $given."))
    allunique(received) || throw(
        ArgumentError("A NameMap can only give each name once, but it gives $received.")
    )
    return NameMap{given,received}()
end

"""
    given_name(name)
    received_name(name)

Return the name an output is given under and the name it is received under, from one
argument of a [`NameMap`](@ref): the two sides of a `:given => :received` pair, or a single
`Symbol`, which is received under the name it is given with.
"""
given_name(name::Symbol) = name
given_name(names::Pair{Symbol,Symbol}) = first(names)
received_name(name::Symbol) = name
received_name(names::Pair{Symbol,Symbol}) = last(names)

#Shows the map as it is written
function Base.show(io::IO, ::NameMap{Tgiven,Treceived}) where {Tgiven,Treceived}
    print(io, "NameMap(")
    join(
        io,
        (
            given === received ? repr(given) : string(repr(given), " => ", repr(received))
            for (given, received) in zip(Tgiven, Treceived)
        ),
        ", ",
    )
    print(io, ")")
    return nothing
end

"""
    ModelHistory(states, inputs, outputs, output_distributions, timestep_sizes)

What has happened to an [`InstantiatedModel`](@ref), which [`simulate!`](@ref) adds to.

Each argument is kept in a field of the same name, such as `history.states`. The names are
the model's own: for a cognitive model the inputs are its observations and the outputs its
actions, and for an environment model the inputs are the actions it received and the outputs
the observations it gave. The model's [`SaveHistory`](@ref) selects what is kept of each
kind, and when it selects nothing at all, the vectors stay empty. Each vector starts with
timestep 0, where the model started, when the `SaveHistory` saves the initial values, which
it does by default, so that element `t + 1` is timestep `t`.

# Arguments
- `states::Vector{Any}`: the states of the model, starting at timestep 0 with its initial
  state. Later states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
- `inputs::Vector{Any}`: what the model received on each timestep, starting at timestep 0
  with every input the model lists as [`BeforeFirstTimestep`](@ref), since it has received
  nothing before the simulation.
- `outputs::Vector{Any}`: what the model gave on each timestep, starting at timestep 0 with
  its previous output, which by default has every output as `BeforeFirstTimestep()`.
- `output_distributions::Vector{Any}`: what the model gave each output to be sampled from,
  starting at timestep 0 with `BeforeFirstTimestep()` for every output, since nothing was
  given yet.
- `timestep_sizes::Vector{Any}`: how much time each timestep covered, starting with
  `BeforeFirstTimestep()` for timestep 0, so that it has an entry for each entry of the
  other vectors.
"""
struct ModelHistory
    states::Vector{Any}
    inputs::Vector{Any}
    outputs::Vector{Any}
    output_distributions::Vector{Any}
    timestep_sizes::Vector{Any}
end

"""
    InstantiatedModel(model, parameters; save_history=SaveHistory())
    InstantiatedModel(model, parameters, state, previous_output; save_history=SaveHistory())

A cognitive model or an environment model with its parameters, current state and previous
output, which [`simulate!`](@ref) can update iteratively.

Each argument is kept in a field of the same name, such as `agent.state`. Parameters that
aren't given take the values in [`default_parameters`](@ref), if the model provides defaults
for them. The state defaults to the output of [`initialize_state`](@ref) with the full
parameters, and the previous output to every output being [`BeforeFirstTimestep`](@ref). The
model must have a method of [`attribute_names`](@ref), and the names of the parameters, of
the previous output, and of the state when it is a `NamedTuple`, must be those it lists, or
an error is thrown.

`agent.history` holds a [`ModelHistory`](@ref) with where the model started, which is
timestep 0, and everything that happens to it, as selected by `save_history`, a
[`SaveHistory`](@ref). By default nothing is selected and the vectors in `agent.history` stay
empty, and `SaveHistory(SaveEverything())` saves everything. Inputs and outputs can only be
selected by name among those the model lists, since the history starts from those names.

The same type serves as the agent and as the environment of a simulation, since both are
models. See also [`simulate!`](@ref) and [`reset!`](@ref).
"""
mutable struct InstantiatedModel{
    Tmodel<:AbstractModel,Tparameters,Tsave_history<:SaveHistory
}
    model::Tmodel
    parameters::Tparameters
    #Not typed, because they can change type, for example from BeforeFirstTimestep() to a Bool inside the output
    state::Any
    previous_output::Any
    history::ModelHistory
    save_history::Tsave_history
end

#Starts the model from its initial state, with nothing given yet
function InstantiatedModel(
    model::AbstractModel, parameters; save_history::SaveHistory=SaveHistory()
)
    parameters = merge_default_parameters(model, parameters)
    return InstantiatedModel(
        model,
        parameters,
        initialize_state(model, parameters),
        initial_output(model);
        save_history,
    )
end

#Fills in the default parameters, checks the names, and starts the history where the model starts, with what is selected of it
function InstantiatedModel(
    model::AbstractModel,
    parameters,
    state,
    previous_output;
    save_history::SaveHistory=SaveHistory(),
)

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check that the parameters, the state and the previous output have the names the model lists
    check_match_attribute_names(model, parameters, state, previous_output)

    #Check that the inputs and outputs selected for the history are among those the model lists
    check_savehistory_selection_names(save_history, model)

    #Start the history from where the model starts
    history = starting_history(model, state, previous_output, save_history)

    return InstantiatedModel(
        model, parameters, state, previous_output, history, save_history
    )
end

"""
    starting_history(model, state, previous_output, save_history)

Return the history that an [`InstantiatedModel`](@ref) starts with: a [`ModelHistory`](@ref)
with what `save_history` selects of where it starts, as timestep 0, and
[`BeforeFirstTimestep`](@ref) as the size of the timestep, since none has run yet. When
nothing is selected, or `save_history` doesn't save the initial values, the vectors are
empty.

A model hasn't received anything when it starts, so that entry has every input the model
lists as `BeforeFirstTimestep()`, and so has every output distribution. [`reset!`](@ref)
starts from the same history as the constructors.
"""
function starting_history(
    model::AbstractModel, state, previous_output, save_history::SaveHistory
)
    if saves_nothing(save_history) || !saves_initial_values(save_history)
        return ModelHistory(Any[], Any[], Any[], Any[], Any[])
    end
    return ModelHistory(
        Any[save_state(model, state, state_selection(save_history))],
        Any[select_names(initial_input(model), input_selection(save_history))],
        Any[select_names(previous_output, output_selection(save_history))],
        Any[select_names(
            initial_output(model), output_distribution_selection(save_history)
        )],
        Any[BeforeFirstTimestep()],
    )
end

"""
    reset!(instantiated_model::InstantiatedModel)

Put `instantiated_model` back to where a new one starts, and return it.

The state becomes the output of [`initialize_state`](@ref) with the parameters it has, and
the previous output has every output as [`BeforeFirstTimestep`](@ref) again. The history is a
new one which starts from there, as timestep 0 when the [`SaveHistory`](@ref) saves the
initial values, so a history that was read before the reset, as in
`history = agent.history`, is kept as it was. The model, the parameters and the
`SaveHistory` stay the same.

An instantiated model that was created with a state or a previous output of its own doesn't
return to those. Create a new one for that.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice)
end

agent = InstantiatedModel(RescorlaWagner(), (; learning_rate = 0.5); save_history = SaveHistory(SaveEverything()))

simulate!(agent, (; reward = 1.0))
simulate!(agent, (; reward = 1.0))
reset!(agent)

agent.state, agent.previous_output, agent.history.states

# output

((expected_reward = 0.5,), (choice = BeforeFirstTimestep(),), Any[(expected_reward = 0.5,)])
```
"""
function reset!(instantiated_model::InstantiatedModel)
    instantiated_model.state = initialize_state(
        instantiated_model.model, instantiated_model.parameters
    )
    instantiated_model.previous_output = initial_output(instantiated_model.model)
    instantiated_model.history = starting_history(
        instantiated_model.model,
        instantiated_model.state,
        instantiated_model.previous_output,
        instantiated_model.save_history,
    )
    return instantiated_model
end
