## Passing names between models ##

"""
    NameMap(names...)

Which outputs of one side of a simulation the other side receives, and under which names.

Each argument is `:given => :received`, for an output which has another name in the
receiver's attributes, or a single `Symbol`, for one which keeps its name. Outputs that
aren't in the map aren't passed on, and each name can only be there once.

[`simulate`](@ref) and [`simulate!`](@ref) take one as `observation_map`, from the
observations the environment model gives to those in the cognitive model's attributes, and
one as `action_map`, from the actions the cognitive model gives to those in the environment
model's attributes. A cognitive model and an environment which name things differently, or
where one gives something that isn't in the other's attributes, can then be used together,
and the map says which outputs the receiver gets: it must give exactly the names in the
receiver's attributes, from [`attribute_names`](@ref). `NameMap()` passes nothing on, for a
receiver without inputs in its attributes. Without a map, the two must have the same names
in their attributes.

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
`Symbol`, which is received under the name it is given with. Used by the constructor of
[`NameMap`](@ref).
"""
given_name(name::Symbol) = name
given_name(names::Pair{Symbol,Symbol}) = first(names)
received_name(name::Symbol) = name
received_name(names::Pair{Symbol,Symbol}) = last(names)

## Instantiated models ##

"""
    InstantiatedModel(model, parameters; save_history=SaveHistory())
    InstantiatedModel(model, parameters, state, previous_output; save_history=SaveHistory())

A cognitive model or an environment model with its parameters, current state and previous
output, which [`simulate!`](@ref) can update iteratively.

Each argument is kept in a field of the same name, such as `agent.state`. Parameters that
aren't given take the values in [`default_parameters`](@ref), if the model provides defaults
for them. The state defaults to the output of [`initialize_state`](@ref) with the full
parameters, and the previous output to every output being [`BeforeFirstTimestep`](@ref). The
previous output and the state must be `NamedTuple`s, and the names of the parameters, of the
previous output and of the state must be those in the model's attributes, from
[`attribute_names`](@ref), or an error is thrown.

`agent.history` holds a [`ModelHistory`](@ref) with where the model started, which is
timestep 0, and everything that happens to it, as selected by `save_history`, a
[`SaveHistory`](@ref). By default nothing is selected and the vectors in `agent.history`
stay empty, and `SaveHistory(SaveEverything())` saves everything. States, inputs and outputs
can only be selected by name among those in the model's attributes, since the history starts
from those names.
`agent.n_timesteps` counts the timesteps the model has simulated, whatever is saved.

In the REPL, an instantiated model shows the kind and the type of its model, how many
timesteps it has simulated, and the values of its parameters and of its state, with only how
many there are when there are more than five. Large arrays are shortened. When the model's
[`stopping_criterion`](@ref) is met, it says so, since it must then be [`reset!`](@ref)
before it can be simulated again. On one line, as inside a vector, it shows the type and the
kind of its model and how many timesteps it has simulated.

An instantiated model is a convenience, and isn't type stable, by design: its state and its
previous output can change type, as an output does from `BeforeFirstTimestep()` to a `Bool`,
and so can what its history saves, so they are kept untyped. Each call of `simulate!` on it
is therefore dynamically dispatched, and the compiler doesn't know what it returns, whatever
is saved. Where that matters, use [`simulate`](@ref) on the models themselves, one
timestep at a time or for a whole simulation.

The same type serves as the agent and as the environment of a simulation, since both are
models. See also [`simulate!`](@ref) and [`reset!`](@ref).

# Examples
```jldoctest
julia> struct LearnsRewards <: AbstractCognitiveModel end

julia> CognitiveModeling.initialize_state(::LearnsRewards, parameters) = (; expected_reward = 0.5);

julia> function CognitiveModeling.apply_timestep_model(::LearnsRewards, context::ModelContext)
           V = context.previous_state.expected_reward
           V = V + context.parameters.learning_rate * (context.input.reward - V)
           return ModelOutput((; choice = Deterministic(V > 0.5)), (; expected_reward = V))
       end;

julia> CognitiveModeling.attribute_names(::LearnsRewards) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice);

julia> agent = InstantiatedModel(LearnsRewards(), (; learning_rate = 0.5));

julia> simulate!(agent, (; reward = 1.0));

julia> agent
-- Instantiated cognitive model --
Type: LearnsRewards
Timesteps simulated: 1
Parameters: learning_rate = 0.5
State: expected_reward = 0.75

julia> print(agent)
LearnsRewards (instantiated cognitive model, 1 timestep simulated)
```
"""
mutable struct InstantiatedModel{
    Tmodel<:AbstractTimestepModel,Tparameters,Tsave_history<:SaveHistory
}
    model::Tmodel
    parameters::Tparameters
    #Not typed, because they can change type, for example from BeforeFirstTimestep() to a Bool inside the output
    state::Any
    previous_output::Any
    history::ModelHistory
    save_history::Tsave_history
    n_timesteps::Int
end

#Starts the model from its initial state, with nothing given yet
function InstantiatedModel(
    model::AbstractTimestepModel, parameters; save_history::SaveHistory=SaveHistory()
)
    parameters = merge_default_parameters(model, parameters)
    return InstantiatedModel(
        model,
        parameters,
        initialize_state(model, parameters),
        output_before_first_timestep(model);
        save_history,
    )
end

#Fills in the default parameters, checks the names, and starts the history where the model starts, with what is selected of it
function InstantiatedModel(
    model::AbstractTimestepModel,
    parameters,
    state,
    previous_output;
    save_history::SaveHistory=SaveHistory(),
)

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check that the parameters, the state and the previous output have the names in the model's attributes, and that each parameter is within its domain
    check_match_attribute_names(model, parameters, state, previous_output)
    check_parameter_domains(model, parameters)

    #Check that the inputs and outputs selected for the history are among those in the model's attributes
    check_savehistory_selection_names(save_history, model)

    #Start the history from where the model starts
    history = starting_history(model, state, previous_output, save_history)

    return InstantiatedModel(
        model, parameters, state, previous_output, history, save_history, 0
    )
end

"""
    starting_history(model, state, previous_output, save_history)

Return the history that an [`InstantiatedModel`](@ref) starts with: a [`ModelHistory`](@ref)
with what `save_history` selects of where it starts, as timestep 0, and
[`BeforeFirstTimestep`](@ref) as the size of the timestep, since none has run yet. When
nothing is selected, or `save_history` doesn't save the initial timestep values, the vectors
are empty.

A model hasn't received anything when it starts, so that entry has every input in the
model's attributes as `BeforeFirstTimestep()`, and so has every output distribution.
[`reset!`](@ref) starts from the same history as the constructors.
"""
function starting_history(
    model::AbstractTimestepModel, state, previous_output, save_history::SaveHistory
)
    if saves_nothing(save_history) || !saves_initial_timestep_values(save_history)
        return ModelHistory(Any[], Any[], Any[], Any[], Any[])
    end
    return ModelHistory(
        Any[select_state_to_save(model, state, state_selection(save_history))],
        Any[select_names(
            input_before_first_timestep(model), input_selection(save_history)
        )],
        Any[select_names(previous_output, output_selection(save_history))],
        Any[select_names(
            output_before_first_timestep(model), output_distribution_selection(save_history)
        )],
        Any[BeforeFirstTimestep()],
    )
end

"""
    reset!(instantiated_model::InstantiatedModel)

Put `instantiated_model` back to where a new one starts, and return it.

The state becomes the output of [`initialize_state`](@ref) with the parameters it has, the
previous output has every output as [`BeforeFirstTimestep`](@ref) again, and `n_timesteps`
is 0. The history is a new one which starts from there, as timestep 0 when the
[`SaveHistory`](@ref) saves the initial timestep values, so a history that was read before
the reset, as in `history = agent.history`, is kept as it was. The model, the parameters and
the `SaveHistory` stay the same.

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

function CognitiveModeling.apply_timestep_model(model::RescorlaWagner, context::ModelContext)
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
    instantiated_model.previous_output = output_before_first_timestep(
        instantiated_model.model
    )
    instantiated_model.history = starting_history(
        instantiated_model.model,
        instantiated_model.state,
        instantiated_model.previous_output,
        instantiated_model.save_history,
    )
    instantiated_model.n_timesteps = 0
    return instantiated_model
end
