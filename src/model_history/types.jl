## Selecting what is saved ##
"""
    SaveEverything()

Marker for saving every attribute of a kind, used in [`SaveHistory`](@ref).
"""
struct SaveEverything end

"""
    SaveHistory(; states=(), inputs=(), outputs=(), output_distributions=(),
                save_initial_timestep_values=true)
    SaveHistory(SaveEverything(); save_initial_timestep_values=true)

Which states, inputs, outputs and output distributions of a model a simulation or a fitted
session saves, by name, and whether it saves them from before the first timestep too.

`states`, `inputs`, `outputs` and `output_distributions` are each the names of that kind to
save, in a `Tuple` of `Symbol`s, a single `Symbol` or a `Vector` of them, or
[`SaveEverything`](@ref) for all of them. The names are those in the model's attributes,
from [`attribute_names`](@ref): for a cognitive model the inputs are its observations and
the outputs its actions, and for an environment model the inputs are the actions it receives
and the outputs the observations it gives. The output distributions are what the model gave
its outputs to be sampled from, named like the outputs. A kind that isn't given saves
nothing, so `SaveHistory()` saves nothing, which is the default everywhere, and
`SaveHistory(SaveEverything())` saves everything. The selections are read as properties,
such as `save_history.states`.

`save_initial_timestep_values` is whether what is saved starts with timestep 0, before the
first timestep: the initial states, and the previous input and output, which are
[`BeforeFirstTimestep`](@ref) unless others are given. It is `true` by default, which
matters when the initial state depends on parameters that are fitted. Since Julia counts
from 1, element `t + 1` of what is saved is then timestep `t`, and with `false`, element `t`
is.

Saving some names of a kind gives a `NamedTuple` with those names on each timestep, and
saving none gives an empty one, `(;)`. Before the first timestep there is no output
distribution, so timestep 0 has `BeforeFirstTimestep()` for each name that is saved.

Each model of a simulation has one: [`simulate_session`](@ref) and [`SessionInfo`](@ref)
take `cognitive_save_history` and `environment_save_history`, and an
[`InstantiatedModel`](@ref) takes `save_history`. The trajectories that a simulation or a
session returns are named by their role: the cognitive model's inputs as `observations` and
outputs as `actions`, and the environment's inputs as `actions` and outputs as
`observations` under `environment`, as [`simulate_session`](@ref) describes.

# Examples
```jldoctest
julia> save_history = SaveHistory(; outputs = :choice, states = (:expected_reward, :surprise))
SaveHistory
States: expected_reward, surprise
Inputs: none
Outputs: choice
Output distributions: none
Initial timestep values: saved

julia> save_history.states
(:expected_reward, :surprise)

julia> SaveHistory(SaveEverything()).inputs
SaveEverything()

julia> SaveHistory(SaveEverything(); save_initial_timestep_values = false).save_initial_timestep_values
false
```
"""
struct SaveHistory{
    Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
} end

#Takes the names by keyword, and puts each selection in the type, and whether the initial timestep values are saved
function SaveHistory(;
    states=(),
    inputs=(),
    outputs=(),
    output_distributions=(),
    save_initial_timestep_values::Bool=true,
)
    return SaveHistory{
        as_selection(states),
        as_selection(inputs),
        as_selection(outputs),
        as_selection(output_distributions),
        save_initial_timestep_values,
    }()
end

#Saves everything of every kind
function SaveHistory(::SaveEverything; save_initial_timestep_values::Bool=true)
    return SaveHistory(;
        states=SaveEverything(),
        inputs=SaveEverything(),
        outputs=SaveEverything(),
        output_distributions=SaveEverything(),
        save_initial_timestep_values,
    )
end

"""
    as_selection(names)

Return `names` as a selection, which is how a [`SaveHistory`](@ref) keeps it:
[`SaveEverything`](@ref) as it is, and otherwise a `Tuple` of `Symbol`s, as
[`as_tuple`](@ref) makes it. Used by the constructor of [`SaveHistory`](@ref).
"""
as_selection(::SaveEverything) = SaveEverything()
as_selection(names) = as_tuple(names)::Tuple{Vararg{Symbol}}

#Reads the selections in the type as properties, and whether the initial timestep values are saved
function Base.getproperty(
    save_history::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values
    },
    name::Symbol,
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_timestep_values}
    if name === :states
        return Tstates
    elseif name === :inputs
        return Tinputs
    elseif name === :outputs
        return Toutputs
    elseif name === :output_distributions
        return Toutput_distributions
    elseif name === :save_initial_timestep_values
        return Tsave_initial_timestep_values
    else
        throw(FieldError(typeof(save_history), name))
    end
end

function Base.propertynames(::SaveHistory)
    return (
        :states, :inputs, :outputs, :output_distributions, :save_initial_timestep_values
    )
end

## What was saved of an instantiated model ##

"""
    ModelHistory(states, inputs, outputs, output_distributions, timestep_sizes)

What has happened to an [`InstantiatedModel`](@ref), which [`simulate_session!`](@ref) adds
to.

Each argument is kept in a field of the same name, such as `history.states`. The names are
the model's own: for a cognitive model the inputs are its observations and the outputs its
actions, and for an environment model the inputs are the actions it received and the outputs
the observations it gave. The model's [`SaveHistory`](@ref) selects what is kept of each
kind, and when it selects nothing at all, the vectors stay empty. Each vector starts with
timestep 0, where the model started, when the `SaveHistory` saves the initial timestep
values, which it does by default, so that element `t + 1` is timestep `t`.

The vectors hold values of any type, since what is saved can change type from one timestep
to the next, as an output does from `BeforeFirstTimestep()` at timestep 0 to a value.

# Arguments
- `states::Vector{Any}`: the states of the model, starting at timestep 0 with its initial
  state. Later states are those after each timestep. All of them are as returned by
  [`prepare_state_for_saving`](@ref).
- `inputs::Vector{Any}`: what the model received on each timestep, starting at timestep 0
  with every input in the model's attributes as [`BeforeFirstTimestep`](@ref), since it has
  received nothing before the simulation.
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

## What was saved of a session ##

"""
    SessionTrajectories(states, observations, actions, action_distributions, environment,
                        timestep_sizes, parameters_valid, n_timesteps, last_timestep)

The trajectories of one session, which are what changed over its timesteps, and how the
session ended. [`simulate_session`](@ref) and [`simulate_session!`](@ref) return one, and a
fitted session returns one for each sample.

Each argument is kept in a field of the same name, such as `trajectories.states`. The
trajectories of each kind are a vector with one element for each timestep, after one for
timestep 0, before the first timestep, when it is saved. Each element is a `NamedTuple` with
what the [`SaveHistory`](@ref) of its model selects of that kind, which is empty when it
selects nothing. `NamedTuple(trajectories)` gives the fields in a `NamedTuple`.

In the REPL, the trajectories of a session show the number of timesteps, the names of what
was saved of each kind, for the cognitive model and for the environment, and whether the
parameters stayed valid. On one line, as inside a vector, they show the number of timesteps.

# Arguments
- `states`: the states of the cognitive model after each timestep, as returned by
  [`prepare_state_for_saving`](@ref).
- `observations`: the observations the cognitive model received.
- `actions`: the actions the cognitive model gave.
- `action_distributions`: what the cognitive model gave its actions to be sampled from.
- `environment`: the trajectories of the environment model in a `NamedTuple`, each in the
  environment's own names: its `states`, the `observations` it gave, the `actions` it
  received, and the `observation_distributions` it gave its observations to be sampled
  from. A fitted session without an environment has [`NoEnvironment`](@ref) instead.
- `timestep_sizes`: how much time each timestep covered: a single size for all of them, or a
  vector with the size of each, after [`BeforeFirstTimestep`](@ref) for timestep 0 when it
  is saved.
- `parameters_valid`: whether the parameters of both models stayed valid on every timestep.
  With `false`, the trajectories end before the timestep on which a model set
  `parameters_valid` to `false` in its [`ModelOutput`](@ref). A fitted session whose models
  can't set it has [`ValidByDefault`](@ref) instead.
- `n_timesteps`: the number of timesteps in the trajectories, without timestep 0. A
  simulation that was masked, under a compiler like Reactant.jl, has every timestep in the
  vectors, of which the first `n_timesteps` were kept, as [`simulate_session`](@ref)
  describes.
- `last_timestep`: the last timestep that was simulated, from a simulation of models that
  can set `parameters_valid`, which is the timestep that turned invalid when
  `parameters_valid` is `false`: its number, under `timestep`, and for each model whether
  its parameters stayed valid on it and its state before and after it, with the cognitive
  model's at the top and the environment's under `environment`. Otherwise `nothing`.

# Examples
```jldoctest
julia> struct LearnsRewards <: AbstractCognitiveModel end

julia> CognitiveModeling.initialize_state(::LearnsRewards, parameters) = (; expected_reward = 0.5);

julia> function CognitiveModeling.timestep_model(::LearnsRewards, context::ModelContext)
           V = context.previous_state.expected_reward
           V = V + context.parameters.learning_rate * (context.input.reward - V)
           return ModelOutput((; choice = Deterministic(V > 0.5)), (; expected_reward = V))
       end;

julia> CognitiveModeling.attribute_names(::LearnsRewards) = ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice);

julia> trajectories = simulate_session(
           LearnsRewards(),
           FixedObservationsEnvironment((; reward = [1.0, 1.0, 0.0]));
           who_starts = EnvironmentStarts(),
           n_steps = 3,
           cognitive_parameters = (; learning_rate = 0.5),
           environment_parameters = (;),
           action_map = NameMap(),
           cognitive_save_history = SaveHistory(; states = :expected_reward, outputs = :choice),
       )
SessionTrajectories
Timesteps: 3, after timestep 0
States: expected_reward
Observations: none
Actions: choice
Action distributions: none
Environment states: none
Environment observations: none
Environment actions: none
Environment observation distributions: none
Parameters valid: true

julia> trajectories.states
4-element Vector{@NamedTuple{expected_reward::Float64}}:
 (expected_reward = 0.5,)
 (expected_reward = 0.75,)
 (expected_reward = 0.875,)
 (expected_reward = 0.4375,)

julia> print(trajectories)
SessionTrajectories (3 timesteps)
```
"""
struct SessionTrajectories{
    Tstates,
    Tobservations,
    Tactions,
    Taction_distributions,
    Tenvironment,
    Ttimestep_sizes,
    Tparameters_valid,
    Tn_timesteps,
    Tlast_timestep,
}
    states::Tstates
    observations::Tobservations
    actions::Tactions
    action_distributions::Taction_distributions
    environment::Tenvironment
    timestep_sizes::Ttimestep_sizes
    parameters_valid::Tparameters_valid
    n_timesteps::Tn_timesteps
    last_timestep::Tlast_timestep
end

#The fields in a NamedTuple, for what a NamedTuple offers and the type doesn't, such as keys and merge
function Base.NamedTuple(trajectories::SessionTrajectories)
    return (;
        trajectories.states,
        trajectories.observations,
        trajectories.actions,
        trajectories.action_distributions,
        trajectories.environment,
        trajectories.timestep_sizes,
        trajectories.parameters_valid,
        trajectories.n_timesteps,
        trajectories.last_timestep,
    )
end

#Two are equal when their fields are, as for a NamedTuple, since a type with vectors in its fields is otherwise only equal to itself
function Base.:(==)(
    first_trajectories::SessionTrajectories, second_trajectories::SessionTrajectories
)
    return NamedTuple(first_trajectories) == NamedTuple(second_trajectories)
end
function Base.isequal(
    first_trajectories::SessionTrajectories, second_trajectories::SessionTrajectories
)
    return isequal(NamedTuple(first_trajectories), NamedTuple(second_trajectories))
end

#Equal trajectories have the same hash, as isequal requires
function Base.hash(trajectories::SessionTrajectories, seed::UInt)
    return hash(NamedTuple(trajectories), hash(SessionTrajectories, seed))
end
