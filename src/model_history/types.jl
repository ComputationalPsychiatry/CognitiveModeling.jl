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
from [`attribute_names`](@ref): for a cognitive model the inputs are its observations
and the outputs its actions, and for an environment model the inputs are the actions it
receives and the outputs the observations it gives. The output distributions are what the
model gave its outputs to be sampled from, named like the outputs. A kind that isn't given
saves nothing, so `SaveHistory()` saves nothing, which is the default everywhere, and
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

Each model of a simulation has one: [`simulate`](@ref) and [`SessionInfo`](@ref) take
`cognitive_save_history` and `environment_save_history`, and an
[`InstantiatedModel`](@ref) takes `save_history`. The trajectories that a simulation or a
session returns are named by their role: the cognitive model's inputs as `observations` and
outputs as `actions`, and the environment's inputs as `actions` and outputs as
`observations` under `environment`, as [`simulate`](@ref) describes.

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

What has happened to an [`InstantiatedModel`](@ref), which [`simulate!`](@ref) adds to.

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
