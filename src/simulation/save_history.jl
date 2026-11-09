## Selecting what is saved ##
"""
    SaveEverything()

Marker for saving every attribute of a kind, used in [`SaveHistory`](@ref).
"""
struct SaveEverything end

"""
    SaveHistory(; states=(), inputs=(), outputs=(), output_distributions=(),
                save_initial_values=true)
    SaveHistory(SaveEverything(); save_initial_values=true)

Which states, inputs, outputs and output distributions of a model a simulation or a fitted
session saves, by name, and whether it saves them from before the first timestep too.

`states`, `inputs`, `outputs` and `output_distributions` are each the names of that kind to
save, in a `Tuple` of `Symbol`s, a single `Symbol` or a `Vector` of them, or
[`SaveEverything`](@ref) for all of them. The names are the model's own, as
[`attribute_names`](@ref) lists them: for a cognitive model the inputs are its observations
and the outputs its actions, and for an environment model the inputs are the actions it
receives and the outputs the observations it gives. The output distributions are what the
model gave its outputs to be sampled from, named like the outputs. A kind that isn't given
saves nothing, so `SaveHistory()` saves nothing, which is the default everywhere, and
`SaveHistory(SaveEverything())` saves everything. The selections are kept in the type, so
that selecting costs nothing in a simulation, and are read as properties, such as
`save_history.states`.

`save_initial_values` is whether what is saved starts with timestep 0, before the first
timestep: the initial states, and the previous input and output, which are
[`BeforeFirstTimestep`](@ref) unless others are given. It is `true` by default, which
matters when the initial state depends on parameters that are fitted. Since Julia counts
from 1, element `t + 1` of what is saved is then timestep `t`, and with `false`, element `t`
is.

Saving some names of a kind gives a `NamedTuple` with those names on each timestep, which
only works when the value is a `NamedTuple`, and saving none gives an empty one, `(;)`, for
any value. Before the first timestep there is no output distribution, so timestep 0 has
`BeforeFirstTimestep()` for each name that is saved.

Each model of a simulation has one: [`simulate`](@ref) and [`SessionInfo`](@ref) take
`cognitive_save_history` and `environment_save_history`, and an
[`InstantiatedModel`](@ref) takes `save_history`. What a simulation or a session returns
names the saved values by their role: the cognitive model's inputs as `observations` and
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
Initial values: saved

julia> save_history.states
(:expected_reward, :surprise)

julia> SaveHistory(SaveEverything()).inputs
SaveEverything()

julia> SaveHistory(SaveEverything(); save_initial_values = false).save_initial_values
false
```
"""
struct SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values} end

#Takes the names by keyword, and puts each selection in the type, and whether the initial values are saved
function SaveHistory(;
    states=(),
    inputs=(),
    outputs=(),
    output_distributions=(),
    save_initial_values::Bool=true,
)
    return SaveHistory{
        as_selection(states),
        as_selection(inputs),
        as_selection(outputs),
        as_selection(output_distributions),
        save_initial_values,
    }()
end

#Saves everything of every kind
function SaveHistory(::SaveEverything; save_initial_values::Bool=true)
    return SaveHistory(;
        states=SaveEverything(),
        inputs=SaveEverything(),
        outputs=SaveEverything(),
        output_distributions=SaveEverything(),
        save_initial_values,
    )
end

"""
    as_selection(names)

Return `names` as a selection, which is how a [`SaveHistory`](@ref) keeps it: a `Tuple` of
`Symbol`s, made from a single `Symbol` or a `Vector` of them too, or
[`SaveEverything`](@ref) as it is.
"""
as_selection(names::Tuple{Vararg{Symbol}}) = names
as_selection(name::Symbol) = (name,)
as_selection(names::AbstractVector{Symbol}) = Tuple(names)
as_selection(::SaveEverything) = SaveEverything()

#Reads the selections in the type as properties, and whether the initial values are saved
function Base.getproperty(
    save_history::SaveHistory{
        Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values
    },
    name::Symbol,
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    if name === :states
        return Tstates
    elseif name === :inputs
        return Tinputs
    elseif name === :outputs
        return Toutputs
    elseif name === :output_distributions
        return Toutput_distributions
    elseif name === :save_initial_values
        return Tsave_initial_values
    else
        throw(FieldError(typeof(save_history), name))
    end
end

function Base.propertynames(::SaveHistory)
    return (:states, :inputs, :outputs, :output_distributions, :save_initial_values)
end

#Shows the selections on one line each in the REPL and notebooks, and whether the initial values are saved
function Base.show(io::IO, ::MIME"text/plain", save_history::SaveHistory)
    print(io, "SaveHistory")
    for (kind, selection) in (
        ("States", save_history.states),
        ("Inputs", save_history.inputs),
        ("Outputs", save_history.outputs),
        ("Output distributions", save_history.output_distributions),
    )
        print(io, "\n", kind, ": ")
        if selection isa SaveEverything
            print(io, "everything")
        elseif isempty(selection)
            print(io, "none")
        else
            join(io, selection, ", ")
        end
    end
    print(
        io, "\nInitial values: ", save_history.save_initial_values ? "saved" : "not saved"
    )
    return nothing
end

## Using a selection ##
"""
    state_selection(save_history)
    input_selection(save_history)
    output_selection(save_history)
    output_distribution_selection(save_history)

Return the selection of one kind in a [`SaveHistory`](@ref) as a `Val`, so that the names are
known at compile time wherever something is selected with it.
"""
function state_selection(
    ::SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    return Val(Tstates)
end
function input_selection(
    ::SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    return Val(Tinputs)
end
function output_selection(
    ::SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    return Val(Toutputs)
end
function output_distribution_selection(
    ::SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    return Val(Toutput_distributions)
end

"""
    saves_initial_values(save_history)

Return whether `save_history` saves the values from before the first timestep, as
timestep 0. It is read from the type, so that it is known at compile time.
"""
function saves_initial_values(
    ::SaveHistory{Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
) where {Tstates,Tinputs,Toutputs,Toutput_distributions,Tsave_initial_values}
    return Tsave_initial_values
end

"""
    select_names(x, selection::Val)

Return what a [`SaveHistory`](@ref) selects of `x`: `x` itself for [`SaveEverything`](@ref),
an empty `NamedTuple` for no names, and the named entries of a `NamedTuple` for a tuple of
names.
"""
select_names(x, ::Val{SaveEverything()}) = x
select_names(x, ::Val{()}) = (;)

#Names can only be selected from a NamedTuple
function select_names(x, ::Val{names}) where {names}
    if x isa NamedTuple
        return NamedTuple{names}(x)
    else
        error(
            "A SaveHistory can only select names from a NamedTuple, but the value to select $names from has type $(typeof(x)). Use SaveEverything() or () for this kind of attribute.",
        )
    end
end

"""
    check_savehistory_selection_names(save_history, model)

Throw an informative error if `save_history` selects an input, an output or an output
distribution by name that [`attribute_names`](@ref) of `model`, a cognitive model or an
environment model, doesn't list, since a history starts from those names. The states aren't
checked, because [`prepare_state_for_saving`](@ref) can save anything.
"""
function check_savehistory_selection_names(save_history::SaveHistory, model::AbstractModel)
    attributes = check_type_attribute_names(attribute_names(model), model)
    check_selected_names(save_history.inputs, attributes.inputs, "inputs", model)
    check_selected_names(save_history.outputs, attributes.outputs, "outputs", model)
    check_selected_names(
        save_history.output_distributions, attributes.outputs, "output distributions", model
    )
    return nothing
end

"""
    check_selected_names(selection, names, kind, model)

Throw an informative error if `selection` has a name that isn't among `names`, those that
`model` lists for `kind`, which is `"inputs"`, `"outputs"` or `"output distributions"`, the
last being named like the outputs. [`SaveEverything`](@ref) can always be selected.
"""
check_selected_names(::SaveEverything, names, kind, model) = nothing
function check_selected_names(selection::Tuple, names, kind, model)
    if !has_every_name(selection, names)
        throw(
            ArgumentError(
                "The SaveHistory selects the $kind $selection, but attribute_names lists the $(kind == "output distributions" ? "outputs" : kind) of $(describe_model(model)) as $names. A history can only select inputs, outputs and output distributions that the model lists.",
            ),
        )
    end
    return nothing
end

"""
    save_state(model, state, selection::Val)

Return what is saved of `state`: what `selection` selects of the output of
[`prepare_state_for_saving`](@ref). When nothing is selected, the state isn't prepared at
all, so that no copy is made of a state which isn't saved.
"""
function save_state(model, state, selection::Val)
    return select_names(prepare_state_for_saving(model, state), selection)
end
save_state(model, state, ::Val{()}) = (;)

"""
    saves_nothing(save_history)

Return whether `save_history` selects nothing of any kind, which is known at compile time.
Whether it saves the initial values doesn't matter, since there is nothing to save of them.
"""
function saves_nothing(
    ::SaveHistory{(),(),(),(),Tsave_initial_values}
) where {Tsave_initial_values}
    return true
end
saves_nothing(::SaveHistory) = false

"""
    saved_timestep_sizes(timestep_sizes, n_steps, save_history)

Return the sizes of `n_steps` timesteps as they are returned with a history: one size for
each timestep when `save_history` saves anything, after [`BeforeFirstTimestep`](@ref) for
timestep 0 when it saves the initial values, and an empty vector when it saves nothing.
"""
function saved_timestep_sizes(timestep_sizes, n_steps, save_history::SaveHistory)
    return returned_timestep_sizes(
        each_timestep_size(timestep_sizes, n_steps),
        n_steps,
        Val(saves_initial_values(save_history)),
    )
end
function saved_timestep_sizes(
    timestep_sizes, n_steps, ::SaveHistory{(),(),(),(),Tsave_initial_values}
) where {Tsave_initial_values}
    return empty(each_timestep_size(timestep_sizes, 0))
end
