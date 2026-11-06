## Selecting what is saved ##
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
any value. For an environment, the states are its own, and the observations and actions are
those it gave and received, which are the agent's unless a [`NameMap`](@ref) selects or
renames them. [`simulate!`](@ref) adds them to the environment's history too, while
[`simulate`](@ref) only uses its states.

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

## Using a selection ##
"""
    observation_selection(save_history)
    action_selection(save_history)
    state_selection(save_history)

Return the selection of one kind in a [`SaveHistory`](@ref) as a `Val`, so that the names are
known at compile time wherever something is selected with it.
"""
observation_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tobservations)
action_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tactions)
state_selection(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = Val(Tstates)

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
        error("A SaveHistory can only select names from a NamedTuple, but the value to select $names from has type $(typeof(x)). Use SaveEverything() or () for this kind of attribute.")
    end
end

"""
    check_savehistory_selection_names(save_history, model)

Throw an informative error if `save_history` selects an observation or an action by name that
[`attribute_names`](@ref) of `model`, a cognitive model or an environment model, doesn't
list, since a history starts from those names. The states aren't checked, because
[`prepare_state_for_saving`](@ref) can save anything.
"""
function check_savehistory_selection_names(save_history::SaveHistory, model::AbstractModel)
    attributes = check_type_attribute_names(attribute_names(model), model)
    check_selected_names(save_history.observations, attributes.observations, "observations", model)
    check_selected_names(save_history.actions, attributes.actions, "actions", model)
    return nothing
end

"""
    check_selected_names(selection, names, kind, model)

Throw an informative error if `selection` has a name that isn't among `names`, those that
`model` lists for `kind`, which is `"observations"` or `"actions"`.
[`SaveEverything`](@ref) can always be selected.
"""
check_selected_names(::SaveEverything, names, kind, model) = nothing
function check_selected_names(selection::Tuple, names, kind, model)
    if !has_every_name(selection, names)
        throw(ArgumentError("The SaveHistory selects the $kind $selection, but attribute_names lists the $kind of $(describe_model(model)) as $names. A history can only select observations and actions that the model lists."))
    end
    return nothing
end

"""
    save_state(model, state, selection::Val)

Return what is saved of `state`: what `selection` selects of the output of
[`prepare_state_for_saving`](@ref). When nothing is selected, the state isn't prepared at
all, so that no copy is made of a state which isn't saved.
"""
save_state(model, state, selection::Val) = select_names(prepare_state_for_saving(model, state), selection)
save_state(model, state, ::Val{()}) = (;)

"""
    saves_nothing(save_history)

Return whether `save_history` selects nothing of any kind, which is known at compile time.
"""
saves_nothing(::SaveHistory{(), (), ()}) = true
saves_nothing(::SaveHistory) = false

"""
    savehistory_states_only(save_history)

Return a [`SaveHistory`](@ref) which selects only the states that `save_history` selects,
which is what [`simulate`](@ref) keeps of an environment's selection.
"""
savehistory_states_only(::SaveHistory{Tobservations, Tactions, Tstates}) where {Tobservations, Tactions, Tstates} = SaveHistory{(), (), Tstates}()

"""
    saved_timestep_sizes(timestep_sizes, n_steps, save_history)

Return the sizes of `n_steps` timesteps as they are returned with a history: one size for
each timestep when `save_history` saves anything, and an empty vector when it saves nothing.
"""
saved_timestep_sizes(timestep_sizes, n_steps, ::SaveHistory) = each_timestep_size(timestep_sizes, n_steps)
saved_timestep_sizes(timestep_sizes, n_steps, ::SaveHistory{(), (), ()}) = empty(each_timestep_size(timestep_sizes, 0))
