## Selecting the values to save ##

"""
    select_names(x, selection::Val)

Return what a [`SaveHistory`](@ref) selects of `x`: `x` itself for [`SaveEverything`](@ref),
an empty `NamedTuple` for no names, and the named entries of a `NamedTuple` for a tuple of
names. Used wherever something is saved, by simulations, instantiated models and fitted
sessions.
"""
select_names(x, ::Val{SaveEverything()}) = x
select_names(x, ::Val{()}) = (;)

#Names can only be selected from a NamedTuple
function select_names(x, ::Val{names}) where {names}
    if x isa NamedTuple
        return NamedTuple{names}(x)
    else
        error(
            "A SaveHistory can only select names from a NamedTuple, but the value to select $names from has type $(typeof(x)). The state that prepare_state_for_saving returns must be a NamedTuple with the states in the model's attributes.",
        )
    end
end

"""
    select_state_to_save(model, state, selection::Val)

Return what is saved of `state`: what `selection` selects of the output of
[`prepare_state_for_saving`](@ref). When nothing is selected, the state isn't prepared at
all, so that no copy is made of a state which isn't saved. Used wherever a state is saved,
by simulations, instantiated models and fitted sessions.
"""
function select_state_to_save(model, state, selection::Val)
    return select_names(prepare_state_for_saving(model, state), selection)
end
select_state_to_save(model, state, ::Val{()}) = (;)
