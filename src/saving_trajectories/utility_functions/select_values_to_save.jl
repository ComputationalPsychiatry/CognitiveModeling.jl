## Selecting the values to save ##

"""
    select_names(x, selection::Val)

Return what a [`SaveTrajectories`](@ref) selects of `x`: `x` itself for
[`SaveEverything`](@ref), an empty `NamedTuple` for no names, and the named entries of a
`NamedTuple` for a tuple of names. A selection with states of submodels, as
[`as_state_selection`](@ref) keeps them, gives those under the names of their submodels, as
[`select_states_of_submodels`](@ref) returns them. Used wherever something is saved, by
simulations, instantiated models and fitted sessions.
"""
select_names(x, ::Val{SaveEverything()}) = x
select_names(x, ::Val{()}) = (;)

#Names can only be selected from a NamedTuple
function select_names(x, ::Val{names}) where {names}
    if x isa NamedTuple
        #The names are of x itself, unless some of them are of the states of submodels
        names isa Tuple{Vararg{Symbol}} && return NamedTuple{names}(x)
        return select_states_of_submodels(x, Val(names))
    else
        error(
            "A SaveTrajectories can only select names from a NamedTuple, but the value to select $names from has type $(typeof(x)). The state that prepare_state_for_saving returns must be a NamedTuple with the states in the model's attributes.",
        )
    end
end

#Generated, so that each state is read by a name that is known when the code is compiled, as NamedTuple{names} reads them
"""
    select_states_of_submodels(state::NamedTuple, ::Val{selection})

Return what `selection` selects of `state`, where `selection` has states of submodels, as
[`as_state_selection`](@ref) keeps them: each state that is selected whole under its name,
and for each submodel a `NamedTuple` under its name with what is selected of its state, in
the same way down through its submodels. Used by [`select_names`](@ref).
"""
@generated function select_states_of_submodels(
    state::NamedTuple, ::Val{selection}
) where {selection}
    return state_selection_expression(:state, selection)
end

"""
    state_selection_expression(state, selection::Tuple)

Return the expression of the `NamedTuple` with what `selection` selects of a state, which
the expression `state` gives, or the expression of an error when `selection` has a name
twice, which a `NamedTuple` can't have. Used while the code of
[`select_states_of_submodels`](@ref) is generated.
"""
function state_selection_expression(state, selection::Tuple)
    names = map(selected_state_name, selection)
    allunique(names) || return :(throw(
        ArgumentError(
            $(
                "A SaveTrajectories can select each state once, but it selects $names, some of them of the states of submodels."
            ),
        ),
    ))
    entries = map(selection) do entry
        value = :(getfield($state, $(QuoteNode(selected_state_name(entry)))))
        entry isa Symbol && return Expr(:kw, entry, value)
        return Expr(:kw, first(entry), state_selection_expression(value, last(entry)))
    end
    return Expr(:tuple, Expr(:parameters, entries...))
end

"""
    selected_state_name(entry)

Return the name of one entry of a selection of states: the state that is saved whole, or
the submodel that some states are saved of, from the tuple that
[`as_state_selection`](@ref) keeps them as, or from `:submodel => states`, as they are
given. Used by [`state_selection_expression`](@ref) and
[`check_state_selection_names`](@ref).
"""
selected_state_name(name::Symbol) = name
selected_state_name(entry::Union{Tuple{Symbol,Any},Pair{Symbol}}) = first(entry)

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
