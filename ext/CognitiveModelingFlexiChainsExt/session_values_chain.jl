### WHAT FITTED SESSIONS SAVED, AS A CHAIN ###

#The results of a full model, or of several sessions, with every entry named after the session it belongs to
function CognitiveModeling.session_values_chain(results::AbstractMatrix, sessions_info::MultipleSessionsInfo; iter_indices::AbstractVector{<:Integer} = 1:size(results, 1), chain_indices::AbstractVector{<:Integer} = 1:size(results, 2))
    check_results(results)
    first_result = first(results)
    entries = empty_entries()

    if first_result isa NamedTuple && hasproperty(first_result, :saved_behavioural_values)
        #The results of a full model have the parameters of every session too
        check_number_of_sessions(first_result.saved_behavioural_values, sessions_info)
        for (i, session_id) in enumerate(sessions_info.session_ids)
            session_name = session_symbol(session_id)
            for parameter in keys(first_result.cognitive_parameters)
                add_entry!(entries, entry_name(session_name, parameter), map(result -> getproperty(result.cognitive_parameters, parameter)[i], results))
            end
            add_session_entries!(entries, results, (session_name,), result -> result.saved_behavioural_values[i])
        end
    elseif first_result isa AbstractVector
        #The results of several sessions are a vector with what each of them saved
        check_number_of_sessions(first_result, sessions_info)
        for (i, session_id) in enumerate(sessions_info.session_ids)
            add_session_entries!(entries, results, (session_symbol(session_id),), result -> result[i])
        end
    else
        #Anything else isn't known
        throw(ArgumentError("With a MultipleSessionsInfo, the results must be what turing_full_model or turing_multiple_behavioural_sessions returns for each sample, but they are of type $(typeof(first_result)). The results of a single session are given without a session info."))
    end

    return chain_of_entries(entries, size(results)...; iter_indices, chain_indices)
end

#The results of a single session, with every entry named without a session identifier
function CognitiveModeling.session_values_chain(results::AbstractMatrix; iter_indices::AbstractVector{<:Integer} = 1:size(results, 1), chain_indices::AbstractVector{<:Integer} = 1:size(results, 2))
    check_results(results)
    first_result = first(results)
    if !(first_result isa NamedTuple && hasproperty(first_result, :states))
        throw(ArgumentError("Without a session info, the results must be what turing_behavioural_session returns for each sample, but they are of type $(typeof(first_result)). The results of a full model or of several sessions are given with their MultipleSessionsInfo."))
    end
    entries = empty_entries()
    add_session_entries!(entries, results, (), identity)
    return chain_of_entries(entries, size(results)...; iter_indices, chain_indices)
end

"""
    add_session_entries!(entries, results, session_prefix, saved_values_of)

Add an entry to `entries` for everything one session saved, where `saved_values_of` gives
the session's saved values from the result of one sample, and `session_prefix` holds the
session's name, or nothing for a single session. There is one entry for each observation,
action and action distribution that was saved, under its kind, and one for each state, or
one for all of them when a saved state isn't a `NamedTuple`. Each entry holds, for every
sample, a vector over the timesteps. Used by [`session_values_chain`](@ref).
"""
function add_session_entries!(entries, results, session_prefix::Tuple{Vararg{Symbol}}, saved_values_of)
    first_saved = saved_values_of(first(results))

    #Each observation, action and action distribution that was saved, under its kind
    for kind in (:observations, :actions, :action_distributions)
        for name in saved_names(getproperty(first_saved, kind))
            add_entry!(entries, entry_name(session_prefix..., kind, name), map(result -> [getproperty(value, name) for value in getproperty(saved_values_of(result), kind)], results))
        end
    end

    #Each state, or all of them in one entry when a saved state isn't a NamedTuple
    states = first_saved.states
    if !isempty(states) && !(first(states) isa NamedTuple)
        add_entry!(entries, entry_name(session_prefix..., :states), map(result -> saved_values_of(result).states, results))
    else
        for name in saved_names(states)
            add_entry!(entries, entry_name(session_prefix..., name), map(result -> [getproperty(state, name) for state in saved_values_of(result).states], results))
        end
    end
    return entries
end

"""
    saved_names(saved_values)

Return the names of what was saved of one kind, a vector with a `NamedTuple` for each
timestep, which has none when nothing of that kind was saved. Used by
[`add_session_entries!`](@ref).
"""
saved_names(saved_values::AbstractVector) = isempty(saved_values) ? () : keys(first(saved_values))

"""
    empty_entries()

Return the entries of a chain before any is added: the values of each entry under its key in
a `Dict`, which has no order, and the names of the entries in the order they are added. Used
by [`session_values_chain`](@ref).
"""
empty_entries() = (; values = Dict{Any, AbstractMatrix}(), names = VarName[])

"""
    add_entry!(entries, name, values)

Add the entry `name` with `values`, one for each sample, to `entries`, and throw an
informative error if an entry has that name already, as when a parameter and a state have
the same name. Used by [`add_session_entries!`](@ref) and [`session_values_chain`](@ref).
"""
function add_entry!(entries, name::VarName, values::AbstractMatrix)
    key = Parameter(name)
    if haskey(entries.values, key)
        throw(ArgumentError("Two values would have the same name in the chain, $name, as when a parameter and a saved state of a session are named the same."))
    end
    entries.values[key] = Matrix(values)
    push!(entries.names, name)
    return entries
end

"""
    chain_of_entries(entries, n_iterations, n_chains; iter_indices, chain_indices)

Return a `FlexiChain` with `entries`, in the order they were added, and with the samples
numbered by `iter_indices` and `chain_indices`. A `FlexiChain` keeps the order of the
dictionary it is made from, and a `Dict` has none, so the chain is indexed with the names in
their order afterwards, which gives a chain with the entries in that order. Used by
[`session_values_chain`](@ref).
"""
function chain_of_entries(entries, n_iterations::Integer, n_chains::Integer; iter_indices, chain_indices)
    chain = FlexiChain{VarName}(n_iterations, n_chains, entries.values; iter_indices, chain_indices)
    return chain[[Parameter(name) for name in entries.names]]
end

"""
    session_symbol(session_id)

Return the identifier of a session as the first part of the names of its entries: as it is
for a single session column, and with the values joined by underscores, as in `S1_2`, for
several. Used by [`session_values_chain`](@ref).
"""
session_symbol(session_id) = Symbol(session_id)
session_symbol(session_id::NamedTuple) = Symbol(join(map(string, values(session_id)), "_"))

"""
    check_results(results)

Throw an informative error if there are no results, or if a sample has `nothing` as its
result, which a model that catches invalid parameters gives for a rejected sample. Used by
[`session_values_chain`](@ref).
"""
function check_results(results::AbstractMatrix)
    isempty(results) && throw(ArgumentError("There are no results to make a chain of."))
    if any(isnothing, results)
        throw(ArgumentError("A sample has nothing as its result, which a model that catches invalid parameters gives for a rejected sample. A chain only holds accepted samples, so make the results with returned(model, chain)."))
    end
    return nothing
end

"""
    check_number_of_sessions(saved_values, sessions_info)

Throw an informative error if the results have another number of sessions than
`sessions_info`. Used by [`session_values_chain`](@ref).
"""
function check_number_of_sessions(saved_values::AbstractVector, sessions_info::MultipleSessionsInfo)
    if length(saved_values) != length(sessions_info.session_ids)
        throw(DimensionMismatch("The results have $(length(saved_values)) sessions, but the session info has $(length(sessions_info.session_ids))."))
    end
    return nothing
end
