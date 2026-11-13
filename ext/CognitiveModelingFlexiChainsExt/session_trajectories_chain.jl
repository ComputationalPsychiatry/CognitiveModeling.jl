### THE TRAJECTORIES OF FITTED SESSIONS, AS A CHAIN ###

#The results of a full model, or of several sessions, with every entry named after the session it belongs to
function CognitiveModeling.session_trajectories_chain(
    results::AbstractMatrix,
    sessions_info::MultipleSessionsInfo;
    iter_indices::AbstractVector{<:Integer}=1:size(results, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(results, 2),
)
    check_results(results)
    first_result = first(results)
    entries = empty_entries()

    if first_result isa NamedTuple &&
        hasproperty(first_result, :saved_behavioral_trajectories)
        #The results of a full model hold what each session saved under saved_behavioral_trajectories
        check_number_of_sessions(first_result.saved_behavioral_trajectories, sessions_info)
        for (i, session_id) in enumerate(sessions_info.session_ids)
            add_session_entries!(
                entries,
                results,
                (session_name(session_id),),
                result -> result.saved_behavioral_trajectories[i],
            )
        end
    elseif first_result isa AbstractVector
        #The results of several sessions are a vector with what each of them saved
        check_number_of_sessions(first_result, sessions_info)
        for (i, session_id) in enumerate(sessions_info.session_ids)
            add_session_entries!(
                entries, results, (session_name(session_id),), result -> result[i]
            )
        end
    else
        #Anything else isn't known
        throw(
            ArgumentError(
                "With a MultipleSessionsInfo, the results must be what turing_full_model or turing_multiple_behavioral_sessions returns for each sample, but they are of type $(typeof(first_result)). The results of a single session are given without a session info.",
            ),
        )
    end

    return chain_of_entries(entries, size(results)...; iter_indices, chain_indices)
end

#The results of a single session, with every entry named without a session identifier
function CognitiveModeling.session_trajectories_chain(
    results::AbstractMatrix;
    iter_indices::AbstractVector{<:Integer}=1:size(results, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(results, 2),
)
    check_results(results)
    first_result = first(results)
    if !(first_result isa NamedTuple && hasproperty(first_result, :states))
        throw(
            ArgumentError(
                "Without a session info, the results must be what turing_behavioral_session returns for each sample, but they are of type $(typeof(first_result)). The results of a full model or of several sessions are given with their MultipleSessionsInfo.",
            ),
        )
    end
    entries = empty_entries()
    add_session_entries!(entries, results, (), identity)
    return chain_of_entries(entries, size(results)...; iter_indices, chain_indices)
end

### THE PARAMETERS OF FITTED SESSIONS, AS A CHAIN ###

#The parameters of every session from the results of a full model, named after the session, with the environment's under environment
function CognitiveModeling.session_parameters_chain(
    results::AbstractMatrix,
    sessions_info::MultipleSessionsInfo;
    iter_indices::AbstractVector{<:Integer}=1:size(results, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(results, 2),
)
    check_results(results)
    first_result = first(results)
    if !(first_result isa NamedTuple && hasproperty(first_result, :cognitive_parameters))
        throw(
            ArgumentError(
                "The results must be what turing_full_model returns for each sample, which holds the parameters of every session, but they are of type $(typeof(first_result)).",
            ),
        )
    end
    check_number_of_sessions(first_result.saved_behavioral_trajectories, sessions_info)
    entries = empty_entries()
    for (i, session_id) in enumerate(sessions_info.session_ids)
        name_of_session = session_name(session_id)
        add_parameter_entries!(
            entries, results, (name_of_session,), result -> result.cognitive_parameters, i
        )
        add_parameter_entries!(
            entries,
            results,
            (name_of_session, :environment),
            result -> result.environment_parameters,
            i,
        )
    end
    return chain_of_entries(entries, size(results)...; iter_indices, chain_indices)
end

### ADDING THE ENTRIES OF EACH SESSION ###

"""
    add_parameter_entries!(entries, results, prefix, parameters_of, i)

Add an entry under `prefix` to `entries` for each parameter of session `i`, where
`parameters_of` gives the parameters of every session from the result of one sample, as a
`NamedTuple` with one vector for each parameter. Used by
[`session_parameters_chain`](@ref).
"""
function add_parameter_entries!(
    entries, results, prefix::Tuple{Vararg{Symbol}}, parameters_of, i
)
    for parameter in keys(parameters_of(first(results)))
        add_entry!(
            entries,
            entry_name(prefix..., parameter),
            map(result -> getproperty(parameters_of(result), parameter)[i], results),
        )
    end
    return entries
end

"""
    add_session_entries!(entries, results, session_prefix, saved_trajectories_of)

Add an entry to `entries` for everything one session saved, where `saved_trajectories_of`
gives the session's saved trajectories from the result of one sample, and `session_prefix`
holds the session's name, or nothing for a single session. The cognitive model's values are
named under the session alone, and the environment's under `environment` after it, as
[`add_model_entries!`](@ref) names them. Used by [`session_trajectories_chain`](@ref).
"""
function add_session_entries!(
    entries, results, session_prefix::Tuple{Vararg{Symbol}}, saved_trajectories_of
)
    add_model_entries!(
        entries, results, session_prefix, saved_trajectories_of, :action_distributions
    )
    if hasproperty(saved_trajectories_of(first(results)), :environment)
        add_model_entries!(
            entries,
            results,
            (session_prefix..., :environment),
            result -> saved_trajectories_of(result).environment,
            :observation_distributions,
        )
    end
    return entries
end

"""
    add_model_entries!(entries, results, prefix, saved_trajectories_of, distribution_kind)

Add an entry under `prefix` to `entries` for everything one model of a session saved, where
`saved_trajectories_of` gives its saved trajectories from the result of one sample, with
`states`, `observations`, `actions` and `distribution_kind`, which is `action_distributions`
for a cognitive model and `observation_distributions` for an environment model. There is one
entry for each observation and action that was saved, named after it alone, one for each
distribution, under its kind, and one for each state. Each entry holds, for every sample, a
vector over the timesteps. Used by [`add_session_entries!`](@ref).
"""
function add_model_entries!(
    entries,
    results,
    prefix::Tuple{Vararg{Symbol}},
    saved_trajectories_of,
    distribution_kind,
)
    first_saved = saved_trajectories_of(first(results))

    #Each observation and action that was saved, named after it alone
    for kind in (:observations, :actions)
        for name in saved_names(getproperty(first_saved, kind))
            add_entry!(
                entries,
                entry_name(prefix..., name),
                map(
                    result -> [
                        getproperty(value, name) for
                        value in getproperty(saved_trajectories_of(result), kind)
                    ],
                    results,
                ),
            )
        end
    end

    #Each distribution that was saved, under its kind
    for name in saved_names(getproperty(first_saved, distribution_kind))
        add_entry!(
            entries,
            entry_name(prefix..., distribution_kind, name),
            map(
                result -> [
                    getproperty(value, name) for value in
                    getproperty(saved_trajectories_of(result), distribution_kind)
                ],
                results,
            ),
        )
    end

    #Each state that was saved, named after it alone
    for name in saved_names(first_saved.states)
        add_entry!(
            entries,
            entry_name(prefix..., name),
            map(
                result -> [
                    getproperty(state, name) for
                    state in saved_trajectories_of(result).states
                ],
                results,
            ),
        )
    end
    return entries
end

"""
    saved_names(saved_trajectories)

Return the names of what was saved of one kind, a vector with a `NamedTuple` for each
timestep, which has none when nothing of that kind was saved. Used by
[`add_session_entries!`](@ref).
"""
function saved_names(saved_trajectories::AbstractVector)
    return isempty(saved_trajectories) ? () : keys(first(saved_trajectories))
end

### BUILDING THE CHAIN ###

"""
    empty_entries()

Return the entries of a chain before any is added: the values of each entry under its key in
a `Dict`, which has no order, and the names of the entries in the order they are added. Used
by [`session_trajectories_chain`](@ref).
"""
empty_entries() = (; values=Dict{Any,AbstractMatrix}(), names=VarName[])

"""
    add_entry!(entries, name, values)

Add the entry `name` with `values`, one for each sample, to `entries`, and throw an
informative error if an entry has that name already, as when a state is named like a kind of
entry. Used by [`add_model_entries!`](@ref) and [`add_parameter_entries!`](@ref).
"""
function add_entry!(entries, name::VarName, values::AbstractMatrix)
    key = Parameter(name)
    if haskey(entries.values, key)
        throw(
            ArgumentError(
                "Two values would have the same name in the chain, $name, as when a saved state of a session is named like a kind of entry.",
            ),
        )
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
[`session_trajectories_chain`](@ref).
"""
function chain_of_entries(
    entries, n_iterations::Integer, n_chains::Integer; iter_indices, chain_indices
)
    chain = FlexiChain{VarName}(
        n_iterations, n_chains, entries.values; iter_indices, chain_indices
    )
    return chain[[Parameter(name) for name in entries.names]]
end

### CHECKS ###

"""
    check_results(results)

Throw an informative error if there are no results, or if a sample has `nothing` as its
result, which a model that catches invalid parameters gives for a rejected sample. Used by
[`session_trajectories_chain`](@ref).
"""
function check_results(results::AbstractMatrix)
    isempty(results) && throw(ArgumentError("There are no results to make a chain of."))
    if any(isnothing, results)
        throw(
            ArgumentError(
                "A sample has nothing as its result, which a model that catches invalid parameters gives for a rejected sample. A chain only holds accepted samples, so make the results with returned(model, chain).",
            ),
        )
    end
    return nothing
end

"""
    check_number_of_sessions(saved_trajectories, sessions_info)

Throw an informative error if the results have another number of sessions than
`sessions_info`. Used by [`session_trajectories_chain`](@ref).
"""
function check_number_of_sessions(
    saved_trajectories::AbstractVector, sessions_info::MultipleSessionsInfo
)
    if length(saved_trajectories) != length(sessions_info.session_ids)
        throw(
            DimensionMismatch(
                "The results have $(length(saved_trajectories)) sessions, but the session info has $(length(sessions_info.session_ids)).",
            ),
        )
    end
    return nothing
end
