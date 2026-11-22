### THE TRAJECTORIES OF FITTED SESSIONS, AS A CHAIN ###

#The returned values of a full model, or of several sessions, with every entry named after the session it belongs to
function CognitiveModeling.get_session_trajectories(
    returned_values::AbstractMatrix,
    sessions_info::MultipleSessionsInfo;
    iter_indices::AbstractVector{<:Integer}=1:size(returned_values, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(returned_values, 2),
)
    check_returned_values(returned_values)
    check_returned_values_of_multiple_sessions(first(returned_values), sessions_info)

    #The returned values of a full model and of several sessions hold what each session saved under saved_trajectories
    entries = empty_entries()
    for (i, session_id) in enumerate(sessions_info.session_ids)
        add_session_entries!(
            entries,
            returned_values,
            (session_id_to_symbol(session_id),),
            returned_value -> returned_value.saved_trajectories[i],
        )
    end
    return chain_of_entries(entries, size(returned_values)...; iter_indices, chain_indices)
end

#The returned values of a single session, with every entry named without the symbol name of a session
function CognitiveModeling.get_session_trajectories(
    returned_values::AbstractMatrix;
    iter_indices::AbstractVector{<:Integer}=1:size(returned_values, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(returned_values, 2),
)
    check_returned_values(returned_values)
    check_returned_values_of_single_session(first(returned_values))
    entries = empty_entries()
    add_session_entries!(
        entries, returned_values, (), returned_value -> returned_value.saved_trajectories
    )
    return chain_of_entries(entries, size(returned_values)...; iter_indices, chain_indices)
end

### THE PARAMETERS OF FITTED SESSIONS, AS A CHAIN ###

#The parameters of every session from the returned values of a full model or of several sessions, named after the session, with the environment's under environment
function CognitiveModeling.get_session_parameters(
    returned_values::AbstractMatrix,
    sessions_info::MultipleSessionsInfo;
    iter_indices::AbstractVector{<:Integer}=1:size(returned_values, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(returned_values, 2),
)
    check_returned_values(returned_values)
    check_returned_values_of_multiple_sessions(first(returned_values), sessions_info)

    #Each parameter has one vector with a value for every session, so session i has the value i of each
    entries = empty_entries()
    for (i, session_id) in enumerate(sessions_info.session_ids)
        symbol_name_of_session = session_id_to_symbol(session_id)
        add_parameter_entries!(
            entries,
            returned_values,
            (symbol_name_of_session,),
            returned_value -> map(
                values_of_sessions -> values_of_sessions[i],
                returned_value.cognitive_parameters,
            ),
        )
        add_parameter_entries!(
            entries,
            returned_values,
            (symbol_name_of_session, :environment),
            returned_value -> map(
                values_of_sessions -> values_of_sessions[i],
                returned_value.environment_parameters,
            ),
        )
    end
    return chain_of_entries(entries, size(returned_values)...; iter_indices, chain_indices)
end

#The parameters of a single session, named after the parameter alone, with the environment's under environment
function CognitiveModeling.get_session_parameters(
    returned_values::AbstractMatrix;
    iter_indices::AbstractVector{<:Integer}=1:size(returned_values, 1),
    chain_indices::AbstractVector{<:Integer}=1:size(returned_values, 2),
)
    check_returned_values(returned_values)
    check_returned_values_of_single_session(first(returned_values))
    entries = empty_entries()
    add_parameter_entries!(
        entries, returned_values, (), returned_value -> returned_value.cognitive_parameters
    )
    add_parameter_entries!(
        entries,
        returned_values,
        (:environment,),
        returned_value -> returned_value.environment_parameters,
    )
    return chain_of_entries(entries, size(returned_values)...; iter_indices, chain_indices)
end

### ADDING THE ENTRIES OF EACH SESSION ###

"""
    add_parameter_entries!(entries, returned_values, prefix, parameters_of)

Add an entry under `prefix` to `entries` for each parameter of one session, where
`parameters_of` gives the session's parameters from the returned value of one sample, as a
`NamedTuple` with the value of each parameter. Used by [`get_session_parameters`](@ref).
"""
function add_parameter_entries!(
    entries, returned_values, prefix::Tuple{Vararg{Symbol}}, parameters_of
)
    for parameter in keys(parameters_of(first(returned_values)))
        add_entry!(
            entries,
            entry_name(prefix..., parameter),
            map(
                returned_value -> getproperty(parameters_of(returned_value), parameter),
                returned_values,
            ),
        )
    end
    return entries
end

"""
    add_session_entries!(entries, returned_values, session_prefix, saved_trajectories_of)

Add an entry to `entries` for everything one session saved, where `saved_trajectories_of`
gives the session's saved trajectories from the returned value of one sample, and
`session_prefix` holds the session's symbol name, or nothing for a single session. The
cognitive model's values are named under the session alone, and the environment's under
`environment` after it, as [`add_model_entries!`](@ref) names them. When the models of the
session can set `parameters_valid`, which a `Bool` shows, an entry `parameters_valid` under
the session says for each sample whether its parameters stayed valid. Used by
[`get_session_trajectories`](@ref).
"""
function add_session_entries!(
    entries, returned_values, session_prefix::Tuple{Vararg{Symbol}}, saved_trajectories_of
)
    add_model_entries!(
        entries,
        returned_values,
        session_prefix,
        saved_trajectories_of,
        :action_distributions,
    )
    first_saved = saved_trajectories_of(first(returned_values))
    if hasproperty(first_saved, :environment)
        add_model_entries!(
            entries,
            returned_values,
            (session_prefix..., :environment),
            returned_value -> saved_trajectories_of(returned_value).environment,
            :observation_distributions,
        )
    end

    #Whether the parameters stayed valid, for models that can set it
    if hasproperty(first_saved, :parameters_valid) && first_saved.parameters_valid isa Bool
        add_entry!(
            entries,
            entry_name(session_prefix..., :parameters_valid),
            map(
                returned_value -> saved_trajectories_of(returned_value).parameters_valid,
                returned_values,
            ),
        )
    end
    return entries
end

"""
    add_model_entries!(entries, returned_values, prefix, saved_trajectories_of, distribution_kind)

Add an entry under `prefix` to `entries` for everything one model of a session saved, where
`saved_trajectories_of` gives its saved trajectories from the returned value of one sample,
with `states`, `observations`, `actions` and `distribution_kind`, which is
`action_distributions` for a cognitive model and `observation_distributions` for an
environment model. There is one entry for each observation and action that was saved, named
after it alone, one for each distribution, under its kind, and one for each state. Each
entry holds, for every sample, a vector over the timesteps. Used by
[`add_session_entries!`](@ref).
"""
function add_model_entries!(
    entries,
    returned_values,
    prefix::Tuple{Vararg{Symbol}},
    saved_trajectories_of,
    distribution_kind,
)
    first_saved = saved_trajectories_of(first(returned_values))

    #Each observation and action that was saved, named after it alone
    for kind in (:observations, :actions)
        for name in saved_names(getproperty(first_saved, kind))
            add_entry!(
                entries,
                entry_name(prefix..., name),
                map(
                    returned_value -> [
                        getproperty(value, name) for value in
                        getproperty(saved_trajectories_of(returned_value), kind)
                    ],
                    returned_values,
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
                returned_value -> [
                    getproperty(value, name) for value in getproperty(
                        saved_trajectories_of(returned_value), distribution_kind
                    )
                ],
                returned_values,
            ),
        )
    end

    #Each state that was saved, named after it alone
    for name in saved_names(first_saved.states)
        add_entry!(
            entries,
            entry_name(prefix..., name),
            map(
                returned_value -> [
                    getproperty(state, name) for
                    state in saved_trajectories_of(returned_value).states
                ],
                returned_values,
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
by [`get_session_trajectories`](@ref).
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
[`get_session_trajectories`](@ref).
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
    check_returned_values(returned_values)

Throw an informative error if there are no returned values, or if a sample has `nothing` as
its returned value. Used by [`get_session_trajectories`](@ref).
"""
function check_returned_values(returned_values::AbstractMatrix)
    isempty(returned_values) &&
        throw(ArgumentError("There are no returned values to make a chain of."))
    if any(isnothing, returned_values)
        throw(
            ArgumentError(
                "A sample has nothing as its returned value. The returned values must be what the model of the sessions returns for every sample, as returned(model, chain) gives them.",
            ),
        )
    end
    return nothing
end

"""
    check_returned_values_of_multiple_sessions(returned_value, sessions_info)
    check_returned_values_of_single_session(returned_value)

Throw an informative error if `returned_value`, the returned value of the first sample,
isn't what the models of several sessions return, [`turing_full_model`](@ref) and
[`turing_multiple_behavioral_sessions`](@ref), with what each session saved in a vector
under `saved_trajectories`, for as many sessions as `sessions_info` has, or what
[`turing_behavioral_session`](@ref) returns, with what the session saved under
`saved_trajectories`. Used by [`get_session_trajectories`](@ref) and
[`get_session_parameters`](@ref).
"""
function check_returned_values_of_multiple_sessions(
    returned_value, sessions_info::MultipleSessionsInfo
)
    if !(
        returned_value isa NamedTuple &&
        hasproperty(returned_value, :saved_trajectories) &&
        returned_value.saved_trajectories isa AbstractVector
    )
        throw(
            ArgumentError(
                "With a MultipleSessionsInfo, the returned values must be what the model of full_model or of multiple_sessions_model returns for each sample, but they are of type $(typeof(returned_value)). The returned values of a single session are given without a session info.",
            ),
        )
    end
    check_number_of_sessions(returned_value.saved_trajectories, sessions_info)
    return nothing
end
function check_returned_values_of_single_session(returned_value)
    if !(
        returned_value isa NamedTuple &&
        hasproperty(returned_value, :saved_trajectories) &&
        returned_value.saved_trajectories isa NamedTuple
    )
        throw(
            ArgumentError(
                "Without a session info, the returned values must be what the model of session_model returns for each sample, but they are of type $(typeof(returned_value)). The returned values of a full model or of several sessions are given with their MultipleSessionsInfo.",
            ),
        )
    end
    return nothing
end

"""
    check_number_of_sessions(saved_trajectories, sessions_info)

Throw an informative error if the returned values have another number of sessions than
`sessions_info`. Used by [`check_returned_values_of_multiple_sessions`](@ref).
"""
function check_number_of_sessions(
    saved_trajectories::AbstractVector, sessions_info::MultipleSessionsInfo
)
    if length(saved_trajectories) != length(sessions_info.session_ids)
        throw(
            DimensionMismatch(
                "The returned values have $(length(saved_trajectories)) sessions, but the session info has $(length(sessions_info.session_ids)).",
            ),
        )
    end
    return nothing
end
