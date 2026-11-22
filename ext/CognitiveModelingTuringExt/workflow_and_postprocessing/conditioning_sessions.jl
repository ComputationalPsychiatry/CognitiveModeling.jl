### SESSIONS WHOSE DATA IS SIMULATED INSTEAD OF FITTED TO, AND THE OTHER WAY ROUND ###

#The models of the package take the data of their sessions as arguments, which Turing.jl's decondition leaves as they are, so a copy of the model is made whose sessions are deconditioned. Turing.jl's own functions then simulate the data, and condition and decondition work on the copy as on any model
function CognitiveModeling.decondition_sessions(
    model::DynamicPPL.Model; check_stopping_criterion::Union{Nothing,Bool}=nothing
)
    #A model of one's own has no session info that the package can find, so its sessions are deconditioned before it is given them
    if !is_session_model(model)
        throw(
            ArgumentError(
                "decondition_sessions(model) takes a model made by full_model, multiple_sessions_model or session_model, but this one is made by $(model.f). For a model of one's own, give it the sessions that decondition_sessions(sessions_info) gives instead.",
            ),
        )
    end
    return with_sessions_simulated(
        model,
        info -> decondition_sessions(info; check_stopping_criterion),
        "decondition_sessions",
    )
end

#Several sessions are named after whether their data is simulated, with name_of_session, while a single session has no name of its own in the model. It is given one with DynamicPPL.jl's prefix, so that Turing.jl's predict draws every variable within it again, whatever a chain of a fit to it holds
"""
    with_sessions_simulated(model, change, function_name)

Return a copy of `model`, a Turing.jl model made by `turing_full_model`,
`turing_multiple_behavioral_sessions` or `turing_behavioral_session`, whose session info is
what `change` makes of its own, with [`with_session_info`](@ref), when `change` makes its
sessions simulate their data. A single session whose data wasn't simulated before has its
variables under the prefix `simulated_session` in the copy, as several sessions have under
`simulated_sessions[i]`. Any other model is an error, which names `function_name`, the
function that was called. Used by `decondition_sessions` and `predict_session_trajectories`.
"""
function with_sessions_simulated(model::DynamicPPL.Model, change, function_name)
    deconditioned = with_session_info(model, change, function_name)
    if model.f === CognitiveModeling.turing_behavioral_session &&
        !simulates_values(info_of(model))
        return DynamicPPL.prefix(deconditioned, @varname(simulated_session))
    end
    return deconditioned
end

#The values go in the data of the sessions, which the models take as arguments, so that the copy is fitted to them as to any data. Turing.jl's condition on the variables gives the same posterior, but looks up every value by its name on each evaluation, which is slower
function CognitiveModeling.condition_sessions(
    model::DynamicPPL.Model,
    values::Union{AbstractDict{<:VarName},DynamicPPL.VarNamedTuple};
    check_stopping_criterion::Union{Nothing,Bool}=nothing,
)
    #A model of one's own has no session info that the package can find, so its sessions are given the values in their data before it is given them
    if !is_session_model(model)
        throw(
            ArgumentError(
                "condition_sessions(model, values) takes a model made by full_model, multiple_sessions_model or session_model, but this one is made by $(model.f). For a model of one's own, give it sessions with the values in their data instead.",
            ),
        )
    end

    #The values under the names of their variables, written out as in simulated_sessions[2].timesteps[3].choice.simulated, and the names of those that go in the data
    values_by_name = Dict{String,Any}(string(name) => values[name] for name in keys(values))
    used_names = Set{String}()
    model_prefix = prefix_of_model(model)
    conditioned_model = with_session_info(
        model,
        info -> with_values_in_data(
            info, model_prefix, values_by_name, used_names; check_stopping_criterion
        ),
        "condition_sessions",
    )

    #A value that has no place in the data is an error, such as that of a parameter, or of a variable that is misspelled
    check_every_value_used(values_by_name, used_names, info_of(model), model_prefix)
    return conditioned_model
end

"""
    prefix_of_model(model)

Return the prefixes that the variables of `model`, a Turing.jl model, have from its context,
as text followed by a dot, such as `simulated_session.` for a single session that
`decondition_sessions` gave, or an empty string when it has none. Used by
`condition_sessions`.
"""
function prefix_of_model(model::DynamicPPL.Model)
    prefixes = last(DynamicPPL.extract_prefixes(model.context))
    return prefixes === nothing ? "" : string(prefixes, ".")
end

"""
    with_values_in_data(info, model_prefix, values_by_name, used_names;
                        check_stopping_criterion)

Return a copy of `info`, a `SessionInfo` or a `MultipleSessionsInfo`, whose sessions have
the values in `values_by_name` in their data, as `condition_sessions` describes, and add the
names of the values that went in to `used_names`. The variables of a single session are
under `model_prefix`, the prefixes of its model, and those of session `i` of several under
[`name_of_session`](@ref) after them. Each session ends after its last timestep with data,
and the columns of a name get the same element type in every session, so that the sessions
keep one type. Used by `condition_sessions`.
"""
function with_values_in_data(
    session_info::SessionInfo,
    model_prefix::String,
    values_by_name::Dict{String,Any},
    used_names::Set{String};
    check_stopping_criterion::Union{Nothing,Bool},
)
    observations, actions = columns_with_values(
        session_info, model_prefix, values_by_name, used_names
    )

    #The session ends after its last timestep with data
    data = cut_after_last_timestep_with_data(
        observations, actions, session_info.timestep_sizes
    )
    return with_data(
        session_info,
        only(with_common_element_types([data.observations])),
        only(with_common_element_types([data.actions]));
        check_stopping_criterion,
        timestep_sizes=data.timestep_sizes,
    )
end

function with_values_in_data(
    sessions_info::MultipleSessionsInfo,
    model_prefix::String,
    values_by_name::Dict{String,Any},
    used_names::Set{String};
    check_stopping_criterion::Union{Nothing,Bool},
)
    #The data of every session with the values in it, where each session ends after its last timestep with data
    data = map(enumerate(sessions_info.sessions)) do (i, session)
        session_prefix = "$(model_prefix)$(name_of_session(sessions_info, i))."
        observations, actions = columns_with_values(
            session, session_prefix, values_by_name, used_names
        )
        return cut_after_last_timestep_with_data(
            observations, actions, session.timestep_sizes
        )
    end

    #The columns of a name get the same element type in every session
    observations = with_common_element_types(map(session -> session.observations, data))
    actions = with_common_element_types(map(session -> session.actions, data))

    #Each session is prepared again with its new data
    sessions = [
        with_data(
            session,
            observations[i],
            actions[i];
            check_stopping_criterion,
            timestep_sizes=data[i].timestep_sizes,
        ) for (i, session) in enumerate(sessions_info.sessions)
    ]

    #The sessions keep their data identifiers, and the rows in the table of the timesteps they have left
    session_rows = if sessions_info.session_rows === nothing
        nothing
    else
        [
            rows[1:(session.n_timesteps)] for
            (rows, session) in zip(sessions_info.session_rows, sessions)
        ]
    end
    return MultipleSessionsInfo(
        sessions, sessions_info.session_ids; table=sessions_info.table, session_rows
    )
end

"""
    columns_with_values(session_info, prefix, values_by_name, used_names)

Return the observations and the actions in the data of `session_info`, a `SessionInfo`, each
a `NamedTuple` with a `Vector{Any}` for each name, in which every entry that is
`ImputedMissingValue()` or `SimulatedMissingValue()` is the value in `values_by_name` of its
variable, when there is one, with [`value_of_entry`](@ref). The variable of an action is
named after `prefix`, such as `simulated_sessions[2].`, and its timestep, as in
`simulated_sessions[2].timesteps[3].choice.simulated`, and that of an observation after the
name of the environment's output that the observation map passes on as it, as in
`simulated_sessions[2].timesteps[3].environment.payoff.simulated`. An output of the
environment that the map doesn't pass on has no place in the data. Used by
`with_values_in_data`.
"""
function columns_with_values(
    session_info::SessionInfo,
    prefix::String,
    values_by_name::Dict{String,Any},
    used_names::Set{String},
)
    #Each action is the value of its variable, under the session and the timestep
    action_names = keys(session_info.actions)
    actions = NamedTuple{action_names}(
        map(action_names) do name
            return Any[
                value_of_entry(
                    entry, "$(prefix)timesteps[$t].$name", values_by_name, used_names
                ) for (t, entry) in enumerate(getproperty(session_info.actions, name))
            ]
        end,
    )

    #Each observation is the value of the variable of the environment's output which the map passes on as it. A session without an environment has none, since nothing samples its observations
    observations = map(column -> Vector{Any}(column), session_info.observations)
    if has_environment(session_info)
        for (environment_name, column) in pairs(session_info.environment_observations)
            observation_column = value_passed_on(
                observations, environment_name, session_info.observation_map, nothing
            )
            observation_column === nothing && continue
            for (t, entry) in enumerate(column)
                observation_column[t] = value_of_entry(
                    entry,
                    "$(prefix)timesteps[$t].environment.$environment_name",
                    values_by_name,
                    used_names,
                )
            end
        end
    end
    return observations, actions
end

#The last parts of the names are those that turing_impute_value gives the variables
"""
    value_of_entry(entry, name_of_value, values_by_name, used_names)

Return the value in `values_by_name` of the variable of `entry`, an action or an observation
in the data, when it is `ImputedMissingValue()` or `SimulatedMissingValue()` and has one,
and add the variable's name to `used_names`. The variable is named `name_of_value` followed
by `imputed` or `simulated`. Any other entry, and one whose variable has no value, is
returned as it is. Used by `columns_with_values`.
"""
value_of_entry(entry, name_of_value::String, values_by_name, used_names) = entry
function value_of_entry(
    entry::LatentValueMarker,
    name_of_value::String,
    values_by_name::Dict{String,Any},
    used_names::Set{String},
)
    variable_name = "$name_of_value.$(entry isa SimulatedMissingValue ? "simulated" : "imputed")"
    haskey(values_by_name, variable_name) || return entry
    push!(used_names, variable_name)
    return values_by_name[variable_name]
end

"""
    check_every_value_used(values_by_name, used_names, info, model_prefix)

Throw an `ArgumentError` when a value in `values_by_name` didn't go in the data of the
sessions of `info`, a `SessionInfo` or a `MultipleSessionsInfo`, because its name isn't in
`used_names`. The error names the first five of them, and how the variable of an action of
the sessions is named, after `model_prefix`, the prefixes of their model. Used by
`condition_sessions`.
"""
function check_every_value_used(
    values_by_name::Dict{String,Any}, used_names::Set{String}, info, model_prefix::String
)
    unused_names = sort!(collect(setdiff(keys(values_by_name), used_names)))
    isempty(unused_names) && return nothing

    #The first five names, and the variable of the first action of the first session, as an example of how they are named
    named = join(first(unused_names, 5), ", ")
    n_more = length(unused_names) - 5
    more = n_more > 0 ? ", and $n_more more" : ""
    first_session = info isa SessionInfo ? info : first(info.sessions)
    session_prefix = if info isa SessionInfo
        model_prefix
    else
        "$(model_prefix)$(name_of_session(info, 1))."
    end
    example = "$(session_prefix)timesteps[1].$(first(keys(first_session.actions))).simulated"
    return throw(
        ArgumentError(
            "condition_sessions(model, values) puts each value in the data of the sessions, in place of an action or an observation that they simulate or impute, but the sessions have none for $named$more. Such variables are named as in $example, and others can be conditioned with Turing.jl's condition.",
        ),
    )
end
