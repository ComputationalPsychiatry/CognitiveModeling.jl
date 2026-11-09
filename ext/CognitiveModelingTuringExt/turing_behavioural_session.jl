### MODEL WHICH HAS ONE COMPONENT FOR THE PARAMETERS OF THE SESSIONS (e.g. a regression), AND THEN THE BEHAVIOURAL SESSIONS ###
# The parameters of the cognitive model and those of the environment model each come from a parameters model. There is also
# an optional third component which can do something with the outcomes of the behavioural sessions, such as a regression
# predicting something (e.g. physiological data) from the cognitive states.

#Creates the Turing model for fitting a cognitive model to several sessions, after warning when both parameters models can sample variables of the same name, and when the sessions save values no downstream model uses
#The warnings are given here, when the model is made, and not in the model, whose body is differentiated: a warning holds a try block, which Mooncake can't differentiate
function CognitiveModeling.turing_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    catch_invalid_parameters::Bool=false,
    parallelisation=NoParallelisation(),
    include_submodel_prefixes::Bool=false,
)
    warn_if_parameter_names_overlap(
        cognitive_parameters_model,
        environment_parameters_model,
        sessions_info,
        include_submodel_prefixes,
    )
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)
    return turing_run_full_model(
        cognitive_parameters_model,
        sessions_info;
        environment_parameters_model,
        downstream_model,
        catch_invalid_parameters,
        parallelisation,
        include_submodel_prefixes,
    )
end

"""
    turing_run_full_model(cognitive_parameters_model, sessions_info;
                          environment_parameters_model, downstream_model,
                          catch_invalid_parameters, parallelisation,
                          include_submodel_prefixes)

Turing.jl model which [`turing_full_model`](@ref) creates, with the same arguments: the
parameters models for the parameters of the cognitive model and of the environment model,
the sessions with those parameters, and the optional downstream model. It is a function of
its own so that `turing_full_model` can check the parameters models when the model is made,
outside the model's body.
"""
Turing.@model function turing_run_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoParameters(),
    downstream_model=NoDownstreamModel(),
    catch_invalid_parameters::Bool=false,
    parallelisation=NoParallelisation(),
    include_submodel_prefixes::Bool=false,
)

    #Give or sample the cognitive parameters of every session, with the variables under cognitive_parameters_model when the prefixes are included
    if include_submodel_prefixes
        cognitive_parameters ~ to_submodel(
            prefix(
                turing_session_parameters_model(cognitive_parameters_model, sessions_info),
                @varname(cognitive_parameters_model)
            ),
            false,
        )
    else
        cognitive_parameters ~ to_submodel(
            turing_session_parameters_model(cognitive_parameters_model, sessions_info),
            false,
        )
    end

    #Give or sample the parameters of the environment of every session, with the variables under environment_parameters_model when the prefixes are included
    if include_submodel_prefixes
        environment_parameters ~ to_submodel(
            prefix(
                turing_session_parameters_model(
                    environment_parameters_model, sessions_info
                ),
                @varname(environment_parameters_model)
            ),
            false,
        )
    else
        environment_parameters ~ to_submodel(
            turing_session_parameters_model(environment_parameters_model, sessions_info),
            false,
        )
    end

    #When the environment's parameters are sampled, every session runs its environment, so that its observations inform them
    sessions_to_run = sessions_for_environment_parameters(
        environment_parameters_model, sessions_info
    )

    #Run all behavioural sessions, inside a try block when invalid parameters are caught. The parameters are sampled outside it, so that they are kept when a draw is rejected, which a sampler needs to start
    sessions = turing_multiple_behavioural_sessions(
        sessions_to_run, cognitive_parameters, environment_parameters; parallelisation
    )
    if catch_invalid_parameters
        saved_behavioural_values ~
        to_submodel(turing_catch_invalid_parameters(sessions), false)

        #A rejected draw leaves nothing for the downstream model, so it returns at once, which Enzyme can differentiate where a value of two types can't be
        if saved_behavioural_values === nothing
            return nothing
        end
    else
        saved_behavioural_values ~ to_submodel(sessions, false)
    end

    #Do the optional downstream model, with its variables under downstream_model when the prefixes are included
    if include_submodel_prefixes
        downstream_values ~ to_submodel(
            prefix(
                turing_downstream_model(
                    downstream_model,
                    saved_behavioural_values,
                    cognitive_parameters,
                    sessions_info,
                ),
                @varname(downstream_model)
            ),
            false,
        )
    else
        downstream_values ~ to_submodel(
            turing_downstream_model(
                downstream_model,
                saved_behavioural_values,
                cognitive_parameters,
                sessions_info,
            ),
            false,
        )
    end

    return (;
        cognitive_parameters,
        environment_parameters,
        saved_behavioural_values,
        downstream_values,
    )
end

### MODEL FOR MULTIPLE SESSIONS ###

#Turing model for running several behavioural sessions, each with its own cognitive parameters, as a submodel with the prefix session[i], and collecting what each returns
#The environment parameters are given in the same form, one vector for each parameter with a value for each session, and only when the sessions have an environment
Turing.@model function CognitiveModeling.turing_multiple_behavioural_sessions(
    sessions_info::MultipleSessionsInfo,
    cognitive_parameters,
    environment_parameters=(;);
    parallelisation=NoParallelisation(),
)
    check_environment_parameters(sessions_info, environment_parameters)
    saved_values ~ to_submodel(
        turing_run_sessions(
            parallelisation, sessions_info, cognitive_parameters, environment_parameters
        ),
        false,
    )
    return saved_values
end

"""
    turing_run_sessions(parallelisation, sessions_info, cognitive_parameters,
                        environment_parameters)

Turing.jl model which runs every session in `sessions_info` as a submodel with the prefix
`session[i]`, and returns a vector with what each returned. The first session runs outside
the loop, so that the vector has its type. Sessions of one type can still save values of
different types, such as an action that is `NoAction()` in one session only, and a session
whose values have another type widens the vector, as [`store_output!`](@ref) does. With
[`NoParallelisation`](@ref) the sessions run one after the other, and with
[`ParallelSessions`](@ref) over the threads Julia was started with, which needs the model
that is evaluated to be marked thread-safe, since the log density is silently wrong
otherwise.
"""
Turing.@model function turing_run_sessions(
    ::NoParallelisation, sessions_info, cognitive_parameters, environment_parameters
)
    sessions = sessions_info.sessions
    first_saved ~ to_submodel(
        prefix(
            turing_behavioural_session(
                sessions[1], row(cognitive_parameters, 1), row(environment_parameters, 1)
            ),
            @varname(session[1])
        ),
        false,
    )
    saved_values = initialize_storage(first_saved, length(sessions))
    for i in 2:length(sessions)
        saved ~ to_submodel(
            prefix(
                turing_behavioural_session(
                    sessions[i],
                    row(cognitive_parameters, i),
                    row(environment_parameters, i),
                ),
                @varname(session[i])
            ),
            false,
        )
        #A session whose saved values have another type widens the vector
        saved_values = store_output!(saved_values, saved, i)
    end
    return saved_values
end

Turing.@model function turing_run_sessions(
    ::ParallelSessions, sessions_info, cognitive_parameters, environment_parameters
)
    if !(__varinfo__ isa DynamicPPL.ThreadSafeVarInfo)
        throw(
            ArgumentError(
                "The sessions run in parallel, so the model that is evaluated must be marked thread-safe with setthreadsafe(model, true) from DynamicPPL. Without it, the log density is wrong with no error.",
            ),
        )
    end
    sessions = sessions_info.sessions
    first_saved ~ to_submodel(
        prefix(
            turing_behavioural_session(
                sessions[1], row(cognitive_parameters, 1), row(environment_parameters, 1)
            ),
            @varname(session[1])
        ),
        false,
    )
    saved_values = initialize_storage(first_saved, length(sessions))
    #A vector can't be widened while other threads write to it, so saved values of another type are put aside until the threads are done
    saved_with_other_types = Vector{Any}(undef, length(sessions))
    #Written as Base.Threads.@threads, which is the same macro, so that DynamicPPL doesn't warn about thread safety when the package loads. The check above takes the warning's place
    Base.Threads.@threads for i in 2:length(sessions)
        saved ~ to_submodel(
            prefix(
                turing_behavioural_session(
                    sessions[i],
                    row(cognitive_parameters, i),
                    row(environment_parameters, i),
                ),
                @varname(session[i])
            ),
            false,
        )
        if saved isa eltype(saved_values)
            saved_values[i] = saved
        else
            saved_with_other_types[i] = saved
        end
    end
    saved_values = include_other_types(saved_values, saved_with_other_types)
    return saved_values
end

"""
    include_other_types(saved_values, saved_with_other_types)

Return `saved_values`, what each session saved, with the values in
`saved_with_other_types` put in their places, widening the vector as
[`store_output!`](@ref) does. A session whose saved values didn't fit the type of
`saved_values` has them in its place in `saved_with_other_types`, and is left unassigned in
`saved_values`. When there are none, `saved_values` is returned as it is. Used by
[`turing_run_sessions`](@ref) with [`ParallelSessions`](@ref), after the threads are done.
"""
function include_other_types(saved_values::Vector, saved_with_other_types::Vector{Any})
    #In the common case, every session saved values of one type
    if !any(i -> isassigned(saved_with_other_types, i), eachindex(saved_with_other_types))
        return saved_values
    end

    #Otherwise, store each session's values again, in order, so that the vector widens as it goes
    widened = initialize_storage(saved_values[1], length(saved_values))
    for i in 2:length(saved_values)
        saved = if isassigned(saved_with_other_types, i)
            saved_with_other_types[i]
        else
            saved_values[i]
        end
        widened = store_output!(widened, saved, i)
    end
    return widened
end

### MODEL FOR A BEHAVIOURAL SESSION ###

#Turing model for running a single behavioural session without an environment: every observation comes from the data
Turing.@model function CognitiveModeling.turing_behavioural_session(
    session_info::SessionInfo{<:Any,NoEnvironment}, cognitive_parameters
)
    cognitive_model = session_info.cognitive_model

    #Fill in the default parameters, and initialise the state with them, with the Turing version so that a model can sample its start
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    initial_cognitive_state ~
    to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))

    #Check the names of the parameters, the state and the previous action against attribute_names, which is decided at compile time
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters,
        initial_cognitive_state,
        session_info.previous_action,
    )

    #Run the timesteps from where the session starts
    previous = (;
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        cognitive_state=initial_cognitive_state,
    )
    saved_values ~ to_submodel(
        turing_run_timesteps(session_info, cognitive_parameters, (;), previous), false
    )
    return saved_values
end

#The same with an environment, which fills in the observations that are missing in the data, and takes its parameters too
Turing.@model function CognitiveModeling.turing_behavioural_session(
    session_info::SessionInfo{<:Any,<:AbstractModel},
    cognitive_parameters,
    environment_parameters=(;),
)
    #A session with nothing missing runs without its environment, whose scores of the observations wouldn't depend on any variable, so it has the variables of a session without one, and an empty record of the environment
    if !session_info.runs_environment
        saved_values ~ to_submodel(
            turing_behavioural_session(
                without_environment(session_info), cognitive_parameters
            ),
            false,
        )
        return with_empty_environment_entry(saved_values)
    end

    cognitive_model = session_info.cognitive_model
    environment = session_info.environment

    #Fill in the default parameters of both models, and initialise both states with them
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    environment_parameters = merge_default_parameters(environment, environment_parameters)
    initial_cognitive_state ~
    to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))
    initial_environment_state ~
    to_submodel(turing_initialize_state(environment, environment_parameters))

    #Check the names of both models against attribute_names, which is decided at compile time
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters,
        initial_cognitive_state,
        session_info.previous_action,
    )
    check_match_attribute_names(
        environment,
        environment_parameters,
        initial_environment_state,
        session_info.previous_observation,
    )

    #Run the timesteps from where the session starts
    previous = (;
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        cognitive_state=initial_cognitive_state,
        environment_state=initial_environment_state,
    )
    saved_values ~ to_submodel(
        turing_run_timesteps(
            session_info, cognitive_parameters, environment_parameters, previous
        ),
        false,
    )
    return saved_values
end

#A session without an environment takes no environment parameters, so an empty NamedTuple passes, which lets several sessions be run alike, and anything else is an error
function CognitiveModeling.turing_behavioural_session(
    session_info::SessionInfo{<:Any,NoEnvironment},
    cognitive_parameters,
    environment_parameters::NamedTuple,
)
    check_environment_parameters(session_info, environment_parameters)
    return turing_behavioural_session(session_info, cognitive_parameters)
end

#Turing model which runs any Turing model inside a try block, and gives the parameters a log density of -Inf if a cognitive model throws an InvalidParametersError
Turing.@model function CognitiveModeling.turing_catch_invalid_parameters(model)
    result = nothing
    try
        result ~ to_submodel(model, false)
    catch error
        error isa InvalidParametersError || rethrow()
        Turing.@addlogprob! -Inf
    end
    return result
end

"""
    turing_run_timesteps(session_info, cognitive_parameters, environment_parameters, previous)

Turing.jl model which runs the timesteps of a session from `previous`, where it starts, each
as [`turing_behavioural_timestep`](@ref) with the prefix `timestep[t]`, saves what the save
histories of `session_info` select, and scores the recorded actions, and the recorded
observations when the environment runs, against the distributions of their timesteps all at
once at the end, as observed values. Before each timestep from
`first_timestep_to_check_stopping` of `session_info` on, it checks the
[`stopping_criterion`](@ref) of the models, and ends the session there when one is met, with
what it saved up to it, which is an error when the data continues. It returns what the
session saved, as [`turing_behavioural_session`](@ref) describes. Both methods of that model
call this one, after setting up the parameters and the initial states with or without an
environment.
"""
Turing.@model function turing_run_timesteps(
    session_info, cognitive_parameters, environment_parameters, previous
)
    n_timesteps = session_info.n_timesteps
    first_timestep_to_check = session_info.first_timestep_to_check_stopping

    #A model which has ended already can't run the first timestep, when the criterion is checked from the start
    if first_timestep_to_check == 1
        check_session_not_ended(
            session_info, previous, cognitive_parameters, environment_parameters
        )
    end

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_saved = initial_session_outputs(
        Val(saves_initial_values(session_info.cognitive_save_history)),
        session_info,
        previous,
    )

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs ~ to_submodel(
        prefix(
            turing_behavioural_timestep(
                session_info, 1, cognitive_parameters, environment_parameters, previous
            ),
            @varname(timestep[1])
        ),
        false,
    )

    #Initialize storage for the saved outputs, which starts with timestep 0 when it is kept, and the distributions of the recorded actions and observations
    storage = initialize_session_storage(
        initial_saved, saved_session_outputs(session_info, outputs), n_timesteps
    )
    offset = initial_saved === nothing ? 0 : 1
    distributions = initialize_recorded_distributions(session_info, outputs)
    distributions = store_recorded_distributions(session_info, distributions, outputs)

    #Run the remaining timesteps, each starting from the outputs of the one before, storing timestep t as element t + offset
    n_timesteps_that_ran = n_timesteps
    for t in 2:n_timesteps

        #Stop when a model has ended, on the timesteps where the criterion is checked, and keep only the timesteps that ran
        if t >= first_timestep_to_check && session_has_ended(
            session_info, outputs, cognitive_parameters, environment_parameters
        )
            check_no_data_after(session_info, t - 1)
            n_timesteps_that_ran = t - 1
            storage = saved_values_until(storage, n_timesteps_that_ran + offset)
            break
        end

        next_outputs ~ to_submodel(
            prefix(
                turing_behavioural_timestep(
                    session_info, t, cognitive_parameters, environment_parameters, outputs
                ),
                @varname(timestep[t])
            ),
            false,
        )
        outputs = next_outputs
        storage = store_session_outputs!(
            storage, saved_session_outputs(session_info, outputs), t + offset
        )
        distributions = store_recorded_distributions(session_info, distributions, outputs)
    end

    #Score the recorded actions, and the recorded observations with an environment, against the distributions of their timesteps, all at once as observed values
    if !isempty(session_info.recorded_actions)
        session_info.recorded_actions ~ product_distribution(
            map(product_distribution, map(scorable_distributions, distributions.actions))
        )
    end
    if has_environment(session_info) && !isempty(session_info.recorded_observations)
        session_info.recorded_observations ~ product_distribution(
            map(
                product_distribution,
                map(scorable_distributions, distributions.observations),
            ),
        )
    end

    return session_saved_values(
        storage,
        timestep_sizes_until(
            session_info.saved_timestep_sizes, n_timesteps_that_ran + offset
        ),
        session_info,
    )
end

"""
    scorable_distributions(distributions)

Return the vector of the distributions of one recorded value as it is, or an empty vector of
`Normal`s for a value the model gave on no timestep and the data recorded on none, whose
vector never got a type. Either scores an empty vector of recorded values as zero, where a
`Vector{Union{}}` can't be made into a product distribution at all. Used by
[`turing_run_timesteps`](@ref) before the recorded values are scored.
"""
scorable_distributions(distributions::AbstractVector) = distributions
scorable_distributions(::Vector{Union{}}) = Normal{Float64}[]

"""
    initialize_recorded_distributions(session_info, outputs)
    store_recorded_distributions(session_info, distributions, outputs)

Keep the distributions that the recorded values of a session are scored against: the
cognitive model's for the actions, and the environment's for the observations when the
environment runs, in a `NamedTuple` with `actions` and `observations`, each a `NamedTuple`
with a vector for each recorded name, as [`initialize_distributions`](@ref) and
[`store_distributions`](@ref) make them from the outputs of a timestep. A
[`Deterministic`](@ref) is kept as the `Dirac` which scores it, see
[`scorable_distribution`](@ref). Used by [`turing_run_timesteps`](@ref).
"""
function initialize_recorded_distributions(session_info::SessionInfo, outputs)
    return (;
        actions=initialize_distributions(
            scorable_distribution_of_each(
                select_names(
                    outputs.action_distribution, recorded_action_names(session_info)
                ),
            ),
        ),
        observations=initialize_distributions(
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, outputs),
                    recorded_observation_names(session_info),
                ),
            ),
        ),
    )
end

function store_recorded_distributions(session_info::SessionInfo, distributions, outputs)
    return (;
        actions=store_distributions(
            distributions.actions,
            scorable_distribution_of_each(
                select_names(
                    outputs.action_distribution, recorded_action_names(session_info)
                ),
            ),
            outputs.recorded_actions_on_timestep,
        ),
        observations=store_distributions(
            distributions.observations,
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, outputs),
                    recorded_observation_names(session_info),
                ),
            ),
            recorded_observations_on_timestep_of(session_info, outputs),
        ),
    )
end

"""
    observation_distribution_of(session_info, outputs)
    recorded_observations_on_timestep_of(session_info, outputs)

Return the distributions the environment gave for the observations of a timestep, and
which observations are recorded on it, from the outputs of the timestep, or an empty
`NamedTuple` for a session without an environment, whose observations aren't scored. Used by
[`store_recorded_distributions`](@ref).
"""
observation_distribution_of(::SessionInfo{<:Any,NoEnvironment}, outputs) = (;)
function observation_distribution_of(::SessionInfo{<:Any,<:AbstractModel}, outputs)
    return outputs.observation_distribution
end
recorded_observations_on_timestep_of(::SessionInfo{<:Any,NoEnvironment}, outputs) = (;)
function recorded_observations_on_timestep_of(::SessionInfo{<:Any,<:AbstractModel}, outputs)
    return outputs.recorded_observations_on_timestep
end

"""
    scorable_distribution(distribution)
    scorable_distribution_of_each(distributions)

Return what a recorded value is scored against: `distribution` as it is, or the `Dirac` at
the value of a [`Deterministic`](@ref), which gives a log density of 0 to that value and
`-Inf` to any other. Only a number or a `Bool` can be scored as a point mass, so a
`Deterministic` with another value is an error, which asks for a distribution. A marker for
no output is kept, since it is only there on timesteps where nothing is recorded, which
[`recorded_on_timestep`](@ref) checks. The second form maps over a `NamedTuple` of
distributions. Used by [`store_recorded_distributions`](@ref).
"""
scorable_distribution(distribution) = distribution
scorable_distribution(deterministic::Deterministic{<:Real}) = Dirac(deterministic.value)
function scorable_distribution(deterministic::Deterministic)
    return throw(
        ArgumentError(
            "A value recorded in the data is scored against what the model gave for it, which is Deterministic($(repr(deterministic.value))), of type $(typeof(deterministic.value)). Only a number or a Bool can be scored as a point mass. Give a distribution for an output that is recorded, or leave it out of the data.",
        ),
    )
end
function scorable_distribution_of_each(distributions::NamedTuple)
    return map(scorable_distribution, distributions)
end

"""
    session_has_ended(session_info, outputs, cognitive_parameters, environment_parameters)

Return whether the [`stopping_criterion`](@ref) of the cognitive model, or of the environment
when the session has one, is met with the states in `outputs`, the outputs of the timestep
before. Used by [`turing_run_timesteps`](@ref) on the timesteps where it is checked.
"""
function session_has_ended(
    session_info::SessionInfo{<:Any,NoEnvironment},
    outputs,
    cognitive_parameters,
    environment_parameters,
)
    return stopping_criterion(
        session_info.cognitive_model, outputs.cognitive_state, cognitive_parameters
    )
end
function session_has_ended(
    session_info::SessionInfo{<:Any,<:AbstractModel},
    outputs,
    cognitive_parameters,
    environment_parameters,
)
    return stopping_criterion(
        session_info.environment, outputs.environment_state, environment_parameters
    ) || stopping_criterion(
        session_info.cognitive_model, outputs.cognitive_state, cognitive_parameters
    )
end

"""
    check_session_not_ended(session_info, previous, cognitive_parameters,
                            environment_parameters)

Throw an informative error if the [`stopping_criterion`](@ref) of a model of the session is
already met before its first timestep, as [`check_not_ended`](@ref) does for a simulation.
Used by [`turing_run_timesteps`](@ref) when the criterion is checked from the first
timestep.
"""
function check_session_not_ended(
    session_info::SessionInfo{<:Any,NoEnvironment},
    previous,
    cognitive_parameters,
    environment_parameters,
)
    check_not_ended(
        session_info.cognitive_model, previous.cognitive_state, cognitive_parameters
    )
    return nothing
end
function check_session_not_ended(
    session_info::SessionInfo{<:Any,<:AbstractModel},
    previous,
    cognitive_parameters,
    environment_parameters,
)
    check_not_ended(
        session_info.environment, previous.environment_state, environment_parameters
    )
    check_not_ended(
        session_info.cognitive_model, previous.cognitive_state, cognitive_parameters
    )
    return nothing
end

"""
    check_no_data_after(session_info, t)

Throw an informative error if the data of a session has a value on a timestep after `t`, the
last one that ran, since a session which ends while its data continues can't score the
recorded values after it. By default the criterion is only checked after the last timestep
with data, so this only happens with `check_stopping_criterion = true`. Used by
[`turing_run_timesteps`](@ref) when a stopping criterion is met.
"""
function check_no_data_after(session_info::SessionInfo, t)
    last_with_data = max(
        last_timestep_with_data(session_info.observations),
        last_timestep_with_data(session_info.actions),
    )
    if last_with_data > t
        throw(
            ArgumentError(
                "A stopping criterion is met before timestep $(t + 1), but the data of the session has values until timestep $last_with_data. A session can't end while its data continues. Give check_stopping_criterion = nothing to check the criterion only after the last timestep with data.",
            ),
        )
    end
    return nothing
end

### MODEL FOR A SINGLE TIMESTEP ###

"""
    turing_behavioural_timestep(session_info, t, cognitive_parameters, environment_parameters,
                                previous)

Turing.jl model which runs timestep `t` of a session from `previous`, the outputs of the
timestep before, and returns what the next one starts from, together with what the session
saves and scores: the observation, the action, the action distribution, the cognitive state,
which actions are recorded on the timestep, and with an environment its state, the
distribution of the observation and which observations are recorded. Each model runs as
[`turing_model_timestep`](@ref), the environment's under the prefix `environment`. Without an
environment, the observation comes from the data: the model acts on this timestep's
observation when the environment starts, and on the previous one when the agent starts. With
an environment, the environment gives the observation in response to the action, before the
model acts when the environment starts, and after when the agent starts.
"""
Turing.@model function turing_behavioural_timestep(
    session_info::SessionInfo{<:Any,NoEnvironment},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
)
    data_observation = row(session_info.observations, t)

    #The model acts on this timestep's observation when the environment starts, and on the previous one when the agent starts
    observation = if session_info.who_starts isa EnvironmentStarts
        data_observation
    else
        previous.observation
    end
    cognitive ~ to_submodel(
        turing_model_timestep(
            session_info.cognitive_model,
            ModelContext(
                observation,
                previous.cognitive_state,
                previous.action,
                cognitive_parameters,
                get_timestep_size(session_info.timestep_sizes, t),
                default_rng(),
                session_info.who_starts,
            ),
            session_info.actions,
            t,
        ),
        false,
    )

    #The next timestep starts from this timestep's observation in the data. The observation and the action can have one of two types, so they go in through entry
    return merge(
        entry(Val(:observation), data_observation),
        entry(Val(:action), cognitive.output),
        (;
            action_distribution=cognitive.output_distribution,
            cognitive_state=cognitive.new_state,
            recorded_actions_on_timestep=cognitive.recorded_on_timestep,
        ),
    )
end

Turing.@model function turing_behavioural_timestep(
    session_info::SessionInfo{<:Any,<:AbstractModel},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
)
    timestep_size = get_timestep_size(session_info.timestep_sizes, t)
    who_starts = session_info.who_starts

    #When the environment starts, it gives this timestep's observation, in response to the previous action
    if who_starts isa EnvironmentStarts
        environment ~ to_submodel(
            prefix(
                turing_model_timestep(
                    session_info.environment,
                    ModelContext(
                        previous.action,
                        previous.environment_state,
                        previous.observation,
                        environment_parameters,
                        timestep_size,
                        default_rng(),
                        who_starts,
                    ),
                    session_info.observations,
                    t,
                ),
                @varname(environment)
            ),
            false,
        )
        observation = environment.output
    else
        observation = previous.observation
    end

    #Run the cognitive model on the observation
    cognitive ~ to_submodel(
        turing_model_timestep(
            session_info.cognitive_model,
            ModelContext(
                observation,
                previous.cognitive_state,
                previous.action,
                cognitive_parameters,
                timestep_size,
                default_rng(),
                who_starts,
            ),
            session_info.actions,
            t,
        ),
        false,
    )
    action = cognitive.output

    #When the agent starts, the environment responds to the action with the next observation
    if who_starts isa AgentStarts
        environment ~ to_submodel(
            prefix(
                turing_model_timestep(
                    session_info.environment,
                    ModelContext(
                        action,
                        previous.environment_state,
                        previous.observation,
                        environment_parameters,
                        timestep_size,
                        default_rng(),
                        who_starts,
                    ),
                    session_info.observations,
                    t,
                ),
                @varname(environment)
            ),
            false,
        )
        observation = environment.output
    end

    #The observation and the action can have one of two types, so they go in through entry
    return merge(
        entry(Val(:observation), observation),
        entry(Val(:action), action),
        (;
            action_distribution=cognitive.output_distribution,
            cognitive_state=cognitive.new_state,
            recorded_actions_on_timestep=cognitive.recorded_on_timestep,
            environment_state=environment.new_state,
            observation_distribution=environment.output_distribution,
            recorded_observations_on_timestep=environment.recorded_on_timestep,
        ),
    )
end

"""
    turing_model_timestep(model, context, data_columns, t)

Turing.jl model for one model's part of timestep `t`, the same for a cognitive model and an
environment model: `model` runs on `context` with [`turing_apply_model`](@ref), and its
outputs are then taken from row `t` of `data_columns`, the data for them, where they are
recorded, filled in from what the model gave where they are [`MissingValue`](@ref), as
[`turing_impute_missing_values`](@ref) does, and kept as the model gave them where nothing
happened. It returns the output, the model's new state, the distribution it gave for the
output and which outputs are recorded on the timestep, as
[`recorded_on_timestep`](@ref) gives it, so that the recorded values can be scored against
the distributions at the end of the session. The row is taken here and not given, since it
can have one of two types when a column holds a `MissingValue`, which the compiler splits
within a function but not across a model's arguments. Used by
[`turing_behavioural_timestep`](@ref).
"""
Turing.@model function turing_model_timestep(model, context, data_columns, t)
    data_output = row(data_columns, t)
    model_output ~ to_submodel(turing_apply_model(model, context), false)
    model_output = check_model_output(model_output, model)
    output ~ to_submodel(
        turing_impute_missing_values(data_output, model_output.output_distribution), false
    )
    recorded = recorded_on_timestep(data_output, model_output.output_distribution, model)

    #The output can have one of two types, so it goes in through entry
    return merge(
        entry(Val(:output), output),
        (;
            new_state=model_output.new_state,
            output_distribution=model_output.output_distribution,
            recorded_on_timestep=recorded,
        ),
    )
end

### IMPUTING THE VALUES THAT ARE MISSING IN THE DATA ###

"""
    turing_impute_missing_values(data, distributions)

Turing.jl model which fills in the outputs of one model on one timestep: each value that is
[`MissingValue`](@ref) in `data` is imputed as a latent variable named after it, such as
`choice.imputed`, with the distribution the model gives for it in `distributions`, and the
rest are taken from the data. The data is returned as it is when nothing is missing on the
timestep, which is decided at compile time when nothing can be. For an environment it runs
under the prefix `environment`, so that an observation and an action with the same name get
a latent variable each.
"""
Turing.@model function turing_impute_missing_values(data, distributions)
    if has_missing_value(data)
        imputed ~ to_submodel(turing_impute_each_value(data, distributions), false)
        return imputed
    else
        return data
    end
end

"""
    turing_impute_each_value(data, distributions)

Turing.jl model which goes through the names of `data` one at a time, with each name taken
from the type, so that the result is compiled knowing the names whether a value is missing
or not, and imputes each value with [`turing_impute_value`](@ref).
"""
Turing.@model function turing_impute_each_value(
    data::NamedTuple{names}, distributions
) where {names}
    name = first(names)
    value ~ to_submodel(
        prefix(
            turing_impute_value(first(data), getproperty(distributions, name)),
            VarName{name}(),
        ),
        false,
    )
    rest ~ to_submodel(turing_impute_each_value(Base.tail(data), distributions), false)
    return merge(entry(Val(name), value), rest)
end

Turing.@model function turing_impute_each_value(data::NamedTuple{()}, distributions)
    return (;)
end

"""
    turing_impute_value(data_value, distribution)

Turing.jl model for one value of the data: a value that is there is taken as it is, and a
[`MissingValue`](@ref) is a latent variable named `imputed` with `distribution`, the value
of a [`Deterministic`](@ref), or the marker itself when the model gives [`NoAction`](@ref)
or [`NoObservation`](@ref) on that timestep.
"""
Turing.@model function turing_impute_value(data_value, distribution)
    return data_value
end

#A missing value is a latent, with the distribution the model gives for it
Turing.@model function turing_impute_value(::MissingValue, distribution)
    imputed ~ distribution
    return imputed
end

#A missing value that the model gives without randomness is that value, with nothing to sample
Turing.@model function turing_impute_value(::MissingValue, deterministic::Deterministic)
    return deterministic.value
end

#A missing value where the model takes no action or gives no observation on this timestep stays as the model gives it
Turing.@model function turing_impute_value(::MissingValue, no_value::AbstractNoValue)
    return no_value
end
