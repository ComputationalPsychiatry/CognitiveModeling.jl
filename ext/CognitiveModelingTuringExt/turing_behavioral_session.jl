### MODEL WHICH HAS ONE COMPONENT FOR THE PARAMETERS OF THE SESSIONS (e.g. a regression), AND THEN THE BEHAVIORAL SESSIONS ###
# The parameters of the cognitive model and those of the environment model each come from a session parameter model. There is also
# an optional third component which can do something with the outcomes of the behavioral sessions, such as a regression
# predicting something (e.g. physiological data) from the cognitive states.

#Creates the Turing model for fitting a cognitive model to several sessions, after warning when both session parameter models can sample variables of the same name, and when the sessions save trajectories no downstream model uses, and after preparing its parts once
#The warnings and the preparation are done here, when the model is made, and not in the model, whose body is differentiated and runs on every evaluation: a warning holds a try block, which Mooncake can't differentiate
function CognitiveModeling.turing_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoSessionParameters(),
    downstream_model=NoDownstreamModel(),
    fixed_cognitive_parameters::NamedTuple=(;),
    fixed_environment_parameters::NamedTuple=(;),
    catch_invalid_parameters::Bool=false,
    parallelization=SerialSessions(),
    include_submodel_prefixes::Bool=false,
)
    warn_if_parameter_names_overlap(
        cognitive_parameters_model,
        environment_parameters_model,
        sessions_info,
        include_submodel_prefixes,
    )
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

    #Prepare the parts once, as every inference backend does: the session parameter models for the sessions, with the values that are fixed, and the sessions with their environment running when its parameters are sampled
    prepared = prepare_full_model(
        cognitive_parameters_model,
        sessions_info;
        environment_parameters_model,
        downstream_model,
        fixed_cognitive_parameters,
        fixed_environment_parameters,
    )

    #Warn when a prior or a transformation can give a parameter a value outside its domain
    first_session = first(sessions_info.sessions)
    warn_if_values_outside_domains(
        cognitive_parameters_model, first_session.cognitive_model
    )
    warn_if_values_outside_domains(environment_parameters_model, first_session.environment)
    return turing_run_full_model(
        prepared.cognitive_parameters_model,
        prepared.sessions_info;
        environment_parameters_model=prepared.environment_parameters_model,
        downstream_model=prepared.downstream_model,
        catch_invalid_parameters,
        parallelization,
        include_submodel_prefixes,
    )
end

"""
    warn_if_parameter_names_overlap(cognitive_parameters_model,
                                    environment_parameters_model, sessions_info,
                                    include_submodel_prefixes)

Warn when the attributes of the cognitive model and of the environment model of the sessions
have a parameter of the same name, both session parameter models sample, and the variables
have no prefixes, since their variables can then have the same names, and a name that both
use is one variable, which `sample` rejects. Whether both sample is decided by their types.
Used by `turing_full_model` when it creates the model, outside the model's body, which is
differentiated.
"""
function warn_if_parameter_names_overlap(
    cognitive_parameters_model,
    environment_parameters_model,
    sessions_info::MultipleSessionsInfo,
    include_submodel_prefixes::Bool,
)
    include_submodel_prefixes && return nothing
    estimates_parameters(cognitive_parameters_model) || return nothing
    estimates_parameters(environment_parameters_model) || return nothing
    session_info = first(sessions_info.sessions)
    has_environment(session_info) || return nothing
    cognitive_names = attribute_names(session_info.cognitive_model).parameters
    environment_names = attribute_names(session_info.environment).parameters
    shared_names = Tuple(name for name in cognitive_names if name in environment_names)
    if !isempty(shared_names)
        @warn "The cognitive model and the environment model both have the parameters $shared_names, and both session parameter models sample, so their variables can have the same names. A name that both use is one variable for both, which sample rejects. Give turing_full_model include_submodel_prefixes = true to keep them apart."
    end
    return nothing
end

"""
    warn_if_saving_without_downstream_model(sessions_info, downstream_model)

Warn when a session in `sessions_info` saves anything and `downstream_model` is
`NoDownstreamModel`, since saving adds a cost to each gradient and nothing in the model
uses the saved trajectories: they are read after the fit with `session_trajectories_chain`,
from sessions that save nothing. Used by `turing_full_model` when it creates the model.
"""
function warn_if_saving_without_downstream_model(
    sessions_info::MultipleSessionsInfo, downstream_model
)
    downstream_model isa NoDownstreamModel || return nothing
    sessions_save_anything(sessions_info) || return nothing
    @warn "The sessions save trajectories, which adds a cost to each gradient, but there is no downstream model to use them. A fit is fastest with sessions that save nothing, and session_trajectories_chain(model, chain; cognitive_save_history, environment_save_history) then gives the trajectories of every sample afterwards."
    return nothing
end

"""
    warn_if_values_outside_domains(parameters_model, model)

Warn when `parameters_model`, a session parameter model, can give a parameter of `model` a
value outside its domain in `parameter_domains`: values below its lower bound or above its
upper bound, or values of a continuous distribution for a discrete parameter. What a
parameter can be given is found by [`sampled_ranges`](@ref). A value outside its domain is
an `InvalidParametersError` when a session runs. Used by `turing_full_model` when it creates
the model.
"""
function warn_if_values_outside_domains(parameters_model, model::AbstractTimestepModel)
    domains = parameter_domains(model)
    isempty(domains) && return nothing
    for (name, range) in sampled_ranges(parameters_model)
        haskey(domains, name) || continue
        domain = getproperty(domains, name)
        lower = get(domain, :lower, -Inf)
        upper = get(domain, :upper, Inf)
        if range.lower < lower || range.upper > upper
            @warn "The parameter $name can be given values from $(range.lower) to $(range.upper), by $(range.source), but its domain in $(describe_model(model)) is from $lower to $upper. A value outside the domain is an InvalidParametersError when a session runs, which stops the fit, or rejects the sample with catch_invalid_parameters = true. A prior or a transformation that stays within the domain avoids this."
        end
        if get(domain, :discrete, false) && !range.discrete
            @warn "The parameter $name is discrete in $(describe_model(model)), but it is given values that aren't whole numbers by $(range.source). Such a value is an InvalidParametersError when a session runs."
        end
    end
    return nothing
end
warn_if_values_outside_domains(parameters_model, ::NoEnvironment) = nothing

"""
    sampled_ranges(parameters_model)

Return the values that `parameters_model`, a session parameter model, can sample for each
parameter, as a vector of pairs from the parameter's name to a `NamedTuple` with `lower`,
`upper`, whether the values are `discrete`, and `source`, a description of where they come
from: the support of a prior, or what a transformation makes of the two ends of the real
line, for a hierarchical parameter with the default normal distribution, and for a
regression with random intercepts or session deviations, which are normal. Parameters that
are given without being sampled, or whose range isn't known, are left out. Used by
[`warn_if_values_outside_domains`](@ref).
"""
sampled_ranges(parameters_model) = Pair{Symbol,NamedTuple}[]
function sampled_ranges(
    parameters_model::Union{IdenticalSessionParameters,IndependentSessionParameters}
)
    ranges = Pair{Symbol,NamedTuple}[]
    for (name, prior) in pairs(parameters_model.priors)
        prior isa UnivariateDistribution || continue
        range = (;
            lower=minimum(prior),
            upper=maximum(prior),
            discrete=prior isa DiscreteUnivariateDistribution,
            source="its prior, $prior",
        )
        push!(ranges, name => range)
    end
    return ranges
end
function sampled_ranges(parameters_model::GroupedHierarchicalSessionParameters)
    ranges = Pair{Symbol,NamedTuple}[]
    for (name, hierarchical_parameter) in pairs(parameters_model.parameters)
        hierarchical_parameter.distribution === nothing || continue
        transformation = hierarchical_parameter.transformation
        ends = try
            (transformation(-Inf), transformation(Inf))
        catch
            continue
        end
        all(value -> value isa Real, ends) || continue
        range = (;
            lower=min(ends...),
            upper=max(ends...),
            discrete=false,
            source="the transformation $transformation of its group-level normal distribution",
        )
        push!(ranges, name => range)
    end
    return ranges
end
function sampled_ranges(parameters_model::HierarchicalRegressionSessionParameters)
    ranges = Pair{Symbol,NamedTuple}[]
    for (name, regression_parameter) in pairs(parameters_model.parameters)

        #A normal random intercept or session deviation can give the sum any value on the real line
        has_normal_term =
            regression_parameter.session_deviation !== nothing ||
            !isempty(regression_parameter.random_intercepts)
        has_normal_term || continue
        transformation = regression_parameter.transformation
        ends = try
            (transformation(-Inf), transformation(Inf))
        catch
            continue
        end
        all(value -> value isa Real, ends) || continue
        range = (;
            lower=min(ends...),
            upper=max(ends...),
            discrete=false,
            source="the transformation $transformation of its regression",
        )
        push!(ranges, name => range)
    end
    return ranges
end
function sampled_ranges(parameters_model::CompositeSessionParameters)
    return reduce(
        vcat, map(sampled_ranges, parameters_model.parts); init=Pair{Symbol,NamedTuple}[]
    )
end

"""
    turing_run_full_model(cognitive_parameters_model, sessions_info;
                          environment_parameters_model, downstream_model,
                          catch_invalid_parameters, parallelization,
                          include_submodel_prefixes)

Turing.jl model which [`turing_full_model`](@ref) creates, with its arguments as
`prepare_full_model` prepared them: the session parameter models for the parameters of the
cognitive model and of the environment model, the sessions with those parameters, and the
optional downstream model. It is a function of its own so that `turing_full_model` can check
and prepare its parts when the model is made, outside the model's body. Used by
[`turing_full_model`](@ref).
"""
Turing.@model function turing_run_full_model(
    cognitive_parameters_model,
    sessions_info::MultipleSessionsInfo;
    environment_parameters_model=NoSessionParameters(),
    downstream_model=NoDownstreamModel(),
    catch_invalid_parameters::Bool=false,
    parallelization=SerialSessions(),
    include_submodel_prefixes::Bool=false,
)

    #Give or sample the cognitive parameters of every session, with the variables under cognitive_parameters_model when the prefixes are included
    if include_submodel_prefixes
        cognitive_parameters ~ to_submodel(
            prefix(
                turing_session_parameters_model(cognitive_parameters_model, sessions_info),
                @varname(cognitive_parameters_model),
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
                @varname(environment_parameters_model),
            ),
            false,
        )
    else
        environment_parameters ~ to_submodel(
            turing_session_parameters_model(environment_parameters_model, sessions_info),
            false,
        )
    end

    #Run all behavioral sessions, inside a try block when invalid parameters are caught. The parameters are sampled outside it, so that they are kept when a draw is rejected, which a sampler needs to start: it needs every variable to exist after its first draw, also when that draw is rejected
    sessions = turing_multiple_behavioral_sessions(
        sessions_info, cognitive_parameters, environment_parameters; parallelization
    )
    if catch_invalid_parameters
        saved_behavioral_trajectories ~
        to_submodel(turing_catch_invalid_parameters(sessions), false)

        #A rejected draw leaves nothing for the downstream model to run on, so the model returns nothing at once, which Enzyme can differentiate where a value of two types can't be. The variables of the downstream model then don't exist for that draw, so a sampler whose first draw is rejected can't start when the downstream model has variables of its own, and needs valid initial parameters
        if saved_behavioral_trajectories === nothing
            return nothing
        end
    else
        saved_behavioral_trajectories ~ to_submodel(sessions, false)
    end

    #Do the optional downstream model, with its variables under downstream_model when the prefixes are included
    if include_submodel_prefixes
        downstream_values ~ to_submodel(
            prefix(
                turing_downstream_model(
                    downstream_model,
                    saved_behavioral_trajectories,
                    cognitive_parameters,
                    sessions_info,
                ),
                @varname(downstream_model),
            ),
            false,
        )
    else
        downstream_values ~ to_submodel(
            turing_downstream_model(
                downstream_model,
                saved_behavioral_trajectories,
                cognitive_parameters,
                sessions_info,
            ),
            false,
        )
    end

    return (;
        cognitive_parameters,
        environment_parameters,
        saved_behavioral_trajectories,
        downstream_values,
    )
end

### MODEL FOR MULTIPLE SESSIONS ###

#Turing model for running several behavioral sessions, each with its own cognitive parameters, as a submodel with the prefix session[i], and collecting what each returns
#The environment parameters are given in the same form, one vector for each parameter with a value for each session, and only when the sessions have an environment
Turing.@model function CognitiveModeling.turing_multiple_behavioral_sessions(
    sessions_info::MultipleSessionsInfo,
    cognitive_parameters,
    environment_parameters=(;);
    parallelization=SerialSessions(),
)
    check_environment_parameters(sessions_info, environment_parameters)
    saved_trajectories ~ to_submodel(
        turing_run_sessions(
            parallelization, sessions_info, cognitive_parameters, environment_parameters
        ),
        false,
    )
    return saved_trajectories
end

"""
    turing_run_sessions(parallelization, sessions_info, cognitive_parameters,
                        environment_parameters)

Turing.jl model which runs every session in `sessions_info` as a submodel with the prefix
`session[i]`, and returns a vector with what each returned. The first session runs outside
the loop, so that the vector has its type. Sessions of one type can still save trajectories
of different types, such as an action that is `NoValue()` in one session only, and a
session whose trajectories have another type widens the vector, as [`store_value!`](@ref)
does. With [`SerialSessions`](@ref) the sessions run one after the other, and with
[`ParallelSessions`](@ref) over the threads Julia was started with, which needs the model
that is evaluated to be marked thread-safe, since the log density is silently wrong
otherwise. Used by [`turing_multiple_behavioral_sessions`](@ref).
"""
Turing.@model function turing_run_sessions(
    ::SerialSessions, sessions_info, cognitive_parameters, environment_parameters
)
    sessions = sessions_info.sessions
    first_saved ~ to_submodel(
        prefix(
            turing_behavioral_session(
                sessions[1],
                select_row(cognitive_parameters, 1),
                select_row(environment_parameters, 1),
            ),
            @varname(session[1]),
        ),
        false,
    )
    saved_trajectories = initialize_storage(first_saved, length(sessions))
    for i in 2:length(sessions)
        saved ~ to_submodel(
            prefix(
                turing_behavioral_session(
                    sessions[i],
                    select_row(cognitive_parameters, i),
                    select_row(environment_parameters, i),
                ),
                @varname(session[i]),
            ),
            false,
        )
        #A session whose saved trajectories have another type widens the vector
        saved_trajectories = store_value!(saved_trajectories, saved, i)
    end
    return saved_trajectories
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
            turing_behavioral_session(
                sessions[1],
                select_row(cognitive_parameters, 1),
                select_row(environment_parameters, 1),
            ),
            @varname(session[1]),
        ),
        false,
    )
    saved_trajectories = initialize_storage(first_saved, length(sessions))
    #A vector can't be widened while other threads write to it, so saved trajectories of another type are put aside until the threads are done
    saved_with_other_types = Vector{Any}(undef, length(sessions))
    #Written as Base.Threads.@threads, which is the same macro, so that DynamicPPL doesn't warn about thread safety when the package loads. The check above takes the warning's place
    Base.Threads.@threads for i in 2:length(sessions)
        saved ~ to_submodel(
            prefix(
                turing_behavioral_session(
                    sessions[i],
                    select_row(cognitive_parameters, i),
                    select_row(environment_parameters, i),
                ),
                @varname(session[i]),
            ),
            false,
        )
        if saved isa eltype(saved_trajectories)
            saved_trajectories[i] = saved
        else
            saved_with_other_types[i] = saved
        end
    end
    saved_trajectories = include_other_types(saved_trajectories, saved_with_other_types)
    return saved_trajectories
end

"""
    include_other_types(saved_trajectories, saved_with_other_types)

Return `saved_trajectories`, what each session saved, with the trajectories in
`saved_with_other_types` put in their places, widening the vector as
[`store_value!`](@ref) does. A session whose saved trajectories didn't fit the type of
`saved_trajectories` has them in its place in `saved_with_other_types`, and is left
unassigned in `saved_trajectories`. When there are none, `saved_trajectories` is returned as
it is. Used by [`turing_run_sessions`](@ref) with [`ParallelSessions`](@ref), after the
threads are done.
"""
function include_other_types(
    saved_trajectories::Vector, saved_with_other_types::Vector{Any}
)
    #In the common case, every session saved trajectories of one type
    if !any(i -> isassigned(saved_with_other_types, i), eachindex(saved_with_other_types))
        return saved_trajectories
    end

    #Otherwise, store each session's trajectories again, in order, so that the vector widens as it goes
    widened = initialize_storage(saved_trajectories[1], length(saved_trajectories))
    for i in 2:length(saved_trajectories)
        saved = if isassigned(saved_with_other_types, i)
            saved_with_other_types[i]
        else
            saved_trajectories[i]
        end
        widened = store_value!(widened, saved, i)
    end
    return widened
end

### MODEL FOR A BEHAVIORAL SESSION ###

#Turing model for running a single behavioral session without an environment: every observation comes from the data
Turing.@model function CognitiveModeling.turing_behavioral_session(
    session_info::SessionInfo{<:Any,NoEnvironment}, cognitive_parameters
)
    cognitive_model = session_info.cognitive_model

    #Fill in the default parameters, and initialize the state with them, with the Turing version so that a model can sample its start
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    initial_cognitive_state ~
    to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))

    #Check the names of the parameters, the state and the previous action against attribute_names, which is decided at compile time, and that each parameter is within its domain
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters,
        initial_cognitive_state,
        session_info.previous_action,
    )
    check_parameter_domains(cognitive_model, cognitive_parameters)

    #Run the timesteps from where the session starts, with the values arranged as those of a timestep
    previous = (;
        state=initial_cognitive_state,
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        environment=(; state=nothing),
    )
    saved_trajectories ~ to_submodel(
        turing_run_timesteps(session_info, cognitive_parameters, (;), previous), false
    )
    return saved_trajectories
end

#The same with an environment, which fills in the observations that are missing in the data, and takes its parameters too
Turing.@model function CognitiveModeling.turing_behavioral_session(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel},
    cognitive_parameters,
    environment_parameters=(;),
)
    #A session with nothing missing runs without its environment, whose scores of the observations wouldn't depend on any variable, so it has the variables of a session without one, and an empty record of the environment
    if !session_info.includes_environment
        saved_trajectories ~ to_submodel(
            turing_behavioral_session(
                remake_session_without_environment(session_info), cognitive_parameters
            ),
            false,
        )
        return with_empty_environment_entry(saved_trajectories)
    end

    cognitive_model = session_info.cognitive_model
    environment = session_info.environment

    #Fill in the default parameters of both models, and initialize both states with them
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
        session_info.previous_environment_observation,
    )

    #Check that each parameter of both models is within its domain
    check_parameter_domains(cognitive_model, cognitive_parameters)
    check_parameter_domains(environment, environment_parameters)

    #Run the timesteps from where the session starts, with the values arranged as those of a timestep: the environment's are under environment, in its own names
    previous = (;
        state=initial_cognitive_state,
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        environment=(;
            state=initial_environment_state,
            observation=session_info.previous_environment_observation,
            action=map_names(session_info.previous_action, session_info.action_map),
        ),
    )
    saved_trajectories ~ to_submodel(
        turing_run_timesteps(
            session_info, cognitive_parameters, environment_parameters, previous
        ),
        false,
    )
    return saved_trajectories
end

#A session without an environment takes no environment parameters, so an empty NamedTuple passes, which lets several sessions be run alike, and anything else is an error
function CognitiveModeling.turing_behavioral_session(
    session_info::SessionInfo{<:Any,NoEnvironment},
    cognitive_parameters,
    environment_parameters::NamedTuple,
)
    check_environment_parameters(session_info, environment_parameters)
    return turing_behavioral_session(session_info, cognitive_parameters)
end

### REJECTING INVALID PARAMETERS ###

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

### THE TIMESTEPS OF A SESSION ###

"""
    turing_run_timesteps(session_info, cognitive_parameters, environment_parameters,
                         previous)

Turing.jl model which runs the timesteps of a session from `previous`, where it starts, each
as [`turing_behavioral_timestep`](@ref) with the prefix `timestep[t]`, saves what the save
histories of `session_info` select, and scores the actions in the data, and the observations
in the data when the environment runs, against the distributions of their timesteps all at
once at the end, as observed values. Before each timestep from
`first_timestep_to_check_stopping` of `session_info` on, it checks the
[`stopping_criterion`](@ref) of the models, and ends the session there when one is met, with
what it saved up to it, which is an error when the data continues. It returns what the
session saved, as [`turing_behavioral_session`](@ref) describes. Both methods of that model
call this one, after setting up the parameters and the initial states with or without an
environment. Used by [`turing_behavioral_session`](@ref).
"""
Turing.@model function turing_run_timesteps(
    session_info, cognitive_parameters, environment_parameters, previous
)
    n_timesteps = session_info.n_timesteps
    first_timestep_to_check = session_info.first_timestep_to_check_stopping

    #No timestep can run when the stopping criterion of either model is met already, when the criterion is checked from the start
    if first_timestep_to_check == 1
        check_stopping_criterion_not_met(
            session_info.environment, previous.environment.state, environment_parameters
        )
        check_stopping_criterion_not_met(
            session_info.cognitive_model, previous.state, cognitive_parameters
        )
    end

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_saved = select_initial_timestep_values_to_save(
        Val(saves_initial_timestep_values(session_info.cognitive_save_history)),
        session_info,
        previous,
    )

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    timestep_values ~ to_submodel(
        prefix(
            turing_behavioral_timestep(
                session_info, 1, cognitive_parameters, environment_parameters, previous
            ),
            @varname(timestep[1]),
        ),
        false,
    )

    #Initialize storage for what is saved, arranged as it is returned, which starts with timestep 0 when it is kept
    storage = initialize_all_storage(
        initial_saved,
        select_timestep_values_to_save(session_info, timestep_values),
        n_timesteps,
    )
    offset = initial_saved === nothing ? 0 : 1

    #Initialize the distributions of the actions and observations to condition on
    distributions = initialize_distributions_for_conditioning(session_info, timestep_values)
    distributions = store_distributions_for_conditioning(
        session_info, distributions, timestep_values
    )

    #Run the remaining timesteps, each starting from the values of the one before, storing timestep t as element t + offset
    n_timesteps_that_ran = n_timesteps
    for t in 2:n_timesteps

        #Stop when the stopping criterion of either model is met, on the timesteps where it is checked
        if t >= first_timestep_to_check && (
            stopping_criterion(
                session_info.environment,
                timestep_values.environment.state,
                environment_parameters,
            ) || stopping_criterion(
                session_info.cognitive_model, timestep_values.state, cognitive_parameters
            )
        )
            check_no_data_after(session_info, t - 1)
            n_timesteps_that_ran = t - 1

            #Keep only the timesteps that ran, shortening the vectors in place. The function uses t and not n_timesteps_that_ran, since a variable that is assigned in two places is boxed when a function uses it: its type is then unknown, and the session slower
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            break
        end

        next_timestep_values ~ to_submodel(
            prefix(
                turing_behavioral_timestep(
                    session_info,
                    t,
                    cognitive_parameters,
                    environment_parameters,
                    timestep_values,
                ),
                @varname(timestep[t]),
            ),
            false,
        )
        timestep_values = next_timestep_values
        storage = store_timestep_values!(
            storage,
            select_timestep_values_to_save(session_info, timestep_values),
            t + offset,
        )
        distributions = store_distributions_for_conditioning(
            session_info, distributions, timestep_values
        )
    end

    #Score the actions in the data, and the observations in the data with an environment, against the distributions of their timesteps, all at once as observed values
    if !isempty(session_info.actions_to_condition_on)
        session_info.actions_to_condition_on ~ product_distribution(
            map(product_distribution, map(scorable_distributions, distributions.actions))
        )
    end
    if has_environment(session_info) && !isempty(session_info.observations_to_condition_on)
        session_info.observations_to_condition_on ~ product_distribution(
            map(
                product_distribution,
                map(scorable_distributions, distributions.observations),
            ),
        )
    end

    return saved_trajectories(
        storage,
        timestep_sizes_until(
            session_info.saved_timestep_sizes, n_timesteps_that_ran + offset
        ),
    )
end

### SCORING THE RECORDED VALUES ###

"""
    scorable_distributions(distributions)

Return the vector of the distributions of one output to condition on as it is, or an empty
vector of `Normal`s for an output the model gave on no timestep and the data has on none,
whose vector never got a type. Either scores an empty vector of values to condition on as
zero, where a `Vector{Union{}}` can't be made into a product distribution at all. Used by
[`turing_run_timesteps`](@ref) before the values to condition on are scored.
"""
scorable_distributions(distributions::AbstractVector) = distributions
scorable_distributions(::Vector{Union{}}) = Normal{Float64}[]

"""
    initialize_distributions_for_conditioning(session_info, timestep_values)
    store_distributions_for_conditioning(session_info, distributions, timestep_values)

Keep the distributions that the values to condition on of a session are scored against: the
cognitive model's for the actions, and the environment's for the observations when the
environment runs, in a `NamedTuple` with `actions` and `observations`, each a `NamedTuple`
with a vector for each name to condition on, as
[`initialize_output_distributions_for_conditioning`](@ref) and
[`store_output_distributions_for_conditioning`](@ref) make them from the values of a
timestep. A [`Deterministic`](@ref) is kept as the `Dirac` which scores it, see
[`scorable_distribution`](@ref). Used by [`turing_run_timesteps`](@ref).
"""
function initialize_distributions_for_conditioning(
    session_info::SessionInfo, timestep_values
)
    return (;
        actions=initialize_output_distributions_for_conditioning(
            scorable_distribution_of_each(
                select_names(
                    timestep_values.action_distribution,
                    action_names_to_condition_on(session_info),
                ),
            ),
        ),
        observations=initialize_output_distributions_for_conditioning(
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, timestep_values),
                    observation_names_to_condition_on(session_info),
                ),
            ),
        ),
    )
end

function store_distributions_for_conditioning(
    session_info::SessionInfo, distributions, timestep_values
)
    return (;
        actions=store_output_distributions_for_conditioning(
            distributions.actions,
            scorable_distribution_of_each(
                select_names(
                    timestep_values.action_distribution,
                    action_names_to_condition_on(session_info),
                ),
            ),
            timestep_values.actions_to_condition_on_at_timestep,
        ),
        observations=store_output_distributions_for_conditioning(
            distributions.observations,
            scorable_distribution_of_each(
                select_names(
                    observation_distribution_of(session_info, timestep_values),
                    observation_names_to_condition_on(session_info),
                ),
            ),
            observations_to_condition_on_at_timestep_of(session_info, timestep_values),
        ),
    )
end

"""
    observation_distribution_of(session_info, timestep_values)
    observations_to_condition_on_at_timestep_of(session_info, timestep_values)

Return the distributions the environment gave for the observations of a timestep, and which
observations to condition on at it, from the environment's part of the values of the
timestep, or an empty `NamedTuple` for a session without an environment, whose observations
aren't scored. Used by [`store_distributions_for_conditioning`](@ref).
"""
observation_distribution_of(::SessionInfo{<:Any,NoEnvironment}, timestep_values) = (;)
function observation_distribution_of(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observation_distribution
end
function observations_to_condition_on_at_timestep_of(
    ::SessionInfo{<:Any,NoEnvironment}, timestep_values
)
    return (;)
end
function observations_to_condition_on_at_timestep_of(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observations_to_condition_on_at_timestep
end

"""
    scorable_distribution(distribution)
    scorable_distribution_of_each(distributions)

Return what a value to condition on is scored against: `distribution` as it is, or the
`Dirac` at the value of a [`Deterministic`](@ref), which gives a log density of 0 to that
value and `-Inf` to any other. Only a number or a `Bool` can be scored as a point mass, so a
`Deterministic` with another value is an error, which asks for a distribution. A marker for
no output is kept, since it is only there on timesteps where the data has no value to
condition on, which [`find_outputs_to_condition_on`](@ref) checks. The second form maps over
a `NamedTuple` of distributions. Used by [`store_distributions_for_conditioning`](@ref).
"""
scorable_distribution(distribution) = distribution
scorable_distribution(deterministic::Deterministic{<:Real}) = Dirac(deterministic.value)
function scorable_distribution(deterministic::Deterministic)
    return throw(
        ArgumentError(
            "A value in the data is scored against what the model gave for it, which is Deterministic($(repr(deterministic.value))), of type $(typeof(deterministic.value)). Only a number or a Bool can be scored as a point mass. Give a distribution for an output that has values in the data, or leave it out of the data.",
        ),
    )
end
function scorable_distribution_of_each(distributions::NamedTuple)
    return map(scorable_distribution, distributions)
end

"""
    check_no_data_after(session_info, t)

Throw an informative error if the data of a session has a value on a timestep after `t`, the
last one that ran, since a session which ends while its data continues can't score the
values in the data after it. By default the criterion is only checked after the last
timestep with data, so this only happens with `check_stopping_criterion = true`. Used by
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
    turing_behavioral_timestep(session_info, t, cognitive_parameters,
                               environment_parameters, previous)

Turing.jl model which runs timestep `t` of a session from `previous`, the values of the
timestep before, and returns the values of this one: what the next one starts from, together
with what the session saves and scores. They are arranged as a simulation arranges them. The
cognitive model's are at the top: its `state`, the `observation` it received, its `action`,
the `action_distribution` it gave, and `actions_to_condition_on_at_timestep`. The
environment's are under `environment`, in its own names: its `state`, its `observation`, the
`action` it received, the `observation_distribution` it gave, and
`observations_to_condition_on_at_timestep`. Without an environment, `environment` only has a
`state`, which is `nothing`.

Each model runs as [`turing_model_timestep`](@ref), the environment's under the prefix
`environment`. Without an environment, the observation comes from the data: the model acts
on this timestep's observation when the environment starts, and on the previous one when the
agent starts. With an environment, the environment gives the observation in response to the
action, before the model acts when the environment starts, and after when the agent starts.
Used by [`turing_run_timesteps`](@ref).
"""
Turing.@model function turing_behavioral_timestep(
    session_info::SessionInfo{<:Any,NoEnvironment},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
)
    data_observation = select_row(session_info.observations, t)

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
                previous.state,
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

    #The next timestep starts from this timestep's observation in the data. The observation and the action can have one of two types, so they go in through build_named_entry
    return merge(
        (; state=cognitive.new_state),
        build_named_entry(Val(:observation), data_observation),
        build_named_entry(Val(:action), cognitive.output),
        (;
            action_distribution=cognitive.output_distribution,
            actions_to_condition_on_at_timestep=cognitive.outputs_to_condition_on,
            environment=(; state=nothing),
        ),
    )
end

Turing.@model function turing_behavioral_timestep(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
)
    timestep_size = get_timestep_size(session_info.timestep_sizes, t)
    who_starts = session_info.who_starts

    #When the environment starts, it gives this timestep's observation, in its own names, in response to the previous action as the action map passes it on
    if who_starts isa EnvironmentStarts
        environment ~ to_submodel(
            prefix(
                turing_model_timestep(
                    session_info.environment,
                    ModelContext(
                        map_names(previous.action, session_info.action_map),
                        previous.environment.state,
                        previous.environment.observation,
                        environment_parameters,
                        timestep_size,
                        default_rng(),
                        who_starts,
                    ),
                    session_info.environment_observations,
                    t,
                ),
                @varname(environment),
            ),
            false,
        )
        environment_observation = environment.output
        observation = map_names(environment_observation, session_info.observation_map)
    else
        observation = previous.observation
    end

    #Run the cognitive model on the observation
    cognitive ~ to_submodel(
        turing_model_timestep(
            session_info.cognitive_model,
            ModelContext(
                observation,
                previous.state,
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
    environment_action = map_names(action, session_info.action_map)

    #When the agent starts, the environment responds to the action, as the action map passes it on, with the next observation in its own names
    if who_starts isa AgentStarts
        environment ~ to_submodel(
            prefix(
                turing_model_timestep(
                    session_info.environment,
                    ModelContext(
                        environment_action,
                        previous.environment.state,
                        previous.environment.observation,
                        environment_parameters,
                        timestep_size,
                        default_rng(),
                        who_starts,
                    ),
                    session_info.environment_observations,
                    t,
                ),
                @varname(environment),
            ),
            false,
        )
        environment_observation = environment.output
        observation = map_names(environment_observation, session_info.observation_map)
    end

    #Collect the environment's values, in its own names. The observations and the actions can have one of two types, so they go in through build_named_entry
    environment_timestep_values = merge(
        (; state=environment.new_state),
        build_named_entry(Val(:observation), environment_observation),
        build_named_entry(Val(:action), environment_action),
        (;
            observation_distribution=environment.output_distribution,
            observations_to_condition_on_at_timestep=environment.outputs_to_condition_on,
        ),
    )

    #Collect the cognitive model's values at the top, with the environment's under environment
    return merge(
        (; state=cognitive.new_state),
        build_named_entry(Val(:observation), observation),
        build_named_entry(Val(:action), action),
        (;
            action_distribution=cognitive.output_distribution,
            actions_to_condition_on_at_timestep=cognitive.outputs_to_condition_on,
        ),
        build_named_entry(Val(:environment), environment_timestep_values),
    )
end

#The row is taken here, since it can have one of two types when a column holds a MissingValue, which the compiler splits within a function but not across a model's arguments
"""
    turing_model_timestep(model, context, data_columns, t)

Turing.jl model for one model's part of timestep `t`, the same for a cognitive model and an
environment model: `model` runs on `context` with [`turing_apply_timestep_model`](@ref), and
its outputs are then taken from row `t` of `data_columns`, the data for them, where it has
values, imputed from what the model gave where they are [`MissingValue`](@ref), as
[`turing_impute_missing_values`](@ref) does, and are the marker that the data has for them
otherwise, such as `UnimputedMissingValue()` or `NoValue()`. It returns the output, the
model's new state, the distribution it gave for the output and which outputs to condition on
at the timestep, as [`find_outputs_to_condition_on`](@ref) gives it, so that their values
can be scored against the distributions at the end of the session. The row is taken here and
not given. Used by [`turing_behavioral_timestep`](@ref).
"""
Turing.@model function turing_model_timestep(model, context, data_columns, t)
    data_output = select_row(data_columns, t)
    model_output ~ to_submodel(turing_apply_timestep_model(model, context), false)
    model_output = check_model_output(model_output, model)
    output ~ to_submodel(
        turing_impute_missing_values(data_output, model_output.output_distribution), false
    )
    outputs_to_condition_on = find_outputs_to_condition_on(
        data_output, model_output.output_distribution, model
    )

    #The output can have one of two types, so it goes in through build_named_entry
    return merge(
        build_named_entry(Val(:output), output),
        (;
            new_state=model_output.new_state,
            output_distribution=model_output.output_distribution,
            outputs_to_condition_on=outputs_to_condition_on,
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
timestep. For an environment it runs under the prefix `environment`, so that an observation
and an action with the same name get a latent variable each. Used by
[`turing_model_timestep`](@ref).
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

Turing.jl model which goes through the names of `data` one at a time, and imputes each value
with [`turing_impute_value`](@ref). Used by [`turing_impute_missing_values`](@ref).
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
    return merge(build_named_entry(Val(name), value), rest)
end

Turing.@model function turing_impute_each_value(data::NamedTuple{()}, distributions)
    return (;)
end

"""
    turing_impute_value(data_value, distribution)

Turing.jl model for one value of the data: a value that is there is taken as it is, and so
is any marker but [`MissingValue`](@ref), which is a latent variable named `imputed` with
`distribution`, the value of a [`Deterministic`](@ref), or the marker itself when the model
gives [`NoValue`](@ref) on that timestep. Used by [`turing_impute_each_value`](@ref).
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
