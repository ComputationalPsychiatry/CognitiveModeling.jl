### MODEL WHICH HAS ONE COMPONENT FOR SAMPLING THE COGNITIVE PARAMETERS (e.g. a regression), AND THEN THE BEHAVIOURAL SESSIONS ###
# There is also an optional third component which can do something with the outcomes of the behavioural sessions,
# such as a regression predicting something (e.g. physiological data) from the cognitive states.

#Turing model for fitting a cognitive model to several sessions, with a population model for their parameters and an optional downstream model
Turing.@model function CognitiveModeling.turing_full_model(population_model, sessions_info::MultipleSessionsInfo; downstream_model = NoDownstreamModel(), environment_parameters = (;), catch_invalid_parameters::Bool = false, parallelisation = NoParallelisation(), include_submodel_prefixes::Bool = false)

    #Sample the cognitive parameters of every session with the population model, with its variables under population_model when the prefixes are included
    if include_submodel_prefixes
        cognitive_parameters ~ to_submodel(prefix(turing_population_model(population_model, sessions_info), @varname(population_model)), false)
    else
        cognitive_parameters ~ to_submodel(turing_population_model(population_model, sessions_info), false)
    end

    #Run all behavioural sessions, inside a try block when invalid parameters are caught. The parameters are sampled outside it, so that they are kept when a draw is rejected, which a sampler needs to start
    sessions = turing_multiple_behavioural_sessions(sessions_info, cognitive_parameters, environment_parameters; parallelisation)
    if catch_invalid_parameters
        saved_behavioural_values ~ to_submodel(turing_catch_invalid_parameters(sessions), false)

        #A rejected draw leaves nothing for the downstream model, so it returns at once, which Enzyme can differentiate where a value of two types can't be
        if saved_behavioural_values === nothing
            return nothing
        end
    else
        saved_behavioural_values ~ to_submodel(sessions, false)
    end

    #Do the optional downstream model, with its variables under downstream_model when the prefixes are included
    if include_submodel_prefixes
        downstream_values ~ to_submodel(prefix(turing_downstream_model(downstream_model, saved_behavioural_values, cognitive_parameters, sessions_info), @varname(downstream_model)), false)
    else
        downstream_values ~ to_submodel(turing_downstream_model(downstream_model, saved_behavioural_values, cognitive_parameters, sessions_info), false)
    end

    return (; cognitive_parameters, saved_behavioural_values, downstream_values)
end


### MODEL FOR MULTIPLE SESSIONS ###

#Turing model for running several behavioural sessions, each with its own cognitive parameters, as a submodel with the prefix session[i], and collecting what each returns
#The environment parameters are the same for every session, and can only be given when the sessions have an environment
Turing.@model function CognitiveModeling.turing_multiple_behavioural_sessions(sessions_info::MultipleSessionsInfo, cognitive_parameters, environment_parameters = (;); parallelisation = NoParallelisation())
    check_environment_parameters(sessions_info, environment_parameters)
    saved_values ~ to_submodel(turing_run_sessions(parallelisation, sessions_info, cognitive_parameters, environment_parameters), false)
    return saved_values
end

"""
    turing_run_sessions(parallelisation, sessions_info, cognitive_parameters,
                        environment_parameters)

Turing.jl model which runs every session in `sessions_info` as a submodel with the prefix
`session[i]`, and returns a vector with what each returned. The first session runs outside
the loop, so that the vector has its type. With [`NoParallelisation`](@ref) the sessions run
one after the other, and with [`ParallelSessions`](@ref) over the threads Julia was started
with, which needs the model that is evaluated to be marked thread-safe, since the log density
is silently wrong otherwise.
"""
Turing.@model function turing_run_sessions(::NoParallelisation, sessions_info, cognitive_parameters, environment_parameters)
    sessions = sessions_info.sessions
    first_saved ~ to_submodel(prefix(turing_behavioural_session(sessions[1], row(cognitive_parameters, 1), environment_parameters), @varname(session[1])), false)
    saved_values = Vector{typeof(first_saved)}(undef, length(sessions))
    saved_values[1] = first_saved
    for i in 2:length(sessions)
        saved ~ to_submodel(prefix(turing_behavioural_session(sessions[i], row(cognitive_parameters, i), environment_parameters), @varname(session[i])), false)
        saved_values[i] = saved
    end
    return saved_values
end

Turing.@model function turing_run_sessions(::ParallelSessions, sessions_info, cognitive_parameters, environment_parameters)
    if !(__varinfo__ isa DynamicPPL.ThreadSafeVarInfo)
        throw(ArgumentError("The sessions run in parallel, so the model that is evaluated must be marked thread-safe with setthreadsafe(model, true) from DynamicPPL. Without it, the log density is wrong with no error."))
    end
    sessions = sessions_info.sessions
    first_saved ~ to_submodel(prefix(turing_behavioural_session(sessions[1], row(cognitive_parameters, 1), environment_parameters), @varname(session[1])), false)
    saved_values = Vector{typeof(first_saved)}(undef, length(sessions))
    saved_values[1] = first_saved
    #Written as Base.Threads.@threads, which is the same macro, so that DynamicPPL doesn't warn about thread safety when the package loads. The check above takes the warning's place
    Base.Threads.@threads for i in 2:length(sessions)
        saved ~ to_submodel(prefix(turing_behavioural_session(sessions[i], row(cognitive_parameters, i), environment_parameters), @varname(session[i])), false)
        saved_values[i] = saved
    end
    return saved_values
end


### MODEL FOR A BEHAVIOURAL SESSION ###

#Turing model for running a single behavioural session without an environment: every observation comes from the data
Turing.@model function CognitiveModeling.turing_behavioural_session(session_info::SessionInfo{<:Any, NoEnvironment}, cognitive_parameters)
    cognitive_model = session_info.cognitive_model

    #Fill in the default parameters, and initialise the state with them, with the Turing version so that a model can sample its start
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    initial_cognitive_state ~ to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))

    #Check the names of the parameters, the state and the previous action against attribute_names, which is decided at compile time
    check_match_attribute_names(cognitive_model, cognitive_parameters, initial_cognitive_state, session_info.previous_action)

    #Run the timesteps from where the session starts
    previous = (; observation = session_info.previous_observation, action = session_info.previous_action, cognitive_state = initial_cognitive_state)
    saved_values ~ to_submodel(turing_run_timesteps(session_info, cognitive_parameters, (;), previous), false)
    return saved_values
end

#The same with an environment, which fills in the observations that are missing in the data, and takes its parameters too
Turing.@model function CognitiveModeling.turing_behavioural_session(session_info::SessionInfo{<:Any, <:AbstractEnvironmentModel}, cognitive_parameters, environment_parameters = (;))
    cognitive_model = session_info.cognitive_model
    environment = session_info.environment

    #Fill in the default parameters of both models, and initialise both states with them
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    environment_parameters = merge_default_parameters(environment, environment_parameters)
    initial_cognitive_state ~ to_submodel(turing_initialize_state(cognitive_model, cognitive_parameters))
    initial_environment_state ~ to_submodel(turing_initialize_state(environment, environment_parameters))

    #Check the names of both models against attribute_names, which is decided at compile time
    check_match_attribute_names(cognitive_model, cognitive_parameters, initial_cognitive_state, session_info.previous_action)
    check_match_attribute_names(environment, environment_parameters, initial_environment_state, session_info.previous_observation)

    #Run the timesteps from where the session starts
    previous = (; observation = session_info.previous_observation, action = session_info.previous_action, cognitive_state = initial_cognitive_state, environment_state = initial_environment_state)
    saved_values ~ to_submodel(turing_run_timesteps(session_info, cognitive_parameters, environment_parameters, previous), false)
    return saved_values
end

#A session without an environment takes no environment parameters, so an empty NamedTuple passes, which lets several sessions be run alike, and anything else is an error
function CognitiveModeling.turing_behavioural_session(session_info::SessionInfo{<:Any, NoEnvironment}, cognitive_parameters, environment_parameters::NamedTuple)
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
    scorable_distributions(distributions)

Return the vector of the distributions of one action as it is, or an empty vector of
`Normal`s for an action the model took on no timestep and the data recorded on none, whose
vector never got a type. Either scores an empty vector of recorded values as zero, where a
`Vector{Union{}}` can't be made into a product distribution at all. Used by
[`turing_run_timesteps`](@ref) before the recorded actions are scored.
"""
scorable_distributions(distributions::AbstractVector) = distributions
scorable_distributions(::Vector{Union{}}) = Normal{Float64}[]

"""
    turing_run_timesteps(session_info, cognitive_parameters, environment_parameters, previous)

Turing.jl model which runs the timesteps of a session from `previous`, where it starts, each
as [`turing_behavioural_timestep`](@ref) with the prefix `timestep[t]`, saves what the
`save_history` of `session_info` selects, and scores the recorded actions against the
distributions of their timesteps all at once at the end. It returns what the session saved,
as [`turing_behavioural_session`](@ref) describes. Both methods of that model call this one,
after setting up the parameters and the initial states with or without an environment.
"""
Turing.@model function turing_run_timesteps(session_info, cognitive_parameters, environment_parameters, previous)
    cognitive_model = session_info.cognitive_model
    n_timesteps = session_info.n_timesteps
    save_history = session_info.save_history

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_saved = initial_session_outputs(Val(saves_initial_values(save_history)), cognitive_model, previous, save_history)

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs ~ to_submodel(prefix(turing_behavioural_timestep(session_info, 1, cognitive_parameters, environment_parameters, previous), @varname(timestep[1])), false)

    #Initialize storage for the saved outputs, which starts with timestep 0 when it is kept, and the distributions of the recorded actions
    storage = initialize_session_storage(initial_saved, saved_session_outputs(cognitive_model, outputs, save_history), n_timesteps)
    offset = initial_saved === nothing ? 0 : 1
    recorded_names = recorded_action_names(session_info)
    distributions = store_distributions(initialize_distributions(select_names(outputs.action_distribution, recorded_names)), select_names(outputs.action_distribution, recorded_names), outputs.recorded_on_timestep)

    #Run the remaining timesteps, each starting from the outputs of the one before, storing timestep t as element t + offset
    for t in 2:n_timesteps
        next_outputs ~ to_submodel(prefix(turing_behavioural_timestep(session_info, t, cognitive_parameters, environment_parameters, outputs), @varname(timestep[t])), false)
        outputs = next_outputs
        storage = store_session_outputs!(storage, saved_session_outputs(cognitive_model, outputs, save_history), t + offset)
        distributions = store_distributions(distributions, select_names(outputs.action_distribution, recorded_names), outputs.recorded_on_timestep)
    end

    #Score the recorded actions against the distributions of their timesteps, all at once
    if !isempty(session_info.recorded)
        session_info.recorded ~ product_distribution(map(product_distribution, map(scorable_distributions, distributions)))
    end

    return (; observations = storage.observations, actions = storage.actions, action_distributions = storage.action_distributions, states = storage.states, timestep_sizes = session_info.saved_timestep_sizes)
end

### MODEL FOR A SINGLE TIMESTEP ###

"""
    turing_behavioural_timestep(session_info, t, cognitive_parameters, environment_parameters,
                                previous)

Turing.jl model which runs timestep `t` of a session from `previous`, the outputs of the
timestep before, and returns what the next one starts from, together with what the session
saves and scores: the observation, the action, the action distribution, the cognitive state,
the environment state when there is an environment, and which actions are recorded on the
timestep. Without an environment, the observation comes from the data: the model acts on
this timestep's observation when the environment starts, and on the previous one when the
agent starts. With an environment, [`turing_environment_timestep`](@ref) gives the
observation in response to the action, before the model acts when the environment starts,
and after when the agent starts.
"""
Turing.@model function turing_behavioural_timestep(session_info::SessionInfo{<:Any, NoEnvironment}, t, cognitive_parameters, environment_parameters, previous)
    cognitive_model = session_info.cognitive_model
    data_observation = row(session_info.observations, t)
    data_action = row(session_info.actions, t)

    #The model acts on this timestep's observation when the environment starts, and on the previous one when the agent starts
    observation = session_info.who_starts isa EnvironmentStarts ? data_observation : previous.observation
    output ~ to_submodel(turing_apply_model(cognitive_model, ModelInput(observation, previous.cognitive_state, previous.action, cognitive_parameters, get_timestep_size(session_info.timestep_sizes, t), default_rng(), session_info.who_starts)), false)
    output = check_model_output(output, cognitive_model)

    #Impute the missing actions, and note which actions are recorded on this timestep
    action ~ to_submodel(turing_impute_missing_values(data_action, output.action_distribution), false)
    recorded = recorded_on_timestep(data_action, output.action_distribution, cognitive_model)

    #The next timestep starts from this timestep's observation in the data. The observation and the action can have one of two types, so they go in through entry
    return merge(
        entry(Val(:observation), data_observation),
        entry(Val(:action), action),
        (; action_distribution = output.action_distribution, cognitive_state = output.new_state, recorded_on_timestep = recorded),
    )
end

Turing.@model function turing_behavioural_timestep(session_info::SessionInfo{<:Any, <:AbstractEnvironmentModel}, t, cognitive_parameters, environment_parameters, previous)
    cognitive_model = session_info.cognitive_model
    data_observation = row(session_info.observations, t)
    data_action = row(session_info.actions, t)
    timestep_size = get_timestep_size(session_info.timestep_sizes, t)

    #When the environment starts, it gives this timestep's observation, in response to the previous action
    if session_info.who_starts isa EnvironmentStarts
        environment_output ~ to_submodel(turing_environment_timestep(session_info.environment, previous.action, previous, environment_parameters, data_observation, timestep_size, session_info.who_starts), false)
        observation = environment_output.observation
        environment_state = environment_output.state
    else
        observation = previous.observation
        environment_state = previous.environment_state
    end

    #Run the cognitive model on the observation
    output ~ to_submodel(turing_apply_model(cognitive_model, ModelInput(observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, default_rng(), session_info.who_starts)), false)
    output = check_model_output(output, cognitive_model)

    #Impute the missing actions, and note which actions are recorded on this timestep
    action ~ to_submodel(turing_impute_missing_values(data_action, output.action_distribution), false)
    recorded = recorded_on_timestep(data_action, output.action_distribution, cognitive_model)

    #When the agent starts, the environment responds to the action with the next observation
    if session_info.who_starts isa AgentStarts
        environment_output ~ to_submodel(turing_environment_timestep(session_info.environment, action, previous, environment_parameters, data_observation, timestep_size, session_info.who_starts), false)
        observation = environment_output.observation
        environment_state = environment_output.state
    end

    #The observation and the action can have one of two types, so they go in through entry
    return merge(
        entry(Val(:observation), observation),
        entry(Val(:action), action),
        (; action_distribution = output.action_distribution, cognitive_state = output.new_state, environment_state, recorded_on_timestep = recorded),
    )
end

"""
    turing_environment_timestep(environment, action, previous, environment_parameters,
                                data_observation, timestep_size, who_starts)

Turing.jl model for the environment's part of a timestep: `environment` runs on `action` as
a submodel with the prefix `environment`, and the observations that are
[`MissingValue`](@ref) in `data_observation` are filled in from what it gives, as
[`turing_impute_missing_values`](@ref) does, under the same prefix. It returns the
observation and the environment's new state.
"""
Turing.@model function turing_environment_timestep(environment, action, previous, environment_parameters, data_observation, timestep_size, who_starts)
    environment_output ~ to_submodel(prefix(turing_apply_model(environment, EnvironmentInput(action, previous.environment_state, previous.observation, environment_parameters, timestep_size, default_rng(), who_starts)), @varname(environment)), false)
    environment_output = check_environment_output(environment_output, environment)
    observation ~ to_submodel(prefix(turing_impute_missing_values(data_observation, environment_output.observation_distribution), @varname(environment)), false)
    return merge(entry(Val(:observation), observation), (; state = environment_output.new_state))
end


### IMPUTING THE VALUES THAT ARE MISSING IN THE DATA ###

"""
    turing_impute_missing_values(data, distributions)

Turing.jl model which fills in the actions or the observations of one timestep: each value
that is [`MissingValue`](@ref) in `data` is imputed as a latent variable named after it, such
as `choice.imputed`, with the distribution the model gives for it in `distributions`, and
the rest are taken from the data. The data is returned as it is when nothing is missing on
the timestep, which is decided at compile time when nothing can be. For observations it
runs under the environment's prefix, so that an observation and an action with the same
name get a latent variable each.
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
Turing.@model function turing_impute_each_value(data::NamedTuple{names}, distributions) where {names}
    name = first(names)
    value ~ to_submodel(prefix(turing_impute_value(first(data), getproperty(distributions, name)), VarName{name}()), false)
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
