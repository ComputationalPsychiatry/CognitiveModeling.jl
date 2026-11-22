### MODEL FOR A BEHAVIORAL SESSION ###

#Turing model for running a single behavioral session without an environment: every observation comes from the data
Turing.@model function CognitiveModeling.turing_behavioral_session(
    session_info::SessionInfo{<:Any,NoEnvironment}, cognitive_parameters
)
    cognitive_model = session_info.cognitive_model

    #Fill in the default parameters, and initialize the state with them, with the Turing version so that a model can sample its start
    cognitive_parameters_with_defaults = merge_default_parameters(
        cognitive_model, cognitive_parameters
    )
    initial_cognitive_state ~ to_submodel(
        turing_initialize_state(cognitive_model, cognitive_parameters_with_defaults)
    )

    #Check the names of the parameters, the state and the previous action against attribute_names, which is decided at compile time, and that each parameter is within its domain
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters_with_defaults,
        initial_cognitive_state,
        session_info.previous_action,
    )
    if session_info.check_parameter_domains
        check_parameters_in_domains(cognitive_model, cognitive_parameters_with_defaults)
    end

    #Run the timesteps from where the session starts, with the values arranged as those of a timestep
    previous = (;
        state=initial_cognitive_state,
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        environment=(; state=nothing),
    )
    saved_trajectories ~ to_submodel(
        turing_run_all_timesteps(
            session_info, cognitive_parameters_with_defaults, (;), previous
        ),
        false,
    )

    #The parameters as they were given, without the defaults, so that a model around the session which transforms its variables into them has them for each sample, with the trajectories the session saved
    return (; cognitive_parameters, environment_parameters=(;), saved_trajectories)
end

#The same with an environment, which fills in the observations that are missing in the data, and takes its parameters too
Turing.@model function CognitiveModeling.turing_behavioral_session(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel},
    cognitive_parameters,
    environment_parameters=(;),
)
    #A session with nothing missing runs without its environment, whose scores of the observations wouldn't depend on any variable, so it has the variables of a session without one, and an empty record of the environment. It returns the environment parameters it was given
    if !session_info.includes_environment
        session_without_environment ~ to_submodel(
            turing_behavioral_session(
                remake_session_without_environment(session_info), cognitive_parameters
            ),
            false,
        )
        return (;
            cognitive_parameters,
            environment_parameters,
            saved_trajectories=with_empty_environment_trajectories(
                session_without_environment.saved_trajectories
            ),
        )
    end

    cognitive_model = session_info.cognitive_model
    environment = session_info.environment

    #Fill in the default parameters of both models, and initialize both states with them
    cognitive_parameters_with_defaults = merge_default_parameters(
        cognitive_model, cognitive_parameters
    )
    environment_parameters_with_defaults = merge_default_parameters(
        environment, environment_parameters
    )
    initial_cognitive_state ~ to_submodel(
        turing_initialize_state(cognitive_model, cognitive_parameters_with_defaults)
    )
    initial_environment_state ~
    to_submodel(turing_initialize_state(environment, environment_parameters_with_defaults))

    #Check the names of both models against attribute_names, which is decided at compile time
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters_with_defaults,
        initial_cognitive_state,
        session_info.previous_action,
    )
    check_match_attribute_names(
        environment,
        environment_parameters_with_defaults,
        initial_environment_state,
        session_info.previous_environment_observation,
    )

    #Check that each parameter of both models is within its domain, unless the session leaves the check out
    if session_info.check_parameter_domains
        check_parameters_in_domains(cognitive_model, cognitive_parameters_with_defaults)
        check_parameters_in_domains(environment, environment_parameters_with_defaults)
    end

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
        turing_run_all_timesteps(
            session_info,
            cognitive_parameters_with_defaults,
            environment_parameters_with_defaults,
            previous,
        ),
        false,
    )

    #The parameters as they were given, without the defaults, with the trajectories the session saved
    return (; cognitive_parameters, environment_parameters, saved_trajectories)
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

#The model of one session under the name that every inference backend has for it
function CognitiveModeling.session_model(
    ::TuringBackend,
    session_info::SessionInfo,
    cognitive_parameters,
    environment_parameters=(;),
)
    return turing_behavioral_session(
        session_info, cognitive_parameters, environment_parameters
    )
end

### THE TIMESTEPS OF A SESSION ###

"""
    turing_run_all_timesteps(session_info, cognitive_parameters, environment_parameters,
                         previous)

Turing.jl model which runs the timesteps of a session from `previous`, where it starts, each
as [`turing_behavioral_timestep`](@ref) with the prefix `timesteps[t]`, saves what the save
histories of `session_info` select, and scores the actions in the data, and the observations
in the data when the environment runs, against the distributions of their timesteps all at
once at the end, as observed values. Before each timestep from
`first_timestep_to_check_stopping` of `session_info` on, it checks the
[`stopping_criterion`](@ref) of the models, and ends the session there when one is met, with
what it saved up to it, which is an error when the data continues. When either model sets
`parameters_valid` to `false` on a timestep, the session gives the parameters a log density
of `-Inf` and ends before that timestep, without scoring the data. It returns the
trajectories the session saved, in a [`SessionTrajectories`](@ref), as
[`turing_behavioral_session`](@ref) describes them, which returns them under
`saved_trajectories`. Both methods of that model call this one, after setting up the
parameters and the initial states with or without an environment. Used by
[`turing_behavioral_session`](@ref).
"""
Turing.@model function turing_run_all_timesteps(
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
            @varname(timesteps[1]),
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

    #A session whose parameters turned invalid on the first timestep gives them a log density of -Inf, keeps nothing of that timestep, and scores nothing
    if !parameters_are_valid(timestep_values.parameters_valid)
        Turing.@addlogprob! -Inf
        map_saved(vector -> resize!(vector, offset), storage)
        return saved_trajectories(
            storage,
            timestep_sizes_until(session_info.saved_timestep_sizes, offset),
            timestep_values.parameters_valid,
            0,
            nothing,
        )
    end

    #Initialize the distributions of the actions and observations to condition on
    distributions_for_conditioning = initialize_distributions_for_conditioning(
        session_info, timestep_values
    )
    distributions_for_conditioning = store_distributions_for_conditioning(
        session_info, distributions_for_conditioning, timestep_values
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
                @varname(timesteps[t]),
            ),
            false,
        )

        #A session whose parameters turned invalid on this timestep gives them a log density of -Inf, ends before the timestep, and scores nothing
        if !parameters_are_valid(next_timestep_values.parameters_valid)
            Turing.@addlogprob! -Inf
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            return saved_trajectories(
                storage,
                timestep_sizes_until(session_info.saved_timestep_sizes, t - 1 + offset),
                next_timestep_values.parameters_valid,
                t - 1,
                nothing,
            )
        end

        timestep_values = next_timestep_values
        storage = store_timestep_values!(
            storage,
            select_timestep_values_to_save(session_info, timestep_values),
            t + offset,
        )
        distributions_for_conditioning = store_distributions_for_conditioning(
            session_info, distributions_for_conditioning, timestep_values
        )
    end

    #Score the actions in the data, and the observations in the data with an environment, against the distributions of their timesteps, all at once as observed values, with the product of distributions that takes the kind of values of each output
    if !isempty(session_info.actions_to_condition_on)
        session_info.actions_to_condition_on ~ product_distribution(
            map(
                product_of_distributions,
                map(scorable_distributions, distributions_for_conditioning.actions),
                session_info.actions_to_condition_on,
            ),
        )
    end
    if has_environment(session_info) && !isempty(session_info.observations_to_condition_on)
        session_info.observations_to_condition_on ~ product_distribution(
            map(
                product_of_distributions,
                map(scorable_distributions, distributions_for_conditioning.observations),
                session_info.observations_to_condition_on,
            ),
        )
    end

    return saved_trajectories(
        storage,
        timestep_sizes_until(
            session_info.saved_timestep_sizes, n_timesteps_that_ran + offset
        ),
        timestep_values.parameters_valid,
        n_timesteps_that_ran,
        nothing,
    )
end
