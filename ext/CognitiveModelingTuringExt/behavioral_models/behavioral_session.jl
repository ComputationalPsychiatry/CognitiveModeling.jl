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

    #The parameters of a model with submodels are bundled, as the model reads them
    bundled_cognitive_parameters = bundle_parameters(
        cognitive_model, cognitive_parameters_with_defaults
    )
    initial_cognitive_state ~
    to_submodel(turing_initialize_state(cognitive_model, bundled_cognitive_parameters))

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
        turing_run_all_timesteps(session_info, bundled_cognitive_parameters, (;), previous),
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

    #The parameters of models with submodels are bundled, as the models read them
    bundled_cognitive_parameters = bundle_parameters(
        cognitive_model, cognitive_parameters_with_defaults
    )
    bundled_environment_parameters = bundle_parameters(
        environment, environment_parameters_with_defaults
    )
    initial_cognitive_state ~
    to_submodel(turing_initialize_state(cognitive_model, bundled_cognitive_parameters))
    initial_environment_state ~
    to_submodel(turing_initialize_state(environment, bundled_environment_parameters))

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
            bundled_cognitive_parameters,
            bundled_environment_parameters,
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
as [`turing_run_timestep_both_models`](@ref) with the prefix `timesteps[t]`, saves what the
`SaveTrajectories` of `session_info` select, and scores the actions in the data, and the
observations in the data when the environment runs, against the distributions of their
timesteps all at once at the end, as observed values. Before each timestep, it checks the
[`stopping_criterion`](@ref) of the models, and ends the session there when one is met, with
what it saved up to it. A criterion that is met already before the first timestep is an
error, from [`check_stopping_criterion_not_met`](@ref). When either model sets
`parameters_valid` to `false` on a timestep, or a criterion is met while the data continues,
which the parameters then can't give, the session gives the parameters a log density of
`-Inf`, keeps nothing of that timestep, scores nothing and ends there. It returns the
trajectories the session saved, in a [`SessionTrajectories`](@ref), as
[`turing_behavioral_session`](@ref) describes them, which returns them under
`saved_trajectories`, with whether the parameters stayed valid as
[`session_parameters_valid`](@ref) gives it. Both methods of that model call this one, after
setting up the parameters and the initial states with or without an environment. Used by
[`turing_behavioral_session`](@ref).
"""
Turing.@model function turing_run_all_timesteps(
    session_info, cognitive_parameters, environment_parameters, previous
)
    n_timesteps = session_info.n_timesteps

    #Which states hold the states of submodels, for the trajectories the session returns
    submodel_states = submodel_states_of(
        session_info.cognitive_model, session_info.environment
    )

    #A criterion that is met already would end the session before its first timestep, which is an error
    check_stopping_criterion_not_met(
        session_info.environment, previous.environment.state, environment_parameters
    )
    check_stopping_criterion_not_met(
        session_info.cognitive_model, previous.state, cognitive_parameters
    )

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_saved = select_initial_timestep_values_to_save(
        Val(saves_initial_timestep_values(session_info.save_cognitive_trajectories)),
        session_info,
        previous,
    )

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    timestep_values ~ to_submodel(
        prefix(
            turing_run_timestep_both_models(
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

    #A session whose parameters turned invalid on the first timestep gives them a log density of -Inf, keeps nothing of the timestep, scores nothing and ends
    if !parameters_are_valid(timestep_values.parameters_valid)
        Turing.@addlogprob! -Inf
        map_saved(vector -> resize!(vector, offset), storage)
        return saved_trajectories(
            storage,
            timestep_sizes_until(session_info.saved_timestep_sizes, offset),
            false,
            0,
            nothing,
            submodel_states,
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

        #Stop when the stopping criterion of either model is met. While the data continues, the parameters can't give it, as when they turn invalid. The criteria are checked in a function that returns a Bool, also when the compiler doesn't know the type of the values of the timestep before, as when an output is NoValue on some timesteps, so that Enzyme.jl doesn't differentiate a state read from them
        if stopping_criterion_of_either_model(
            session_info.cognitive_model,
            session_info.environment,
            timestep_values,
            cognitive_parameters,
            environment_parameters,
        )
            if data_continues_after(session_info, t - 1)
                Turing.@addlogprob! -Inf
                map_saved(vector -> resize!(vector, t - 1 + offset), storage)
                return saved_trajectories(
                    storage,
                    timestep_sizes_until(session_info.saved_timestep_sizes, t - 1 + offset),
                    session_parameters_valid(timestep_values.parameters_valid, false),
                    t - 1,
                    nothing,
                    submodel_states,
                )
            end
            n_timesteps_that_ran = t - 1

            #Keep only the timesteps that ran, shortening the vectors in place. The function uses t and not n_timesteps_that_ran, since a variable that is assigned in two places is boxed when a function uses it: its type is then unknown, and the session slower
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            break
        end

        next_timestep_values ~ to_submodel(
            prefix(
                turing_run_timestep_both_models(
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

        #A session whose parameters turned invalid on this timestep gives them a log density of -Inf, keeps nothing of the timestep, scores nothing and ends before it
        if !parameters_are_valid(next_timestep_values.parameters_valid)
            Turing.@addlogprob! -Inf
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            return saved_trajectories(
                storage,
                timestep_sizes_until(session_info.saved_timestep_sizes, t - 1 + offset),
                session_parameters_valid(next_timestep_values.parameters_valid, true),
                t - 1,
                nothing,
                submodel_states,
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

    #Score the actions in the data, and the observations in the data with an environment, against the distributions of their timesteps, all at once as observed values, in a product of the outputs, each with the product of distributions that takes its kind of values
    if !isempty(session_info.actions_to_condition_on)
        session_info.actions_to_condition_on ~ product_of_outputs(
            distributions_for_conditioning.actions, session_info.actions_to_condition_on
        )
    end
    if has_environment(session_info) && !isempty(session_info.observations_to_condition_on)
        session_info.observations_to_condition_on ~ product_of_outputs(
            distributions_for_conditioning.observations,
            session_info.observations_to_condition_on,
        )
    end

    return saved_trajectories(
        storage,
        timestep_sizes_until(
            session_info.saved_timestep_sizes, n_timesteps_that_ran + offset
        ),
        session_parameters_valid(timestep_values.parameters_valid, true),
        n_timesteps_that_ran,
        nothing,
        submodel_states,
    )
end
