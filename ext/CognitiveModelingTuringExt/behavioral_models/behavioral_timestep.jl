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

Each model runs as [`turing_timestep_one_perspective`](@ref), the environment's under the
prefix `environment`. Without an environment, the observation comes from the data: the
model acts on this timestep's observation when the environment starts, and on the previous
one when the agent starts. With an environment, the environment gives the observation in
response to the action, before the model acts when the environment starts, and after when
the agent starts. Used by [`turing_run_all_timesteps`](@ref).
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
        turing_timestep_one_perspective(
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
                turing_timestep_one_perspective(
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
        turing_timestep_one_perspective(
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
                turing_timestep_one_perspective(
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
    turing_timestep_one_perspective(model, context, data_columns, t)

Turing.jl model for one model's part of timestep `t`, the same for a cognitive model and an
environment model: `model` runs on `context` with [`turing_timestep_model`](@ref), and its
outputs are then taken from row `t` of `data_columns`, the data for them, where it has
values, imputed from what the model gave where they are [`MissingValue`](@ref), as
[`turing_impute_missing_values`](@ref) does, and are the marker that the data has for them
otherwise, such as `UnimputedMissingValue()` or `NoValue()`. It returns the output, the
model's new state, the distribution it gave for the output and which outputs to condition on
at the timestep, as [`find_outputs_to_condition_on`](@ref) gives it, so that their values
can be scored against the distributions at the end of the session. The row is taken here and
not given. Used by [`turing_behavioral_timestep`](@ref).
"""
Turing.@model function turing_timestep_one_perspective(model, context, data_columns, t)
    #Take this timestep's row of the data for the model's outputs
    data_output = select_row(data_columns, t)

    #Run the model's timestep with the Turing version, so that a model can sample within it, and check that it returned a ModelOutput
    model_output ~ to_submodel(turing_timestep_model(model, context), false)
    model_output = check_model_output(model_output, model)

    #Take the outputs from the data where it has them, and impute those that are missing from the distributions the model gave
    output ~ to_submodel(
        turing_impute_missing_values(data_output, model_output.output_distribution), false
    )

    #Find which outputs have values in the data to condition on, which the session scores against the distributions the model gave, all at once at its end
    outputs_to_condition_on = find_outputs_to_condition_on(
        data_output, model_output.output_distribution, model
    )

    #Return the outputs, the model's new state, the distributions it gave and which outputs to condition on. The outputs can have one of two types, so they go in through build_named_entry
    return merge(
        build_named_entry(Val(:output), output),
        (;
            new_state=model_output.new_state,
            output_distribution=model_output.output_distribution,
            outputs_to_condition_on=outputs_to_condition_on,
        ),
    )
end
