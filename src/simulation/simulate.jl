## Sampling outputs ##
"""
    sample_output(rng, output_distribution)

Sample an output from what a model returned, using `rand`. Each output in a `NamedTuple` is
sampled on its own, so a `NamedTuple` inside it gives a `NamedTuple` of samples, and an
entry that is [`NoValue`](@ref) is kept. A number, a string or a symbol is a value and not
something to sample from, and throws an `ArgumentError` which
names [`Deterministic`](@ref). Used on each timestep of a simulation, by
[`simulate_timestep`](@ref).
"""
function sample_output(rng, output_distribution)
    return rand(rng, output_distribution)
end

#Samples each named output on its own
function sample_output(rng, output_distributions::NamedTuple)
    return map(
        output_distribution -> sample_output(rng, output_distribution), output_distributions
    )
end

#An output that isn't given has nothing to sample
function sample_output(rng, no_value::AbstractNoValue)
    return no_value
end

#A number, a string or a symbol is a value, which rand would read as something else or not at all
function sample_output(rng, value::Union{Number,AbstractString,Symbol})
    return throw(
        ArgumentError(
            "A model gave $(repr(value)), of type $(typeof(value)), as what to sample an output from, but it is a value. An output that isn't random is given in a Deterministic, as in Deterministic($(repr(value))).",
        ),
    )
end

## Starting a simulation ##
"""
    merge_default_parameters(model, parameters)

Return `parameters` with the values from [`default_parameters`](@ref) of `model`, a
cognitive model or an environment model, for the parameters it doesn't have, if `model`
provides defaults for them. Used by [`simulate`](@ref), [`InstantiatedModel`](@ref) and
fitted sessions.
"""
function merge_default_parameters(model::AbstractTimestepModel, parameters)
    return merge(default_parameters(model), parameters)
end

## Passing outputs between the model and the environment ##
"""
    map_names(x, name_map)

Return what the receiver gets of the outputs `x`: all of `x` without a map, and with a
[`NameMap`](@ref) the outputs it takes, under the names it gives them. Used on each timestep
of a simulation, by [`simulate_timestep`](@ref), and for timestep 0 by
[`collect_initial_values`](@ref).
"""
map_names(x, ::Nothing) = x

#The names are in the type, so both the selection and the new names are known at compile time
function map_names(x::NamedTuple, ::NameMap{Tgiven,Treceived}) where {Tgiven,Treceived}
    return NamedTuple{Treceived}(values(NamedTuple{Tgiven}(x)))
end

"""
    check_names_fit(cognitive_model, environment, observation_map=nothing,
                    action_map=nothing)

Throw an informative error if `cognitive_model` and `environment` can't be simulated
together. Without a map, the outputs of each must be the inputs of the other in
[`attribute_names`](@ref): the observations the environment gives are what the cognitive
model lists as its inputs, and the actions the cognitive model gives are what the
environment lists as its inputs. With a [`NameMap`](@ref) for the observations or the
actions, the map must take names the giver lists, and give exactly the names the receiver
lists. Used by [`simulate`](@ref), [`simulate!`](@ref), and [`SessionInfo`](@ref) for a
session with an environment.
"""
function check_names_fit(
    cognitive_model::AbstractTimestepModel,
    environment::AbstractTimestepModel,
    observation_map=nothing,
    action_map=nothing,
)
    cognitive_attributes = check_type_attribute_names(
        attribute_names(cognitive_model), cognitive_model
    )
    environment_attributes = check_type_attribute_names(
        attribute_names(environment), environment
    )
    check_observations_fit(
        cognitive_model,
        environment,
        cognitive_attributes.inputs,
        environment_attributes.outputs,
        observation_map,
    )
    check_actions_fit(
        cognitive_model,
        environment,
        cognitive_attributes.outputs,
        environment_attributes.inputs,
        action_map,
    )
    return nothing
end

"""
    check_observations_fit(cognitive_model, environment, cognitive_model_names,
                           environment_names, observation_map)
    check_actions_fit(cognitive_model, environment, cognitive_model_names,
                      environment_names, action_map)

Throw an informative error if the observations `environment` gives, its outputs, don't fit
those `cognitive_model` lists as its inputs, or the actions `cognitive_model` gives, its
outputs, don't fit those `environment` lists as its inputs, as [`check_names_fit`](@ref)
describes, with or without a [`NameMap`](@ref). Used by [`check_names_fit`](@ref).
"""
function check_observations_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(cognitive_model))) lists the observations $cognitive_model_names as its inputs, but $(describe_model(environment)) gives $environment_names. Without a map, the two must list the same observations. A NameMap as observation_map can select and rename them.",
            ),
        )
    end
    return nothing
end

#Without a map, the actions the cognitive model gives must be those the environment lists as its inputs. The method above is for the observations
function check_actions_fit(
    cognitive_model, environment, cognitive_model_names, environment_names, ::Nothing
)
    if !same_names(cognitive_model_names, environment_names)
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(cognitive_model))) gives the actions $cognitive_model_names, but $(describe_model(environment)) lists $environment_names as its inputs. Without a map, the two must list the same actions. A NameMap as action_map can select and rename them.",
            ),
        )
    end
    return nothing
end

#With a map, it must take observations the environment gives, and give exactly those the cognitive model lists as its inputs
function check_observations_fit(
    cognitive_model,
    environment,
    cognitive_model_names,
    environment_names,
    ::NameMap{Tgiven,Treceived},
) where {Tgiven,Treceived}
    if !has_every_name(Tgiven, environment_names)
        throw(
            ArgumentError(
                "The observation_map takes the observations $Tgiven, but $(describe_model(environment)) gives $environment_names. A map can only take observations the environment gives.",
            ),
        )
    end
    if !same_names(Treceived, cognitive_model_names)
        throw(
            ArgumentError(
                "The observation_map gives the observations $Treceived, but $(describe_model(cognitive_model)) lists $cognitive_model_names as its inputs. A map must give exactly the observations the cognitive model lists.",
            ),
        )
    end
    return nothing
end

#With a map, it must take actions the cognitive model gives, and give exactly those the environment lists as its inputs
function check_actions_fit(
    cognitive_model,
    environment,
    cognitive_model_names,
    environment_names,
    ::NameMap{Tgiven,Treceived},
) where {Tgiven,Treceived}
    if !has_every_name(Tgiven, cognitive_model_names)
        throw(
            ArgumentError(
                "The action_map takes the actions $Tgiven, but $(describe_model(cognitive_model)) gives $cognitive_model_names. A map can only take actions the cognitive model gives.",
            ),
        )
    end
    if !same_names(Treceived, environment_names)
        throw(
            ArgumentError(
                "The action_map gives the actions $Treceived, but $(describe_model(environment)) lists $environment_names as its inputs. A map must give exactly the actions the environment lists.",
            ),
        )
    end
    return nothing
end

## The values of a timestep ##
"""
    get_timestep_size(timestep_sizes, t)

Return the size of timestep `t`. A vector has one size for each timestep, and anything else
is the size of all of them. Used on each timestep, by the simulation loop and by fitted
sessions.
"""
function get_timestep_size(timestep_sizes::AbstractVector, t)
    return timestep_sizes[t]
end

function get_timestep_size(timestep_size, t)
    return timestep_size
end

#Its own function, so that the compiler can split the call when an output can have two types, like an observation with an entry that is sometimes NoValue
"""
    collect_timestep_values(environment_output, cognitive_output, environment_observation,
                            observation, action, environment_action, cognitive_model,
                            environment, cognitive_save_history, environment_save_history)

Collect the values of a timestep in a `NamedTuple`: the state and the action of the
cognitive model, and under `environment` the state and the observation of the environment,
which the next timestep starts from, and under `saved` what `cognitive_save_history` and
`environment_save_history` select, as [`map_saved`](@ref) arranges them.

Each side has the values in its own names: `observation` is the observation as the cognitive
model receives it and `environment_observation` as the environment gave it, and `action` is
the action as the cognitive model gave it and `environment_action` as the environment
receives it. The two of a pair differ when a [`NameMap`](@ref) selects or renames them. Used
by [`simulate_timestep`](@ref).
"""
function collect_timestep_values(
    environment_output,
    cognitive_output,
    environment_observation,
    observation,
    action,
    environment_action,
    cognitive_model,
    environment,
    cognitive_save_history,
    environment_save_history,
)
    return (;
        state=cognitive_output.new_state,
        action,
        environment=environment_values(
            environment_output.new_state, environment_observation
        ),
        saved=(;
            states=save_state(
                cognitive_model,
                cognitive_output.new_state,
                state_selection(cognitive_save_history),
            ),
            observations=select_names(observation, input_selection(cognitive_save_history)),
            actions=select_names(action, output_selection(cognitive_save_history)),
            action_distributions=select_names(
                cognitive_output.output_distribution,
                output_distribution_selection(cognitive_save_history),
            ),
            environment=(;
                states=save_state(
                    environment,
                    environment_output.new_state,
                    state_selection(environment_save_history),
                ),
                observations=select_names(
                    environment_observation, output_selection(environment_save_history)
                ),
                actions=select_names(
                    environment_action, input_selection(environment_save_history)
                ),
                observation_distributions=select_names(
                    environment_output.output_distribution,
                    output_distribution_selection(environment_save_history),
                ),
            ),
        ),
    )
end

#A call of its own, so that an observation which can have two types, like one with an entry that is sometimes NoValue, gives one of two NamedTuples with concrete fields, as the call is split by type
"""
    environment_values(state, observation)

Return the values of a timestep that the environment continues from, its state and its
observation, in a `NamedTuple`. Used by [`collect_timestep_values`](@ref).
"""
environment_values(state, observation) = (; state, observation)

"""
    collect_initial_values(save_initial_values::Val, previous_values, cognitive_model,
                           environment, observation_map, action_map,
                           cognitive_save_history, environment_save_history)

Return what is saved from before the first timestep, as timestep 0, arranged as the `saved`
values of [`collect_timestep_values`](@ref), or `nothing` when `save_initial_values` is
`Val(false)`.

The observation and the action are the previous ones, as each side receives them, the
output distributions are [`BeforeFirstTimestep`](@ref) for every output, since none was
given yet, and the states are the states before the first timestep. They must be saved
before the first timestep is simulated, since it can change a state in place. Used by
[`simulate_all_timesteps`](@ref).
"""
function collect_initial_values(
    ::Val{false},
    previous_values,
    cognitive_model,
    environment,
    observation_map,
    action_map,
    cognitive_save_history,
    environment_save_history,
)
    return nothing
end

function collect_initial_values(
    ::Val{true},
    previous_values,
    cognitive_model,
    environment,
    observation_map,
    action_map,
    cognitive_save_history,
    environment_save_history,
)
    return (;
        states=save_state(
            cognitive_model, previous_values.state, state_selection(cognitive_save_history)
        ),
        observations=select_names(
            map_names(previous_values.environment.observation, observation_map),
            input_selection(cognitive_save_history),
        ),
        actions=select_names(
            previous_values.action, output_selection(cognitive_save_history)
        ),
        action_distributions=select_names(
            output_before_first_timestep(cognitive_model),
            output_distribution_selection(cognitive_save_history),
        ),
        environment=(;
            states=save_state(
                environment,
                previous_values.environment.state,
                state_selection(environment_save_history),
            ),
            observations=select_names(
                previous_values.environment.observation,
                output_selection(environment_save_history),
            ),
            actions=select_names(
                map_names(previous_values.action, action_map),
                input_selection(environment_save_history),
            ),
            observation_distributions=select_names(
                output_before_first_timestep(environment),
                output_distribution_selection(environment_save_history),
            ),
        ),
    )
end

## Storing what is saved ##
#The function and the number of NamedTuples are type parameters, since Julia doesn't specialize on varargs which are only passed on, and boxes them otherwise
#Always inlined, since Enzyme can't prove the types of the vectors it creates otherwise, and throws an EnzymeNoTypeError
"""
    map_saved(f, saved...)

Apply `f` to each kind of saved value in `saved`, and return the results arranged in the
same way: the cognitive model's `states`, `observations`, `actions` and
`action_distributions`, and under `environment` the environment's `states`, `observations`,
`actions` and `observation_distributions`, as [`simulate`](@ref) returns them. What a fitted
session without an environment saves has no `environment`, and is arranged without one.
With several, `f` takes the values of the same kind from each. Used to create, fill and
shorten the vectors of the storage of a simulation and of a fitted session.
"""
@inline function map_saved(f::F, saved::Vararg{NamedTuple,N}) where {F,N}

    #Whether there is an environment is read from the names, which are in the types
    if haskey(first(saved), :environment)
        environments = map(each -> each.environment, saved)
        return (;
            states=f(map(each -> each.states, saved)...),
            observations=f(map(each -> each.observations, saved)...),
            actions=f(map(each -> each.actions, saved)...),
            action_distributions=f(map(each -> each.action_distributions, saved)...),
            environment=(;
                states=f(map(each -> each.states, environments)...),
                observations=f(map(each -> each.observations, environments)...),
                actions=f(map(each -> each.actions, environments)...),
                observation_distributions=f(
                    map(each -> each.observation_distributions, environments)...
                ),
            ),
        )
    end
    return (;
        states=f(map(each -> each.states, saved)...),
        observations=f(map(each -> each.observations, saved)...),
        actions=f(map(each -> each.actions, saved)...),
        action_distributions=f(map(each -> each.action_distributions, saved)...),
    )
end

"""
    initialize_all_storage(initial_values, saved, n_steps)

Create the vectors for saving `n_steps` timesteps, arranged as [`map_saved`](@ref) arranges
them, and typed by `saved`, what the first timestep saved. When `initial_values` isn't
`nothing`, each vector starts with what it has for timestep 0, from
[`collect_initial_values`](@ref). Used by [`simulate_all_timesteps`](@ref) and by fitted
sessions.
"""
function initialize_all_storage(::Nothing, saved, n_steps)
    return map_saved(value -> initialize_storage(value, n_steps), saved)
end

function initialize_all_storage(initial_values::NamedTuple, saved, n_steps)
    return map_saved(
        (initial_value, value) -> initialize_storage(initial_value, value, n_steps),
        initial_values,
        saved,
    )
end

"""
    store_all_values!(storage, saved, t)

Store `saved`, what timestep `t` saved, as element `t` of each vector in `storage`, with
[`store_value!`](@ref), and return the vectors arranged in the same way, in which a vector
is wider when a value didn't fit it. Used by [`simulate_remaining_timesteps`](@ref) and by
fitted sessions.
"""
function store_all_values!(storage, saved, t)
    return map_saved((vector, value) -> store_value!(vector, value, t), storage, saved)
end

"""
    saved_trajectories(storage, timestep_sizes)

Return the saved trajectories of a simulation as [`simulate`](@ref) returns them: the
vectors in `storage`, with the cognitive model's under `states`, `observations`, `actions`
and `action_distributions` and the environment's under `environment`, and then
`timestep_sizes` as given. Used by [`simulate`](@ref), [`simulate!`](@ref) and fitted
sessions.
"""
function saved_trajectories(storage, timestep_sizes)
    return merge(storage, (; timestep_sizes))
end

## Simulating timesteps ##
#Always inlined, because a call here makes the loop several times slower
"""
    simulate_timestep(::EnvironmentStarts, cognitive_model, environment,
                      cognitive_parameters, environment_parameters,
                      cognitive_save_history, environment_save_history, observation_map,
                      action_map, rng, timestep_size, previous)

Simulate a timestep where the environment moves first, starting from `previous`, the values
of the timestep before, and return its values, from [`collect_timestep_values`](@ref). Used
by [`simulate_all_timesteps`](@ref) and [`simulate_remaining_timesteps`](@ref).
"""
@inline function simulate_timestep(
    ::EnvironmentStarts,
    cognitive_model,
    environment,
    cognitive_parameters,
    environment_parameters,
    cognitive_save_history,
    environment_save_history,
    observation_map,
    action_map,
    rng,
    timestep_size,
    previous,
)

    #The environment receives the previous action as action_map passes it on
    previous_environment_action = map_names(previous.action, action_map)

    #Update the environment
    environment_output = check_model_output(
        @inline(
            apply_timestep_model(
                environment,
                ModelContext(
                    previous_environment_action,
                    previous.environment.state,
                    previous.environment.observation,
                    environment_parameters,
                    timestep_size,
                    rng,
                    EnvironmentStarts(),
                ),
            )
        ),
        environment,
    )

    #Sample observation
    environment_observation = sample_output(rng, environment_output.output_distribution)

    #The cognitive model receives the observation as observation_map passes it on
    observation = map_names(environment_observation, observation_map)

    #Update the cognitive model
    cognitive_output = check_model_output(
        @inline(
            apply_timestep_model(
                cognitive_model,
                ModelContext(
                    observation,
                    previous.state,
                    previous.action,
                    cognitive_parameters,
                    timestep_size,
                    rng,
                    EnvironmentStarts(),
                ),
            )
        ),
        cognitive_model,
    )

    #Sample action
    action = sample_output(rng, cognitive_output.output_distribution)

    #The environment's history has the action as action_map passes it on
    environment_action = map_names(action, action_map)

    return collect_timestep_values(
        environment_output,
        cognitive_output,
        environment_observation,
        observation,
        action,
        environment_action,
        cognitive_model,
        environment,
        cognitive_save_history,
        environment_save_history,
    )
end

#Always inlined, because a call here makes the loop several times slower
"""
    simulate_timestep(::AgentStarts, cognitive_model, environment, cognitive_parameters,
                      environment_parameters, cognitive_save_history,
                      environment_save_history, observation_map, action_map, rng,
                      timestep_size, previous)

Simulate a timestep where the agent moves first, starting from `previous`, the values of the
timestep before, and return its values, from [`collect_timestep_values`](@ref). Used by
[`simulate_all_timesteps`](@ref) and [`simulate_remaining_timesteps`](@ref).
"""
@inline function simulate_timestep(
    ::AgentStarts,
    cognitive_model,
    environment,
    cognitive_parameters,
    environment_parameters,
    cognitive_save_history,
    environment_save_history,
    observation_map,
    action_map,
    rng,
    timestep_size,
    previous,
)

    #The cognitive model receives the previous observation as observation_map passes it on
    previous_observation = map_names(previous.environment.observation, observation_map)

    #Update the cognitive model
    cognitive_output = check_model_output(
        @inline(
            apply_timestep_model(
                cognitive_model,
                ModelContext(
                    previous_observation,
                    previous.state,
                    previous.action,
                    cognitive_parameters,
                    timestep_size,
                    rng,
                    AgentStarts(),
                ),
            )
        ),
        cognitive_model,
    )

    #Sample action
    action = sample_output(rng, cognitive_output.output_distribution)

    #The environment receives the action as action_map passes it on
    environment_action = map_names(action, action_map)

    #Update the environment
    environment_output = check_model_output(
        @inline(
            apply_timestep_model(
                environment,
                ModelContext(
                    environment_action,
                    previous.environment.state,
                    previous.environment.observation,
                    environment_parameters,
                    timestep_size,
                    rng,
                    AgentStarts(),
                ),
            )
        ),
        environment,
    )

    #Sample observation
    environment_observation = sample_output(rng, environment_output.output_distribution)

    #The cognitive model's history has the observation as observation_map passes it on
    observation = map_names(environment_observation, observation_map)

    return collect_timestep_values(
        environment_output,
        cognitive_output,
        environment_observation,
        observation,
        action,
        environment_action,
        cognitive_model,
        environment,
        cognitive_save_history,
        environment_save_history,
    )
end

"""
    check_stopping_criterion_not_met(model, state, parameters)

Throw an informative error if [`stopping_criterion`](@ref) of `model` is already met with
`state`, before the first timestep, since no timestep can then run. Used for each model by
[`simulate_all_timesteps`](@ref), and by fitted sessions, where there is nothing to check
for an environment a session doesn't have.
"""
function check_stopping_criterion_not_met(model::AbstractTimestepModel, state, parameters)
    if stopping_criterion(model, state, parameters)
        throw(
            ArgumentError(
                "The stopping criterion of $(describe_model(model)) is already met before the first timestep, so no timestep can run. An InstantiatedModel whose stopping criterion is met starts again after reset!.",
            ),
        )
    end
    return nothing
end

#Its own function, so that it is compiled for the types of the values of the first timestep
"""
    simulate_remaining_timesteps(storage, timestep_values, who_starts, cognitive_model,
                                 environment, cognitive_parameters, environment_parameters,
                                 cognitive_save_history, environment_save_history,
                                 observation_map, action_map, rng, timestep_sizes, n_steps,
                                 offset, check_stopping_criterion)

Simulate the timesteps after the first, and return the vectors with what the simulated
timesteps saved, followed by the values of the last one. When `check_stopping_criterion` is
`true`, the simulation stops before `n_steps` timesteps if [`stopping_criterion`](@ref) of
either model is met. `offset` is 1 when the vectors start with timestep 0, and 0 otherwise,
so that timestep `t` is element `t + offset`. Used by [`simulate_all_timesteps`](@ref).
"""
function simulate_remaining_timesteps(
    storage,
    timestep_values,
    who_starts,
    cognitive_model,
    environment,
    cognitive_parameters,
    environment_parameters,
    cognitive_save_history,
    environment_save_history,
    observation_map,
    action_map,
    rng,
    timestep_sizes,
    n_steps,
    offset,
    check_stopping_criterion,
)

    #Loop through timesteps
    for t in 2:n_steps

        #Stop if the stopping criterion of either model is met, and keep only the timesteps that were simulated
        if check_stopping_criterion && (
            stopping_criterion(
                environment, timestep_values.environment.state, environment_parameters
            ) || stopping_criterion(
                cognitive_model, timestep_values.state, cognitive_parameters
            )
        )
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            return storage, timestep_values
        end

        #Simulate the timestep, starting from the values of the one before
        timestep_values = simulate_timestep(
            who_starts,
            cognitive_model,
            environment,
            cognitive_parameters,
            environment_parameters,
            cognitive_save_history,
            environment_save_history,
            observation_map,
            action_map,
            rng,
            get_timestep_size(timestep_sizes, t),
            timestep_values,
        )

        #Store what the timestep saved
        storage = store_all_values!(storage, timestep_values.saved, t + offset)
    end

    #Return what was stored, and the values of the last timestep for continuing from
    return storage, timestep_values
end

"""
    simulate_all_timesteps(previous_values, who_starts, cognitive_model, environment,
                           cognitive_parameters, environment_parameters,
                           cognitive_save_history, environment_save_history,
                           observation_map, action_map, rng, timestep_sizes, n_steps,
                           save_initial_values::Val, check_stopping_criterion)

Simulate up to `n_steps` timesteps starting from `previous_values`, and return the vectors
with what the simulated timesteps saved, followed by the values of the last one. When
`check_stopping_criterion` is `true`, the simulation stops before `n_steps` timesteps if
[`stopping_criterion`](@ref) of either model is met, and none can be simulated if it is met
already. When `save_initial_values` is `Val(true)`, each vector starts with what is saved
from before the first timestep, as timestep 0.

`previous_values` and the values returned are those of a timestep: the state and the action
of the cognitive model, and under `environment` the state and the observation of the
environment, each in its own names. The vectors are arranged as [`map_saved`](@ref)
arranges them, with what `cognitive_save_history` and `environment_save_history` select.

[`simulate!`](@ref) continues from the values of the last timestep instead of from what was
saved, because what is saved of a state can be a selection or a copy of it, and because the
next simulation can change a state in place. Used by [`simulate`](@ref) and
[`simulate!`](@ref).
"""
function simulate_all_timesteps(
    previous_values,
    who_starts,
    cognitive_model,
    environment,
    cognitive_parameters,
    environment_parameters,
    cognitive_save_history,
    environment_save_history,
    observation_map,
    action_map,
    rng,
    timestep_sizes,
    n_steps,
    save_initial_values,
    check_stopping_criterion,
)

    #The first timestep sets the types of the storage, so there must be at least one
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, but it is $n_steps."))

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_steps
        throw(
            DimensionMismatch(
                "timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and n_steps is $n_steps.",
            ),
        )
    end

    #No timestep can be simulated when the stopping criterion of either model is met already
    if check_stopping_criterion
        check_stopping_criterion_not_met(
            environment, previous_values.environment.state, environment_parameters
        )
        check_stopping_criterion_not_met(
            cognitive_model, previous_values.state, cognitive_parameters
        )
    end

    #Save what is kept from before the first timestep before it is simulated, since it can change a state in place
    initial_values = collect_initial_values(
        save_initial_values,
        previous_values,
        cognitive_model,
        environment,
        observation_map,
        action_map,
        cognitive_save_history,
        environment_save_history,
    )

    #Simulate the first timestep outside the loop, since its inputs can have other types than all the rest
    timestep_values = simulate_timestep(
        who_starts,
        cognitive_model,
        environment,
        cognitive_parameters,
        environment_parameters,
        cognitive_save_history,
        environment_save_history,
        observation_map,
        action_map,
        rng,
        get_timestep_size(timestep_sizes, 1),
        previous_values,
    )

    #Initialize storage for the trajectories that are saved, which start with timestep 0 when it is kept
    storage = initialize_all_storage(initial_values, timestep_values.saved, n_steps)
    offset = initial_values === nothing ? 0 : 1

    #Simulate the remaining timesteps
    return simulate_remaining_timesteps(
        storage,
        timestep_values,
        who_starts,
        cognitive_model,
        environment,
        cognitive_parameters,
        environment_parameters,
        cognitive_save_history,
        environment_save_history,
        observation_map,
        action_map,
        rng,
        timestep_sizes,
        n_steps,
        offset,
        check_stopping_criterion,
    )
end

## Simulating a cognitive model with an environment ##
"""
    simulate(cognitive_model, environment; <keyword arguments>)

Simulate `cognitive_model` interacting with `environment`, and return the trajectories that
were saved, which are what changed over the timesteps, in a `NamedTuple`: `states`,
`observations`, `actions` and `action_distributions` of the cognitive model, `environment`
with the `states`, `observations`, `actions` and `observation_distributions` of the
environment, and `timestep_sizes`. Each is a vector with one element for timestep 0, before
the first timestep, and one for each timestep after it.

Each element is what happened on that timestep, and the states are those after it. The
observations are those the cognitive model received, and the actions those it gave, and
under `environment` the observations are those the environment gave and the actions those it
received, which differ from the cognitive model's when a [`NameMap`](@ref) selects or
renames them. The action distributions are what the cognitive model gave its actions to be
sampled from, and the observation distributions what the environment gave its observations
to be sampled from. The element for timestep 0 has the previous observation and action,
which are [`BeforeFirstTimestep`](@ref) unless others are given, `BeforeFirstTimestep()` for
every distribution, and the states before the first timestep. Since Julia counts from 1,
element `t + 1` is timestep `t`.

`cognitive_save_history` and `environment_save_history` select what is kept of each element
of each model, which is nothing by default: give `SaveHistory(SaveEverything())` to keep
everything, and `SaveHistory(SaveEverything(); save_initial_values = false)` to leave out
timestep 0, so that element `t` is timestep `t`. The vectors of both models have the same
length, which `cognitive_save_history` decides. The sizes of the timesteps are returned as
they were given: a single size as it is, and a vector of sizes with the sizes of the
timesteps that ran, after `BeforeFirstTimestep()` for timestep 0 when it is there.

Parameters that aren't given take the values in [`default_parameters`](@ref), if
`cognitive_model` or `environment` provides defaults for them. The names of the parameters,
the states, the previous action and the previous observation must be those in
[`attribute_names`](@ref), and the outputs of each model must be the inputs of the other,
or an `ArgumentError` is thrown before the first timestep. When the two name things
differently, or one gives something the other doesn't list, `observation_map` and
`action_map` say which outputs of one the other receives, and under which names.

The simulation stops before `n_steps` timesteps if [`stopping_criterion`](@ref) of either
model is met, unless `check_stopping_criterion` is `false`, and everything returned then has
the timesteps that ran.

# Arguments
- `cognitive_model::AbstractTimestepModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `environment::AbstractTimestepModel`: the environment model, usually an
  [`AbstractEnvironmentModel`](@ref). A cognitive model can serve as the environment of
  another when its outputs are what the other lists as its inputs, and vice versa.
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep. It
  has no default, because a wrong choice shifts the model by a timestep without any error.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1. It is the
  most timesteps there can be when a model has a `stopping_criterion`.
- `timestep_sizes=1`: how much time each timestep covers, which both `cognitive_model` and
  `environment` receive. A value that isn't a vector, such as a single number, is the size
  of every timestep. A vector has one size for each timestep, so its length must be
  `n_steps`.
- `cognitive_parameters::NamedTuple`: the parameters of `cognitive_model`, with its default
  values for those that aren't given, if it provides any.
- `environment_parameters::NamedTuple`: the parameters of `environment`, with its default
  values for those that aren't given, if it provides any.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample actions and
  observations from, which `cognitive_model` and `environment` also receive in their
  contexts.
- `cognitive_save_history::SaveHistory=SaveHistory()`: which states, inputs, outputs and
  output distributions of `cognitive_model` are saved, by name, and whether timestep 0 is.
  Nothing is saved by default, and `SaveHistory(SaveEverything())` saves everything. See
  [`SaveHistory`](@ref).
- `environment_save_history::SaveHistory=SaveHistory()`: the same for `environment`, whose
  inputs are the actions it receives and whose outputs are the observations it gives.
  Whether timestep 0 is saved follows `cognitive_save_history`, so that the vectors of both
  models have the same length.
- `observation_map::Union{Nothing, NameMap}=nothing`: which observations of `environment`
  are passed on to `cognitive_model`, and under which names, as a [`NameMap`](@ref) from
  the names `environment` gives to exactly those `cognitive_model` lists. Without one, the
  two must list the same observations, which `cognitive_model` receives under the same
  names.
- `action_map::Union{Nothing, NameMap}=nothing`: which actions of `cognitive_model` are
  passed on to `environment`, and under which names, as a `NameMap` from the names
  `cognitive_model` gives to exactly those `environment` lists. Without one, the two must
  list the same actions, which `environment` receives under the same names.
- `check_stopping_criterion::Bool=true`: whether [`stopping_criterion`](@ref) of the two
  models is checked before each timestep. With `false`, all of `n_steps` timesteps run.
- `previous_cognitive_state=initialize_state(cognitive_model, ...)`: the state of
  `cognitive_model` before the first timestep, by default its initial state from the full
  parameters. A state that `cognitive_model` changes in place will be changed.
- `previous_action=output_before_first_timestep(cognitive_model)`: the action before the
  first timestep. By default every action is [`BeforeFirstTimestep`](@ref), with the names
  from [`attribute_names`](@ref).
- `previous_environment_state=initialize_state(environment, ...)`: the state of
  `environment` before the first timestep, by default its initial state from the full
  parameters.
- `previous_observation=output_before_first_timestep(environment)`: the observation before
  the first timestep. By default every observation is `BeforeFirstTimestep()`, with the
  names from `attribute_names` of `environment`.

See also [`simulate!`](@ref), which continues from where the last simulation left off.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli, BernoulliLogit

#A cognitive model which learns the expected reward
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_timestep_model(model::RescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput((; choice = BernoulliLogit(Vₜ - 0.5)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice)
end

#An environment which takes the choice, and gives a reward with a fixed probability whatever it is
struct BernoulliRewards <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(environment::BernoulliRewards, parameters)
    return (;)
end

function CognitiveModeling.apply_timestep_model(environment::BernoulliRewards, context::ModelContext)
    return ModelOutput((; reward = Bernoulli(context.parameters.reward_probability)))
end

function CognitiveModeling.attribute_names(environment::BernoulliRewards)
    return ModelAttributes(; parameters = :reward_probability, states = (), inputs = :choice, outputs = :reward)
end

simulation = simulate(
    RescorlaWagner(),
    BernoulliRewards();
    who_starts = EnvironmentStarts(),
    n_steps = 3,
    cognitive_parameters = (; learning_rate = 0.5),
    environment_parameters = (; reward_probability = 1.0),
    cognitive_save_history = SaveHistory(SaveEverything()),
)

#The initial state, as timestep 0, and the state after each timestep
simulation.states

# output

4-element Vector{@NamedTuple{expected_reward::Float64}}:
 (expected_reward = 0.5,)
 (expected_reward = 0.75,)
 (expected_reward = 0.875,)
 (expected_reward = 0.9375,)
```

# Extended help
The return type is concrete as long as the types of the outputs don't change between
timesteps. If they do, like an observation with an entry that is sometimes
[`NoValue`](@ref), the vector for that output holds a `Union` of the types. The same
goes for timestep 0 when it has other types than the timesteps after it, which the
observations and actions usually do, since they are `BeforeFirstTimestep()` there.

Nothing is copied when outputs are saved. A model or environment which changes an object in
its state in place, instead of returning a new one, must add a method of
[`prepare_state_for_saving`](@ref), or all the saved states show the final values, with no
error. Observations and actions have no such function, so they must be new
objects on each timestep.
"""
function simulate(
    cognitive_model::AbstractTimestepModel,
    environment::AbstractTimestepModel;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    n_steps::Int,
    timestep_sizes=1,
    cognitive_parameters::NamedTuple,
    environment_parameters::NamedTuple,
    rng::AbstractRNG=default_rng(),
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
    observation_map::Union{Nothing,NameMap}=nothing,
    action_map::Union{Nothing,NameMap}=nothing,
    check_stopping_criterion::Bool=true,
    previous_cognitive_state=initialize_state(
        cognitive_model, merge_default_parameters(cognitive_model, cognitive_parameters)
    ),
    previous_action=output_before_first_timestep(cognitive_model),
    previous_environment_state=initialize_state(
        environment, merge_default_parameters(environment, environment_parameters)
    ),
    previous_observation=output_before_first_timestep(environment),
)

    #Fill in the default parameters
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    environment_parameters = merge_default_parameters(environment, environment_parameters)

    #Check the names of everything against attribute_names, and that the cognitive model and the environment fit together
    check_match_attribute_names(
        cognitive_model, cognitive_parameters, previous_cognitive_state, previous_action
    )
    check_match_attribute_names(
        environment,
        environment_parameters,
        previous_environment_state,
        previous_observation,
    )
    check_names_fit(cognitive_model, environment, observation_map, action_map)

    #Check that each parameter is within its domain
    check_parameter_domains(cognitive_model, cognitive_parameters)
    check_parameter_domains(environment, environment_parameters)
    check_savehistory_selection_names(cognitive_save_history, cognitive_model)
    check_savehistory_selection_names(environment_save_history, environment)

    #Collect the values from before the first timestep, each side's in its own names
    previous_values = (;
        state=previous_cognitive_state,
        action=previous_action,
        environment=(; state=previous_environment_state, observation=previous_observation),
    )

    #Simulate the timesteps, starting with timestep 0 when the cognitive model's save history saves the initial values
    save_initial_values = Val(saves_initial_values(cognitive_save_history))
    storage, _ = simulate_all_timesteps(
        previous_values,
        who_starts,
        cognitive_model,
        environment,
        cognitive_parameters,
        environment_parameters,
        cognitive_save_history,
        environment_save_history,
        observation_map,
        action_map,
        rng,
        timestep_sizes,
        n_steps,
        save_initial_values,
        check_stopping_criterion,
    )

    #The sizes of the timesteps that were simulated, since a model can stop the simulation early
    n_timesteps_simulated =
        length(storage.states) - (saves_initial_values(cognitive_save_history) ? 1 : 0)
    timestep_sizes_simulated = returned_timestep_sizes(
        timestep_sizes, n_timesteps_simulated, save_initial_values
    )

    #Return what was saved
    return saved_trajectories(storage, timestep_sizes_simulated)
end

"""
    simulate!(agent::InstantiatedModel, environment::InstantiatedModel; <keyword arguments>)

Simulate `agent` interacting with `environment` from where they are, update them both, and
return the trajectories of this call in the same `NamedTuple` as [`simulate`](@ref),
without timestep 0: what `agent.save_history` selects of the states, observations, actions
and action distributions on each timestep, what `environment.save_history` selects of the
environment's under `environment`, and the sizes of the timesteps. What the two select is
also added to `agent.history` and `environment.history`, in each model's own names, together
with the sizes of the timesteps, unless a selection is empty. A history has timestep 0 from
where the model started, when its [`SaveHistory`](@ref) saves the initial values.

The two models must fit together, as for `simulate`, directly or through `observation_map`
and `action_map`. The agent's history holds the observations as its model received them,
and the environment's history the actions as its model did.

The simulation stops before `n_steps` timesteps if [`stopping_criterion`](@ref) of either
model is met, unless `check_stopping_criterion` is `false`. A model where it is met already
can't be simulated further, and throws an error until it is [`reset!`](@ref).

An instantiated model isn't type stable, as [`InstantiatedModel`](@ref) describes. Use
`simulate` where that matters, or if most calls are for a single timestep.

# Arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1. It is the
  most timesteps there can be when a model has a `stopping_criterion`.
- `timestep_sizes=1`: how much time each timestep covers. A value that isn't a vector, such
  as a single number, is the size of every timestep of this call. A vector has one size for
  each timestep of this call, so its length must be `n_steps`.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.
- `observation_map::Union{Nothing, NameMap}=nothing`: which observations of the environment
  the agent receives, and under which names, as for `simulate`.
- `action_map::Union{Nothing, NameMap}=nothing`: which actions of the agent the environment
  receives, and under which names, as for `simulate`.
- `check_stopping_criterion::Bool=true`: whether [`stopping_criterion`](@ref) of the two
  models is checked before each timestep, as for `simulate`.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RandomChoices <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RandomChoices, parameters)
    return (;)
end

function CognitiveModeling.apply_timestep_model(model::RandomChoices, context::ModelContext)
    return ModelOutput((; choice = Bernoulli(0.5)))
end

function CognitiveModeling.attribute_names(model::RandomChoices)
    return ModelAttributes(; parameters = (), states = (), inputs = :timestep, outputs = :choice)
end

struct TimestepCounter <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(model::TimestepCounter, parameters)
    return (; n_timesteps = 0)
end

function CognitiveModeling.apply_timestep_model(model::TimestepCounter, context::ModelContext)
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput((; timestep = Deterministic(n_timesteps)), (; n_timesteps))
end

function CognitiveModeling.attribute_names(model::TimestepCounter)
    return ModelAttributes(; parameters = (), states = :n_timesteps, inputs = :choice, outputs = :timestep)
end

agent = InstantiatedModel(RandomChoices(), (;))
environment = InstantiatedModel(TimestepCounter(), (;))

simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)

environment.state

# output

(n_timesteps = 5,)
```
"""
function simulate!(
    agent::InstantiatedModel,
    environment::InstantiatedModel;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    n_steps::Int,
    timestep_sizes=1,
    rng::AbstractRNG=default_rng(),
    observation_map::Union{Nothing,NameMap}=nothing,
    action_map::Union{Nothing,NameMap}=nothing,
    check_stopping_criterion::Bool=true,
)

    #Check that the two models fit together
    check_names_fit(agent.model, environment.model, observation_map, action_map)

    #Collect where the agent and the environment are, each in its own names
    previous_values = (;
        state=agent.state,
        action=agent.previous_output,
        environment=(; state=environment.state, observation=environment.previous_output),
    )

    #Simulate the timesteps, without timestep 0, which the histories have from where the agent and the environment started
    storage, last_values = simulate_all_timesteps(
        previous_values,
        who_starts,
        agent.model,
        environment.model,
        agent.parameters,
        environment.parameters,
        agent.save_history,
        environment.save_history,
        observation_map,
        action_map,
        rng,
        timestep_sizes,
        n_steps,
        Val(false),
        check_stopping_criterion,
    )

    #Update agent and environment
    agent.state = last_values.state
    agent.previous_output = last_values.action
    environment.state = last_values.environment.state
    environment.previous_output = last_values.environment.observation

    #The sizes of the timesteps that were simulated, since a model can stop the simulation early
    n_timesteps_simulated = length(storage.states)
    timestep_sizes_simulated = returned_timestep_sizes(
        timestep_sizes, n_timesteps_simulated, Val(false)
    )
    sizes_for_histories = each_timestep_size(
        timestep_sizes_simulated, n_timesteps_simulated
    )

    #Count the timesteps that were simulated in both
    agent.n_timesteps += n_timesteps_simulated
    environment.n_timesteps += n_timesteps_simulated

    #Add to the agent's history, in its own names
    if !saves_nothing(agent.save_history)
        append!(agent.history.states, storage.states)
        append!(agent.history.inputs, storage.observations)
        append!(agent.history.outputs, storage.actions)
        append!(agent.history.output_distributions, storage.action_distributions)
        append!(agent.history.timestep_sizes, sizes_for_histories)
    end

    #Add to the environment's history, in its own names
    if !saves_nothing(environment.save_history)
        append!(environment.history.states, storage.environment.states)
        append!(environment.history.inputs, storage.environment.actions)
        append!(environment.history.outputs, storage.environment.observations)
        append!(
            environment.history.output_distributions,
            storage.environment.observation_distributions,
        )
        append!(environment.history.timestep_sizes, sizes_for_histories)
    end

    #Return what was saved
    return saved_trajectories(storage, timestep_sizes_simulated)
end

## Simulating a model on a single input ##
#Its own function, so that it is compiled for the types in an instantiated model's untyped fields, which makes simulate! on a single input several times faster
"""
    simulate_timestep(model, input, previous_state, previous_output, parameters,
                      timestep_size, rng, save_history)

Simulate a timestep of `model` on `input`, without the other side of a simulation, and
return the output sampled from what it gave, its new state, and under `saved` what
`save_history` selects of its `states`, `inputs`, `outputs` and `output_distributions`, in a
`NamedTuple`. The input must have the names that `model` lists. [`simulate`](@ref) and
[`simulate!`](@ref) on a single input both use this.
"""
function simulate_timestep(
    model,
    input,
    previous_state,
    previous_output,
    parameters,
    timestep_size,
    rng,
    save_history,
)

    #Check that the input has the names the model lists
    check_input_names(model, input)

    #Update the model
    model_output = check_model_output(
        @inline(
            apply_timestep_model(
                model,
                ModelContext(
                    input,
                    previous_state,
                    previous_output,
                    parameters,
                    timestep_size,
                    rng,
                    UnknownWhoStarts(),
                ),
            )
        ),
        model,
    )

    #Sample the output
    output = sample_output(rng, model_output.output_distribution)

    return (;
        output,
        state=model_output.new_state,
        saved=(;
            states=save_state(model, model_output.new_state, state_selection(save_history)),
            inputs=select_names(input, input_selection(save_history)),
            outputs=select_names(output, output_selection(save_history)),
            output_distributions=select_names(
                model_output.output_distribution,
                output_distribution_selection(save_history),
            ),
        ),
    )
end

"""
    simulate(model, input; <keyword arguments>)

Run a single timestep of `model`, a cognitive model or an environment model, on `input`,
without the other side of a simulation, and return the sampled output and the new state of
the model.

Nothing is kept between calls, so pass the returned state and output on as `previous_state`
and `previous_output` to run the next timestep. [`simulate!`](@ref) on a single input does
the same for an [`InstantiatedModel`](@ref), which keeps them. This is type stable, which an
instantiated model isn't, so it is the faster of the two.

The output is in a `NamedTuple` with the names the model gave it, as in `output.choice`. The
input must be a `NamedTuple` with every input the model lists in [`attribute_names`](@ref),
or an `ArgumentError` is thrown. Parameters that aren't given take the values in
[`default_parameters`](@ref), if `model` provides defaults for them, and the names of the
parameters, the state and the previous output are checked as in a full simulation.

# Arguments
- `model::AbstractTimestepModel`: the model.
- `input::NamedTuple`: what the model receives on this timestep, such as `(; reward = 1.0)`
  for a cognitive model. An entry that isn't there is [`NoValue`](@ref).
- `parameters::NamedTuple`: the parameters of `model`, with its default values for those
  that aren't given, if it provides any.
- `timestep_size=1`: how much time this timestep covers.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.
- `previous_state=initialize_state(model, ...)`: the state of `model` before this timestep,
  by default its initial state from the full parameters. A state that `model` changes in
  place will be changed.
- `previous_output=output_before_first_timestep(model)`: the output before this timestep.
  By default every output is [`BeforeFirstTimestep`](@ref).

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_timestep_model(model::RescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice)
end

parameters = (; learning_rate = 0.5)

#The first timestep starts from the initial state
action, state = simulate(RescorlaWagner(), (; reward = 1.0); parameters)

#The next one continues from what the first returned
action, state = simulate(
    RescorlaWagner(),
    (; reward = 1.0);
    parameters,
    previous_state = state,
    previous_output = action,
)

state

# output

(expected_reward = 0.875,)
```
"""
function simulate(
    model::AbstractTimestepModel,
    input;
    parameters::NamedTuple,
    timestep_size=1,
    rng::AbstractRNG=default_rng(),
    previous_state=initialize_state(model, merge_default_parameters(model, parameters)),
    previous_output=output_before_first_timestep(model),
)

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check the names of the parameters, the state and the previous output against attribute_names, and that each parameter is within its domain
    check_match_attribute_names(model, parameters, previous_state, previous_output)
    check_parameter_domains(model, parameters)

    #Simulate the timestep, where nothing is selected for saving, since everything is returned
    timestep_values = simulate_timestep(
        model,
        input,
        previous_state,
        previous_output,
        parameters,
        timestep_size,
        rng,
        SaveHistory(),
    )

    return timestep_values.output, timestep_values.state
end

"""
    simulate!(instantiated_model::InstantiatedModel, input; timestep_size=1,
              rng=default_rng())

Run a single timestep of `instantiated_model` on `input`, update it, and return the sampled
output. What `instantiated_model.save_history` selects of the state, the input, the output
and the output distribution is added to `instantiated_model.history`, together with the size
of the timestep.

The output is in a `NamedTuple` with the names the model gave it, as in `output.choice`, as
for [`simulate`](@ref). The input must be a `NamedTuple` with every input the model lists in
[`attribute_names`](@ref), or an `ArgumentError` is thrown.

# Arguments
- `instantiated_model::InstantiatedModel`: the model to update.
- `input::NamedTuple`: what the model receives on this timestep, such as `(; reward = 1.0)`
  for a cognitive model. An entry that isn't there is [`NoValue`](@ref).
- `timestep_size=1`: how much time this timestep covers.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_timestep_model(model::RescorlaWagner, context::ModelContext)
    V₋₁ = context.previous_state.expected_reward
    Vₜ = V₋₁ + context.parameters.learning_rate * (context.input.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, inputs = :reward, outputs = :choice)
end

agent = InstantiatedModel(RescorlaWagner(), (; learning_rate = 0.5); save_history = SaveHistory(SaveEverything()))

simulate!(agent, (; reward = 1.0))
simulate!(agent, (; reward = 1.0))

agent.history.states

# output

3-element Vector{Any}:
 (expected_reward = 0.5,)
 (expected_reward = 0.75,)
 (expected_reward = 0.875,)
```
"""
function simulate!(
    instantiated_model::InstantiatedModel,
    input;
    timestep_size=1,
    rng::AbstractRNG=default_rng(),
)

    #Simulate the timestep
    timestep_values = simulate_timestep(
        instantiated_model.model,
        input,
        instantiated_model.state,
        instantiated_model.previous_output,
        instantiated_model.parameters,
        timestep_size,
        rng,
        instantiated_model.save_history,
    )

    #Update the instantiated model
    instantiated_model.state = timestep_values.state
    instantiated_model.previous_output = timestep_values.output
    instantiated_model.n_timesteps += 1

    #Add to its history
    if !saves_nothing(instantiated_model.save_history)
        push!(instantiated_model.history.states, timestep_values.saved.states)
        push!(instantiated_model.history.inputs, timestep_values.saved.inputs)
        push!(instantiated_model.history.outputs, timestep_values.saved.outputs)
        push!(
            instantiated_model.history.output_distributions,
            timestep_values.saved.output_distributions,
        )
        push!(instantiated_model.history.timestep_sizes, timestep_size)
    end

    return timestep_values.output
end
