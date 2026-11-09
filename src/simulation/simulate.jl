## Sampling outputs ##
"""
    sample_output(rng, output_distribution)

Sample an output from what a model returned, using `rand`. Each output in a `NamedTuple` is
sampled on its own, so a `NamedTuple` inside it gives a `NamedTuple` of samples, and an entry
that is [`NoAction`](@ref) or [`NoObservation`](@ref) is kept. A number, a string or a symbol
is a value and not something to sample from, and throws an `ArgumentError` which names
[`Deterministic`](@ref).
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
provides defaults for them.
"""
function merge_default_parameters(model::AbstractModel, parameters)
    return merge(default_parameters(model), parameters)
end

## Passing outputs between the model and the environment ##
"""
    map_names(x, name_map)

Return what the receiver gets of the outputs `x`: all of `x` without a map, and with a
[`NameMap`](@ref) the outputs it takes, under the names it gives them.
"""
map_names(x, ::Nothing) = x

#The names are in the type, so both the selection and the new names are known at compile time
function map_names(x::NamedTuple, ::NameMap{Tgiven,Treceived}) where {Tgiven,Treceived}
    return NamedTuple{Treceived}(values(NamedTuple{Tgiven}(x)))
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

## Saving outputs ##
"""
    initialize_storage(first_value, n_steps)
    initialize_storage(initial_value, first_value, n_steps)

Create the vector for saving one kind of output over `n_steps` timesteps, using the value of
the first timestep to set its type. With `initial_value`, what is saved from before the
first timestep, the vector starts with it, as timestep 0, and its type holds both.
"""
function initialize_storage(first_value, n_steps)

    #Create the vector and put the value in it
    storage = Vector{typeof(first_value)}(undef, n_steps)
    storage[1] = first_value

    return storage
end

function initialize_storage(initial_value, first_value, n_steps)

    #Create the vector with room for timestep 0, and put both values in it
    storage = Vector{Union{typeof(initial_value),typeof(first_value)}}(undef, n_steps + 1)
    storage[1] = initial_value
    storage[2] = first_value

    return storage
end

"""
    store_output!(storage, value, t)

Save `value` as element `t` of `storage`, and return the vector.

If the value doesn't fit the type of `storage`, the outputs so far are moved to a wider
vector which can hold both types, and that vector is returned instead.
"""
function store_output!(storage::Vector{T}, value, t) where {T}

    #If it has the type of the vector, which the compiler can usually see, store it
    if value isa T
        storage[t] = value
        return storage
    end

    #Otherwise, move the outputs so far to a wider vector which can hold both types
    wider_storage = Vector{Union{T,typeof(value)}}(undef, length(storage))
    copyto!(wider_storage, 1, storage, 1, t - 1)
    wider_storage[t] = value

    return wider_storage
end

"""
    push_output!(storage, value)

Add `value` at the end of `storage`, and return the vector. As for [`store_output!`](@ref), a
value that doesn't fit the type of `storage` moves the outputs so far to a wider vector,
which is returned instead. Starting from `Vector{Union{}}()` lets the first value set the
type.
"""
function push_output!(storage::Vector{T}, value) where {T}

    #If it has the type of the vector, add it
    if value isa T
        push!(storage, value)
        return storage
    end

    #Otherwise, move the outputs so far to a wider vector which can hold both types
    wider_storage = Vector{Union{T,typeof(value)}}(undef, length(storage) + 1)
    copyto!(wider_storage, 1, storage, 1, length(storage))
    wider_storage[end] = value

    return wider_storage
end

## Running timesteps ##
"""
    get_timestep_size(timestep_sizes, t)

Return the size of timestep `t`. A vector has one size for each timestep, and anything else
is the size of all of them.
"""
function get_timestep_size(timestep_sizes::AbstractVector, t)
    return timestep_sizes[t]
end

function get_timestep_size(timestep_size, t)
    return timestep_size
end

"""
    each_timestep_size(timestep_sizes, n_steps)

Return the size of each of `n_steps` timesteps as a vector, for a history. A vector of sizes
is returned as it is, and anything else is repeated for every timestep.
"""
function each_timestep_size(timestep_sizes::AbstractVector, n_steps)
    return timestep_sizes
end

function each_timestep_size(timestep_size, n_steps)
    return fill(timestep_size, n_steps)
end

"""
    returned_timestep_sizes(timestep_sizes, n_timesteps, save_initial_values::Val)

Return the sizes of the `n_timesteps` timesteps that ran, as a simulation returns them. A
vector of sizes is cut to those timesteps, since a model can end a simulation early, and
starts with [`BeforeFirstTimestep`](@ref) for timestep 0 when `save_initial_values` is
`Val(true)`, so that it has an element for each element of what is saved. Anything else is
returned as it is, since it is the size of every timestep.
"""
function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{false})
    return timestep_sizes[1:n_timesteps]
end

function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{true})

    #Create the vector with room for timestep 0, and put the marker and the sizes in it
    sizes = Vector{Union{BeforeFirstTimestep,eltype(timestep_sizes)}}(
        undef, n_timesteps + 1
    )
    sizes[1] = BeforeFirstTimestep()
    copyto!(sizes, 2, timestep_sizes, 1, n_timesteps)

    return sizes
end

function returned_timestep_sizes(timestep_size, n_timesteps, save_initial_values)
    return timestep_size
end

"""
    collect_timestep_outputs(environment_output, cognitive_output,
                             observation_from_environment, observation, action,
                             action_to_environment, cognitive_model, environment,
                             cognitive_save_history, environment_save_history)

Collect the outputs of a timestep in a `NamedTuple`, each both as it is, which the next
timestep starts from, and as it is saved, which `cognitive_save_history` and
`environment_save_history` select.

`observation_from_environment` is the observation sampled from what the environment gave,
and `observation` is the same observation as the cognitive model receives it.
`action_to_environment` is `action` as the environment receives it. Each pair differs when a
[`NameMap`](@ref) selects or renames them. The cognitive model's history holds the
observation as the cognitive model receives it, and the environment's the action as the
environment does.

This is its own function so that if an output can have two types, like an observation with
an entry that is sometimes [`NoObservation`](@ref), the compiler can split the call and get a
concrete type for each.
"""
function collect_timestep_outputs(
    environment_output,
    cognitive_output,
    observation_from_environment,
    observation,
    action,
    action_to_environment,
    cognitive_model,
    environment,
    cognitive_save_history,
    environment_save_history,
)
    return (;
        observation_from_environment,
        action,
        cognitive_state=cognitive_output.new_state,
        environment_state=environment_output.new_state,
        saved_state=save_state(
            cognitive_model,
            cognitive_output.new_state,
            state_selection(cognitive_save_history),
        ),
        saved_observation=select_names(
            observation, input_selection(cognitive_save_history)
        ),
        saved_action=select_names(action, output_selection(cognitive_save_history)),
        saved_action_distribution=select_names(
            cognitive_output.output_distribution,
            output_distribution_selection(cognitive_save_history),
        ),
        saved_environment_state=save_state(
            environment,
            environment_output.new_state,
            state_selection(environment_save_history),
        ),
        saved_environment_observation=select_names(
            observation_from_environment, output_selection(environment_save_history)
        ),
        saved_environment_action=select_names(
            action_to_environment, input_selection(environment_save_history)
        ),
        saved_environment_observation_distribution=select_names(
            environment_output.output_distribution,
            output_distribution_selection(environment_save_history),
        ),
    )
end

"""
    collect_initial_outputs(save_initial_values::Val, previous_outputs, cognitive_model,
                            environment, observation_map, action_map,
                            cognitive_save_history, environment_save_history)

Return what is saved from before the first timestep, as timestep 0, in a `NamedTuple` with
the names of the saved outputs of [`collect_timestep_outputs`](@ref), or `nothing` when
`save_initial_values` is `Val(false)`.

The observation and the action are the previous ones, as each side receives them, the
output distributions are [`BeforeFirstTimestep`](@ref) for every output, since none was
given yet, and the states are the states before the first timestep. They must be saved
before the first timestep runs, since it can change a state in place.
"""
function collect_initial_outputs(
    ::Val{false},
    previous_outputs,
    cognitive_model,
    environment,
    observation_map,
    action_map,
    cognitive_save_history,
    environment_save_history,
)
    return nothing
end

function collect_initial_outputs(
    ::Val{true},
    previous_outputs,
    cognitive_model,
    environment,
    observation_map,
    action_map,
    cognitive_save_history,
    environment_save_history,
)
    return (;
        saved_state=save_state(
            cognitive_model,
            previous_outputs.cognitive_state,
            state_selection(cognitive_save_history),
        ),
        saved_observation=select_names(
            map_names(previous_outputs.observation_from_environment, observation_map),
            input_selection(cognitive_save_history),
        ),
        saved_action=select_names(
            previous_outputs.action, output_selection(cognitive_save_history)
        ),
        saved_action_distribution=select_names(
            initial_output(cognitive_model),
            output_distribution_selection(cognitive_save_history),
        ),
        saved_environment_state=save_state(
            environment,
            previous_outputs.environment_state,
            state_selection(environment_save_history),
        ),
        saved_environment_observation=select_names(
            previous_outputs.observation_from_environment,
            output_selection(environment_save_history),
        ),
        saved_environment_action=select_names(
            map_names(previous_outputs.action, action_map),
            input_selection(environment_save_history),
        ),
        saved_environment_observation_distribution=select_names(
            initial_output(environment),
            output_distribution_selection(environment_save_history),
        ),
    )
end

"""
    initialize_all_storage(initial_outputs, outputs, n_steps)

Create the `NamedTuple` of vectors for saving the outputs of `n_steps` timesteps, typed by
`outputs`, the outputs of the first timestep. When `initial_outputs` isn't `nothing`, each
vector starts with what it has for timestep 0, from [`collect_initial_outputs`](@ref).
"""
function initialize_all_storage(::Nothing, outputs, n_steps)
    return (;
        states=initialize_storage(outputs.saved_state, n_steps),
        observations=initialize_storage(outputs.saved_observation, n_steps),
        actions=initialize_storage(outputs.saved_action, n_steps),
        action_distributions=initialize_storage(outputs.saved_action_distribution, n_steps),
        environment_states=initialize_storage(outputs.saved_environment_state, n_steps),
        environment_observations=initialize_storage(
            outputs.saved_environment_observation, n_steps
        ),
        environment_actions=initialize_storage(outputs.saved_environment_action, n_steps),
        environment_observation_distributions=initialize_storage(
            outputs.saved_environment_observation_distribution, n_steps
        ),
    )
end

function initialize_all_storage(initial_outputs::NamedTuple, outputs, n_steps)
    return (;
        states=initialize_storage(
            initial_outputs.saved_state, outputs.saved_state, n_steps
        ),
        observations=initialize_storage(
            initial_outputs.saved_observation, outputs.saved_observation, n_steps
        ),
        actions=initialize_storage(
            initial_outputs.saved_action, outputs.saved_action, n_steps
        ),
        action_distributions=initialize_storage(
            initial_outputs.saved_action_distribution,
            outputs.saved_action_distribution,
            n_steps,
        ),
        environment_states=initialize_storage(
            initial_outputs.saved_environment_state,
            outputs.saved_environment_state,
            n_steps,
        ),
        environment_observations=initialize_storage(
            initial_outputs.saved_environment_observation,
            outputs.saved_environment_observation,
            n_steps,
        ),
        environment_actions=initialize_storage(
            initial_outputs.saved_environment_action,
            outputs.saved_environment_action,
            n_steps,
        ),
        environment_observation_distributions=initialize_storage(
            initial_outputs.saved_environment_observation_distribution,
            outputs.saved_environment_observation_distribution,
            n_steps,
        ),
    )
end

"""
    store_all_outputs!(storage, outputs, t)

Save the saved outputs of timestep `t` in `outputs`, from [`collect_timestep_outputs`](@ref),
as element `t` of each vector in `storage`, with [`store_output!`](@ref), and return the
`NamedTuple` of vectors, in which a vector is wider when a value didn't fit it.
"""
function store_all_outputs!(storage, outputs, t)
    return (;
        states=store_output!(storage.states, outputs.saved_state, t),
        observations=store_output!(storage.observations, outputs.saved_observation, t),
        actions=store_output!(storage.actions, outputs.saved_action, t),
        action_distributions=store_output!(
            storage.action_distributions, outputs.saved_action_distribution, t
        ),
        environment_states=store_output!(
            storage.environment_states, outputs.saved_environment_state, t
        ),
        environment_observations=store_output!(
            storage.environment_observations, outputs.saved_environment_observation, t
        ),
        environment_actions=store_output!(
            storage.environment_actions, outputs.saved_environment_action, t
        ),
        environment_observation_distributions=store_output!(
            storage.environment_observation_distributions,
            outputs.saved_environment_observation_distribution,
            t,
        ),
    )
end

"""
    saved_values(storage, timestep_sizes)

Return the saved values of a simulation as [`simulate`](@ref) returns them, from the
`NamedTuple` of vectors in `storage`: the cognitive model's under `states`, `observations`,
`actions` and `action_distributions`, the environment's under `environment`, and
`timestep_sizes` as given.
"""
function saved_values(storage, timestep_sizes)
    return (;
        storage.states,
        storage.observations,
        storage.actions,
        storage.action_distributions,
        environment=(;
            states=storage.environment_states,
            observations=storage.environment_observations,
            actions=storage.environment_actions,
            observation_distributions=storage.environment_observation_distributions,
        ),
        timestep_sizes,
    )
end

"""
    check_model_output(output, model)

Return `output` if it is the `ModelOutput` that an `apply_model` method of `model` must
return, and throw an error otherwise.
"""
function check_model_output(output::ModelOutput, ::AbstractModel)
    return output
end

#Untyped, so that there is always a method, whatever the model is inferred as
function check_model_output(output, model)
    return error(
        "The apply_model method of $(describe_model(model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).",
    )
end

"""
    timestep(::EnvironmentStarts, cognitive_model, environment, cognitive_parameters,
             environment_parameters, cognitive_save_history, environment_save_history,
             observation_map, action_map, rng, timestep_size, previous)

Run a timestep where the environment moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(
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
    previous_action_to_environment = map_names(previous.action, action_map)

    #Update the environment
    environment_output = check_model_output(
        @inline(
            apply_model(
                environment,
                ModelContext(
                    previous_action_to_environment,
                    previous.environment_state,
                    previous.observation_from_environment,
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
    observation_from_environment = sample_output(
        rng, environment_output.output_distribution
    )

    #The cognitive model receives the observation as observation_map passes it on
    observation = map_names(observation_from_environment, observation_map)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(
        @inline(
            apply_model(
                cognitive_model,
                ModelContext(
                    observation,
                    previous.cognitive_state,
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
    action_to_environment = map_names(action, action_map)

    return collect_timestep_outputs(
        environment_output,
        cognitive_output,
        observation_from_environment,
        observation,
        action,
        action_to_environment,
        cognitive_model,
        environment,
        cognitive_save_history,
        environment_save_history,
    )
end

"""
    timestep(::AgentStarts, cognitive_model, environment, cognitive_parameters,
             environment_parameters, cognitive_save_history, environment_save_history,
             observation_map, action_map, rng, timestep_size, previous)

Run a timestep where the agent moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(
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
    previous_observation = map_names(previous.observation_from_environment, observation_map)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(
        @inline(
            apply_model(
                cognitive_model,
                ModelContext(
                    previous_observation,
                    previous.cognitive_state,
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
    action_to_environment = map_names(action, action_map)

    #Update the environment
    environment_output = check_model_output(
        @inline(
            apply_model(
                environment,
                ModelContext(
                    action_to_environment,
                    previous.environment_state,
                    previous.observation_from_environment,
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
    observation_from_environment = sample_output(
        rng, environment_output.output_distribution
    )

    #The cognitive model's history has the observation as observation_map passes it on
    observation = map_names(observation_from_environment, observation_map)

    return collect_timestep_outputs(
        environment_output,
        cognitive_output,
        observation_from_environment,
        observation,
        action,
        action_to_environment,
        cognitive_model,
        environment,
        cognitive_save_history,
        environment_save_history,
    )
end

"""
    has_ended(cognitive_model, environment, cognitive_parameters, environment_parameters,
              outputs)

Return whether [`stopping_criterion`](@ref) of either model is met, with the states in
`outputs`, the outputs of the timestep before. Used by the simulation loop before each
timestep, when the criteria are checked.
"""
function has_ended(
    cognitive_model, environment, cognitive_parameters, environment_parameters, outputs
)
    return stopping_criterion(
        environment, outputs.environment_state, environment_parameters
    ) || stopping_criterion(
        cognitive_model, outputs.cognitive_state, cognitive_parameters
    )
end

"""
    check_not_ended(model, state, parameters)

Throw an informative error if [`stopping_criterion`](@ref) of `model` is already met with
`state`, before the first timestep of a simulation, since no timestep can then run.
"""
function check_not_ended(model, state, parameters)
    if stopping_criterion(model, state, parameters)
        throw(
            ArgumentError(
                "The stopping criterion of $(describe_model(model)) is already met before the first timestep, so no timestep can run. An InstantiatedModel which has ended starts again after reset!.",
            ),
        )
    end
    return nothing
end

"""
    run_remaining_timesteps(storage, outputs, who_starts, cognitive_model, environment,
                            cognitive_parameters, environment_parameters,
                            cognitive_save_history, environment_save_history,
                            observation_map, action_map, rng, timestep_sizes, n_steps,
                            offset, check_stopping_criterion)

Run the timesteps after the first, and return the `NamedTuple` of vectors with the saved
outputs of the timesteps that ran, followed by the outputs of the last one. When
`check_stopping_criterion` is `true`, the timesteps end before `n_steps` if
[`stopping_criterion`](@ref) of either model is met. `offset` is 1 when the vectors start
with timestep 0, and 0 otherwise, so that timestep `t` is element `t + offset`.

This is its own function so that it's compiled for the types of the first outputs.
"""
function run_remaining_timesteps(
    storage,
    outputs,
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

        #Stop if either model has ended, and keep only the timesteps that ran
        if check_stopping_criterion && has_ended(
            cognitive_model,
            environment,
            cognitive_parameters,
            environment_parameters,
            outputs,
        )
            foreach(vector -> resize!(vector, t - 1 + offset), storage)
            return storage, outputs
        end

        #Run the timestep, starting from the outputs of the one before
        outputs = timestep(
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
            outputs,
        )

        #Store trial outputs
        storage = store_all_outputs!(storage, outputs, t + offset)
    end

    #Return stored outputs, and the last ones for continuing from
    return storage, outputs
end

"""
    run_timesteps(previous_outputs, who_starts, cognitive_model, environment,
                  cognitive_parameters, environment_parameters, cognitive_save_history,
                  environment_save_history, observation_map, action_map, rng,
                  timestep_sizes, n_steps, save_initial_values::Val,
                  check_stopping_criterion)

Run up to `n_steps` timesteps starting from `previous_outputs`, and return a `NamedTuple` of
vectors with the saved outputs of the timesteps that ran, followed by the outputs of the
last one. When `check_stopping_criterion` is `true`, the timesteps end before `n_steps` if
[`stopping_criterion`](@ref) of either model is met, and a model where it is met already
can't run any. When `save_initial_values` is `Val(true)`, each vector starts with what is
saved from before the first timestep, as timestep 0.

The vectors are `states`, `observations`, `actions` and `action_distributions` with what
`cognitive_save_history` selects of the cognitive model's, and `environment_states`,
`environment_observations`, `environment_actions` and
`environment_observation_distributions` with what `environment_save_history` selects of the
environment's, as [`saved_values`](@ref) arranges them for what is returned.

[`simulate!`](@ref) continues from the last outputs instead of from what was saved, because
[`prepare_state_for_saving`](@ref) can return anything, and because the next simulation can
change a state in place.
"""
function run_timesteps(
    previous_outputs,
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

    #A model which has ended already can't run the first timestep
    if check_stopping_criterion
        check_not_ended(
            environment, previous_outputs.environment_state, environment_parameters
        )
        check_not_ended(
            cognitive_model, previous_outputs.cognitive_state, cognitive_parameters
        )
    end

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_outputs = collect_initial_outputs(
        save_initial_values,
        previous_outputs,
        cognitive_model,
        environment,
        observation_map,
        action_map,
        cognitive_save_history,
        environment_save_history,
    )

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs = timestep(
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
        previous_outputs,
    )

    #Initialize storage for the saved outputs, which starts with timestep 0 when it is kept
    storage = initialize_all_storage(initial_outputs, outputs, n_steps)
    offset = initial_outputs === nothing ? 0 : 1

    #Run the remaining timesteps
    return run_remaining_timesteps(
        storage,
        outputs,
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

"""
    simulate(cognitive_model, environment; <keyword arguments>)

Simulate `cognitive_model` interacting with `environment`, and return what was saved in a
`NamedTuple`: `states`, `observations`, `actions` and `action_distributions` of the
cognitive model, `environment` with the `states`, `observations`, `actions` and
`observation_distributions` of the environment, and `timestep_sizes`. Each is a vector with
one element for timestep 0, before the first timestep, and one for each timestep after it.

Each element is what happened on that timestep, and the states are those after it. The
observations are those the cognitive model received, and the actions those it gave, and
under `environment` the observations are those the environment gave and the actions those it
received, which differ from the cognitive model's when a [`NameMap`](@ref) selects or renames
them. The action distributions are what the cognitive model gave its actions to be sampled
from, and the observation distributions what the environment gave its observations to be
sampled from. The element for timestep 0 has the previous observation and action, which are
[`BeforeFirstTimestep`](@ref) unless others are given, `BeforeFirstTimestep()` for every
distribution, and the states before the first timestep. Since Julia counts from 1, element
`t + 1` is timestep `t`.

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

The simulation ends before `n_steps` timesteps if [`stopping_criterion`](@ref) of either
model is met, unless `check_stopping_criterion` is `false`, and everything returned then has
the timesteps that ran.

# Arguments
- `cognitive_model::AbstractModel`: the cognitive model, usually an
  [`AbstractCognitiveModel`](@ref).
- `environment::AbstractModel`: the environment model, usually an
  [`AbstractEnvironmentModel`](@ref). A cognitive model can serve as the environment of
  another when its outputs are what the other lists as its inputs, and the other way round.
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
- `previous_cognitive_state=initialize_state(cognitive_model, ...)`: the state of
  `cognitive_model` before the first timestep, by default its initial state from the full
  parameters. A state that `cognitive_model` changes in place will be changed.
- `previous_action=initial_output(cognitive_model)`: the action before the first timestep.
  By default every action is [`BeforeFirstTimestep`](@ref), with the names from
  [`attribute_names`](@ref).
- `previous_environment_state=initialize_state(environment, ...)`: the state of
  `environment` before the first timestep, by default its initial state from the full
  parameters.
- `previous_observation=initial_output(environment)`: the observation before the first
  timestep. By default every observation is `BeforeFirstTimestep()`, with the names from
  `attribute_names` of `environment`.
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

See also [`simulate!`](@ref), which continues from where the last simulation ended.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli, BernoulliLogit

#A cognitive model which learns the expected reward
struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, context::ModelContext)
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

function CognitiveModeling.apply_model(environment::BernoulliRewards, context::ModelContext)
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
[`NoObservation`](@ref), the vector for that output holds a `Union` of the types, and the
simulation is a little slower. The same goes for timestep 0 when it has other types than the
timesteps after it, which the observations and actions usually do, since they are
`BeforeFirstTimestep()` there.

Nothing is copied when outputs are saved. A model or environment which changes an object in
its state in place, instead of returning a new one, must add a method of
[`prepare_state_for_saving`](@ref), or all the saved states show the final values, with no
error. Observations and actions have no such function, so they must be new
objects on each timestep.
"""
function simulate(
    cognitive_model::AbstractModel,
    environment::AbstractModel;
    who_starts::Union{AgentStarts,EnvironmentStarts},
    n_steps::Int,
    timestep_sizes=1,
    cognitive_parameters::NamedTuple,
    environment_parameters::NamedTuple,
    rng::AbstractRNG=default_rng(),
    previous_cognitive_state=initialize_state(
        cognitive_model, merge_default_parameters(cognitive_model, cognitive_parameters)
    ),
    previous_action=initial_output(cognitive_model),
    previous_environment_state=initialize_state(
        environment, merge_default_parameters(environment, environment_parameters)
    ),
    previous_observation=initial_output(environment),
    cognitive_save_history::SaveHistory=SaveHistory(),
    environment_save_history::SaveHistory=SaveHistory(),
    observation_map::Union{Nothing,NameMap}=nothing,
    action_map::Union{Nothing,NameMap}=nothing,
    check_stopping_criterion::Bool=true,
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

    #Collect what was left from before the first timestep
    previous_outputs = (;
        observation_from_environment=previous_observation,
        action=previous_action,
        cognitive_state=previous_cognitive_state,
        environment_state=previous_environment_state,
    )

    #Run the timesteps, starting with timestep 0 when the cognitive model's save history saves the initial values
    save_initial_values = Val(saves_initial_values(cognitive_save_history))
    storage, _ = run_timesteps(
        previous_outputs,
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

    #The sizes of the timesteps that ran, since a model can end the simulation early
    n_timesteps_that_ran =
        length(storage.states) - (saves_initial_values(cognitive_save_history) ? 1 : 0)
    timestep_sizes_that_ran = returned_timestep_sizes(
        timestep_sizes, n_timesteps_that_ran, save_initial_values
    )

    #Return stored outputs
    return saved_values(storage, timestep_sizes_that_ran)
end

"""
    simulate!(agent::InstantiatedModel, environment::InstantiatedModel; <keyword arguments>)

Simulate `agent` interacting with `environment` from where they are, update them both, and
return the same `NamedTuple` as [`simulate`](@ref), without timestep 0: what
`agent.save_history` selects of the states, observations, actions and action distributions
on each timestep of this call, what `environment.save_history` selects of the environment's
under `environment`, and the sizes of the timesteps. What the two select is also added to
`agent.history` and `environment.history`, in each model's own names, together with the
sizes of the timesteps, unless a selection is empty. A history has timestep 0 from where the
model started, when its [`SaveHistory`](@ref) saves the initial values.

The two models must fit together, as for `simulate`, directly or through `observation_map`
and `action_map`. The agent's history holds the observations as its model received them,
and the environment's history the actions as its model did.

The simulation ends before `n_steps` timesteps if [`stopping_criterion`](@ref) of either
model is met, unless `check_stopping_criterion` is `false`. A model where it is met already
can't be simulated further, and throws an error until it is [`reset!`](@ref).

Reading the untyped fields of the instantiated models costs a dynamic dispatch on each call,
which makes the return type unknown to the compiler. The loop inside is still fully typed.
Use `simulate` if the return type matters, or if most calls are for a single timestep.

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

function CognitiveModeling.apply_model(model::RandomChoices, context::ModelContext)
    return ModelOutput((; choice = Bernoulli(0.5)))
end

function CognitiveModeling.attribute_names(model::RandomChoices)
    return ModelAttributes(; parameters = (), states = (), inputs = :timestep, outputs = :choice)
end

struct TimestepCounter <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(model::TimestepCounter, parameters)
    return (; n_timesteps = 0)
end

function CognitiveModeling.apply_model(model::TimestepCounter, context::ModelContext)
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

    #Collect where the agent and the environment are
    previous_outputs = (;
        observation_from_environment=environment.previous_output,
        action=agent.previous_output,
        cognitive_state=agent.state,
        environment_state=environment.state,
    )

    #Run the timesteps, without timestep 0, which the histories have from where the agent and the environment started
    storage, last_outputs = run_timesteps(
        previous_outputs,
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
    agent.state = last_outputs.cognitive_state
    agent.previous_output = last_outputs.action
    environment.state = last_outputs.environment_state
    environment.previous_output = last_outputs.observation_from_environment

    #The sizes of the timesteps that ran, since a model can end the simulation early
    n_timesteps_that_ran = length(storage.states)
    timestep_sizes_that_ran = returned_timestep_sizes(
        timestep_sizes, n_timesteps_that_ran, Val(false)
    )
    sizes_for_histories = each_timestep_size(timestep_sizes_that_ran, n_timesteps_that_ran)

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
        append!(environment.history.states, storage.environment_states)
        append!(environment.history.inputs, storage.environment_actions)
        append!(environment.history.outputs, storage.environment_observations)
        append!(
            environment.history.output_distributions,
            storage.environment_observation_distributions,
        )
        append!(environment.history.timestep_sizes, sizes_for_histories)
    end

    #Return stored outputs
    return saved_values(storage, timestep_sizes_that_ran)
end

"""
    timestep(model, input, previous_state, previous_output, parameters, timestep_size, rng,
             save_history)

Run a timestep of `model` on `input`, without the other side of a simulation, and return the
output sampled from what it gave, its new state, and what `save_history` selects of the
state, the input, the output and the output distribution, in a `NamedTuple`. The input must
have the names that `model` lists.

[`simulate`](@ref) and [`simulate!`](@ref) on a single input both run this. It is its own
function so that it's compiled for the types in an instantiated model's untyped fields,
which makes `simulate!` on a single input several times faster, and makes the check of the
input's names cost nothing.
"""
function timestep(
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

    #Run timestep of the model
    model_output = check_model_output(
        @inline(
            apply_model(
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
        saved_state=save_state(
            model, model_output.new_state, state_selection(save_history)
        ),
        saved_input=select_names(input, input_selection(save_history)),
        saved_output=select_names(output, output_selection(save_history)),
        saved_output_distribution=select_names(
            model_output.output_distribution, output_distribution_selection(save_history)
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
the same for an [`InstantiatedModel`](@ref), which keeps them. The compiler knows the types
here, which it doesn't for an instantiated model, so this is the faster of the two.

The output is in a `NamedTuple` with the names the model gave it, as in `output.choice`. The
input must be a `NamedTuple` with every input the model lists in [`attribute_names`](@ref),
or an `ArgumentError` is thrown. Parameters that aren't given take the values in
[`default_parameters`](@ref), if `model` provides defaults for them, and the names of the
parameters, the state and the previous output are checked as in a full simulation.

# Arguments
- `model::AbstractModel`: the model.
- `input::NamedTuple`: what the model receives on this timestep, such as `(; reward = 1.0)`
  for a cognitive model. An entry that isn't there is [`NoObservation`](@ref) or
  [`NoAction`](@ref).
- `parameters::NamedTuple`: the parameters of `model`, with its default values for those
  that aren't given, if it provides any.
- `timestep_size=1`: how much time this timestep covers.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.
- `previous_state=initialize_state(model, ...)`: the state of `model` before this timestep,
  by default its initial state from the full parameters. A state that `model` changes in
  place will be changed.
- `previous_output=initial_output(model)`: the output before this timestep. By default every
  output is [`BeforeFirstTimestep`](@ref).

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, context::ModelContext)
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
    model::AbstractModel,
    input;
    parameters::NamedTuple,
    timestep_size=1,
    rng::AbstractRNG=default_rng(),
    previous_state=initialize_state(model, merge_default_parameters(model, parameters)),
    previous_output=initial_output(model),
)

    #Fill in the default parameters
    parameters = merge_default_parameters(model, parameters)

    #Check the names of the parameters, the state and the previous output against attribute_names
    check_match_attribute_names(model, parameters, previous_state, previous_output)

    #Run the timestep, where nothing is selected for saving, since everything is returned
    outputs = timestep(
        model,
        input,
        previous_state,
        previous_output,
        parameters,
        timestep_size,
        rng,
        SaveHistory(),
    )

    return outputs.output, outputs.state
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
  for a cognitive model. An entry that isn't there is [`NoObservation`](@ref) or
  [`NoAction`](@ref).
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

function CognitiveModeling.apply_model(model::RescorlaWagner, context::ModelContext)
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

    #Run the timestep
    outputs = timestep(
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
    instantiated_model.state = outputs.state
    instantiated_model.previous_output = outputs.output

    #Add to its history
    if !saves_nothing(instantiated_model.save_history)
        push!(instantiated_model.history.states, outputs.saved_state)
        push!(instantiated_model.history.inputs, outputs.saved_input)
        push!(instantiated_model.history.outputs, outputs.saved_output)
        push!(
            instantiated_model.history.output_distributions,
            outputs.saved_output_distribution,
        )
        push!(instantiated_model.history.timestep_sizes, timestep_size)
    end

    return outputs.output
end
