## Sampling actions and observations ##
"""
    sample_action(rng, action_distribution)

Sample an action from what a model returned, using `rand`. Each action in a `NamedTuple` is
sampled on its own, so a `NamedTuple` inside it gives a `NamedTuple` of samples, and an
entry that is [`NoAction`](@ref) is kept. A number, a string or a symbol is a value and not
something to sample from, and throws an `ArgumentError` which names [`Deterministic`](@ref).
"""
function sample_action(rng, action_distribution)
    return rand(rng, action_distribution)
end

#Samples each named action on its own
function sample_action(rng, action_distributions::NamedTuple)
    return map(action_distribution -> sample_action(rng, action_distribution), action_distributions)
end

#An action that isn't taken has nothing to sample
function sample_action(rng, ::NoAction)
    return NoAction()
end

#A number, a string or a symbol is a value, which rand would read as something else or not at all
function sample_action(rng, value::Union{Number, AbstractString, Symbol})
    throw(ArgumentError("A cognitive model gave $(repr(value)), of type $(typeof(value)), as what to sample an action from, but it is a value. An action that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))."))
end

"""
    sample_observation(rng, observation_distribution)

Sample an observation from what an environment model returned, using `rand`, as
[`sample_action`](@ref) does for an action. Each observation in a `NamedTuple` is sampled on
its own, and an entry that is [`NoObservation`](@ref) is kept. A number, a string or a
symbol throws an `ArgumentError` here too.
"""
function sample_observation(rng, observation_distribution)
    return rand(rng, observation_distribution)
end

#Samples each named observation on its own
function sample_observation(rng, observation_distributions::NamedTuple)
    return map(observation_distribution -> sample_observation(rng, observation_distribution), observation_distributions)
end

#An observation that isn't made has nothing to sample
function sample_observation(rng, ::NoObservation)
    return NoObservation()
end

#A number, a string or a symbol is a value, which rand would read as something else or not at all
function sample_observation(rng, value::Union{Number, AbstractString, Symbol})
    throw(ArgumentError("An environment model gave $(repr(value)), of type $(typeof(value)), as what to sample an observation from, but it is a value. An observation that isn't random is given in a Deterministic, as in Deterministic($(repr(value)))."))
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
function map_names(x::NamedTuple, ::NameMap{Tgiven, Treceived}) where {Tgiven, Treceived}
    return NamedTuple{Treceived}(values(NamedTuple{Tgiven}(x)))
end

#With a map, it must take observations the environment gives, and give exactly those the cognitive model lists
function check_observations_fit(cognitive_model, environment, cognitive_model_names, environment_names, ::NameMap{Tgiven, Treceived}) where {Tgiven, Treceived}
    if !has_every_name(Tgiven, environment_names)
        throw(ArgumentError("The observation_map takes the observations $Tgiven, but an environment model of type $(typeof(environment)) gives $environment_names. A map can only take observations the environment gives."))
    end
    if !same_names(Treceived, cognitive_model_names)
        throw(ArgumentError("The observation_map gives the observations $Treceived, but a cognitive model of type $(typeof(cognitive_model)) lists $cognitive_model_names. A map must give exactly the observations the cognitive model lists."))
    end
    return nothing
end

#With a map, it must take actions the cognitive model gives, and give exactly those the environment lists
function check_actions_fit(cognitive_model, environment, cognitive_model_names, environment_names, ::NameMap{Tgiven, Treceived}) where {Tgiven, Treceived}
    if !has_every_name(Tgiven, cognitive_model_names)
        throw(ArgumentError("The action_map takes the actions $Tgiven, but a cognitive model of type $(typeof(cognitive_model)) gives $cognitive_model_names. A map can only take actions the cognitive model gives."))
    end
    if !same_names(Treceived, environment_names)
        throw(ArgumentError("The action_map gives the actions $Treceived, but an environment model of type $(typeof(environment)) lists $environment_names. A map must give exactly the actions the environment lists."))
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
    storage = Vector{Union{typeof(initial_value), typeof(first_value)}}(undef, n_steps + 1)
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
    wider_storage = Vector{Union{T, typeof(value)}}(undef, length(storage))
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
    wider_storage = Vector{Union{T, typeof(value)}}(undef, length(storage) + 1)
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
vector of sizes is cut to those timesteps, since an environment can end a simulation early,
and starts with [`BeforeFirstTimestep`](@ref) for timestep 0 when `save_initial_values` is
`Val(true)`, so that it has an element for each element of what is saved. Anything else is
returned as it is, since it is the size of every timestep.
"""
function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{false})
    return timestep_sizes[1:n_timesteps]
end

function returned_timestep_sizes(timestep_sizes::AbstractVector, n_timesteps, ::Val{true})

    #Create the vector with room for timestep 0, and put the marker and the sizes in it
    sizes = Vector{Union{BeforeFirstTimestep, eltype(timestep_sizes)}}(undef, n_timesteps + 1)
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
                             save_history, save_environment_history)

Collect the outputs of a timestep in a `NamedTuple`, each both as it is, which the next
timestep starts from, and as it is saved, which `save_history` and
`save_environment_history` select.

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
function collect_timestep_outputs(environment_output, cognitive_output, observation_from_environment, observation, action, action_to_environment, cognitive_model, environment, save_history, save_environment_history)
    return (;
        observation_from_environment,
        action,
        cognitive_state = cognitive_output.new_state,
        environment_state = environment_output.new_state,
        saved_observation = select_names(observation, observation_selection(save_history)),
        saved_action = select_names(action, action_selection(save_history)),
        saved_cognitive_state = save_state(cognitive_model, cognitive_output.new_state, state_selection(save_history)),
        saved_environment_state = save_state(environment, environment_output.new_state, state_selection(save_environment_history)),
        saved_observation_from_environment = select_names(observation_from_environment, observation_selection(save_environment_history)),
        saved_action_to_environment = select_names(action_to_environment, action_selection(save_environment_history)),
    )
end

"""
    collect_initial_outputs(save_initial_values::Val, previous_outputs, cognitive_model,
                            environment, observation_map, action_map, save_history,
                            save_environment_history)

Return what is saved from before the first timestep, as timestep 0, in a `NamedTuple` with
the names of the saved outputs of [`collect_timestep_outputs`](@ref), or `nothing` when
`save_initial_values` is `Val(false)`.

The observation and the action are the previous ones, as each side receives them, and the
states are the states before the first timestep. They must be saved before the first
timestep runs, since it can change a state in place.
"""
function collect_initial_outputs(::Val{false}, previous_outputs, cognitive_model, environment, observation_map, action_map, save_history, save_environment_history)
    return nothing
end

function collect_initial_outputs(::Val{true}, previous_outputs, cognitive_model, environment, observation_map, action_map, save_history, save_environment_history)
    return (;
        saved_observation = select_names(map_names(previous_outputs.observation_from_environment, observation_map), observation_selection(save_history)),
        saved_action = select_names(previous_outputs.action, action_selection(save_history)),
        saved_cognitive_state = save_state(cognitive_model, previous_outputs.cognitive_state, state_selection(save_history)),
        saved_environment_state = save_state(environment, previous_outputs.environment_state, state_selection(save_environment_history)),
        saved_observation_from_environment = select_names(previous_outputs.observation_from_environment, observation_selection(save_environment_history)),
        saved_action_to_environment = select_names(map_names(previous_outputs.action, action_map), action_selection(save_environment_history)),
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
        observations = initialize_storage(outputs.saved_observation, n_steps),
        actions = initialize_storage(outputs.saved_action, n_steps),
        cognitive_states = initialize_storage(outputs.saved_cognitive_state, n_steps),
        environment_states = initialize_storage(outputs.saved_environment_state, n_steps),
        observations_from_environment = initialize_storage(outputs.saved_observation_from_environment, n_steps),
        actions_to_environment = initialize_storage(outputs.saved_action_to_environment, n_steps),
    )
end

function initialize_all_storage(initial_outputs::NamedTuple, outputs, n_steps)
    return (;
        observations = initialize_storage(initial_outputs.saved_observation, outputs.saved_observation, n_steps),
        actions = initialize_storage(initial_outputs.saved_action, outputs.saved_action, n_steps),
        cognitive_states = initialize_storage(initial_outputs.saved_cognitive_state, outputs.saved_cognitive_state, n_steps),
        environment_states = initialize_storage(initial_outputs.saved_environment_state, outputs.saved_environment_state, n_steps),
        observations_from_environment = initialize_storage(initial_outputs.saved_observation_from_environment, outputs.saved_observation_from_environment, n_steps),
        actions_to_environment = initialize_storage(initial_outputs.saved_action_to_environment, outputs.saved_action_to_environment, n_steps),
    )
end

"""
    check_model_output(output, cognitive_model)

Return `output` if it is the `ModelOutput` that an `apply_model` method of `cognitive_model`
must return, and throw an error otherwise.
"""
function check_model_output(output::ModelOutput, ::AbstractCognitiveModel)
    return output
end

#Untyped, so that there is always a method, whatever the cognitive model is inferred as
function check_model_output(output, cognitive_model)
    error("The apply_model method of a cognitive model of type $(typeof(cognitive_model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).")
end

"""
    check_environment_output(output, environment)

Return `output` if it is the `EnvironmentOutput` that an `apply_model` method of
`environment` must return, and throw an error otherwise.
"""
function check_environment_output(output::EnvironmentOutput, ::AbstractEnvironmentModel)
    return output
end

#Untyped, so that there is always a method, whatever the environment is inferred as
function check_environment_output(output, environment)
    error("The apply_model method of an environment model of type $(typeof(environment)) must return a CognitiveModeling.EnvironmentOutput, but it returned a value of type $(typeof(output)).")
end

"""
    timestep(::EnvironmentStarts, cognitive_model, environment, cognitive_parameters,
             environment_parameters, save_history, save_environment_history,
             observation_map, action_map, rng, timestep_size, previous)

Run a timestep where the environment moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::EnvironmentStarts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, timestep_size, previous)

    #The environment receives the previous action as action_map passes it on
    previous_action_to_environment = map_names(previous.action, action_map)

    #Update the environment
    environment_output = check_environment_output(@inline(apply_model(environment, EnvironmentInput(previous_action_to_environment, previous.environment_state, previous.observation_from_environment, environment_parameters, timestep_size, rng))), environment)

    #Sample observation
    observation_from_environment = sample_observation(rng, environment_output.observation_distribution)

    #The cognitive model receives the observation as observation_map passes it on
    observation = map_names(observation_from_environment, observation_map)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(@inline(apply_model(cognitive_model, ModelInput(observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), cognitive_model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    #The environment's history has the action as action_map passes it on
    action_to_environment = map_names(action, action_map)

    return collect_timestep_outputs(environment_output, cognitive_output, observation_from_environment, observation, action, action_to_environment, cognitive_model, environment, save_history, save_environment_history)
end

"""
    timestep(::AgentStarts, cognitive_model, environment, cognitive_parameters,
             environment_parameters, save_history, save_environment_history,
             observation_map, action_map, rng, timestep_size, previous)

Run a timestep where the agent moves first, starting from the outputs of the timestep
before, and return its outputs.

This is always inlined, because a call here makes the loop several times slower.
"""
@inline function timestep(::AgentStarts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, timestep_size, previous)

    #The cognitive model receives the previous observation as observation_map passes it on
    previous_observation = map_names(previous.observation_from_environment, observation_map)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(@inline(apply_model(cognitive_model, ModelInput(previous_observation, previous.cognitive_state, previous.action, cognitive_parameters, timestep_size, rng))), cognitive_model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    #The environment receives the action as action_map passes it on
    action_to_environment = map_names(action, action_map)

    #Update the environment
    environment_output = check_environment_output(@inline(apply_model(environment, EnvironmentInput(action_to_environment, previous.environment_state, previous.observation_from_environment, environment_parameters, timestep_size, rng))), environment)

    #Sample observation
    observation_from_environment = sample_observation(rng, environment_output.observation_distribution)

    #The cognitive model's history has the observation as observation_map passes it on
    observation = map_names(observation_from_environment, observation_map)

    return collect_timestep_outputs(environment_output, cognitive_output, observation_from_environment, observation, action, action_to_environment, cognitive_model, environment, save_history, save_environment_history)
end

"""
    run_remaining_timesteps(storage, outputs, who_starts, cognitive_model, environment,
                            cognitive_parameters, environment_parameters, save_history,
                            save_environment_history, observation_map, action_map, rng,
                            timestep_sizes, n_steps, offset)

Run the timesteps after the first, and return the `NamedTuple` of vectors with the saved
outputs of the timesteps that ran, followed by the outputs of the last one. The timesteps
end before `n_steps` if [`stopping_criterion`](@ref) of `environment` is met. `offset` is 1
when the vectors start with timestep 0, and 0 otherwise, so that timestep `t` is element
`t + offset`.

This is its own function so that it's compiled for the types of the first outputs.
"""
function run_remaining_timesteps(storage, outputs, who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, timestep_sizes, n_steps, offset)

    #Loop through timesteps
    for t in 2:n_steps

        #Stop if the environment has ended, and keep only the timesteps that ran
        if stopping_criterion(environment, outputs.environment_state, environment_parameters)
            foreach(vector -> resize!(vector, t - 1 + offset), storage)
            return storage, outputs
        end

        #Run the timestep, starting from the outputs of the one before
        outputs = timestep(who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, get_timestep_size(timestep_sizes, t), outputs)

        #Store trial outputs
        storage = (;
            observations = store_output!(storage.observations, outputs.saved_observation, t + offset),
            actions = store_output!(storage.actions, outputs.saved_action, t + offset),
            cognitive_states = store_output!(storage.cognitive_states, outputs.saved_cognitive_state, t + offset),
            environment_states = store_output!(storage.environment_states, outputs.saved_environment_state, t + offset),
            observations_from_environment = store_output!(storage.observations_from_environment, outputs.saved_observation_from_environment, t + offset),
            actions_to_environment = store_output!(storage.actions_to_environment, outputs.saved_action_to_environment, t + offset),
        )
    end

    #Return stored outputs, and the last ones for continuing from
    return storage, outputs
end

"""
    run_timesteps(previous_outputs, who_starts, cognitive_model, environment,
                  cognitive_parameters, environment_parameters, save_history,
                  save_environment_history, observation_map, action_map, rng,
                  timestep_sizes, n_steps, save_initial_values::Val)

Run up to `n_steps` timesteps starting from `previous_outputs`, and return a `NamedTuple` of
vectors with the saved outputs of the timesteps that ran, followed by the outputs of the
last one. The timesteps end before `n_steps` if [`stopping_criterion`](@ref) of
`environment` is met, and an environment where it is met already can't run any. When
`save_initial_values` is `Val(true)`, each vector starts with what is saved from before the
first timestep, as timestep 0.

The vectors are `observations`, `actions`, `cognitive_states` and `environment_states`, as
`save_history` and `save_environment_history` select them, and
`observations_from_environment` and `actions_to_environment` with what
`save_environment_history` selects of the observations the environment gave and the actions
it received, for the environment's history.

[`simulate!`](@ref) continues from the last outputs instead of from what was saved, because
[`prepare_state_for_saving`](@ref) can return anything, and because the next simulation can
change a state in place.
"""
function run_timesteps(previous_outputs, who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, timestep_sizes, n_steps, save_initial_values)

    #The first timestep sets the types of the storage, so there must be at least one
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, but it is $n_steps."))

    #A vector of timestep sizes must have one size for each timestep
    if timestep_sizes isa AbstractVector && length(timestep_sizes) != n_steps
        throw(DimensionMismatch("timestep_sizes must have one size for each timestep, but it has $(length(timestep_sizes)) sizes and n_steps is $n_steps."))
    end

    #An environment which has ended already can't run the first timestep
    if stopping_criterion(environment, previous_outputs.environment_state, environment_parameters)
        throw(ArgumentError("The stopping criterion of an environment model of type $(typeof(environment)) is already met before the first timestep, so no timestep can run. An Environment which has ended starts again after reset!."))
    end

    #Save what is kept from before the first timestep before it runs, since it can change a state in place
    initial_outputs = collect_initial_outputs(save_initial_values, previous_outputs, cognitive_model, environment, observation_map, action_map, save_history, save_environment_history)

    #Run the first timestep outside the loop, since its inputs can have other types than all the rest
    outputs = timestep(who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, get_timestep_size(timestep_sizes, 1), previous_outputs)

    #Initialize storage for the saved outputs, which starts with timestep 0 when it is kept
    storage = initialize_all_storage(initial_outputs, outputs, n_steps)
    offset = initial_outputs === nothing ? 0 : 1

    #Run the remaining timesteps
    return run_remaining_timesteps(storage, outputs, who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, save_environment_history, observation_map, action_map, rng, timestep_sizes, n_steps, offset)
end

"""
    simulate(cognitive_model, environment; <keyword arguments>)

Simulate `cognitive_model` interacting with `environment`, and return the vectors
`observations`, `actions`, `cognitive_states` and `environment_states`, with one element for
timestep 0, before the first timestep, and one for each timestep after it, followed by
`timestep_sizes`.

Each element is what happened on that timestep, and the states are those after it. The
element for timestep 0 has the previous observation and action, which are
[`BeforeFirstTimestep`](@ref) unless others are given, and the states before the first
timestep. Since Julia counts from 1, element `t + 1` is timestep `t`. `save_history` and
`save_environment_history` select what is kept of each element, which is nothing by
default: give `SaveHistory(SaveEverything())` to keep everything, and
`SaveHistory(SaveEverything(); save_initial_values = false)` to leave out timestep 0, so
that element `t` is timestep `t`. The sizes of the timesteps are returned as they were
given: a single size as it is, and a vector of sizes with the sizes of the timesteps that
ran, after `BeforeFirstTimestep()` for timestep 0 when it is there.

Parameters that aren't given take the values in [`default_parameters`](@ref), if
`cognitive_model` or `environment` provides defaults for them. The names of the parameters,
the states, the previous action and the previous observation must be those in
[`attribute_names`](@ref), and `cognitive_model` and `environment` must list the same
observations and the same actions, or an `ArgumentError` is thrown before the first
timestep. When the two name things differently, or one gives something the other doesn't
list, `observation_map` and `action_map` say which outputs of one the other receives, and
under which names. The observations are then returned as `cognitive_model` received them,
and the actions as it gave them.

The simulation ends before `n_steps` timesteps if [`stopping_criterion`](@ref) of
`environment` is met, and everything returned then has the timesteps that ran.

# Arguments
- `cognitive_model::AbstractCognitiveModel`: the cognitive model.
- `environment::AbstractEnvironmentModel`: the environment model.
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep. It
  has no default, because a wrong choice shifts the model by a timestep without any error.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1. It is the
  most timesteps there can be when `environment` has a `stopping_criterion`.
- `timestep_sizes=1`: how much time each timestep covers, which both `cognitive_model` and
  `environment` receive. A value that isn't a vector, such as a single number, is the size
  of every timestep. A vector has one size for each timestep, so its length must be
  `n_steps`.
- `cognitive_parameters::NamedTuple`: the parameters of `cognitive_model`, with its default
  values for those that aren't given, if it provides any.
- `environment_parameters::NamedTuple`: the parameters of `environment`, with its default
  values for those that aren't given, if it provides any.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample actions and
  observations from, which `cognitive_model` and `environment` also receive in their inputs.
- `previous_cognitive_state=initialize_state(cognitive_model, ...)`: the state of
  `cognitive_model` before the first timestep, by default its initial state from the full
  parameters. A state that `cognitive_model` changes in place will be changed.
- `previous_action=initial_action(cognitive_model)`: the action before the first timestep.
  By default every action is [`BeforeFirstTimestep`](@ref), with the names from
  [`attribute_names`](@ref).
- `previous_environment_state=initialize_state(environment, ...)`: the state of
  `environment` before the first timestep, by default its initial state from the full
  parameters.
- `previous_observation=initial_observation(environment)`: the observation before the first
  timestep. By default every observation is `BeforeFirstTimestep()`, with the names from
  `attribute_names` of `environment`.
- `save_history::SaveHistory=SaveHistory()`: which observations, actions and states of
  `cognitive_model` are saved, by name, and whether timestep 0 is. Nothing is saved by
  default, and `SaveHistory(SaveEverything())` saves everything. See [`SaveHistory`](@ref).
- `save_environment_history::SaveHistory=SaveHistory()`: which states of `environment` are
  saved. Only its `states` is used here, since the observations and actions are returned
  once, as `save_history` selects them, and so that the four vectors have the same length,
  `save_history` decides whether they start with timestep 0.
- `observation_map::Union{Nothing, NameMap}=nothing`: which observations of `environment`
  are passed on to `cognitive_model`, and under which names, as a [`NameMap`](@ref) from
  the names `environment` gives to exactly those `cognitive_model` lists. Without one, the
  two must list the same observations, which `cognitive_model` receives under the same
  names.
- `action_map::Union{Nothing, NameMap}=nothing`: which actions of `cognitive_model` are
  passed on to `environment`, and under which names, as a `NameMap` from the names
  `cognitive_model` gives to exactly those `environment` lists. Without one, the two must
  list the same actions, which `environment` receives under the same names.

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

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = BernoulliLogit(Vₜ - 0.5)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
end

#An environment which takes the choice, and gives a reward with a fixed probability whatever it is
struct BernoulliRewards <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(environment::BernoulliRewards, parameters)
    return (;)
end

function CognitiveModeling.apply_model(environment::BernoulliRewards, input::EnvironmentInput)
    return EnvironmentOutput((; reward = Bernoulli(input.parameters.reward_probability)))
end

function CognitiveModeling.attribute_names(environment::BernoulliRewards)
    return ModelAttributes(; parameters = :reward_probability, states = (), observations = :reward, actions = :choice)
end

observations, actions, cognitive_states, environment_states, timestep_sizes = simulate(
    RescorlaWagner(),
    BernoulliRewards();
    who_starts = EnvironmentStarts(),
    n_steps = 3,
    cognitive_parameters = (; learning_rate = 0.5),
    environment_parameters = (; reward_probability = 1.0),
    save_history = SaveHistory(SaveEverything()),
)

#The initial state, as timestep 0, and the state after each timestep
cognitive_states

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
        cognitive_model::AbstractCognitiveModel,
        environment::AbstractEnvironmentModel;
        who_starts::Union{AgentStarts, EnvironmentStarts},
        n_steps::Int,
        timestep_sizes = 1,
        cognitive_parameters::NamedTuple,
        environment_parameters::NamedTuple,
        rng::AbstractRNG = default_rng(),
        previous_cognitive_state = initialize_state(cognitive_model, merge_default_parameters(cognitive_model, cognitive_parameters)),
        previous_action = initial_action(cognitive_model),
        previous_environment_state = initialize_state(environment, merge_default_parameters(environment, environment_parameters)),
        previous_observation = initial_observation(environment),
        save_history::SaveHistory = SaveHistory(),
        save_environment_history::SaveHistory = SaveHistory(),
        observation_map::Union{Nothing, NameMap} = nothing,
        action_map::Union{Nothing, NameMap} = nothing,
    )

    #Fill in the default parameters
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    environment_parameters = merge_default_parameters(environment, environment_parameters)

    #Check the names of everything against attribute_names, and that the cognitive model and the environment fit together
    check_match_attribute_names(cognitive_model, cognitive_parameters, previous_cognitive_state, previous_action)
    check_match_attribute_names(environment, environment_parameters, previous_environment_state, previous_observation)
    check_names_fit(cognitive_model, environment, observation_map, action_map)

    #Collect what was left from before the first timestep
    previous_outputs = (;
        observation_from_environment = previous_observation,
        action = previous_action,
        cognitive_state = previous_cognitive_state,
        environment_state = previous_environment_state,
    )

    #Run the timesteps, keeping only the states of the environment's selection, since the observations and actions are returned once, starting with timestep 0 when save_history saves the initial values
    save_initial_values = Val(saves_initial_values(save_history))
    storage, _ = run_timesteps(previous_outputs, who_starts, cognitive_model, environment, cognitive_parameters, environment_parameters, save_history, savehistory_states_only(save_environment_history), observation_map, action_map, rng, timestep_sizes, n_steps, save_initial_values)

    #The sizes of the timesteps that ran, since the environment can end the simulation early
    n_timesteps_that_ran = length(storage.observations) - (saves_initial_values(save_history) ? 1 : 0)
    timestep_sizes_that_ran = returned_timestep_sizes(timestep_sizes, n_timesteps_that_ran, save_initial_values)

    #Return stored outputs
    return storage.observations, storage.actions, storage.cognitive_states, storage.environment_states, timestep_sizes_that_ran
end

"""
    simulate!(agent::CognitiveAgent, environment::Environment; <keyword arguments>)

Simulate `agent` interacting with `environment` from where they are, update them both, and
return the same as [`simulate`](@ref), without timestep 0: the vectors with what
`agent.save_history` selects of the observations, actions and states on each timestep of
this call, and what `environment.save_history` selects of the environment's states,
followed by the sizes of the timesteps. What the two select is also added to
`agent.history` and `environment.history`, together with the sizes of the timesteps, unless
a selection is empty. A history has timestep 0 from where the agent or the environment
started, when its [`SaveHistory`](@ref) saves the initial values.

The agent's model and the environment's must fit together, as for `simulate`, directly or
through `observation_map` and `action_map`. The agent's history holds the observations as
its model received them, and the environment's history the actions as its model did.

The simulation ends before `n_steps` timesteps if [`stopping_criterion`](@ref) of the
environment's model is met. An environment where it is met already can't be simulated
further, and throws an error until it is [`reset!`](@ref).

Reading the agent's untyped fields costs a dynamic dispatch on each call, which makes the
return type unknown to the compiler. The loop inside is still fully typed. Use `simulate` if
the return type matters, or if most calls are for a single timestep.

# Arguments
- `who_starts::Union{AgentStarts, EnvironmentStarts}`: who moves first on each timestep.
- `n_steps::Int`: the number of timesteps to simulate, which must be at least 1. It is the
  most timesteps there can be when the environment's model has a `stopping_criterion`.
- `timestep_sizes=1`: how much time each timestep covers. A value that isn't a vector, such
  as a single number, is the size of every timestep of this call. A vector has one size for
  each timestep of this call, so its length must be `n_steps`.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.
- `observation_map::Union{Nothing, NameMap}=nothing`: which observations of the environment
  the agent receives, and under which names, as for `simulate`.
- `action_map::Union{Nothing, NameMap}=nothing`: which actions of the agent the environment
  receives, and under which names, as for `simulate`.

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RandomChoices <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RandomChoices, parameters)
    return (;)
end

function CognitiveModeling.apply_model(model::RandomChoices, input::ModelInput)
    return ModelOutput((; choice = Bernoulli(0.5)))
end

function CognitiveModeling.attribute_names(model::RandomChoices)
    return ModelAttributes(; parameters = (), states = (), observations = :n_timesteps, actions = :choice)
end

struct TimestepCounter <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(model::TimestepCounter, parameters)
    return (; n_timesteps = 0)
end

function CognitiveModeling.apply_model(model::TimestepCounter, input::EnvironmentInput)
    n_timesteps = input.previous_state.n_timesteps + 1
    return EnvironmentOutput((; n_timesteps = Deterministic(n_timesteps)), (; n_timesteps))
end

function CognitiveModeling.attribute_names(model::TimestepCounter)
    return ModelAttributes(; parameters = (), states = :n_timesteps, observations = :n_timesteps, actions = :choice)
end

agent = CognitiveAgent(RandomChoices(), (;))
environment = Environment(TimestepCounter(), (;))

simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 2)
simulate!(agent, environment; who_starts = EnvironmentStarts(), n_steps = 3)

environment.state

# output

(n_timesteps = 5,)
```
"""
function simulate!(
        agent::CognitiveAgent,
        environment::Environment;
        who_starts::Union{AgentStarts, EnvironmentStarts},
        n_steps::Int,
        timestep_sizes = 1,
        rng::AbstractRNG = default_rng(),
        observation_map::Union{Nothing, NameMap} = nothing,
        action_map::Union{Nothing, NameMap} = nothing,
    )

    #Check that the cognitive model and the environment fit together
    check_names_fit(agent.cognitive_model, environment.model, observation_map, action_map)

    #Collect where the agent and the environment are
    previous_outputs = (;
        observation_from_environment = environment.previous_observation,
        action = agent.previous_action,
        cognitive_state = agent.cognitive_state,
        environment_state = environment.state,
    )

    #Run the timesteps, without timestep 0, which the histories have from where the agent and the environment started
    storage, last_outputs = run_timesteps(previous_outputs, who_starts, agent.cognitive_model, environment.model, agent.cognitive_parameters, environment.parameters, agent.save_history, environment.save_history, observation_map, action_map, rng, timestep_sizes, n_steps, Val(false))

    #Update agent and environment
    agent.cognitive_state = last_outputs.cognitive_state
    agent.previous_action = last_outputs.action
    environment.state = last_outputs.environment_state
    environment.previous_observation = last_outputs.observation_from_environment

    #The sizes of the timesteps that ran, since the environment can end the simulation early
    n_timesteps_that_ran = length(storage.observations)
    timestep_sizes_that_ran = returned_timestep_sizes(timestep_sizes, n_timesteps_that_ran, Val(false))

    #Add to the agent's history
    if !saves_nothing(agent.save_history)
        append!(agent.history.observations, storage.observations)
        append!(agent.history.actions, storage.actions)
        append!(agent.history.states, storage.cognitive_states)
        append!(agent.history.timestep_sizes, each_timestep_size(timestep_sizes_that_ran, n_timesteps_that_ran))
    end

    #Add to the environment's history
    if !saves_nothing(environment.save_history)
        append!(environment.history.observations, storage.observations_from_environment)
        append!(environment.history.actions, storage.actions_to_environment)
        append!(environment.history.states, storage.environment_states)
        append!(environment.history.timestep_sizes, each_timestep_size(timestep_sizes_that_ran, n_timesteps_that_ran))
    end

    #Return stored outputs
    return storage.observations, storage.actions, storage.cognitive_states, storage.environment_states, timestep_sizes_that_ran
end

"""
    timestep(cognitive_model, observation, previous_cognitive_state, previous_action,
             cognitive_parameters, timestep_size, rng, save_history)

Run a timestep of `cognitive_model` on `observation`, without an environment, and return the
action sampled from its output, its new state, and what `save_history` selects of the
observation, the action and the state, in a `NamedTuple`. The observation must have the
names that `cognitive_model` lists.

[`simulate`](@ref) and [`simulate!`](@ref) on a single observation both run this. It is its
own function so that it's compiled for the types in an agent's untyped fields, which makes
`simulate!` on a single observation several times faster, and makes the check of the
observation's names cost nothing.
"""
function timestep(cognitive_model, observation, previous_cognitive_state, previous_action, cognitive_parameters, timestep_size, rng, save_history)

    #Check that the observation has the names the cognitive model lists
    check_observation_names(cognitive_model, observation)

    #Run timestep of cognitive model
    cognitive_output = check_model_output(@inline(apply_model(cognitive_model, ModelInput(observation, previous_cognitive_state, previous_action, cognitive_parameters, timestep_size, rng))), cognitive_model)

    #Sample action
    action = sample_action(rng, cognitive_output.action_distribution)

    return (;
        action,
        cognitive_state = cognitive_output.new_state,
        saved_observation = select_names(observation, observation_selection(save_history)),
        saved_action = select_names(action, action_selection(save_history)),
        saved_cognitive_state = save_state(cognitive_model, cognitive_output.new_state, state_selection(save_history)),
    )
end

"""
    simulate(cognitive_model, observation; <keyword arguments>)

Run a single timestep of `cognitive_model` on `observation`, without an environment, and
return the sampled action and the new state of the model.

Nothing is kept between calls, so pass the returned state and action on as
`previous_cognitive_state` and `previous_action` to run the next timestep.
[`simulate!`](@ref) on a single observation does the same for a [`CognitiveAgent`](@ref),
which keeps them. The compiler knows the types here, which it doesn't for an agent, so this
is the faster of the two.

The action is in a `NamedTuple` with the name the model gave it, as in `action.choice`. The
observation must be a `NamedTuple` with every observation the model lists in
[`attribute_names`](@ref), or an `ArgumentError` is thrown. Parameters that aren't given
take the values in [`default_parameters`](@ref), if `cognitive_model` provides defaults for
them, and the names of the parameters, the state and the previous action are checked as
when there is an environment.

# Arguments
- `cognitive_model::AbstractCognitiveModel`: the cognitive model.
- `observation::NamedTuple`: what the model observes on this timestep, such as
  `(; reward = 1.0)`. An observation that isn't made is [`NoObservation`](@ref).
- `cognitive_parameters::NamedTuple`: the parameters of `cognitive_model`, with its default
  values for those that aren't given, if it provides any.
- `timestep_size=1`: how much time this timestep covers.
- `rng::AbstractRNG=default_rng()`: the random number generator to sample from.
- `previous_cognitive_state=initialize_state(cognitive_model, ...)`: the state of
  `cognitive_model` before this timestep, by default its initial state from the full
  parameters. A state that `cognitive_model` changes in place will be changed.
- `previous_action=initial_action(cognitive_model)`: the action before this timestep. By
  default every action is [`BeforeFirstTimestep`](@ref).

# Examples
```jldoctest
using CognitiveModeling
using Distributions: Bernoulli

struct RescorlaWagner <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::RescorlaWagner, parameters)
    return (; expected_reward = 0.5)
end

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
end

cognitive_parameters = (; learning_rate = 0.5)

#The first timestep starts from the initial state
action, cognitive_state = simulate(RescorlaWagner(), (; reward = 1.0); cognitive_parameters)

#The next one continues from what the first returned
action, cognitive_state = simulate(
    RescorlaWagner(),
    (; reward = 1.0);
    cognitive_parameters,
    previous_cognitive_state = cognitive_state,
    previous_action = action,
)

cognitive_state

# output

(expected_reward = 0.875,)
```
"""
function simulate(
        cognitive_model::AbstractCognitiveModel,
        observation;
        cognitive_parameters::NamedTuple,
        timestep_size = 1,
        rng::AbstractRNG = default_rng(),
        previous_cognitive_state = initialize_state(cognitive_model, merge_default_parameters(cognitive_model, cognitive_parameters)),
        previous_action = initial_action(cognitive_model),
    )

    #Fill in the default parameters
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)

    #Check the names of the parameters, the state and the previous action against attribute_names
    check_match_attribute_names(cognitive_model, cognitive_parameters, previous_cognitive_state, previous_action)

    #Run the timestep, where nothing is selected for saving, since everything is returned
    outputs = timestep(cognitive_model, observation, previous_cognitive_state, previous_action, cognitive_parameters, timestep_size, rng, SaveHistory())

    return outputs.action, outputs.cognitive_state
end

"""
    simulate!(agent::CognitiveAgent, observation; timestep_size=1, rng=default_rng())

Run a single timestep of `agent` on `observation`, update it, and return the sampled action.
What `agent.save_history` selects of the observation, the action and the state is added to
`agent.history`, together with the size of the timestep.

The action is in a `NamedTuple` with the name the model gave it, as in `action.choice`, as
for [`simulate`](@ref). The observation must be a `NamedTuple` with every observation the
model lists in [`attribute_names`](@ref), or an `ArgumentError` is thrown.

# Arguments
- `agent::CognitiveAgent`: the agent to update.
- `observation::NamedTuple`: what the agent observes on this timestep, such as
  `(; reward = 1.0)`. An observation that isn't made is [`NoObservation`](@ref).
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

function CognitiveModeling.apply_model(model::RescorlaWagner, input::ModelInput)
    V₋₁ = input.previous_state.expected_reward
    Vₜ = V₋₁ + input.parameters.learning_rate * (input.observation.reward - V₋₁)
    return ModelOutput((; choice = Bernoulli(Vₜ)), (; expected_reward = Vₜ))
end

function CognitiveModeling.attribute_names(model::RescorlaWagner)
    return ModelAttributes(; parameters = :learning_rate, states = :expected_reward, observations = :reward, actions = :choice)
end

agent = CognitiveAgent(RescorlaWagner(), (; learning_rate = 0.5); save_history = SaveHistory(SaveEverything()))

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
function simulate!(agent::CognitiveAgent, observation; timestep_size = 1, rng::AbstractRNG = default_rng())

    #Run the timestep
    outputs = timestep(agent.cognitive_model, observation, agent.cognitive_state, agent.previous_action, agent.cognitive_parameters, timestep_size, rng, agent.save_history)

    #Update the agent
    agent.cognitive_state = outputs.cognitive_state
    agent.previous_action = outputs.action

    #Add to the agent's history
    if !saves_nothing(agent.save_history)
        push!(agent.history.observations, outputs.saved_observation)
        push!(agent.history.actions, outputs.saved_action)
        push!(agent.history.states, outputs.saved_cognitive_state)
        push!(agent.history.timestep_sizes, timestep_size)
    end

    return outputs.action
end
