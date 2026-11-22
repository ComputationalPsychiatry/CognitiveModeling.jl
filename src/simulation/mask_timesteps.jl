## Masking the timesteps of a simulation, for flags that a compiler traces ##

#Types with a single value, such as markers, can't be given to ifelse with a traced condition, and are the same either way
"""
    select_values(condition, values_if_true, values_if_false)

Return `values_if_true` when `condition` is true and `values_if_false` otherwise, selected
with `ifelse` one number at a time, through `NamedTuple`s, tuples and arrays, which must
have the same layout in both. A value of a type that has a single value, such as
[`NoValue`](@ref), is the same in both and is returned as it is. Used by the masking loop,
where `condition` can be a traced Bool, which nothing can branch on.
"""
function select_values(
    condition, values_if_true::NamedTuple{names}, values_if_false::NamedTuple{names}
) where {names}
    return NamedTuple{names}(
        select_values(condition, values(values_if_true), values(values_if_false))
    )
end
function select_values(condition, values_if_true::Tuple, values_if_false::Tuple)
    return map(
        (value_if_true, value_if_false) ->
            select_values(condition, value_if_true, value_if_false),
        values_if_true,
        values_if_false,
    )
end
function select_values(
    condition, values_if_true::AbstractArray, values_if_false::AbstractArray
)
    return map(
        (value_if_true, value_if_false) ->
            select_values(condition, value_if_true, value_if_false),
        values_if_true,
        values_if_false,
    )
end
function select_values(condition, value_if_true, value_if_false)
    if Base.issingletontype(typeof(value_if_true)) && value_if_true === value_if_false
        return value_if_true
    end
    return ifelse(condition, value_if_true, value_if_false)
end

"""
    values_to_continue_from(timestep_values)

Return the values of a timestep that the next one starts from, from what
[`collect_timestep_values`](@ref) collected: the state and the action of the cognitive
model, and under `environment` the state and the observation of the environment. Used by
the masking loop, which selects them with [`select_values`](@ref).
"""
function values_to_continue_from(timestep_values)
    return (;
        state=timestep_values.state,
        action=timestep_values.action,
        environment=(;
            state=timestep_values.environment.state,
            observation=timestep_values.environment.observation,
        ),
    )
end

"""
    stopping_criterion_of_either_model(cognitive_model, environment, timestep_values,
                                       cognitive_parameters, environment_parameters)

Return whether [`stopping_criterion`](@ref) of either model is met on `timestep_values`, the
values of the timestep before, combined with `|`, so that a criterion that a compiler
traces is combined too. Used by the masking loop, and by
[`simulate_all_timesteps`](@ref) to choose it.
"""
function stopping_criterion_of_either_model(
    cognitive_model,
    environment,
    timestep_values,
    cognitive_parameters,
    environment_parameters,
)
    return stopping_criterion(
        environment, timestep_values.environment.state, environment_parameters
    ) | stopping_criterion(cognitive_model, timestep_values.state, cognitive_parameters)
end

"""
    still_valid(parameters_valid, running, timestep_parameters_valid)

Return whether the parameters of both models are still valid after a timestep of the
masking loop: `parameters_valid` until the timestep, and `false` when the timestep ran, as
`running` says, and its flag, `timestep_parameters_valid`, is false. With
[`ValidByDefault`](@ref), `parameters_valid` stays as it is. Used by
[`count_masked_timestep`](@ref).
"""
still_valid(parameters_valid, _, ::ValidByDefault) = parameters_valid
function still_valid(parameters_valid, running, timestep_parameters_valid)
    return parameters_valid & (!running | timestep_parameters_valid)
end

"""
    valid_on_first_timestep(parameters_valid, stopped_before_first)

Return whether the parameters of both models are valid after the first timestep of a masked
simulation: what they gave, `parameters_valid`, unless the stopping criterion was met before
it, as `stopped_before_first` says, so that it didn't count. With [`ValidByDefault`](@ref),
`true`. Used by [`masking_after_first_timestep`](@ref).
"""
valid_on_first_timestep(::ValidByDefault, _) = true
function valid_on_first_timestep(parameters_valid, stopped_before_first)
    return parameters_valid | stopped_before_first
end

#What the masking loop carries from one timestep to the next, in place of the branches of the stopping loop
"""
    masking_after_first_timestep(timestep_values, stopped_before_first)

Return what a masked simulation carries after its first timestep, whose values are
`timestep_values`, in a `NamedTuple`: whether it is still `running`, which it isn't when the
stopping criterion was met before it, as `stopped_before_first` says, or the parameters
turned invalid on it, whether the parameters stayed valid, under `parameters_valid`, the
number of timesteps kept, under `n_timesteps`, the record of the last timestep simulated,
under `last_record`, and the values that the next timestep starts from, under `previous`.
The timesteps after a first timestep that didn't count start from its values, since the
values from before it can have other types. Used by [`simulate_all_timesteps`](@ref).
"""
function masking_after_first_timestep(timestep_values, stopped_before_first)
    running = !stopped_before_first & parameters_are_valid(timestep_values.parameters_valid)
    return (;
        running,
        parameters_valid=valid_on_first_timestep(
            timestep_values.parameters_valid, stopped_before_first
        ),
        n_timesteps=ifelse(running, 1, 0),
        last_record=numbered_record(timestep_values.record, 1),
        previous=values_to_continue_from(timestep_values),
    )
end

"""
    masking_before_timestep(timestep_values, t)

Return what a simulation that is masked from timestep `t` on carries before it, as
[`masking_after_first_timestep`](@ref) arranges it: it is running, with valid parameters and
the `t - 1` timesteps before kept, the last of which has the values `timestep_values`. Used
by [`simulate_remaining_timesteps`](@ref) when a flag on timestep `t` must be masked.
"""
function masking_before_timestep(timestep_values, t)
    return (;
        running=true,
        parameters_valid=true,
        n_timesteps=t - 1,
        last_record=numbered_record(timestep_values.record, t - 1),
        previous=values_to_continue_from(timestep_values),
    )
end

"""
    count_masked_timestep(masking, timestep_values, t)

Return what a masked simulation carries after timestep `t`, from `masking`, what it carried
before it, as [`masking_after_first_timestep`](@ref) arranges it, and the values of the
timestep, `timestep_values`. The timestep is the last simulated when the simulation was
running before it, and it is kept when the parameters also stayed valid on it. Otherwise the
simulation isn't running after it, the number of timesteps kept stays as it was, and the
next timestep starts from the values the timestep started from, which
[`select_values`](@ref) selects. Used by the masking loop.
"""
function count_masked_timestep(masking, timestep_values, t)
    timestep_valid = parameters_are_valid(timestep_values.parameters_valid)
    running = masking.running & timestep_valid
    return (;
        running,
        parameters_valid=still_valid(
            masking.parameters_valid, masking.running, timestep_values.parameters_valid
        ),
        n_timesteps=masking.n_timesteps + ifelse(running, 1, 0),
        last_record=select_values(
            masking.running, numbered_record(timestep_values.record, t), masking.last_record
        ),
        previous=select_values(
            running, values_to_continue_from(timestep_values), masking.previous
        ),
    )
end

"""
    simulate_remaining_timesteps_masked(storage, masking, first_timestep, who_starts,
                                        cognitive_model, environment, cognitive_parameters,
                                        environment_parameters, cognitive_save_history,
                                        environment_save_history, observation_map,
                                        action_map, rng, timestep_sizes, n_steps, offset,
                                        check_stopping_criterion)

Simulate every timestep from `first_timestep` on, also those after the simulation ended,
and return what [`simulate_all_timesteps`](@ref) returns, with `masked=Val(true)`. It
continues a simulation from the first timestep whose flag must be masked, as
[`masks_flag`](@ref) says, such as the traced Bools of a compiler like Reactant.jl, which
nothing can branch on: the `parameters_valid` of a model, or what its
[`stopping_criterion`](@ref) returned.

Instead of stopping, the simulation carries `masking`, as
[`masking_after_first_timestep`](@ref) arranges it, whose running flag turns false when the
stopping criterion of either model is met before a timestep, when
`check_stopping_criterion` is `true`, or when the parameters of either model turn invalid
on a timestep, and stays false. From then on, every timestep starts from the values of the
last timestep that was kept, and nothing of it counts, as
[`count_masked_timestep`](@ref) works out, so that `n_timesteps`, `parameters_valid` and
`last_record` are what the stopping loop returns. The vectors in `storage` keep every
timestep, so those after `n_timesteps` hold what the timesteps after the end gave, which
belongs to no simulation. Used by [`simulate_all_timesteps`](@ref) and
[`simulate_remaining_timesteps`](@ref).
"""
function simulate_remaining_timesteps_masked(
    storage,
    masking,
    first_timestep,
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
    for t in first_timestep:n_steps

        #The simulation ends before this timestep when the stopping criterion of either model is met
        if check_stopping_criterion
            criterion_met = stopping_criterion_of_either_model(
                cognitive_model,
                environment,
                masking.previous,
                cognitive_parameters,
                environment_parameters,
            )
            masking = merge(masking, (; running=masking.running & !criterion_met))
        end

        #Simulate the timestep, also after the simulation ended, starting from the values of the last timestep that was kept
        timestep_values = simulate_both_models(
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
            masking.previous,
        )

        #Store what the timestep saved, which belongs to no simulation once it ended, and count it
        storage = store_timestep_values!(storage, timestep_values.saved, t + offset)
        masking = count_masked_timestep(masking, timestep_values, t)
    end

    #Return what was stored, which has every timestep, with where the simulation ended
    return (;
        storage,
        last_kept_values=nothing,
        masking.n_timesteps,
        masking.parameters_valid,
        masking.last_record,
        masked=Val(true),
    )
end
