## Masking the timesteps of a fitted session, for flags that a compiler traces ##
#An inference backend which a compiler like Reactant.jl compiles can't stop a session where a traced flag says, so the
#session runs every timestep, as the masked simulation does, and these carry where it ended

"""
    values_to_continue_fitted_session_from(timestep_values)

Return the values of a timestep of a fitted session that the next one starts from:
everything in `timestep_values` but what is scored or checked of the timestep itself, its
`action_distribution`, `actions_to_condition_on_at_timestep` and `parameters_valid`, and
under `environment` its `observation_distribution` and
`observations_to_condition_on_at_timestep`. Used by the masking of fitted sessions, which
selects them with [`select_values`](@ref).
"""
function values_to_continue_fitted_session_from(timestep_values::NamedTuple)
    kept = Base.structdiff(
        timestep_values,
        NamedTuple{(
            :action_distribution, :actions_to_condition_on_at_timestep, :parameters_valid
        )},
    )
    return merge(
        kept,
        (;
            environment=Base.structdiff(
                timestep_values.environment,
                NamedTuple{(
                    :observation_distribution, :observations_to_condition_on_at_timestep
                )},
            ),
        ),
    )
end

#What the masking of a fitted session carries from one timestep to the next, in place of the branches of a session that stops
"""
    masking_fitted_session_after_first_timestep(timestep_values, stopped_before_first,
                                                data_continues_before_first)

Return what a masked fitted session carries after its first timestep, whose values are
`timestep_values`, in a `NamedTuple`: whether it is still `running`, which it isn't when the
stopping criterion was met before it, as `stopped_before_first` says, or the parameters
turned invalid on it, whether the parameters stayed valid, under `parameters_valid`, which
they didn't when the criterion was met before the first timestep while the data continues,
as `data_continues_before_first` says, the number of timesteps kept, under `n_timesteps`,
and the values that the next timestep starts from, under `previous`. The timesteps after a
first timestep that didn't count start from its values, since the values from before it can
have other types. Used by fitted sessions that a compiler like Reactant.jl compiles.
"""
function masking_fitted_session_after_first_timestep(
    timestep_values, stopped_before_first, data_continues_before_first::Bool
)
    first_runs = !stopped_before_first
    running = first_runs & parameters_are_valid(timestep_values.parameters_valid)
    return (;
        running,
        parameters_valid=still_valid(
            !(stopped_before_first & data_continues_before_first),
            first_runs,
            timestep_values.parameters_valid,
        ),
        n_timesteps=ifelse(running, 1, 0),
        previous=values_to_continue_fitted_session_from(timestep_values),
    )
end

"""
    masking_fitted_session_before_timestep(timestep_values, t)

Return what a fitted session that is masked from timestep `t` on carries before it, as
[`masking_fitted_session_after_first_timestep`](@ref) arranges it: it is running, with valid
parameters and the `t - 1` timesteps before kept, the last of which has the values
`timestep_values`. Used by fitted sessions which switch to masking where a flag first must
be masked.
"""
function masking_fitted_session_before_timestep(timestep_values, t)
    return (;
        running=true,
        parameters_valid=true,
        n_timesteps=t - 1,
        previous=values_to_continue_fitted_session_from(timestep_values),
    )
end

"""
    stop_masked_fitted_session(masking, criterion_met, data_continues::Bool)

Return `masking`, what a masked fitted session carries, as
[`masking_fitted_session_after_first_timestep`](@ref) arranges it, after the stopping
criterion of either model was checked before a timestep, as `criterion_met` says: the session
is no longer running when it was met, and its parameters are invalid when it was met while
the session ran and its data continues, as `data_continues` says. Used by fitted sessions
that a compiler like Reactant.jl compiles.
"""
function stop_masked_fitted_session(masking, criterion_met, data_continues::Bool)
    return merge(
        masking,
        (;
            running=masking.running & !criterion_met,
            parameters_valid=masking.parameters_valid &
                             !(masking.running & criterion_met & data_continues),
        ),
    )
end

"""
    count_masked_fitted_timestep(masking, timestep_values)

Return what a masked fitted session carries after a timestep whose values are
`timestep_values`, from `masking`, what it carried before it. The timestep is kept when the
session was running before it and the parameters stayed valid on it. Otherwise the session
isn't running after it, the number of timesteps kept stays as it was, and the next timestep
starts from the values the timestep started from, which [`select_values`](@ref) selects.
Used by fitted sessions that a compiler like Reactant.jl compiles.
"""
function count_masked_fitted_timestep(masking, timestep_values)
    running = masking.running & parameters_are_valid(timestep_values.parameters_valid)
    return (;
        running,
        parameters_valid=still_valid(
            masking.parameters_valid, masking.running, timestep_values.parameters_valid
        ),
        n_timesteps=masking.n_timesteps + ifelse(running, 1, 0),
        previous=select_values(
            running,
            values_to_continue_fitted_session_from(timestep_values),
            masking.previous,
        ),
    )
end

"""
    log_density_of_masked_ending(masking)

Return what a masked fitted session adds to the log density for how it ended, from
`masking`, what it carried after its last timestep: `0.0` when the parameters stayed valid,
and `-Inf` otherwise, also when a stopping criterion ended the session while
its data continues. The session scores every timestep, also those after it ended, since
data after the end only exists when the parameters are invalid, and imputed values after it
keep their scores, so that each stays a variable with a density. Used by fitted sessions
that a compiler like Reactant.jl compiles.
"""
function log_density_of_masked_ending(masking)
    return ifelse(masking.parameters_valid, 0.0, -Inf)
end
