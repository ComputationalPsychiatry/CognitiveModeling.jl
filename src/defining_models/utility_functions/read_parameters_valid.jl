## Whether the parameters of a model are valid ##

"""
    parameters_are_valid(parameters_valid)

Return whether `parameters_valid`, what a model gave in its [`ModelOutput`](@ref), says that
its parameters are valid: `true` for [`ValidByDefault`](@ref), and the value itself
otherwise. Used after each timestep by the simulation and by fitted sessions, which read the
flag only through this function.
"""
parameters_are_valid(::ValidByDefault) = true
parameters_are_valid(parameters_valid) = parameters_valid

"""
    combine_parameters_valid(first_parameters_valid, second_parameters_valid)

Return whether the parameters of two models are valid, from what each gave in its
[`ModelOutput`](@ref): [`ValidByDefault`](@ref) when both gave it, the other one when one of
them gave it, and both combined with `&` otherwise, which also works for a flag that isn't a
`Bool`. Used on each timestep to combine the cognitive model's flag with the environment's.
"""
combine_parameters_valid(::ValidByDefault, ::ValidByDefault) = ValidByDefault()
combine_parameters_valid(::ValidByDefault, parameters_valid) = parameters_valid
combine_parameters_valid(parameters_valid, ::ValidByDefault) = parameters_valid
function combine_parameters_valid(first_parameters_valid, second_parameters_valid)
    return first_parameters_valid & second_parameters_valid
end

#A compiler like Reactant.jl gives a traced Bool, whose value isn't known when the code is compiled, so nothing can branch on it. The package defines no such type, and recognizes one by whether ifelse takes it as a condition
"""
    masks_flag(parameters_valid)

Return whether `parameters_valid`, what a model gave in its [`ModelOutput`](@ref), must be
masked with `ifelse` instead of branched on: `false` for [`ValidByDefault`](@ref), and else
what [`masks_bool`](@ref) says. Used by the simulation loops, which mask the timesteps from
the first whose flag must be masked.
"""
masks_flag(::ValidByDefault) = false
masks_flag(parameters_valid) = masks_bool(parameters_valid)

"""
    masks_bool(value)

Return whether `value`, what a [`stopping_criterion`](@ref) returned or the flag that
[`parameters_are_valid`](@ref) read, must be masked with `ifelse` instead of branched on:
`false` for a `Bool`, and `true` for a flag that `ifelse` takes as a condition, such as a
traced Bool of Reactant.jl. Anything else is an `ArgumentError`. It has no method for
[`ValidByDefault`](@ref), which such a value never is, so that the compiler knows that a
value it doesn't mask is a `Bool`. Used by the simulation loops and by
[`masks_flag`](@ref).
"""
masks_bool(::Bool) = false
function masks_bool(value)
    applicable(ifelse, value, true, false) && return true
    return throw(
        ArgumentError(
            "parameters_valid and stopping_criterion must give a Bool, or a Bool that a compiler like Reactant.jl traces, which ifelse takes as a condition, but one gave $(repr(value)) of type $(typeof(value)).",
        ),
    )
end
