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
