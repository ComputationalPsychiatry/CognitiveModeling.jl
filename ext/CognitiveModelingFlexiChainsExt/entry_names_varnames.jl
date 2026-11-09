### NAMES OF THE ENTRIES, WITH FLEXICHAINS 0.7 ###

#FlexiChains.jl 0.7 names its entries with the VarName of VarNames.jl, made with the macro @vn
using FlexiChains: @vn

"""
    entry_name(names::Symbol...)

Return the name of an entry in the chain, which reads `names` one after the other, as in
`P1.choice` or `P1.environment.observation_distributions.reward`, made with the macro of the
installed FlexiChains.jl. Used by [`add_model_entries!`](@ref) and
[`add_parameter_entries!`](@ref).
"""
entry_name(first_name::Symbol) = @vn($first_name)
entry_name(first_name::Symbol, second_name::Symbol) = @vn($first_name.$second_name)
function entry_name(first_name::Symbol, second_name::Symbol, third_name::Symbol)
    @vn($first_name.$second_name.$third_name)
end
function entry_name(
    first_name::Symbol, second_name::Symbol, third_name::Symbol, fourth_name::Symbol
)
    @vn($first_name.$second_name.$third_name.$fourth_name)
end
