### NAMES OF THE ENTRIES, WITH FLEXICHAINS 0.6 ###

#FlexiChains.jl 0.6, which Turing.jl needs, names its entries with the VarName of AbstractPPL.jl, made with the macro @varname.
#This file can go when Turing.jl moves to VarNames.jl and FlexiChains.jl 0.7
using FlexiChains: @varname

"""
    entry_name(names::Symbol...)

Return the name of an entry in the chain, which reads `names` one after the other, as in
`P1.actions.choice`, made with the macro of the installed FlexiChains.jl. Used by
[`add_session_entries!`](@ref) and [`session_values_chain`](@ref).
"""
entry_name(first_name::Symbol) = @varname($first_name)
entry_name(first_name::Symbol, second_name::Symbol) = @varname($first_name.$second_name)
function entry_name(first_name::Symbol, second_name::Symbol, third_name::Symbol)
    @varname($first_name.$second_name.$third_name)
end
