## Comparing names ##
"""
    has_name(name, names)
    has_every_name(names, other_names)

Return whether `name` is among the tuple `names`, and whether every name in the tuple
`names` is among `other_names`. Used by [`unique_names`](@ref), and by the checks that a
name map, a save history or the data has names in a model's attributes.
"""
has_name(name::Symbol, names::Tuple{}) = false
function has_name(name::Symbol, names::Tuple)
    return name === first(names) || has_name(name, Base.tail(names))
end
has_every_name(names::Tuple{}, other_names::Tuple) = true
function has_every_name(names::Tuple, other_names::Tuple)
    return has_name(first(names), other_names) &&
           has_every_name(Base.tail(names), other_names)
end

"""
    same_names(names, other_names)

Return whether two tuples of names have the same names, in any order. Used by the checks of
names in simulations and fitted sessions, such as [`check_names_fit`](@ref) and
[`check_match_attribute_names`](@ref).
"""
function same_names(names, other_names)
    return has_every_name(names, other_names) && has_every_name(other_names, names)
end
