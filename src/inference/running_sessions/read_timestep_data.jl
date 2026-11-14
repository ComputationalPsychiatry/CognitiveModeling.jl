## Reading the data of one timestep ##

#A call of its own, so that a value which can have two types gives one of two NamedTuples with a concrete field each: (; a = x) written out is inferred as a NamedTuple whose field type is the Union, which nothing is compiled for, while a call with a Union-typed argument is split by type
"""
    build_named_entry(::Val{name}, value)

Return `(; name = value)`. A `NamedTuple` which holds a value that can have two types is
built with `merge` of these, as [`select_row`](@ref) does. Used by [`select_row`](@ref), by
[`select_timestep_values_to_save`](@ref) and by the timesteps of fitted sessions.
"""
build_named_entry(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    select_row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of
vectors with one entry for each timestep, built with [`build_named_entry`](@ref) so that a
column which holds two types, such as an action that is [`MissingValue`](@ref) on some
timesteps, gives one of two `NamedTuple`s with concrete fields rather than one without a
definite type. Used by the timesteps of fitted sessions, and by the models of several
sessions, which read the parameters of one session with it.
"""
select_row(columns::NamedTuple{()}, t) = (;)
function select_row(columns::NamedTuple{names}, t) where {names}
    return merge(
        build_named_entry(Val(first(names)), first(columns)[t]),
        select_row(Base.tail(columns), t),
    )
end

"""
    has_missing_value(data)

Return whether any entry of `data`, the `NamedTuple` of the actions or the observations of
one timestep, is [`MissingValue`](@ref). Used by the timesteps of fitted sessions, to skip
imputing when nothing is missing.
"""
has_missing_value(data::NamedTuple) = any(value -> value isa MissingValue, Tuple(data))
