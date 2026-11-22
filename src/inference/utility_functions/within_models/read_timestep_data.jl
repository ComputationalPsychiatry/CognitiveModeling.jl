## Reading the data of one timestep ##

#A call of its own, so that a value which can have two types gives one of two NamedTuples with a concrete field each: (; a = x) written out is inferred as a NamedTuple whose field type is the Union, which nothing is compiled for, while a call with a Union-typed argument is split by type
"""
    build_named_entry(::Val{name}, value)

Return `(; name = value)`. A `NamedTuple` which holds a value that can have two types is
built with `merge` of these. Used by [`select_timestep_values_to_save`](@ref) and by the
timesteps of fitted sessions.
"""
build_named_entry(::Val{name}, value) where {name} = NamedTuple{(name,)}((value,))

"""
    select_row(columns, t)

Return the `NamedTuple` of the values at timestep `t` of `columns`, a `NamedTuple` of
vectors with one entry for each timestep. The values are read as a tuple with
[`read_row_values`](@ref), so that a column which holds two types, such as an action that is
[`ImputedMissingValue`](@ref) on some timesteps, gives one of two `NamedTuple`s with
concrete fields rather than one without a definite type. Used by the timesteps of fitted
sessions, by the models of several sessions, which read the parameters of one session with
it, and by a [`FixedObservationsEnvironment`](@ref), which reads the observations of a
timestep.
"""
function select_row(columns::NamedTuple{names}, t) where {names}

    #The values are read as a tuple, which keeps the type of the row with any number of columns. A recursive merge of NamedTuples isn't inferred from three columns on, so that every read allocated, and Enzyme aborted inside the sessions of a Turing.jl model
    return NamedTuple{names}(read_row_values(Tuple(columns), t))
end

"""
    read_row_values(columns, t)

Return the values at timestep `t` of `columns`, a tuple of vectors, as a tuple. Each value
is put in front of the values of the remaining columns by [`prepend_value`](@ref), a call
of its own, so that a value which can have two types gives one of two tuples with concrete
elements, since the compiler splits a call over the types of its arguments. Used by
[`select_row`](@ref).
"""
read_row_values(columns::Tuple{}, t) = ()
function read_row_values(columns::Tuple, t)
    return prepend_value(first(columns)[t], read_row_values(Base.tail(columns), t))
end

"""
    prepend_value(value, values)

Return the tuple `values` with `value` in front. Used by [`read_row_values`](@ref).
"""
prepend_value(value, values::Tuple) = (value, values...)

"""
    has_missing_value(data)

Return whether any entry of `data`, the `NamedTuple` of the actions or the observations of
one timestep, is [`ImputedMissingValue`](@ref) or [`SimulatedMissingValue`](@ref). Used by
the timesteps of fitted sessions, to skip imputing when nothing is missing.
"""
has_missing_value(data::NamedTuple) = any(value -> value isa LatentValueMarker, Tuple(data))
