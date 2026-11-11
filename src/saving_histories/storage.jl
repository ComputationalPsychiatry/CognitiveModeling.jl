"""
    initialize_storage(first_value, n_steps)
    initialize_storage(initial_value, first_value, n_steps)

Create the vector for saving one kind of value over `n_steps` timesteps, using the value of
the first timestep to set its type. With `initial_value`, what is saved from before the
first timestep, the vector starts with it, as timestep 0, and its type holds both. Used by
simulations and by fitted sessions.
"""
function initialize_storage(first_value, n_steps)

    #Create the vector and put the value in it
    storage = Vector{typeof(first_value)}(undef, n_steps)
    storage[1] = first_value

    return storage
end

function initialize_storage(initial_value, first_value, n_steps)

    #Create the vector with room for timestep 0, and put both values in it
    storage = Vector{Union{typeof(initial_value),typeof(first_value)}}(undef, n_steps + 1)
    storage[1] = initial_value
    storage[2] = first_value

    return storage
end

"""
    store_value!(storage, value, t)

Save `value` as element `t` of `storage`, and return the vector.

If the value doesn't fit the type of `storage`, the values so far are moved to a wider
vector which can hold both types, and that vector is returned instead. Used by simulations
and by fitted sessions.
"""
function store_value!(storage::Vector{T}, value, t) where {T}

    #If it has the type of the vector, which the compiler can usually see, store it
    if value isa T
        storage[t] = value
        return storage
    end

    #Otherwise, move the values so far to a wider vector which can hold both types
    wider_storage = Vector{Union{T,typeof(value)}}(undef, length(storage))
    copyto!(wider_storage, 1, storage, 1, t - 1)
    wider_storage[t] = value

    return wider_storage
end

"""
    push_value!(storage, value)

Add `value` at the end of `storage`, and return the vector. As for [`store_value!`](@ref), a
value that doesn't fit the type of `storage` moves the values so far to a wider vector,
which is returned instead. Starting from `Vector{Union{}}()` lets the first value set the
type. Used by fitted sessions.
"""
function push_value!(storage::Vector{T}, value) where {T}

    #If it has the type of the vector, add it
    if value isa T
        push!(storage, value)
        return storage
    end

    #Otherwise, move the values so far to a wider vector which can hold both types
    wider_storage = Vector{Union{T,typeof(value)}}(undef, length(storage) + 1)
    copyto!(wider_storage, 1, storage, 1, length(storage))
    wider_storage[end] = value

    return wider_storage
end
