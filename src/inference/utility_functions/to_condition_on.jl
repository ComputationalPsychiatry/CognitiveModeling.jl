## Values to condition on ##

"""
    find_outputs_to_condition_on(data_output, output_distribution, model)

Return whether each output of `model` on one timestep has a value to condition on in the
data, as a `NamedTuple` of `Bool`s with the names of `data_output`: the actions of a
cognitive model, or the observations an environment model gives. Such a value is scored
against the distribution the model gave for it in `output_distribution`, so it is an error
when the model gave nothing for it, [`NoValue`](@ref). Used by the timesteps of fitted
sessions.
"""
function find_outputs_to_condition_on(
    data_output::NamedTuple{names}, output_distribution::NamedTuple, model
) where {names}
    return NamedTuple{names}(
        find_outputs_to_condition_on(names, Tuple(data_output), output_distribution, model)
    )
end

function find_outputs_to_condition_on(
    names::Tuple{}, data_values::Tuple{}, output_distribution, model
)
    return ()
end
function find_outputs_to_condition_on(
    names::Tuple, data_values::Tuple, output_distribution, model
)
    name = first(names)
    value = first(data_values)
    to_condition_on = is_value_to_condition_on(value)
    if to_condition_on && getproperty(output_distribution, name) isa AbstractNoValue
        throw(
            ArgumentError(
                "$(uppercasefirst(describe_model(model))) gave no output $name on a timestep where the data has $value for it.",
            ),
        )
    end
    return (
        to_condition_on,
        find_outputs_to_condition_on(
            Base.tail(names), Base.tail(data_values), output_distribution, model
        )...,
    )
end

"""
    initialize_output_distributions_for_conditioning(distributions_for_conditioning)

Return an empty vector for the distributions of each output that has values to condition on
in a session, typed by the distribution the model gave on the first timestep, so that the
vectors keep one type through the session when every timestep gives the same type. An output
the model doesn't give on the first timestep gets a `Vector{Union{}}()`, which the first
distribution kept sets the type of, as for [`push_value!`](@ref). Used by fitted sessions,
after their first timestep.
"""
function initialize_output_distributions_for_conditioning(
    distributions_for_conditioning::NamedTuple
)
    return map(
        distribution -> if distribution isa AbstractNoValue
            Vector{Union{}}()
        else
            Vector{typeof(distribution)}()
        end, distributions_for_conditioning
    )
end

"""
    store_output_distributions_for_conditioning(distributions,
                                                distributions_for_conditioning,
                                                outputs_to_condition_on)

Add the distributions of the outputs to condition on at one timestep, which
`outputs_to_condition_on` says, to the vectors in `distributions`, one for each output that
has values to condition on in the session, and return them. A distribution of another type
than its vector widens the vector, as for [`push_value!`](@ref). Used by fitted sessions, on
every timestep.
"""
function store_output_distributions_for_conditioning(
    distributions::NamedTuple{names},
    distributions_for_conditioning::NamedTuple,
    outputs_to_condition_on::NamedTuple,
) where {names}
    return NamedTuple{names}(
        map(
            name -> if getproperty(outputs_to_condition_on, name)
                push_value!(
                    getproperty(distributions, name),
                    getproperty(distributions_for_conditioning, name),
                )
            else
                getproperty(distributions, name)
            end,
            names,
        ),
    )
end
