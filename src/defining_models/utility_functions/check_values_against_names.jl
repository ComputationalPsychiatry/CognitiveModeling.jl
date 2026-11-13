## Checking values against a model's names ##
"""
    check_match_attribute_names(model, parameters, state, previous_output)

Throw an informative error if the names of the parameters, of the previous output or of the
state aren't those in the attributes of `model`, a cognitive model or an environment model,
from [`attribute_names`](@ref), or if the previous output or the state isn't a
`NamedTuple`. Used by [`InstantiatedModel`](@ref), [`simulate`](@ref) and fitted sessions,
before anything runs.
"""
function check_match_attribute_names(
    model::AbstractTimestepModel, parameters, state, previous_output
)
    attributes = check_type_attribute_names(attribute_names(model), model)
    if !same_names(keys(parameters), attributes.parameters)
        throw(
            ArgumentError(
                "The parameters are named $(keys(parameters)), but the parameters in the attributes of $(describe_model(model)) are $(attributes.parameters).",
            ),
        )
    end
    if !(
        previous_output isa NamedTuple &&
        same_names(keys(previous_output), attributes.outputs)
    )
        throw(
            ArgumentError(
                "The previous output is $previous_output, but it must be a NamedTuple with the outputs in the attributes of $(describe_model(model)), $(attributes.outputs).",
            ),
        )
    end
    if !(state isa NamedTuple)
        throw(
            ArgumentError(
                "The state has type $(typeof(state)), but it must be a NamedTuple with the states in the attributes of $(describe_model(model)), $(attributes.states), also when it is a single object.",
            ),
        )
    end
    if !same_names(keys(state), attributes.states)
        throw(
            ArgumentError(
                "The state is named $(keys(state)), but the states in the attributes of $(describe_model(model)) are $(attributes.states).",
            ),
        )
    end
    return nothing
end

"""
    check_input_names(model, input)

Throw an informative error if `input`, given to a model without the other side of a
simulation, isn't a `NamedTuple` with every input in the attributes of `model`, from
[`attribute_names`](@ref). It can have more than those. Used by [`simulate`](@ref) and
[`simulate!`](@ref) on a single input.
"""
function check_input_names(model::AbstractTimestepModel, input)
    attributes = check_type_attribute_names(attribute_names(model), model)
    if !(input isa NamedTuple && has_every_name(attributes.inputs, keys(input)))
        throw(
            ArgumentError(
                "The input is $input, but it must be a NamedTuple with the inputs in the attributes of $(describe_model(model)), $(attributes.inputs).",
            ),
        )
    end
    return nothing
end
