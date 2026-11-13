## Checking a model's own methods ##
"""
    check_type_attribute_names(attributes, model)

Return `attributes` if it is the [`ModelAttributes`](@ref) that a method of
[`attribute_names`](@ref) must return, and throw an informative error otherwise. `model` is
the cognitive model or the environment model whose method returned it. Used wherever the
names of a model are read: by the checks of simulations, instantiated models, save histories
and fitted sessions, and by the display of a model.
"""
function check_type_attribute_names(attributes::ModelAttributes, ::AbstractTimestepModel)
    return attributes
end

function check_type_attribute_names(attributes, model::AbstractTimestepModel)
    return error(
        "The attribute_names method of $(describe_model(model)) must return a ModelAttributes, but it returned a value of type $(typeof(attributes)).",
    )
end

"""
    check_model_output(output, model)

Return `output` if it is the `ModelOutput` that an `apply_timestep_model` method of `model`
must return, and throw an error otherwise. Used on each timestep of [`simulate`](@ref),
[`simulate!`](@ref) and fitted sessions.
"""
function check_model_output(output::ModelOutput, ::AbstractTimestepModel)
    return output
end

#Untyped, so that there is always a method, whatever the model is inferred as
function check_model_output(output, model)
    return error(
        "The apply_timestep_model method of $(describe_model(model)) must return a CognitiveModeling.ModelOutput, but it returned a value of type $(typeof(output)).",
    )
end

"""
    has_attribute_names(model)

Return whether `model`, a cognitive model or an environment model, has its own method of
[`attribute_names`](@ref), instead of the fallback which throws. Used by the display of a
model, which shows one without a method of its own as `not specified`.
"""
function has_attribute_names(model::AbstractTimestepModel)
    method_signature = which(attribute_names, (typeof(model),)).sig
    return method_signature != Tuple{typeof(attribute_names),AbstractTimestepModel}
end
