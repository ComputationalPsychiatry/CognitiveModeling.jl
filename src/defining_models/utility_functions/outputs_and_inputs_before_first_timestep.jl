## Outputs and inputs before the first timestep ##
"""
    output_before_first_timestep(model)
    input_before_first_timestep(model)
    output_before_first_timestep(attributes)
    input_before_first_timestep(attributes)

Return the output and the input from before the first timestep: a `NamedTuple` with the
output or the input names in the attributes of `model`, a cognitive model or an environment
model, from [`attribute_names`](@ref), or those that `attributes`, a
[`ModelAttributes`](@ref), has, each [`BeforeFirstTimestep`](@ref). A model's names are
checked to be a `ModelAttributes` first.
Used wherever a model starts: by [`InstantiatedModel`](@ref), [`reset!`](@ref),
[`simulate`](@ref) and fitted sessions, for the previous output and for the output
distributions saved as timestep 0.
"""
function output_before_first_timestep(model::AbstractTimestepModel)
    return output_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

function input_before_first_timestep(model::AbstractTimestepModel)
    return input_before_first_timestep(
        check_type_attribute_names(attribute_names(model), model)
    )
end

function output_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Toutputs}(map(_ -> BeforeFirstTimestep(), Toutputs))
end

function input_before_first_timestep(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return NamedTuple{Tinputs}(map(_ -> BeforeFirstTimestep(), Tinputs))
end
