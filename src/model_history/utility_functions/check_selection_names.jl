## Checking the names a save history selects ##

"""
    check_savehistory_selection_names(save_history, model)

Throw an informative error if `save_history` selects a state, an input, an output or an
output distribution by name that isn't in the attributes of `model`, a cognitive model or an
environment model, from [`attribute_names`](@ref), since a history starts from those names,
and [`prepare_state_for_saving`](@ref) returns a state with them. Used by
[`InstantiatedModel`](@ref), [`simulate_session`](@ref) and [`SessionInfo`](@ref), before
anything is saved.
"""
function check_savehistory_selection_names(
    save_history::SaveHistory, model::AbstractTimestepModel
)
    attributes = check_type_attribute_names(attribute_names(model), model)
    check_selection_names(save_history.states, attributes.states, "states", model)
    check_selection_names(save_history.inputs, attributes.inputs, "inputs", model)
    check_selection_names(save_history.outputs, attributes.outputs, "outputs", model)
    check_selection_names(
        save_history.output_distributions, attributes.outputs, "output distributions", model
    )
    return nothing
end

"""
    check_selection_names(selection, names, kind, model)

Throw an informative error if `selection` has a name that isn't among `names`, those of
`kind` in the attributes of `model`, where `kind` is `"states"`, `"inputs"`, `"outputs"` or
`"output distributions"`, the last being named like the outputs. [`SaveEverything`](@ref)
can always be selected. Used by [`check_savehistory_selection_names`](@ref) for each kind.
"""
check_selection_names(::SaveEverything, names, kind, model) = nothing
function check_selection_names(selection::Tuple, names, kind, model)
    if !has_every_name(selection, names)
        throw(
            ArgumentError(
                "The SaveHistory selects the $kind $selection, but the $(kind == "output distributions" ? "outputs" : kind) in the attributes of $(describe_model(model)) are $names. A history can only select names in the model's attributes.",
            ),
        )
    end
    return nothing
end
