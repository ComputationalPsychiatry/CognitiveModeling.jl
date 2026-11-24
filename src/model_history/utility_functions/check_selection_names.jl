## Checking the names a save history selects ##

"""
    check_savehistory_selection_names(save_history, model)

Throw an informative error if `save_history` selects a state, an input, an output or an
output distribution by name that isn't in the attributes of `model`, a cognitive model or an
environment model, as [`get_attribute_names`](@ref) gives them with the name of the state of
each submodel, since a history starts from those names, and
[`prepare_state_for_saving`](@ref) returns a state with them. The states that it selects of
a submodel are checked against the attributes of the submodel, as
[`check_state_selection_names`](@ref) describes. Used by [`InstantiatedModel`](@ref),
[`simulate_session`](@ref) and [`SessionInfo`](@ref), before anything is saved.
"""
function check_savehistory_selection_names(
    save_history::SaveHistory, model::AbstractTimestepModel
)
    attributes = get_attribute_names(model)
    check_state_selection_names(save_history.states, model, ())
    check_selection_names(save_history.inputs, attributes.inputs, "inputs", model)
    check_selection_names(save_history.outputs, attributes.outputs, "outputs", model)
    check_selection_names(
        save_history.output_distributions, attributes.outputs, "output distributions", model
    )
    return nothing
end

"""
    check_state_selection_names(selection, model, submodel_names)

Throw an informative error if the states that a [`SaveHistory`](@ref) selects of `model`,
in `selection`, as `save_history.states` gives them, aren't states in its attributes, as
[`get_attribute_names`](@ref) gives them, if one is selected twice, or if states are
selected of one that isn't the state of a submodel. The states that are selected of a
submodel are checked against the submodel in the same way, down through its submodels, with
`submodel_names` the names of the submodels that lead to `model`, which are none for the
model of the `SaveHistory` itself. [`SaveEverything`](@ref) can always be selected. Used by
[`check_savehistory_selection_names`](@ref).
"""
check_state_selection_names(::SaveEverything, model, submodel_names) = nothing
function check_state_selection_names(
    selection::Tuple, model::AbstractSubmodel, submodel_names::Tuple{Vararg{Symbol}}
)
    names = get_attribute_names(model).states
    selected_names = map(selected_state_name, selection)
    of_submodel =
        isempty(submodel_names) ? "" : " of the submodel $(join(submodel_names, "."))"

    #Every name must be a state in the attributes of the model, or the name of a submodel with a state
    if !has_every_name(selected_names, names)
        throw(
            ArgumentError(
                "The SaveHistory selects the states $selected_names$of_submodel, but the states in the attributes of $(describe_model(model)) are $names. A history can only select names in the model's attributes.$(state_of_submodel_hint(selected_names, names, model))",
            ),
        )
    end

    #Each can be selected once, so the states to save of a submodel are given together
    if !allunique(selected_names)
        given_together = if all(entry -> entry isa Symbol, selection)
            ""
        else
            " The states to save of a submodel are given together, as in :perception => (:expected_reward, :prediction_error)."
        end
        throw(
            ArgumentError(
                "The SaveHistory selects the states $selected_names$of_submodel, some of them twice.$given_together",
            ),
        )
    end

    #The states that are selected of a submodel must be states of that submodel
    submodels_of_model = submodels(model)
    for entry in selection
        entry isa Symbol && continue
        name = first(entry)
        if !haskey(submodels_of_model, name)
            throw(
                ArgumentError(
                    "The SaveHistory selects the states $(map(selected_state_name, last(entry))) of $(repr(name))$of_submodel, but $(repr(name)) isn't a submodel of $(describe_model(model)). It is a state of its own, which is selected by its name alone.",
                ),
            )
        end
        check_state_selection_names(
            last(entry), getfield(submodels_of_model, name), (submodel_names..., name)
        )
    end
    return nothing
end

"""
    state_of_submodel_hint(selected_names, names, model)

Return a sentence on how to select the first of `selected_names` that isn't among `names`,
the states in the attributes of `model`, when it is a state of one of the submodels of
`model`, and an empty string otherwise. Used by [`check_state_selection_names`](@ref), for
its error.
"""
function state_of_submodel_hint(selected_names, names, model)
    submodels_of_model = submodels(model)
    for selected_name in selected_names
        has_name(selected_name, names) && continue
        for submodel_name in keys(submodels_of_model)
            submodel = getfield(submodels_of_model, submodel_name)
            if has_name(selected_name, get_attribute_names(submodel).states)
                return " $(repr(selected_name)) is a state of the submodel $(repr(submodel_name)), which is selected with :$submodel_name => :$selected_name."
            end
        end
    end
    return ""
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
