## The defaults and the domains of the parameters of a model with submodels ##

#Generated, so that the names are known when the code is compiled
"""
    prefixed_names(::Val{prefix}, values::NamedTuple)

Return `values` with `prefix` and `__` before the name of each entry, as the flat names of
the parameters of a submodel named `prefix` start when its model uses prefixes. Used by
[`collect_parameter_values`](@ref).
"""
@generated function prefixed_names(
    ::Val{prefix}, values::NamedTuple{names}
) where {prefix,names}
    new_names = map(name -> Symbol(prefix, SUBMODEL_PREFIX_SEPARATOR, name), names)
    return :(NamedTuple{$new_names}(Tuple(values)))
end

"""
    collect_parameter_values(read_values, model::AbstractSubmodel)

Return what `read_values` gives for `model` and for each of its submodels, a `NamedTuple`
with an entry under the name of each of their own parameters that has one, together in one
`NamedTuple` under the flat names of the parameters, as [`get_attribute_names`](@ref) gives
them: the model's own first, and then those of each submodel, in the same way down through
its submodels. Used by [`collected_default_parameters`](@ref) and
[`collected_parameter_domains`](@ref).
"""
function collect_parameter_values(read_values::F, model::AbstractSubmodel) where {F}
    submodels_of_model = submodels(model)
    return merge(
        read_values(model)::NamedTuple,
        prefixed_each(
            Val(keys(submodels_of_model)),
            Val(uses_submodel_prefixes(model)),
            values_of_submodels(read_values, values(submodels_of_model)),
        )...,
    )
end

#One submodel at a time off the front of the tuple, without their names, as for model_layout: Julia limits how far it
#infers a function that calls itself, and a model whose submodels have submodels runs into that limit when the names are
#peeled off with them
"""
    values_of_submodels(read_values, submodels_of_model::Tuple)

Return, in a tuple, what [`collect_parameter_values`](@ref) gives for each submodel in
`submodels_of_model`, in the same order.
"""
values_of_submodels(read_values, ::Tuple{}) = ()
function values_of_submodels(read_values::F, submodels_of_model::Tuple) where {F}
    return (
        collect_parameter_values(read_values, first(submodels_of_model))::NamedTuple,
        values_of_submodels(read_values, Base.tail(submodels_of_model))...,
    )
end

"""
    prefixed_each(::Val{names}, ::Val{prefixes}, values_of_each::Tuple)

Return `values_of_each`, what [`collect_parameter_values`](@ref) gives for each submodel,
with the names of each prefixed by the submodel's name in `names` and `__`, with
[`prefixed_names`](@ref), when `prefixes` is `true`, and as they are otherwise.
"""
prefixed_each(::Val, ::Val{false}, values_of_each::Tuple) = values_of_each
function prefixed_each(::Val{names}, ::Val{true}, values_of_each::Tuple) where {names}
    return ntuple(i -> prefixed_names(Val(names[i]), values_of_each[i]), Val(length(names)))
end

"""
    collected_default_parameters(model::AbstractSubmodel)

Return the default values of the parameters of `model` and of its submodels, from
[`default_parameters`](@ref) of each, under the flat names of the parameters, without
checking the submodels. Used by [`get_default_parameters`](@ref), which checks them first,
and by [`merge_default_parameters`](@ref).
"""
function collected_default_parameters(model::AbstractSubmodel)
    return collect_parameter_values(default_parameters, model)
end

"""
    collected_parameter_domains(model::AbstractSubmodel)

Return the [`Domain`](@ref)s of the parameters of `model` and of its submodels, from
[`parameter_domains`](@ref) of each, which are checked against the names of each, under the
flat names of the parameters, without checking the submodels. Used by
[`get_parameter_domains`](@ref), which checks them first, and by
[`discrete_parameter_names`](@ref).
"""
function collected_parameter_domains(model::AbstractSubmodel)
    return collect_parameter_values(checked_parameter_domains, model)
end

"""
    checked_parameter_domains(model::AbstractSubmodel)

Return the domains of the parameters of `model`, from [`parameter_domains`](@ref), after
checking them against its names with [`check_domains_fit_model`](@ref). Used by
[`collected_parameter_domains`](@ref).
"""
function checked_parameter_domains(model::AbstractSubmodel)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    return domains
end

## Checks on each submodel ##

"""
    check_parameters_of_submodels_in_domains(model::AbstractSubmodel, parameters::NamedTuple)

Throw a [`ParameterDomainError`](@ref) if a parameter of `model` or of one of its submodels
is outside its domain, from [`parameter_domains`](@ref) of each, where `parameters` are the
parameters of `model` as [`bundle_parameters`](@ref) gives them. Each submodel is checked
with its own parameters, so that an error names the submodel. Used by
[`check_parameters_in_domains`](@ref).
"""
function check_parameters_of_submodels_in_domains(
    model::AbstractSubmodel, parameters::NamedTuple
)
    domains = parameter_domains(model)
    check_domains_fit_model(model, domains)
    check_each_parameter_in_domain(model, parameters, domains)
    submodels_of_model = submodels(model)
    check_each_submodel_in_domains(
        Val(keys(submodels_of_model)), values(submodels_of_model), parameters
    )
    return nothing
end
"""
    check_each_submodel_in_domains(::Val{names}, submodels_of_model::Tuple, parameters)

Check the parameters of each submodel in `submodels_of_model`, named in `names`, which
`parameters` has under its name, with [`check_parameters_of_submodels_in_domains`](@ref),
one submodel at a time off the front of the tuple.
"""
check_each_submodel_in_domains(::Val{()}, ::Tuple{}, parameters) = nothing
function check_each_submodel_in_domains(
    ::Val{names}, submodels_of_model::Tuple, parameters
) where {names}
    check_parameters_of_submodels_in_domains(
        first(submodels_of_model), getfield(parameters, first(names))
    )
    return check_each_submodel_in_domains(
        Val(Base.tail(names)), Base.tail(submodels_of_model), parameters
    )
end

"""
    check_submodel_states(model::AbstractSubmodel, state::NamedTuple)

Throw an informative error if the state of a submodel of `model` with a state, which `state`
keeps under the submodel's name, isn't a `NamedTuple` with the names of the states that
[`get_attribute_names`](@ref) gives for the submodel, in the same way down through its
submodels. Used by [`check_match_attribute_names`](@ref), after the names of `state` itself
are checked.
"""
function check_submodel_states(model::AbstractSubmodel, state::NamedTuple)
    submodels_of_model = submodels(model)
    check_each_submodel_state(
        model, Val(keys(submodels_of_model)), values(submodels_of_model), state
    )
    return nothing
end
"""
    check_each_submodel_state(model, ::Val{names}, submodels_of_model::Tuple, state)

Check the state that `state` keeps under the name of each submodel of `model` with a state,
in `submodels_of_model`, named in `names`, one submodel at a time off the front of the
tuple. Used by [`check_submodel_states`](@ref).
"""
check_each_submodel_state(model, ::Val{()}, ::Tuple{}, state) = nothing
function check_each_submodel_state(
    model, ::Val{names}, submodels_of_model::Tuple, state
) where {names}
    name = first(names)
    submodel = first(submodels_of_model)
    states_of_submodel = get_attribute_names(submodel).states
    if !isempty(states_of_submodel)
        submodel_state = getfield(state, name)
        if !(
            submodel_state isa NamedTuple &&
            same_names(keys(submodel_state), states_of_submodel)
        )
            throw(
                ArgumentError(
                    "The state of $(describe_model(model)) has $(submodel_state isa NamedTuple ? "names $(keys(submodel_state))" : "a value of type $(typeof(submodel_state))") under $(repr(name)), but that must be a NamedTuple with the states of its submodel $(repr(name)), $(describe_model(submodel)), which are $states_of_submodel.",
                ),
            )
        end
        check_submodel_states(submodel, submodel_state)
    end
    return check_each_submodel_state(
        model, Val(Base.tail(names)), Base.tail(submodels_of_model), state
    )
end

## Saving the states of submodels ##

"""
    prepare_submodel_states_for_saving(model::AbstractSubmodel, state)

Return `state` with the state of each submodel of `model` that has one, which `state` keeps
under the submodel's name, as [`prepare_state_for_saving`](@ref) of that submodel returns
it, so that a submodel which changes its state in place has its own method used. A model
without submodels gets its state as it is. Used by the default method of
`prepare_state_for_saving`.
"""
function prepare_submodel_states_for_saving(model::AbstractSubmodel, state)
    submodels_of_model = submodels(model)
    isempty(submodels_of_model) && return state
    return merge(
        state,
        prepared_states_of_submodels(
            Val(keys(submodels_of_model)), values(submodels_of_model), state
        ),
    )
end
"""
    prepared_states_of_submodels(::Val{names}, submodels_of_model::Tuple, state)

Return, in a `NamedTuple` under their names, the states that `state` keeps of the submodels
in `submodels_of_model`, named in `names`, each as [`prepare_state_for_saving`](@ref) of its
submodel returns it, leaving out submodels without a state. Used by
[`prepare_submodel_states_for_saving`](@ref).
"""
prepared_states_of_submodels(::Val{()}, ::Tuple{}, state) = (;)
function prepared_states_of_submodels(
    ::Val{names}, submodels_of_model::Tuple, state
) where {names}
    name = first(names)
    others = prepared_states_of_submodels(
        Val(Base.tail(names)), Base.tail(submodels_of_model), state
    )
    hasfield(typeof(state), name) || return others
    prepared = prepare_state_for_saving(first(submodels_of_model), getfield(state, name))
    return merge(NamedTuple{(name,)}((prepared,)), others)
end
