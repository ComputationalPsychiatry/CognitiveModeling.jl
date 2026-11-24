## The names of a model with submodels ##

"""
    get_attribute_names(model; with_default_parameters=true)::ModelAttributes

Return the names of the attributes of `model` together with those of its submodels, in a
[`ModelAttributes`](@ref): the parameters of the model and of every submodel, under the flat
names that they are given with, which start with the submodel's name and `__` when the model
uses prefixes, as [`CognitiveModeling.uses_submodel_prefixes`](@ref) says, the model's own
states and the name of each submodel that has a state, under which that state is kept, and
the model's own inputs and outputs. For a model without submodels, these are the names that
[`attribute_names`](@ref) gives.

These are the names that a model is given its parameters and its state with, so they are
the ones to give priors and fixed values for. With `with_default_parameters = false`, the
parameters with a default value in [`default_parameters`](@ref) of the model or of its
submodel are left out, which leaves those that a fit needs priors or fixed values for.
Before the names are collected, the submodels are checked, as [`check_submodels`](@ref)
describes.

# Examples
```jldoctest
julia> struct Perception <: AbstractSubmodel end

julia> CognitiveModeling.attribute_names(::Perception) = ModelAttributes(;
           parameters = :learning_rate, states = :expected_reward, inputs = (), outputs = ());

julia> struct Response <: AbstractSubmodel end

julia> CognitiveModeling.attribute_names(::Response) = ModelAttributes(;
           parameters = :inverse_temperature, states = (), inputs = (), outputs = ());

julia> CognitiveModeling.default_parameters(::Response) = (; inverse_temperature = 1.0);

julia> struct PerceptionAndResponse <: AbstractCognitiveModel
           perception::Perception
           response::Response
       end

julia> CognitiveModeling.submodel_fields(::PerceptionAndResponse) = (:perception, :response);

julia> CognitiveModeling.attribute_names(::PerceptionAndResponse) = ModelAttributes(;
           parameters = :lapse_rate, states = (), inputs = :reward, outputs = :choice);

julia> model = PerceptionAndResponse(Perception(), Response());

julia> CognitiveModeling.get_attribute_names(model)
ModelAttributes
Parameters: lapse_rate, learning_rate, inverse_temperature
States: perception
Inputs: reward
Outputs: choice

julia> CognitiveModeling.get_attribute_names(model; with_default_parameters = false)
ModelAttributes
Parameters: lapse_rate, learning_rate
States: perception
Inputs: reward
Outputs: choice
```
"""
function get_attribute_names(model::AbstractSubmodel; with_default_parameters::Bool=true)
    return checked_attribute_names(model, Val(with_default_parameters))
end

#A method for each value of the keyword, so that the names with every parameter, which the checks of a simulation and
#of a fitted session read, compile to the same code as without the keyword
"""
    checked_attribute_names(model::AbstractSubmodel, ::Val{with_default_parameters})

Return the names that [`get_attribute_names`](@ref) gives for `model`, after checking its
submodels with [`check_submodels`](@ref): with every parameter for `Val(true)`, and without
the parameters that have a default value for `Val(false)`, as
[`without_default_parameters`](@ref) leaves them out.
"""
function checked_attribute_names(model::AbstractSubmodel, ::Val{true})
    check_submodels(model)
    return collected_attribute_names(model_layout(model))
end
function checked_attribute_names(model::AbstractSubmodel, ::Val{false})
    return without_default_parameters(
        checked_attribute_names(model, Val(true)), collected_default_parameters(model)
    )
end

#Generated, so that the names are known when the code is compiled, as those with the default parameters are
"""
    without_default_parameters(attributes::ModelAttributes, defaults::NamedTuple)

Return `attributes` without the parameters that have a value in `defaults`, the default
values of the parameters under their flat names, as
[`collected_default_parameters`](@ref) gives them. Used by
[`checked_attribute_names`](@ref), for [`get_attribute_names`](@ref) with
`with_default_parameters = false`.
"""
@generated function without_default_parameters(
    ::ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}, ::NamedTuple{default_names}
) where {Tparameters,Tstates,Tinputs,Toutputs,default_names}
    parameters = Tuple(name for name in Tparameters if !(name in default_names))
    return :(ModelAttributes{$parameters,$Tstates,$Tinputs,$Toutputs}())
end

"""
    attribute_names_in_type(::Type{<:ModelAttributes})

Return the names of each kind in a type of [`ModelAttributes`](@ref), in a `NamedTuple` with
`parameters`, `states`, `inputs` and `outputs`. Used while code that reads a
[`ModelLayout`](@ref) is generated.
"""
function attribute_names_in_type(
    ::Type{ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return (; parameters=Tparameters, states=Tstates, inputs=Tinputs, outputs=Toutputs)
end

"""
    has_state(::Type{<:ModelLayout})

Return whether a model of a layout type has a state: states of its own, or a submodel that
has a state. Used while the code of [`collected_attribute_names`](@ref) is generated.
"""
function has_state(
    ::Type{ModelLayout{Tattributes,prefixes,Tsubmodels}}
) where {Tattributes,prefixes,Tsubmodels}
    return !isempty(attribute_names_in_type(Tattributes).states) ||
           any(has_state, fieldtypes(Tsubmodels))
end

#Generated, so that the names are known when the code is compiled, as those of a model without submodels are
"""
    collected_attribute_names(layout::ModelLayout)

Return the names that [`get_attribute_names`](@ref) gives for a model of the
[`ModelLayout`](@ref) `layout`, without checking them.
"""
@generated function collected_attribute_names(
    layout::ModelLayout{Tattributes,prefixes,Tsubmodels}
) where {Tattributes,prefixes,Tsubmodels}
    own = attribute_names_in_type(Tattributes)
    parameters = Tuple(first.(parameter_paths(layout)))
    states = (
        own.states...,
        (
            name for
            (name, Tsubmodel) in zip(fieldnames(Tsubmodels), fieldtypes(Tsubmodels)) if
            has_state(Tsubmodel)
        )...,
    )
    return :(ModelAttributes{$parameters,$states,$(own.inputs),$(own.outputs)}())
end

## The states of the submodels of a model ##

#Generated, so that the names are known when the code is compiled, and the trajectories that hold them have one type
"""
    submodel_state_names(model::AbstractSubmodel)

Return the names of the submodels of `model` that have a state, in a `NamedTuple` with the
names of the submodels with a state of each under its name, such as `(; perception = (;))`:
the state of `model` keeps the state of each under its name, in the same way down through
the submodels. A model without submodels that have a state gives `(;)`. Used by
[`submodel_states_of`](@ref).
"""
submodel_state_names(model::AbstractSubmodel) = submodel_state_names(model_layout(model))
@generated function submodel_state_names(layout::ModelLayout)
    return submodel_state_names_expression(layout)
end

"""
    submodel_state_names_expression(::Type{<:ModelLayout})

Return the expression of the `NamedTuple` that [`submodel_state_names`](@ref) gives for a
layout type, with [`has_state`](@ref) to find the submodels with a state. Used while the
code of `submodel_state_names` is generated.
"""
function submodel_state_names_expression(
    ::Type{ModelLayout{Tattributes,prefixes,Tsubmodels}}
) where {Tattributes,prefixes,Tsubmodels}
    entries = Any[
        Expr(:kw, name, submodel_state_names_expression(Tsubmodel)) for
        (name, Tsubmodel) in zip(fieldnames(Tsubmodels), fieldtypes(Tsubmodels)) if
        has_state(Tsubmodel)
    ]
    return Expr(:tuple, Expr(:parameters, entries...))
end

"""
    submodel_states_of(cognitive_model, environment)

Return which states of `cognitive_model` and of `environment` hold the state of a
submodel, as [`SessionTrajectories`](@ref) keeps them under `submodel_states`: those of the
cognitive model under `cognitive_model`, and those of the environment under `environment`,
each from [`submodel_state_names`](@ref). Used by simulations and fitted sessions, for the
trajectories they return.
"""
function submodel_states_of(cognitive_model, environment)
    return (;
        cognitive_model=submodel_state_names(cognitive_model),
        environment=submodel_state_names(environment),
    )
end

## Checking the submodels of a model ##

"""
    check_submodels(model::AbstractSubmodel)

Throw an informative error if the submodels of `model` aren't declared or named as they must
be, and else return `nothing`. Each name that
[`CognitiveModeling.submodel_fields`](@ref) gives must be a field of `model` that holds an
[`AbstractSubmodel`](@ref), and each of [`CognitiveModeling.submodels`](@ref) must be one.
The name of a submodel must differ from the model's own names in [`attribute_names`](@ref),
since it is the name of the submodel's parameters and state. No two parameters may get the
same flat name, which happens when two submodels, or a submodel and the model, have a
parameter of the same name, and the model doesn't use prefixes: the error then says to give
it a method of [`CognitiveModeling.uses_submodel_prefixes`](@ref). Last, every name of
[`get_attribute_names`](@ref) must be used once. The submodels of each submodel are checked
in the same way. Used by `get_attribute_names`.
"""
function check_submodels(model::AbstractSubmodel)
    check_submodel_fields(model, submodel_fields(model))
    submodels_of_model = submodels(model)
    check_submodel_names(model, submodels_of_model)
    foreach(check_submodels, values(submodels_of_model))
    check_unique_flat_parameter_names(model, model_layout(model))
    check_unique_attribute_names(collected_attribute_names(model_layout(model)))
    return nothing
end

"""
    check_submodel_fields(model, fields)

Throw an informative error if a name in `fields`, from
[`CognitiveModeling.submodel_fields`](@ref), isn't a field of `model`. Used by
[`check_submodels`](@ref), before the submodels are read from the fields.
"""
check_submodel_fields(model, ::Tuple{}) = nothing
function check_submodel_fields(model, fields::Tuple)
    field = first(fields)
    if !hasfield(typeof(model), field)
        throw(
            ArgumentError(
                "CognitiveModeling.submodel_fields of $(describe_model(model)) names $(repr(field)), but it has no field $(repr(field)). Its fields are $(fieldnames(typeof(model))).",
            ),
        )
    end
    return check_submodel_fields(model, Base.tail(fields))
end

"""
    check_submodel_names(model, submodels_of_model::NamedTuple)

Throw an informative error if a submodel of `model`, in `submodels_of_model`, isn't an
[`AbstractSubmodel`](@ref), or if its name is also one of the model's own names in
[`attribute_names`](@ref). Used by [`check_submodels`](@ref).
"""
function check_submodel_names(model, submodels_of_model::NamedTuple{names}) where {names}
    attributes = attribute_names(model)
    own_names = (
        attributes.parameters...,
        attributes.states...,
        attributes.inputs...,
        attributes.outputs...,
    )
    for name in names
        submodel = getfield(submodels_of_model, name)
        if !(submodel isa AbstractSubmodel)
            throw(
                ArgumentError(
                    "The submodel $(repr(name)) of $(describe_model(model)) is a $(typeof(submodel)), which isn't an AbstractSubmodel.",
                ),
            )
        end
        if name in own_names
            throw(
                ArgumentError(
                    "$(uppercasefirst(describe_model(model))) has a submodel named $(repr(name)), which is also one of its own names in attribute_names. A submodel's name is the name of its parameters and its state, so it must differ from the model's own names.",
                ),
            )
        end
    end
    return nothing
end

#Generated, so that the check is decided when the code is compiled, and costs nothing when the names are unique
"""
    check_unique_flat_parameter_names(model, layout::ModelLayout)

Throw an informative error if two parameters of `model` and its submodels, in its
[`ModelLayout`](@ref) `layout`, have the same flat name, naming where each comes from and
saying to give the model a method of [`CognitiveModeling.uses_submodel_prefixes`](@ref).
Used by [`check_submodels`](@ref).
"""
@generated function check_unique_flat_parameter_names(model, layout::ModelLayout)
    paths = parameter_paths(layout)
    flat_names = first.(paths)
    repeated = unique(filter(name -> count(==(name), flat_names) > 1, flat_names))
    isempty(repeated) && return :(nothing)
    name = first(repeated)
    places = [join(path, ".") for (flat_name, path) in paths if flat_name === name]
    message = "has two parameters with the flat name $(repr(name)): $(join(places, " and ")). Give it a method of CognitiveModeling.uses_submodel_prefixes which returns true, so that the names of its submodels' parameters start with the name of the submodel, as in perception__learning_rate."
    return :(throw(
        ArgumentError(string(uppercasefirst(describe_model(model)), " ", $message))
    ))
end
