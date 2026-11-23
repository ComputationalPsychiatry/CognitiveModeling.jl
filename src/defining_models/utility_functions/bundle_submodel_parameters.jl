## The parameters of a model with submodels ##

#The names of the fields are in the type of the Val, and ntuple with a Val unrolls, so that each field is read with a
#constant name and the NamedTuple is inferred
"""
    fields_as_named_tuple(model, ::Val{fields})

Return the fields of `model` named in `fields`, a tuple of `Symbol`s, in a `NamedTuple` under
the same names. Used by [`CognitiveModeling.submodels`](@ref), with the names that
[`CognitiveModeling.submodel_fields`](@ref) gives.
"""
function fields_as_named_tuple(model, ::Val{fields}) where {fields}
    return NamedTuple{fields}(ntuple(i -> getfield(model, fields[i]), Val(length(fields))))
end

#What separates the name of a submodel from the name of a parameter in the flat names, when a model uses prefixes
const SUBMODEL_PREFIX_SEPARATOR = "__"

#Everything is in the type, so that the code that reads it is decided when it is compiled
"""
    ModelLayout{Tattributes,prefixes,Tsubmodels}

How a model and its submodels are arranged: `Tattributes`, the type of the
[`ModelAttributes`](@ref) with the names of the model's own attributes, `prefixes`, whether
the flat names of its submodels' parameters start with the submodel's name, and in its field
`submodels` the layout of each submodel under its name. [`model_layout`](@ref) makes it,
and [`bundle_parameters`](@ref), [`flat_parameter_names`](@ref) and
[`flatten_parameters`](@ref) read it.
"""
struct ModelLayout{Tattributes<:ModelAttributes,prefixes,Tsubmodels<:NamedTuple}
    submodels::Tsubmodels
end
function ModelLayout{Tattributes,prefixes}(layouts::NamedTuple) where {Tattributes,prefixes}
    return ModelLayout{Tattributes,prefixes,typeof(layouts)}(layouts)
end

"""
    own_parameter_names(::Type{<:ModelAttributes})

Return the names of the parameters in a type of [`ModelAttributes`](@ref). Used while the
code that reads a [`ModelLayout`](@ref) is generated.
"""
function own_parameter_names(
    ::Type{ModelAttributes{Tparameters,Tstates,Tinputs,Toutputs}}
) where {Tparameters,Tstates,Tinputs,Toutputs}
    return Tparameters
end

#Built by going through the submodels one at a time, each of a simpler type than the model that holds it. Julia limits
#how far it infers a function that calls itself, and a map over the NamedTuple of submodels runs into that limit,
#which peeling the submodels off a tuple doesn't
"""
    model_layout(model::AbstractSubmodel)

Return the [`ModelLayout`](@ref) of `model`: the names of its own attributes, from
[`attribute_names`](@ref), whether it uses prefixes, from
[`CognitiveModeling.uses_submodel_prefixes`](@ref), and the layout of each of its
submodels, from [`CognitiveModeling.submodels`](@ref), under their names. Used to bundle
and flatten the parameters of a model with submodels.
"""
function model_layout(model::AbstractSubmodel)
    submodels_of_model = submodels(model)
    return layout_with(
        check_type_attribute_names(attribute_names(model), model),
        Val(uses_submodel_prefixes(model)),
        NamedTuple{keys(submodels_of_model)}(layouts_of(values(submodels_of_model))),
    )
end

"""
    layouts_of(submodels_of_model::Tuple)

Return the [`ModelLayout`](@ref) of each submodel in `submodels_of_model`, in a tuple in the
same order, taking one submodel at a time off the front of the tuple. Used by
[`model_layout`](@ref).
"""
layouts_of(::Tuple{}) = ()
function layouts_of(submodels_of_model::Tuple)
    return (
        model_layout(first(submodels_of_model)),
        layouts_of(Base.tail(submodels_of_model))...,
    )
end

"""
    layout_with(attributes::ModelAttributes, ::Val{prefixes}, layouts::NamedTuple)

Return the [`ModelLayout`](@ref) of a model with the names of its own `attributes`,
`prefixes`, and the `layouts` of its submodels. Used by [`model_layout`](@ref).
"""
function layout_with(
    attributes::ModelAttributes, ::Val{prefixes}, layouts::NamedTuple
) where {prefixes}
    return ModelLayout{typeof(attributes),prefixes}(layouts)
end

## Reading a layout's type, while code is generated ##
#A generated function sees types only, and the methods of the functions above may be defined after it, so the layout
#reaches it through the type of a value
"""
    parameter_paths(::Type{<:ModelLayout})

Return every parameter of a layout type as a pair of its flat name and its path in the
bundled parameters: the names to go through from the top, ending with the parameter's own
name. Used while the code of [`flat_parameter_names`](@ref) and
[`flatten_parameters`](@ref) is generated.
"""
function parameter_paths(
    ::Type{ModelLayout{Tattributes,prefixes,Tsubmodels}}
) where {Tattributes,prefixes,Tsubmodels}
    paths = Pair{Symbol,Tuple}[name => (name,) for name in own_parameter_names(Tattributes)]
    for (submodel_name, Tsubmodel) in zip(fieldnames(Tsubmodels), fieldtypes(Tsubmodels))
        for (flat_name, path) in parameter_paths(Tsubmodel)
            name = if prefixes
                Symbol(submodel_name, SUBMODEL_PREFIX_SEPARATOR, flat_name)
            else
                flat_name
            end
            push!(paths, name => (submodel_name, path...))
        end
    end
    return paths
end

"""
    bundling_expression(::Type{<:ModelLayout}, prefix::String)

Return the expression that builds the bundled parameters of a layout type from the flat ones
in a variable named `flat`, where `prefix` is what the prefixes of the models above it add to
the flat names. Used while the code of [`bundle_parameters`](@ref) is generated.
"""
function bundling_expression(
    ::Type{ModelLayout{Tattributes,prefixes,Tsubmodels}}, prefix::String
) where {Tattributes,prefixes,Tsubmodels}
    entries = Any[
        Expr(
            :kw, name, :(read_flat_parameter(flat, Val($(QuoteNode(Symbol(prefix, name))))))
        ) for name in own_parameter_names(Tattributes)
    ]
    for (submodel_name, Tsubmodel) in zip(fieldnames(Tsubmodels), fieldtypes(Tsubmodels))
        prefix_of_submodel = if prefixes
            string(prefix, submodel_name, SUBMODEL_PREFIX_SEPARATOR)
        else
            prefix
        end
        push!(
            entries,
            Expr(:kw, submodel_name, bundling_expression(Tsubmodel, prefix_of_submodel)),
        )
    end
    return Expr(:tuple, Expr(:parameters, entries...))
end

## Flat parameters to bundled parameters, and back ##

"""
    bundle_parameters(model::AbstractSubmodel, flat::NamedTuple)
    bundle_parameters(layout::ModelLayout, flat::NamedTuple)

Return the parameters of `model` as it reads them, from `flat`, its parameters and those of
its submodels in one `NamedTuple`, under the names that [`flat_parameter_names`](@ref)
gives: the model's own parameters under their names, and those of each submodel bundled in a
`NamedTuple` under the submodel's name, in the same way down through its submodels. A model
without submodels gets its parameters as they are given. The code is generated from the
[`ModelLayout`](@ref), so that nothing is looked up while it runs, and a parameter that
isn't given is an informative error.
"""
function bundle_parameters(model::AbstractSubmodel, flat::NamedTuple)
    return bundle_parameters(model_layout(model), flat)
end
@generated function bundle_parameters(
    layout::ModelLayout{Tattributes,prefixes,Tsubmodels}, flat::NamedTuple
) where {Tattributes,prefixes,Tsubmodels}
    fieldcount(Tsubmodels) == 0 && return :(flat)
    return bundling_expression(layout, "")
end

#The check of whether the parameter is there is on the type, so it costs nothing when it is
"""
    read_flat_parameter(flat::NamedTuple, ::Val{name})

Return the parameter `name` of `flat`, the parameters of a model with submodels as they are
given, and throw an informative error if it isn't there. Used by the code of
[`bundle_parameters`](@ref).
"""
function read_flat_parameter(flat::NamedTuple, ::Val{name}) where {name}
    hasfield(typeof(flat), name) && return getfield(flat, name)
    return throw(
        ArgumentError(
            "The parameters of a model with submodels are given flat, under the names that CognitiveModeling.get_attribute_names gives, but $(repr(name)) isn't among them, which are $(keys(flat)).",
        ),
    )
end

"""
    flat_parameter_names(model::AbstractSubmodel)
    flat_parameter_names(layout::ModelLayout)

Return the names of the parameters of `model` and of its submodels, as they are given to it
flat, in a tuple: its own parameters first, and then those of each submodel in the order of
its submodels, each with the submodel's name and `__` before it when `model` uses prefixes,
as [`CognitiveModeling.uses_submodel_prefixes`](@ref) says.
"""
function flat_parameter_names(model::AbstractSubmodel)
    return flat_parameter_names(model_layout(model))
end
@generated function flat_parameter_names(layout::ModelLayout)
    return Tuple(first.(parameter_paths(layout)))
end

"""
    flatten_parameters(model::AbstractSubmodel, bundled::NamedTuple)
    flatten_parameters(layout::ModelLayout, bundled::NamedTuple)

Return the parameters `bundled`, as [`bundle_parameters`](@ref) gives them, as flat ones,
under the names that [`flat_parameter_names`](@ref) gives, so that the two undo each other.
"""
function flatten_parameters(model::AbstractSubmodel, bundled::NamedTuple)
    return flatten_parameters(model_layout(model), bundled)
end
@generated function flatten_parameters(layout::ModelLayout, bundled::NamedTuple)
    entries = map(parameter_paths(layout)) do (flat_name, path)
        value = foldl(
            (object, name) -> :(getfield($object, $(QuoteNode(name)))), path; init=:bundled
        )
        return Expr(:kw, flat_name, value)
    end
    return Expr(:tuple, Expr(:parameters, entries...))
end
