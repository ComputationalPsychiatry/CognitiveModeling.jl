"""
    get_attributes(model) -> (; parameters, states, observations, actions)

Return all the attributes of `model`, as given by [`get_parameters`](@ref),
[`get_states`](@ref), [`get_observations`](@ref) and [`get_actions`](@ref).

This throws an error if `model` has no method for one of them.
"""
function get_attributes(model::AbstractCognitiveModel)
    return (parameters = get_parameters(model), states = get_states(model), observations = get_observations(model), actions = get_actions(model))
end

"""
    show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

Print a summary of `model` and its attributes.

This is what the REPL and notebooks show. Everywhere else, such as in `print` or inside a
vector, a model is printed on one line by Julia's default `show(io, model)`.

An attribute shows as `not specified` if `model` has no method for it, and as `none` if the
method returns an empty list.

# Examples
```jldoctest
julia> struct ExampleModel <: AbstractCognitiveModel end

julia> CognitiveModeling.get_parameters(model::ExampleModel) = (:learning_rate, :noise);

julia> CognitiveModeling.get_states(model::ExampleModel) = (:expected_reward,);

julia> ExampleModel()
-- Cognitive model --
Type: ExampleModel
Parameters: learning_rate, noise
States: expected_reward
Observations: not specified
Actions: not specified
```
"""
function Base.show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

    #Print the type of the model
    print(io, "-- Cognitive model --\nType: ", nameof(typeof(model)))

    #Go through each type of attribute
    for (attribute_type, get_names) in (("Parameters", get_parameters), ("States", get_states), ("Observations", get_observations), ("Actions", get_actions))

        #Start a new line with the type of attribute
        print(io, "\n", attribute_type, ": ")

        #Print that the model has no method of its own, that there are no names, or the names
        if which(get_names, (typeof(model),)).sig == Tuple{typeof(get_names), AbstractCognitiveModel}
            print(io, "not specified")
        elseif isempty(get_names(model))
            print(io, "none")
        else
            join(io, get_names(model), ", ")
        end
    end

    return nothing
end
