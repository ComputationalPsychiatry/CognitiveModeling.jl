"""
    get_attributes(model) -> (; parameters, states, observations, actions)

Return all the attributes of `model`, as given by [`get_parameters`](@ref),
[`get_states`](@ref), [`get_observations`](@ref) and [`get_actions`](@ref).
"""
function get_attributes(model::AbstractCognitiveModel)
    return (parameters = get_parameters(model), states = get_states(model), observations = get_observations(model), actions = get_actions(model))
end

"""
    show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

Print a summary of `model` and its attributes.

This is what the REPL and notebooks show. Everywhere else, such as in `print` or inside a
vector, a model is printed on one line by Julia's default `show(io, model)`.

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
Observations: none
Actions: none
```
"""
function Base.show(io::IO, ::MIME"text/plain", model::AbstractCognitiveModel)

    #Print the type of the model
    print(io, "-- Cognitive model --\nType: ", nameof(typeof(model)))

    #Go through each type of attribute
    for (attribute_type, attribute_names) in pairs(get_attributes(model))

        #Start a new line with the type of attribute
        print(io, "\n", uppercasefirst(string(attribute_type)), ": ")

        #Print the names of the attributes, or that there are none
        if isempty(attribute_names)
            print(io, "none")
        else
            join(io, attribute_names, ", ")
        end
    end

    return nothing
end
