## Describing a model in messages ##
"""
    describe_model(model)

Return how `model` is named in an error message: as a cognitive model or as an environment
model, with its type. Used in the error messages of the checks on models, in simulations and
in fitted sessions.
"""
function describe_model(cognitive_model::AbstractCognitiveModel)
    return "a cognitive model of type $(typeof(cognitive_model))"
end
function describe_model(environment::AbstractEnvironmentModel)
    return "an environment model of type $(typeof(environment))"
end
describe_model(model::AbstractTimestepModel) = "a model of type $(typeof(model))"
