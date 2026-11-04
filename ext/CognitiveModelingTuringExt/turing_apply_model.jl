#Default implementation of turing_apply_model, which just calls the normal apply_model function
#Can be extended for specific cognitive models that sample within a timestep
Turing.@model function CognitiveModeling.turing_apply_model(model::AbstractCognitiveModel, input::ModelInput)
    return apply_model(model, input)
end
