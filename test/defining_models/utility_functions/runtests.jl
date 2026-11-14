#A cognitive model and an environment model with names, which the tests of several files use. The inputs of an
#environment are the actions it receives, and its outputs the observations it gives
struct ModelWithAttributes <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithAttributes)
    return ModelAttributes(;
        parameters=(:learning_rate, :action_noise),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end
struct EnvironmentWithAttributes <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(environment::EnvironmentWithAttributes)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

@testset "Utility functions" begin
    include("describe_model.jl")
    include("check_model_methods.jl")
    include("compare_names.jl")
    include("check_values_against_names.jl")
    include("check_parameter_domains.jl")
    include("outputs_and_inputs_before_first_timestep.jl")
    include("display_models.jl")
end
