#Distributions written by hand, which stand in for the ones a model gives, so that the tests need no package of
#distributions. The type parameter lets two of them have different types, as Normal{Float64} and Normal{Float32} do
struct NormalByHand{T<:Real}
    mean::T
    standard_deviation::T
end
struct BernoulliByHand
    probability::Float64
end

#A model for the error messages about scoring, and an environment for the record of one
struct ScoringModel <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::ScoringModel)
    return ModelAttributes(;
        parameters=(), states=:expected_reward, inputs=:reward, outputs=:choice
    )
end
struct ScoringEnvironment <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::ScoringEnvironment)
    return ModelAttributes(; parameters=(), states=:level, inputs=:choice, outputs=:reward)
end

@testset "Running sessions" begin
    include("read_session_setup.jl")
    include("read_timestep_data.jl")
    include("to_condition_on.jl")
    include("save_fitted_session_trajectories.jl")
end
