## Functions that a model can implement ##

"""
    turing_apply_model(model::AbstractCognitiveModel, input::ModelInput)
    turing_apply_model(environment::AbstractEnvironmentModel, input::EnvironmentInput)

Create a Turing.jl model for a single timestep of `model`, to be used as a submodel, which
returns a [`ModelOutput`](@ref), or of `environment`, which returns an
[`EnvironmentOutput`](@ref). This needs Turing.jl to be loaded.

# Implementation
A model can add a method of this function. The default calls [`apply_model`](@ref). A
model which samples within a timestep needs to add a method, however; or if any other
changes are desired within Turing. It must sample with `~` instead of from `input.rng`.
Otherwise the model has a different log density each time it is evaluated, with no error.
The same holds for an environment model, which fitting uses to fill in observations that
are [`MissingObservation`](@ref) in the data.

# Examples
```julia
using CognitiveModeling
using Turing

struct NoisyRescorlaWagner <: AbstractCognitiveModel end

Turing.@model function CognitiveModeling.turing_apply_model(
        model::NoisyRescorlaWagner,
        input::ModelInput,
    )

    ## Unpack input ##
    #Reward observed on this timestep
    r = input.observation.reward
    #Expected reward from the previous state
    V₋₁ = input.previous_state.expected_reward
    #Learning rate parameter
    α = input.parameters.learning_rate

    ## Update state ##
    #Sample the noise in the update
    ϵ ~ Normal(0, input.parameters.noise)
    #Update expected reward using the Rescorla-Wagner rule
    Vₜ = V₋₁ + α * (r - V₋₁) + ϵ

    return ModelOutput((; choice = BernoulliLogit(Vₜ - 0.5)), (; expected_reward = Vₜ))
end
```
"""
function turing_apply_model end
