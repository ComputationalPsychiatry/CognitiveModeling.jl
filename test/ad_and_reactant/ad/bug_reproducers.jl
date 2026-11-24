#Cases that reproduce a bug in a backend which the shared simulation cases don't hit, kept so that a fix shows up as an
#unexpected pass of its known failure. Their models are written here, since the bug depends on their exact code

#A choice that is sampled: true with a probability, and false otherwise
struct SampledChoice{T}
    probability::T
end
Base.rand(rng::AbstractRNG, choice::SampledChoice) = rand(rng) < choice.probability

#A reward that is sampled: 1.0 with a probability, and 0.0 otherwise
struct SampledReward{T}
    probability::T
end
Base.rand(rng::AbstractRNG, reward::SampledReward) = float(rand(rng) < reward.probability)

#A Rescorla-Wagner model which gives its action as a Taction of its probability. Its state starts as a Float64, whatever
#number type the parameters have
struct LearnsRewards{Taction} <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsRewards, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::LearnsRewards{Taction}, context::ModelContext
) where {Taction}
    V₋₁ = context.previous_state.expected_reward
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + α * (context.input.reward - V₋₁)
    end
    return ModelOutput(
        (; choice=Taction(inv(1 + exp(-β * (Vₜ - 0.5))))), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::LearnsRewards)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

#An environment which takes the choice, and whose reward is sampled whatever it is
struct SampledRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::SampledRewards, parameters) = (;)
function CognitiveModeling.timestep_model(
    environment::SampledRewards, context::ModelContext
)
    return ModelOutput((; reward=SampledReward(context.parameters.reward_probability)))
end
function CognitiveModeling.attribute_names(environment::SampledRewards)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

#A simulation of 30 timesteps where the agent starts, and its choices and the rewards are sampled from one seeded
#Xoshiro, with the parameters from a vector, as the sum of the expected rewards. Enzyme reverse gives wrong gradients here,
#with no error, which Enzyme.jl issue 3912 reports. The number of timesteps is a constant because that's part of what
#triggers it
function simulate_sampled_for_30_timesteps(x)
    simulation = simulate_session(
        LearnsRewards{SampledChoice}(),
        SampledRewards();
        who_starts=AgentStarts(),
        n_steps=30,
        cognitive_parameters=(; learning_rate=x[1], inverse_temperature=x[2]),
        environment_parameters=(; reward_probability=0.8),
        rng=Xoshiro(1),
        save_cognitive_trajectories=SaveTrajectories(SaveEverything()),
    )
    total = zero(first(simulation.states).expected_reward)
    for state in simulation.states
        total += state.expected_reward
    end
    return total
end
const BUG_REPRODUCERS = [(;
    name="Choices and rewards sampled from a Xoshiro for 30 timesteps, agent starts",
    f=simulate_sampled_for_30_timesteps,
    x=[0.3, 2.0],
),]
