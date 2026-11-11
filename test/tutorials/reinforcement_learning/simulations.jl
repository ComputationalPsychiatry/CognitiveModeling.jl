#The tutorial defines its models without simulating them, so they are simulated here, to check that they run with the
#current interface. The environment is a random walk written against ReinforcementLearningBase.jl, as the environments of
#ReinforcementLearning.jl are, and given to the models through CommonRLInterface.jl, as the tutorial describes
using CommonRLInterface: CommonRLInterface
using Random: Xoshiro

#A walk over seven positions which starts in the middle and ends at either end, with a reward at the right end
mutable struct RandomWalk <: RLBase.AbstractEnv
    position::Int
    reward::Float64
end
RLBase.state(rl_env::RandomWalk) = rl_env.position
RLBase.action_space(rl_env::RandomWalk) = Base.OneTo(2)
RLBase.reward(rl_env::RandomWalk) = rl_env.reward
RLBase.is_terminated(rl_env::RandomWalk) = rl_env.position in (1, 7)
function RLBase.reset!(rl_env::RandomWalk)
    rl_env.position = 4
    rl_env.reward = 0.0
    return nothing
end
function RLBase.act!(rl_env::RandomWalk, action)
    rl_env.position += action == 1 ? -1 : 1
    rl_env.reward = rl_env.position == 7 ? 1.0 : 0.0
    return nothing
end

#The walk as an environment model, through CommonRLInterface.jl
function random_walk_environment()
    rl_env = convert(CommonRLInterface.AbstractEnv, RandomWalk(4, 0.0))
    return CommonRLEnvironment(rl_env; action_name=:move)
end

@testset "Both models simulate an episode of an environment written for ReinforcementLearning.jl, and learn from it" begin
    for (cognitive_model, cognitive_parameters, table_of) in (
        (
            TemporalDifferenceAgent(7, 2),
            (; learning_rate=0.5, discount=0.9, epsilon=0.1),
            state -> state.rl_agent.policy.learner.approximator.model,
        ),
        (
            TemporalDifferenceLearner(7, 2),
            (; learning_rate=0.5, discount=0.9, inverse_temperature=2.0),
            state -> state.rl_learner.approximator.model,
        ),
    )
        simulation = simulate(
            cognitive_model,
            random_walk_environment();
            who_starts=EnvironmentStarts(),
            n_steps=200,
            cognitive_parameters,
            environment_parameters=(;),
            cognitive_save_history=SaveHistory(SaveEverything()),
            rng=Xoshiro(1),
        )
        #The episode ends at one end of the walk, which stops the simulation
        @test simulation.observations[end].terminated
        @test all(action -> action.move in (1, 2), simulation.actions[2:end])
        #Each saved state is a copy, whose table of action values starts at zero and changes as the model learns
        tables = map(table_of, simulation.states)
        @test tables[1] == zeros(2, 7)
        @test length(unique(tables)) > 1
    end
end
