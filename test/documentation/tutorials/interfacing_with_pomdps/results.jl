#The tutorial defines its model without simulating it, so it is simulated here, to check that it runs with the current
#interface, with the tiger problem of POMDPModels.jl as the environment, through CommonRLInterface.jl as the tutorial
#describes
using CommonRLInterface: CommonRLInterface
using POMDPModels: TigerPOMDP
using POMDPTools: POMDPCommonRLEnv
using Random: Xoshiro

@testset "The model simulates the tiger problem, and its beliefs are those the updater of POMDPTools.jl gives" begin
    simulation = simulate(
        BayesianObserver(TigerPOMDP()),
        CommonRLEnvironment(POMDPCommonRLEnv{Any}(TigerPOMDP()); action_name=:choice);
        who_starts=EnvironmentStarts(),
        n_steps=30,
        cognitive_parameters=(; inverse_temperature=2.0),
        environment_parameters=(;),
        observation_map=NameMap(:observation),
        cognitive_save_history=SaveHistory(SaveEverything()),
        rng=Xoshiro(1),
    )
    @test length(simulation.states) == 31
    @test all(action -> action.choice in (0, 1, 2), simulation.actions[2:end])

    #The first timestep has no previous action, so the belief stays the initial one, and every later one is updated
    #with the previous action and the observation that followed it
    pomdp = simulation.states[1].belief.pomdp
    updater = DiscreteUpdater(pomdp)
    belief = POMDPs.initialize_belief(updater, POMDPs.initialstate(pomdp))
    @test simulation.states[2].belief.b == belief.b
    for t in 3:length(simulation.states)
        belief = POMDPs.update(
            updater,
            belief,
            simulation.actions[t - 1].choice,
            simulation.observations[t].observation,
        )
        @test simulation.states[t].belief.b ≈ belief.b
    end
end
