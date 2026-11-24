#What the tutorial on models with submodels gave when it ran: the flat names of the parameters, the parameters as the
#model reads them, what a simulation saved of the perception's state, and the entries of a fit's trajectories

@testset "The parameters are flat, and those without a default are the ones to give" begin
    @test parameter_names == (:learning_rate, :initial_value, :inverse_temperature)
    @test names_to_give == (:learning_rate, :inverse_temperature)
    @test default_parameters == (; initial_value=0.5)
    @test bundled_parameters == (;
        perception=(; learning_rate=0.3, initial_value=0.5),
        response=(; inverse_temperature=3.0),
    )
end

@testset "A simulation saves the one state of the perception that is selected, under its name" begin
    @test keys(last_saved_state) == (:n_rewards, :perception)
    @test keys(last_saved_state.perception) == (:expected_reward,)
    @test all(simulation -> length(simulation.states) == 80, simulations)
    @test last_saved_state.n_rewards == sum(table.reward[1:80])
end

@testset "The trajectories of the fit have the state of the perception for each participant, and the fit recovers who learns faster" begin
    @test trajectory_names == [
        "P1.perception.expected_reward",
        "P1.parameters_valid",
        "P2.perception.expected_reward",
        "P2.parameters_valid",
    ]
    @test 0 < final_expected_reward < 1
    learning_rate_samples = chain[@varname(learning_rate.sessions)]
    @test mean(first, learning_rate_samples) < mean(last, learning_rate_samples)
end
