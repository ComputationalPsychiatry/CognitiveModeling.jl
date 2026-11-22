#What the tutorial on predictive checks and parameter recovery gave when it ran: the simulated choices under the names
#of deconditioned sessions, shares of choices between 0 and 1 in both ways of simulating, the same log joint from
#Turing.jl's condition and from condition_sessions, and a learning rate between 0 and 1 for each participant from the
#recovery. The fits are short, so the recovered learning rates aren't checked against those that simulated the choices

@testset "The deconditioned model simulates every choice, as a variable named apart from an imputed one" begin
    @test all(
        ==(SimulatedMissingValue()),
        decondition_sessions(sessions_info).sessions[1].actions.choice,
    )
    @test @varname(session[3].timestep[40].choice.simulated) in
        Turing.FlexiChains.parameters(prior_predictions)
    @test @varname(session[3].timestep[40].choice.simulated) in
        Turing.FlexiChains.parameters(posterior_predictions)
end

@testset "The shares of choices are between 0 and 1, in the data and in the simulations" begin
    @test all(share -> 0 <= share <= 1, share_in_data)
    @test all(shares -> all(share -> 0 <= share <= 1, shares), prior_shares)
    for participant in participants
        @test all(
            choices -> length(choices) == 40 && 0 <= mean(choices) <= 1,
            vec(prior_trajectories[@varname($participant.choice)]),
        )
        @test length(vec(posterior_trajectories[@varname($participant.choice)])) == 300
    end
end

@testset "Turing.jl's condition and condition_sessions give the same log joint" begin
    @test logjoint(turing_recovery_model, simulated_parameters) ≈
        logjoint(package_recovery_model, simulated_parameters)
    @test isfinite(logjoint(package_recovery_model, simulated_parameters))
end

@testset "The recovery gives a learning rate between 0 and 1 for each participant" begin
    for participant in participants
        @test 0 < mean(recovered_parameters[@varname($participant.learning_rate)]) < 1
    end
end
