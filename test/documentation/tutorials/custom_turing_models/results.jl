#What the tutorial on Turing.jl models of one's own gave when it ran: a learning rate between 0 and 1 for each
#participant, read from what the model returned, the expected rewards of each participant from timestep 0 on, and the
#parameters of the single session under their own names. The tutorial is a sketch, so its fits aren't checked against
#the values it simulated with

@testset "Each participant has a learning rate between 0 and 1" begin
    for participant in Symbol.(sessions_info.session_ids)
        @test 0 < mean(parameters[@varname($participant.learning_rate)]) < 1
    end
end

@testset "Each sample has the expected rewards of a participant from timestep 0 on" begin
    for participant in Symbol.(sessions_info.session_ids)
        @test all(
            draw -> length(draw) == 51,
            vec(trajectories[@varname($participant.expected_reward)]),
        )
    end
end

@testset "The single session's parameters are named without a session" begin
    single_parameters = get_session_parameters(single_results)
    @test 0 < mean(single_parameters[@varname(learning_rate)]) < 1
    @test mean(single_parameters[@varname(inverse_temperature)]) > 0
end
