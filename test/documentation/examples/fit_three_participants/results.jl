#What the example of fitting three participants gave when it ran: a learning rate between 0 and 1 and a positive noise
#for each participant, read with get_session_parameters, the expected values of each participant from before the first
#trial on, and the figure of them. The example is a sketch, so the values of its fit aren't checked

@testset "Each participant has a learning rate between 0 and 1 and a positive noise" begin
    for participant in participants
        @test 0 < mean(session_parameters[@varname($participant.learning_rate)]) < 1
        @test mean(session_parameters[@varname($participant.noise)]) > 0
    end
end

@testset "Each sample has the expected values of a participant from before the first trial on" begin
    for participant in participants
        @test all(
            draw -> length(draw) == 6,
            vec(session_trajectories[@varname($participant.expected_value)]),
        )
    end
end

@testset "The figure of the expected values is made" begin
    @test figure isa Figure
end
