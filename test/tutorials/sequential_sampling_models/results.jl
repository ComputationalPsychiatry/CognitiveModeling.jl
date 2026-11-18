#What the model of the tutorial gave when it ran: responses with a choice and a response time, read from two columns of
#the data, a fit that recovers the drift rate and the non-decision time of each participant, and a missing response that
#is left out
using Statistics: mean

@testset "Each response is a NamedTuple of a choice and a response time" begin
    responses = [action.response for action in trajectories.actions]
    @test length(responses) == 5
    @test all(response -> response.choice in (1, 2) && response.rt > 0.3, responses)
    @test eltype(first(sessions_info.sessions).actions.response) ==
        @NamedTuple{choice::Int64, rt::Float64}
end

@testset "NUTS recovers the drift rate and the non-decision time of each participant" begin
    drift_rates = [
        mean(session_parameters[@varname($participant.drift_rate)]) for
        participant in (:P1, :P2, :P3)
    ]
    @test issorted(drift_rates)
    @test all(abs.(drift_rates .- true_drift_rates) .< 0.6)
    non_decision_times = [
        mean(session_parameters[@varname($participant.non_decision_time)]) for
        participant in (:P1, :P2, :P3)
    ]
    @test all(abs.(non_decision_times .- 0.3) .< 0.05)
end

@testset "A response with a missing column is missing, and left out" begin
    responses = first(sessions_info_with_missing.sessions).actions.response
    is_left_out(response) = response isa CognitiveModeling.UnimputedMissingValue
    @test count(is_left_out, responses) == 1
    @test is_left_out(responses[5])
end
