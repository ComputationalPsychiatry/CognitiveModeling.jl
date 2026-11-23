#What the model of the tutorial gave when it ran: responses with a choice and a response time, read from two columns of
#the data, a fit that recovers the drift rate and the non-decision time of each participant, a log-likelihood for each
#response, which add up to that of the sample, and a PSIS-LOO that prefers parameters for each participant to shared
#ones, and whose estimates are unreliable only at the fastest responses, a missing response that is left out, and
#responses with only one of their parts, scored by that part
using SequentialSamplingModels: SequentialSamplingModels
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

@testset "Each response has a log-likelihood, which add up to that of the sample, and PSIS-LOO prefers parameters for each participant" begin
    @test size(log_likelihood) == (500, 1, 3 * n_trials)
    @test sum(log_likelihood[1, 1, :]) ≈
        loglikelihood(model, Turing.FlexiChains.parameters_at(chain; iter=1, chain=1))
    @test isfinite(independent_parameters_loo.estimates.elpd)
    #The estimates that are unreliable are those of the fastest responses, as the tutorial says
    @test !isempty(unreliable_responses) &&
        issubset(unreliable_responses, fastest_responses)
    @test comparison.rank[findfirst(
        ==(:independent_parameters), collect(comparison.name)
    )] == 1
end

@testset "A response with a missing column is missing, and left out" begin
    responses = first(sessions_info_with_missing.sessions).actions.response
    is_left_out(response) = response isa CognitiveModeling.UnimputedMissingValue
    @test count(is_left_out, responses) == 1
    @test is_left_out(responses[5])
end

@testset "A response with only one of its parts keeps it, is scored by it, and counts in the fit" begin
    session_responses = first(sessions_info_with_parts.sessions).actions.response
    @test session_responses[5].choice == data.choice[5] &&
        ismissing(session_responses[5].rt)
    @test ismissing(session_responses[6].choice) && session_responses[6].rt == data.rt[6]
    @test !any(
        response -> response isa CognitiveModeling.AbstractNoValue, session_responses
    )

    #The probability of a choice is the DDM's, which its cumulative distribution gives at a long time
    ddm = DDM(1.0, 1.2, 0.5, 0.25)
    for choice in (1, 2)
        @test choice_probability(ddm, choice) ≈
            SequentialSamplingModels.cdf(ddm, choice, 60.0)
    end

    #Each part is scored as by hand
    distribution = ResponseWithMissingParts(ddm)
    @test logpdf(distribution, (; choice=1, rt=missing)) ≈ log(choice_probability(ddm, 1))
    @test logpdf(distribution, (; choice=missing, rt=0.5)) ≈
        log(pdf(ddm, 1, 0.5) + pdf(ddm, 2, 0.5))
    @test logpdf(distribution, (; choice=2, rt=0.5)) ≈ logpdf(ddm, 2, 0.5)

    #A response missing both parts is missing as a whole, and left out
    responses_with_both_missing = copy(responses)
    responses_with_both_missing[1] = (; choice=missing, rt=missing)
    both_missing = MultipleSessionsInfo(
        DriftDiffusionWithMissingParts(),
        merge(data, (; response=responses_with_both_missing));
        observation_columns=:stimulus,
        action_columns=:response,
        session_columns=:participant,
        who_starts=EnvironmentStarts(),
        missing_action_values=(missing, response -> all(ismissing, response)),
        impute_missing_actions=false,
    )
    @test first(both_missing.sessions).actions.response[1] isa
        CognitiveModeling.UnimputedMissingValue

    #The fit recovers the drift rates as the first one does
    parameters_with_parts = get_session_parameters(model_with_parts, chain_with_parts)
    drift_rates = [
        mean(parameters_with_parts[@varname($participant.drift_rate)]) for
        participant in (:P1, :P2, :P3)
    ]
    @test issorted(drift_rates)
    @test all(abs.(drift_rates .- true_drift_rates) .< 0.6)
end
