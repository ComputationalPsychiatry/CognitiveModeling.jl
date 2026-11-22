#What the models of the tutorial gave when it ran: the PID controller follows the target, the fit recovers the gains of
#each session, and the Kalman filter's beliefs are those of the same filter written by hand
using Statistics: mean

@testset "The PID controller keeps the plant near the target" begin
    errors = [
        observation.target - observation.position for observation in simulation.observations
    ]
    @test length(errors) == 150
    @test sqrt(mean(abs2, errors)) < 0.5
end

@testset "NUTS recovers the gains of each session" begin
    for (name, variable, tolerance) in (
        (:K, @varname(K.session), 0.3),
        (:Ti, @varname(Ti.session), 0.3),
        (:Td, @varname(Td.session), 0.05),
    )
        draws = vec(chain[variable])
        estimates = [mean(draw[session] for draw in draws) for session in 1:3]
        truths = [getproperty(parameters, name) for parameters in true_parameters]
        @test all(abs.(estimates .- truths) .< tolerance .* max.(truths, 1))
    end
end

@testset "The Kalman filter believes what the same filter written by hand does" begin
    with_distributions = simulate_session(
        KalmanTracker(),
        DriftingTarget();
        who_starts=EnvironmentStarts(),
        n_steps=100,
        cognitive_parameters=(; assumed_drift=0.3, assumed_cue_noise=0.5, motor_noise=0.1),
        environment_parameters=(; drift=0.3, cue_noise=0.5),
        cognitive_save_history=SaveHistory(
            SaveEverything(); save_initial_timestep_values=false
        ),
        rng=Xoshiro(2),
    )
    believed_mean, believed_variance = 0.0, 1.0
    for t in eachindex(with_distributions.observations)
        cue = with_distributions.observations[t].cue
        gain = believed_variance / (believed_variance + 0.5^2)
        believed_mean += gain * (cue - believed_mean)
        @test mean(with_distributions.action_distributions[t].report) ≈ believed_mean
        believed_variance = (1 - gain) * believed_variance + 0.3^2
    end
end
