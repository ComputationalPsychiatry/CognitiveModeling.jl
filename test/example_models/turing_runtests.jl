#The example models in the Turing.jl suites, with the distributions of Distributions.jl: the log density of a session of
#each model which samples something of its own or gives a distribution written in turing_methods.jl is the log likelihood
#of its outputs by hand, with what it sampled at values drawn from the session, plus the log density of those values. The
#environment runs in each session, so that its observations are scored too. A runner includes this file after
#ExampleModels.jl and turing_methods.jl

include("example_pairs.jl")

#The pairs which need what turing_methods.jl adds
const PAIRS_WITH_TURING_METHODS = map(
    pair_named,
    [
        "SamplingWithinTimestep",
        "SampledInitialState",
        "SymbolOutput",
        "NamedTupleAttributes",
        "VectorAttributes",
        "Baseline with SamplingWithinTimestepEnvironment",
        "EveryFeature",
    ],
)

#The values of one name on each timestep, from a vector of NamedTuples, as a column of the data
column(values, name) = [value[name] for value in values]
function columns(values)
    return NamedTuple{keys(first(values))}(
        map(name -> column(values, name), keys(first(values)))
    )
end

@testset "Example models with Turing.jl" begin
    @testset "$(pair.name), $(nameof(typeof(who_starts)))" for pair in
                                                               PAIRS_WITH_TURING_METHODS,
        who_starts in (EnvironmentStarts(), AgentStarts())

        #The data of a simulation with the same distributions
        simulation = simulate_pair(pair, DistributionsJl(), who_starts)
        n_timesteps = length(simulation.actions) - 1
        keywords = get(pair, :keywords, (;))
        timestep_sizes = sizes_of_timesteps(keywords, n_timesteps)
        model = pair.model(DistributionsJl())
        environment = pair.environment(DistributionsJl())
        session_info = SessionInfo(
            model,
            columns(simulation.observations[2:end]),
            columns(simulation.actions[2:end]);
            who_starts,
            environment,
            include_environment=true,
            merge(keywords, (; timestep_sizes))...,
        )
        session = session_model(
            TuringBackend(),
            session_info,
            pair.cognitive_parameters,
            pair.environment_parameters,
        )

        #The log density at values drawn from the session, and the same by hand
        values = rand(Xoshiro(2), session)
        cognitive_parameters = merge(
            CognitiveModeling.default_parameters(model), pair.cognitive_parameters
        )
        environment_parameters = merge(
            CognitiveModeling.default_parameters(environment), pair.environment_parameters
        )
        cognitive_latents = latents_from_variables(model, values, n_timesteps)
        environment_latents = latents_from_variables(environment, values, n_timesteps)
        cognitive_timesteps = timesteps_of(
            inputs_received(simulation.observations, who_starts isa AgentStarts),
            simulation.actions,
            timestep_sizes,
        )
        environment_timesteps = timesteps_of(
            inputs_received(
                simulation.environment.actions, who_starts isa EnvironmentStarts
            ),
            simulation.environment.observations,
            timestep_sizes,
        )
        by_hand =
            log_likelihood_by_hand(
                model, cognitive_parameters, cognitive_timesteps, cognitive_latents
            ) +
            log_density_of_latents_by_hand(model, cognitive_parameters, cognitive_latents) +
            log_likelihood_by_hand(
                environment,
                environment_parameters,
                environment_timesteps,
                environment_latents,
            ) +
            log_density_of_latents_by_hand(
                environment, environment_parameters, environment_latents
            )
        @test logjoint(session, values) ≈ by_hand
    end
end
