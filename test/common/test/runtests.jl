#The example models on their own, so that a broken example model fails here and not in a suite that uses it. Each
#simulates with its partner, with the distributions written by hand, with both starting, and the log likelihood of what
#each did, by hand, must be the log densities of the distributions it gives when it is run again on the same timesteps,
#from the states it saved. The models that can give the means of their distributions also simulate with them, without
#randomness

using Random: Xoshiro

#The log likelihood of the outputs that a model gave in a simulation, from the distributions that it gives when it is run
#again on each timestep from the state it saved before it, with what it sampled of its own in the simulation. The inputs
#are those it received on each timestep, and the outputs and the states are saved with timestep 0 first, which holds the
#output before the first timestep and the initial state
function log_likelihood_by_running(
    model, parameters, inputs, outputs, states, timestep_sizes
)
    latents = latents_from_states(model, states)
    log_likelihood = 0.0
    for t in eachindex(inputs)
        context = ModelContext(;
            input=inputs[t],
            previous_state=deepcopy(states[t]),
            previous_output=outputs[t],
            parameters,
            timestep_size=timestep_sizes[t],
            rng=Xoshiro(0),
        )
        output_distribution =
            timestep_model_with_latents(model, context, latents, t).output_distribution
        for name in keys(output_distribution)
            distribution = output_distribution[name]
            distribution isa AbstractNoValue && continue
            value = outputs[t + 1][name]
            log_likelihood += if distribution isa Deterministic
                value == distribution.value ? 0.0 : -Inf
            else
                CommonTestFunctionality.log_density(distribution, value)
            end
        end
    end
    return log_likelihood
end

@testset "Example models" begin
    @testset "The distributions written by hand" begin
        #The log densities at known points, and the samples in their support
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.BernoulliByHand(0.8), true
        ) ≈ log(0.8)
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.BernoulliByHand(0.8), false
        ) ≈ log(0.2)
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.NormalByHand(1.0, 2.0), 3.0
        ) ≈ -1 / 2 - log(2.0) - log(2π) / 2
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.CategoricalByHand([0.2, 0.3, 0.5]), 2
        ) ≈ log(0.3)
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.SymbolChoiceByHand(0.8), :left
        ) ≈ log(0.2)
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.ChoiceAndTimeByHand(0.8, 2.0), (; choice=2, time=0.5)
        ) ≈ log(0.2) + log(2.0) - 1.0
        @test CommonTestFunctionality.log_density(
            CommonTestFunctionality.RewardAndCueByHand(0.8, 0.5, 0.1),
            (; reward=true, cue=0.5),
        ) ≈ log(0.8) - log(0.1) - log(2π) / 2
        rng = Xoshiro(1)
        @test all(
            option -> option in 1:3,
            [
                rand(rng, CommonTestFunctionality.CategoricalByHand([0.2, 0.3, 0.5])) for
                _ in 1:100
            ],
        )
        @test all(
            response -> response.choice in (1, 2) && response.time > 0,
            [
                rand(rng, CommonTestFunctionality.ChoiceAndTimeByHand(0.8, 2.0)) for
                _ in 1:100
            ],
        )
    end

    @testset "$(pair.name), $(nameof(typeof(who_starts)))" for pair in MODEL_PAIRS,
        who_starts in (EnvironmentStarts(), AgentStarts())

        simulation = simulate_pair(pair, ByHand(), who_starts)
        n_timesteps = length(simulation.actions) - 1
        @test 1 <= n_timesteps <= 10
        timestep_sizes = sizes_of_timesteps(get(pair, :keywords, (;)), n_timesteps)

        #The cognitive model's outputs, by hand and by running it
        model = pair.model(ByHand())
        parameters = merge(
            CognitiveModeling.default_parameters(model), pair.cognitive_parameters
        )
        latents = latents_from_states(model, simulation.states)
        observations = inputs_received(simulation.observations, who_starts isa AgentStarts)
        by_hand = log_likelihood_by_hand(
            model,
            parameters,
            timesteps_of(observations, simulation.actions, timestep_sizes),
            latents,
        )
        @test by_hand ≈ log_likelihood_by_running(
            model,
            parameters,
            observations,
            simulation.actions,
            simulation.states,
            timestep_sizes,
        )
        if !isempty(CognitiveModeling.attribute_names(model).outputs)
            @test !iszero(by_hand)
        end

        #The environment's outputs, from its own side of the simulation
        environment = pair.environment(ByHand())
        environment_parameters = merge(
            CognitiveModeling.default_parameters(environment), pair.environment_parameters
        )
        sides = simulation.environment
        environment_latents = latents_from_states(environment, sides.states)
        actions = inputs_received(sides.actions, who_starts isa EnvironmentStarts)
        environment_by_hand = log_likelihood_by_hand(
            environment,
            environment_parameters,
            timesteps_of(actions, sides.observations, timestep_sizes),
            environment_latents,
        )
        @test environment_by_hand ≈ log_likelihood_by_running(
            environment,
            environment_parameters,
            actions,
            sides.observations,
            sides.states,
            timestep_sizes,
        )
    end

    @testset "$(pair.name) with the means of its distributions" for pair in PAIRS_WITH_MEANS
        #The same simulation twice, with different random number generators, gives the same outputs
        first_simulation = simulate_pair(pair, Means(), EnvironmentStarts(); rng=Xoshiro(1))
        second_simulation = simulate_pair(
            pair, Means(), EnvironmentStarts(); rng=Xoshiro(2)
        )
        @test isequal(first_simulation.actions, second_simulation.actions)
        @test isequal(first_simulation.observations, second_simulation.observations)
    end

    @testset "The features that a simulation shows" begin
        #WithInvalidParameters stays valid with the parameters of its pair, and turns invalid with a learning rate of 1 on the
        #first rewarded timestep, and WithDomains throws with a learning rate above 1
        with_invalid_parameters = pair_named("WithInvalidParameters")
        @test simulate_pair(with_invalid_parameters, ByHand(), EnvironmentStarts()).parameters_valid
        turning_invalid = simulate_pair(
            merge(
                with_invalid_parameters,
                (; cognitive_parameters=(; learning_rate=1.0, inverse_temperature=2.0)),
            ),
            ByHand(),
            EnvironmentStarts();
            warn_on_invalid_parameters=false,
        )
        @test !turning_invalid.parameters_valid &&
            turning_invalid.last_timestep.state.expected_reward == 1.0
        with_domains = pair_named("WithDomains")
        @test_throws ParameterDomainError simulate_pair(
            merge(
                with_domains,
                (; cognitive_parameters=(; learning_rate=1.5, inverse_temperature=2.0)),
            ),
            ByHand(),
            EnvironmentStarts(),
        )

        #OutputNotEveryTimestep gives nothing on the even timesteps, and its environment none on them either
        simulation = simulate_pair(
            pair_named("OutputNotEveryTimestep"), ByHand(), EnvironmentStarts()
        )
        @test all(t -> simulation.actions[t + 1].choice isa NoValue, 2:2:10)
        @test all(t -> simulation.actions[t + 1].choice isa Bool, 1:2:10)
        simulation = simulate_pair(
            pair_named("Baseline with OutputNotEveryTimestepEnvironment"),
            ByHand(),
            AgentStarts(),
        )
        @test all(
            t -> simulation.environment.observations[t + 1].reward isa NoValue, 2:2:10
        )

        #NoOutputBeforeFirstInputEnvironment gives nothing before the first choice, when it starts
        simulation = simulate_pair(
            pair_named("Baseline with NoOutputBeforeFirstInputEnvironment"),
            ByHand(),
            EnvironmentStarts(),
        )
        @test simulation.observations[2].reward isa NoValue
        @test all(t -> simulation.observations[t + 1].reward isa Bool, 2:10)

        #The stopping criteria end the simulations: the environment's after its last timestep
        simulation = simulate_pair(
            pair_named("Baseline with WithStoppingCriterionEnvironment"),
            ByHand(),
            AgentStarts(),
        )
        @test length(simulation.actions) == 7

        #StateChangesType starts with an integer, which the first timestep makes a floating-point number
        simulation = simulate_pair(
            pair_named("StateChangesType"), ByHand(), EnvironmentStarts()
        )
        @test simulation.states[1].expected_reward isa Int
        @test simulation.states[2].expected_reward isa Float64

        #StateChangedInPlace saves a copy of its state on every timestep, so the saved states differ
        simulation = simulate_pair(
            pair_named("StateChangedInPlace"), ByHand(), EnvironmentStarts()
        )
        @test allunique(state.expected_reward[1] for state in simulation.states)

        #DependsOnPreviousOutput doesn't look at the choice before the first timestep
        model = DependsOnPreviousOutput(ByHand())
        parameters = (; learning_rate=0.3, inverse_temperature=2.0, stickiness=1.0)
        first_timestep(previous_choice) = CognitiveModeling.timestep_model(
            model,
            ModelContext(;
                input=(; reward=true),
                previous_state=CognitiveModeling.initialize_state(model, parameters),
                previous_output=(; choice=previous_choice),
                parameters,
            ),
        ).output_distribution.choice
        @test first_timestep(true) ==
            first_timestep(false) ==
            first_timestep(BeforeFirstTimestep())

        #MethodsFromSupertype has the methods of its family, and none of its own
        @test MethodsFromSupertype(ByHand()) isa BaselineFamily
        @test CognitiveModeling.attribute_names(MethodsFromSupertype(ByHand())) ==
            CognitiveModeling.attribute_names(Baseline(ByHand()))
    end
end
