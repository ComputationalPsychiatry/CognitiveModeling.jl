@testset "Saving what a fitted session returns" begin
    @testset "What a session saves of a timestep is selected by the save histories, for the cognitive model alone or for both models" begin
        #The values of a timestep have the cognitive model's at the top, and without an environment only an empty state under environment
        timestep_values = (;
            state=(; expected_reward=0.5),
            observation=(; reward=1.0),
            action=(; choice=true),
            action_distribution=(; choice=BernoulliByHand(0.7)),
            environment=(; state=nothing),
        )
        #The SaveHistory is in the cognitive model's own names, so the action distributions are its output distributions
        session_info = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            cognitive_save_history=SaveHistory(;
                output_distributions=:choice, states=SaveEverything()
            ),
        )
        #What is saved is arranged as the session returns it, without an entry for an environment
        saved = CognitiveModeling.select_timestep_values_to_save(
            session_info, timestep_values
        )
        @test saved === (;
            states=(; expected_reward=0.5),
            observations=(;),
            actions=(;),
            action_distributions=(; choice=BernoulliByHand(0.7)),
        )
        #Timestep 0 has BeforeFirstTimestep for every action distribution
        previous = (;
            state=(; expected_reward=0.5),
            observation=(; reward=BeforeFirstTimestep()),
            action=(; choice=BeforeFirstTimestep()),
            environment=(; state=nothing),
        )
        saving_everything = SessionInfo(
            session_info; cognitive_save_history=SaveHistory(SaveEverything())
        )
        initial = CognitiveModeling.select_initial_timestep_values_to_save(
            Val(true), saving_everything, previous
        )
        @test initial.action_distributions === (; choice=BeforeFirstTimestep()) &&
            initial.actions === (; choice=BeforeFirstTimestep())
        @test CognitiveModeling.select_initial_timestep_values_to_save(
            Val(false), saving_everything, previous
        ) === nothing
        #The storage of a simulation takes the arrangement without an environment too: it starts with timestep 0 when it is kept, and later timesteps go in by index
        storage = CognitiveModeling.initialize_all_storage(initial, saved, 2)
        @test keys(storage) == (:states, :observations, :actions, :action_distributions)
        @test length(storage.action_distributions) == 3 &&
            storage.action_distributions[1] === (; choice=BeforeFirstTimestep())
        storage = CognitiveModeling.store_timestep_values!(storage, saved, 3)
        @test storage.action_distributions[3] === (; choice=BernoulliByHand(0.7)) &&
            storage.states[3] === (; expected_reward=0.5)
        @test length(CognitiveModeling.initialize_all_storage(nothing, saved, 2).states) ==
            2
        #What is returned has the vectors as they are stored, with the timestep sizes
        returned_trajectories = CognitiveModeling.saved_trajectories(storage, [1, 1, 1])
        @test keys(returned_trajectories) ==
            (:states, :observations, :actions, :action_distributions, :timestep_sizes)
        @test returned_trajectories.action_distributions === storage.action_distributions
        #With an environment, its record is saved too, as its save history selects it, under environment
        with_environment = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
            environment=ScoringEnvironment(),
            cognitive_save_history=SaveHistory(; states=SaveEverything()),
            environment_save_history=SaveHistory(;
                states=:level, outputs=:reward, output_distributions=SaveEverything()
            ),
        )
        #The environment's observation and action are in its own names, which are the same here
        values_with_environment = merge(
            timestep_values,
            (;
                environment=(;
                    state=(; level=0.3),
                    observation=timestep_values.observation,
                    action=timestep_values.action,
                    observation_distribution=(; reward=NormalByHand(0.0, 1.0)),
                ),
            ),
        )
        saved_both = CognitiveModeling.select_timestep_values_to_save(
            with_environment, values_with_environment
        )
        @test saved_both === (;
            states=(; expected_reward=0.5),
            observations=(;),
            actions=(;),
            action_distributions=(;),
            environment=(;
                states=(; level=0.3),
                observations=(; reward=1.0),
                actions=(;),
                observation_distributions=(; reward=NormalByHand(0.0, 1.0)),
            ),
        )
        initial_both = CognitiveModeling.select_initial_timestep_values_to_save(
            Val(true),
            with_environment,
            merge(
                previous,
                (;
                    environment=(;
                        state=(; level=0.0),
                        observation=previous.observation,
                        action=previous.action,
                    ),
                ),
            ),
        )
        @test initial_both.environment.observation_distributions ===
              (; reward=BeforeFirstTimestep()) &&
            initial_both.environment.states === (; level=0.0)
        storage_both = CognitiveModeling.initialize_all_storage(initial_both, saved_both, 2)
        storage_both = CognitiveModeling.store_timestep_values!(storage_both, saved_both, 3)
        returned_both = CognitiveModeling.saved_trajectories(storage_both, [1, 1, 1])
        @test keys(returned_both) == (
            :states,
            :observations,
            :actions,
            :action_distributions,
            :environment,
            :timestep_sizes,
        )
        @test keys(returned_both.environment) ==
            (:states, :observations, :actions, :observation_distributions)
        @test returned_both.environment.states ==
            [(; level=0.0), (; level=0.3), (; level=0.3)]
        #A session whose environment isn't included gets an empty record of it, with the entries in the same order
        with_empty = CognitiveModeling.with_empty_environment_entry(returned_trajectories)
        @test keys(with_empty) == keys(returned_both)
        @test with_empty.environment.states == fill((;), 3) &&
            with_empty.environment.observation_distributions == fill((;), 3)
        #A session that ends early has its vectors shortened in place, with or without an environment, and its timestep sizes cut
        CognitiveModeling.map_saved(vector -> resize!(vector, 2), storage)
        @test length(storage.states) == 2 && length(storage.action_distributions) == 2
        CognitiveModeling.map_saved(vector -> resize!(vector, 2), storage_both)
        @test length(storage_both.states) == 2 &&
            length(storage_both.environment.states) == 2
        @test CognitiveModeling.timestep_sizes_until([BeforeFirstTimestep(), 1, 1], 2) ==
            [BeforeFirstTimestep(), 1]
        @test CognitiveModeling.timestep_sizes_until(Int[], 2) == Int[]
    end
end
