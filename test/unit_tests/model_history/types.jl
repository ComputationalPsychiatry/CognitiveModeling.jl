@testset "SaveHistory" begin
    @testset "It keeps each selection as a tuple of names, or as the marker for everything" begin
        save_history = SaveHistory(;
            inputs=SaveEverything(),
            outputs=(:choice,),
            states=(:expected_reward, :surprise),
        )
        @test save_history.inputs === SaveEverything()
        @test save_history.outputs === (:choice,)
        @test save_history.states === (:expected_reward, :surprise)
        @test propertynames(save_history) == (
            :states, :inputs, :outputs, :output_distributions, :save_initial_timestep_values
        )
        @test save_history.output_distributions === ()
        @test_throws FieldError save_history.parameters
        #The kinds are named as the model's own, not by their role in a simulation
        @test_throws FieldError save_history.observations
        @test_throws FieldError save_history.actions
    end

    @testset "It saves the initial timestep values by default, and can leave them out" begin
        @test SaveHistory().save_initial_timestep_values === true
        @test SaveHistory(SaveEverything()).save_initial_timestep_values === true
        @test SaveHistory(; states=:expected_reward, save_initial_timestep_values=false).save_initial_timestep_values ===
            false
        @test SaveHistory(SaveEverything(); save_initial_timestep_values=false).states ===
            SaveEverything()
        @test CognitiveModeling.saves_initial_timestep_values(SaveHistory()) &&
            !CognitiveModeling.saves_initial_timestep_values(
            SaveHistory(; save_initial_timestep_values=false)
        )
        @test_throws TypeError SaveHistory(; save_initial_timestep_values=1)
        @test SaveHistory(SaveEverything()).output_distributions === SaveEverything()
        @test SaveHistory(; output_distributions=:choice).output_distributions ===
            (:choice,)
        @test CognitiveModeling.saves_nothing(
            SaveHistory(; save_initial_timestep_values=false)
        )
    end

    @testset "A single name or a vector of names is put in a tuple" begin
        save_history = SaveHistory(; outputs=:choice, states=[:expected_reward, :surprise])
        @test save_history.outputs === (:choice,)
        @test save_history.states === (:expected_reward, :surprise)
    end

    @testset "A kind that isn't given saves nothing, and SaveEverything() saves everything" begin
        @test SaveHistory().states ===
            SaveHistory().inputs ===
            SaveHistory().outputs ===
            SaveHistory().output_distributions ===
            ()
        @test SaveHistory(; states=:expected_reward).inputs === ()
        everything = SaveHistory(SaveEverything())
        @test everything.states ===
            everything.inputs ===
            everything.outputs ===
            everything.output_distributions ===
            SaveEverything()
        @test everything === SaveHistory(;
            states=SaveEverything(),
            inputs=SaveEverything(),
            outputs=SaveEverything(),
            output_distributions=SaveEverything(),
        )
    end

    @testset "It only takes names" begin
        @test_throws TypeError SaveHistory(; states="expected_reward")
        @test_throws TypeError SaveHistory(; states=(:expected_reward, 1))
        @test_throws TypeError SaveHistory(; states=true)
    end

    @testset "The selections are in the type, so constructing it from literals is inferred, and holding it costs nothing" begin
        from_literals() =
            SaveHistory(; outputs=:choice, states=(:expected_reward, :surprise))
        @test Base.return_types(from_literals, Tuple{})[1] ===
            SaveHistory{(:expected_reward, :surprise),(),(:choice,),(),true}
        without_initial_values() =
            SaveHistory(; states=:expected_reward, save_initial_timestep_values=false)
        @test Base.return_types(without_initial_values, Tuple{})[1] ===
            SaveHistory{(:expected_reward,),(),(),(),false}
        @test @inferred(SaveHistory(SaveEverything())) isa SaveHistory
        @test @inferred(SaveHistory()) isa SaveHistory
        @test isbitstype(typeof(SaveHistory(; states=:expected_reward)))
        @test sizeof(SaveHistory(; states=(:expected_reward, :surprise))) == 0
    end
end

@testset "ModelHistory" begin
    @testset "It keeps what was saved of each kind in a vector of its own" begin
        history = CognitiveModeling.ModelHistory(
            Any[(; expected_reward=0.5)], Any[(;)], Any[(;)], Any[(;)], Any[1]
        )
        @test history.states == [(; expected_reward=0.5)]
        @test fieldnames(CognitiveModeling.ModelHistory) ==
            (:states, :inputs, :outputs, :output_distributions, :timestep_sizes)
    end
end

@testset "SessionTrajectories" begin
    #The trajectories of a session of two timesteps, from timestep 0, with the given states of the cognitive model
    function trajectories_for_test(
        states=[(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.6)]
    )
        return SessionTrajectories(
            states,
            [(; reward=BeforeFirstTimestep()), (; reward=1.0), (; reward=0.0)],
            fill((;), 3),
            fill((;), 3),
            (;
                states=fill((;), 3),
                observations=fill((;), 3),
                actions=fill((;), 3),
                observation_distributions=fill((;), 3),
            ),
            1,
            true,
            2,
            nothing,
        )
    end

    @testset "It keeps each argument in a field of the same name" begin
        trajectories = trajectories_for_test()
        @test fieldnames(SessionTrajectories) == (
            :states,
            :observations,
            :actions,
            :action_distributions,
            :environment,
            :timestep_sizes,
            :parameters_valid,
            :n_timesteps,
            :last_timestep,
        )
        @test trajectories.states[2] === (; expected_reward=0.75)
        @test trajectories.observations[1] === (; reward=BeforeFirstTimestep())
        @test trajectories.environment.states == fill((;), 3)
        @test trajectories.timestep_sizes === 1 &&
            trajectories.parameters_valid === true &&
            trajectories.n_timesteps === 2 &&
            trajectories.last_timestep === nothing
        #The names can be destructured
        (; states, n_timesteps) = trajectories
        @test states === trajectories.states && n_timesteps === 2
        #The types of the fields are in the type, so constructing it is inferred
        @test @inferred(trajectories_for_test()) isa SessionTrajectories
        @test isconcretetype(typeof(trajectories))
    end

    @testset "NamedTuple gives the fields in a NamedTuple" begin
        trajectories = trajectories_for_test()
        fields = @inferred NamedTuple(trajectories)
        @test keys(fields) == fieldnames(SessionTrajectories)
        @test fields.states === trajectories.states &&
            fields.environment === trajectories.environment &&
            fields.n_timesteps === 2
        @test merge(fields, (; n_timesteps=3)).n_timesteps == 3
    end

    @testset "Two are equal when their fields are" begin
        trajectories = trajectories_for_test()
        same = trajectories_for_test()
        @test trajectories == same && isequal(trajectories, same)
        @test hash(trajectories) == hash(same)
        @test length(unique([trajectories, same])) == 1
        #The types of the vectors don't matter, as for vectors
        widened = trajectories_for_test(
            Any[(; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.6)]
        )
        @test typeof(widened) != typeof(trajectories)
        @test trajectories == widened && hash(trajectories) == hash(widened)
        #A different value in any field makes them different
        other_states = trajectories_for_test([
            (; expected_reward=0.5), (; expected_reward=0.75), (; expected_reward=0.1)
        ])
        @test trajectories != other_states && !isequal(trajectories, other_states)
        #They aren't equal to the NamedTuple of their fields
        @test trajectories != NamedTuple(trajectories)
    end
end
