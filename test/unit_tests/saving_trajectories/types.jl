@testset "SaveTrajectories" begin
    @testset "It keeps each selection as a tuple of names, or as the marker for everything" begin
        save_trajectories = SaveTrajectories(;
            inputs=SaveEverything(),
            outputs=(:choice,),
            states=(:expected_reward, :surprise),
        )
        @test save_trajectories.inputs === SaveEverything()
        @test save_trajectories.outputs === (:choice,)
        @test save_trajectories.states === (:expected_reward, :surprise)
        @test propertynames(save_trajectories) == (
            :states, :inputs, :outputs, :output_distributions, :save_initial_timestep_values
        )
        @test save_trajectories.output_distributions === ()
        @test_throws FieldError save_trajectories.parameters
        #The kinds are named as the model's own, not by their role in a simulation
        @test_throws FieldError save_trajectories.observations
        @test_throws FieldError save_trajectories.actions
    end

    @testset "It saves the initial timestep values by default, and can leave them out" begin
        @test SaveTrajectories().save_initial_timestep_values === true
        @test SaveTrajectories(SaveEverything()).save_initial_timestep_values === true
        @test SaveTrajectories(;
            states=:expected_reward, save_initial_timestep_values=false
        ).save_initial_timestep_values === false
        @test SaveTrajectories(SaveEverything(); save_initial_timestep_values=false).states ===
            SaveEverything()
        @test CognitiveModeling.saves_initial_timestep_values(SaveTrajectories()) &&
            !CognitiveModeling.saves_initial_timestep_values(
            SaveTrajectories(; save_initial_timestep_values=false)
        )
        @test_throws TypeError SaveTrajectories(; save_initial_timestep_values=1)
        @test SaveTrajectories(SaveEverything()).output_distributions === SaveEverything()
        @test SaveTrajectories(; output_distributions=:choice).output_distributions ===
            (:choice,)
        @test CognitiveModeling.saves_nothing(
            SaveTrajectories(; save_initial_timestep_values=false)
        )
    end

    @testset "A single name or a vector of names is put in a tuple" begin
        save_trajectories = SaveTrajectories(;
            outputs=:choice, states=[:expected_reward, :surprise]
        )
        @test save_trajectories.outputs === (:choice,)
        @test save_trajectories.states === (:expected_reward, :surprise)
    end

    @testset "A kind that isn't given saves nothing, and SaveEverything() saves everything" begin
        @test SaveTrajectories().states ===
            SaveTrajectories().inputs ===
            SaveTrajectories().outputs ===
            SaveTrajectories().output_distributions ===
            ()
        @test SaveTrajectories(; states=:expected_reward).inputs === ()
        everything = SaveTrajectories(SaveEverything())
        @test everything.states ===
            everything.inputs ===
            everything.outputs ===
            everything.output_distributions ===
            SaveEverything()
        @test everything === SaveTrajectories(;
            states=SaveEverything(),
            inputs=SaveEverything(),
            outputs=SaveEverything(),
            output_distributions=SaveEverything(),
        )
    end

    @testset "It only takes names" begin
        @test_throws TypeError SaveTrajectories(; states="expected_reward")
        @test_throws TypeError SaveTrajectories(; states=(:expected_reward, 1))
        @test_throws TypeError SaveTrajectories(; states=true)
        #Also for a submodel and for its states
        @test_throws TypeError SaveTrajectories(; states=:perception => 1)
        @test_throws TypeError SaveTrajectories(;
            states=("perception" => :expected_reward,)
        )
        @test_throws TypeError SaveTrajectories(;
            states=:perception => (:expected_reward, "surprise")
        )
    end

    @testset "The states of a submodel are selected with its name and =>, down through its submodels" begin
        save_trajectories = SaveTrajectories(;
            states=(:surprise, :perception => :expected_reward)
        )
        @test save_trajectories.states === (:surprise, :perception => (:expected_reward,))
        #A single one without a tuple around it, and several states of the submodel in a tuple or a vector
        @test SaveTrajectories(;
            states=:perception => (:expected_reward, :prediction_error)
        ).states === (:perception => (:expected_reward, :prediction_error),)
        @test SaveTrajectories(;
            states=[:surprise, :perception => [:expected_reward, :prediction_error]]
        ) === SaveTrajectories(;
            states=(:surprise, :perception => (:expected_reward, :prediction_error))
        )
        #The states of a submodel of the submodel, with one => after the other or inside a tuple
        nested = SaveTrajectories(; states=:perception => :learning => :expected_reward)
        @test nested.states === (:perception => (:learning => (:expected_reward,),),)
        @test nested ===
            SaveTrajectories(; states=(:perception => (:learning => :expected_reward,),))
        #In the type, a submodel is a tuple of its name and the selection of its states
        @test CognitiveModeling.state_selection(nested) ===
            Val(((:perception, ((:learning, (:expected_reward,)),)),))
        @test CognitiveModeling.state_selection(save_trajectories) ===
            Val((:surprise, (:perception, (:expected_reward,))))
        #A submodel with everything is saved whole, as by its name alone, and one with nothing is left out
        @test SaveTrajectories(; states=(:surprise, :perception => SaveEverything())) ===
            SaveTrajectories(; states=(:surprise, :perception))
        @test SaveTrajectories(; states=(:surprise, :perception => ())) ===
            SaveTrajectories(; states=:surprise)
        @test CognitiveModeling.saves_nothing(SaveTrajectories(; states=:perception => ()))
        #The other kinds have no submodels
        @test_throws TypeError SaveTrajectories(; inputs=:perception => :reward)
        @test_throws TypeError SaveTrajectories(; outputs=(:choice, :response => :choice))
    end

    @testset "The selections are in the type, so constructing it from literals is inferred, and holding it costs nothing" begin
        from_literals() =
            SaveTrajectories(; outputs=:choice, states=(:expected_reward, :surprise))
        @test Base.return_types(from_literals, Tuple{})[1] ===
            SaveTrajectories{(:expected_reward, :surprise),(),(:choice,),(),true}
        without_initial_values() =
            SaveTrajectories(; states=:expected_reward, save_initial_timestep_values=false)
        @test Base.return_types(without_initial_values, Tuple{})[1] ===
            SaveTrajectories{(:expected_reward,),(),(),(),false}
        with_states_of_submodels() = SaveTrajectories(;
            states=(:surprise, :perception => (:expected_reward, :learning => :volatility)),
        )
        @test Base.return_types(with_states_of_submodels, Tuple{})[1] === SaveTrajectories{
            (:surprise, (:perception, (:expected_reward, (:learning, (:volatility,))))),
            (),
            (),
            (),
            true,
        }
        @test sizeof(with_states_of_submodels()) == 0
        @test @inferred(SaveTrajectories(SaveEverything())) isa SaveTrajectories
        @test @inferred(SaveTrajectories()) isa SaveTrajectories
        @test isbitstype(typeof(SaveTrajectories(; states=:expected_reward)))
        @test sizeof(SaveTrajectories(; states=(:expected_reward, :surprise))) == 0
    end
end

@testset "TrajectoryHistory" begin
    @testset "It keeps what was saved of each kind in a vector of its own" begin
        trajectories = CognitiveModeling.TrajectoryHistory(
            Any[(; expected_reward=0.5)], Any[(;)], Any[(;)], Any[(;)], Any[1]
        )
        @test trajectories.states == [(; expected_reward=0.5)]
        @test fieldnames(CognitiveModeling.TrajectoryHistory) ==
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
            :submodel_states,
        )
        @test trajectories.states[2] === (; expected_reward=0.75)
        @test trajectories.observations[1] === (; reward=BeforeFirstTimestep())
        @test trajectories.environment.states == fill((;), 3)
        @test trajectories.timestep_sizes === 1 &&
            trajectories.parameters_valid === true &&
            trajectories.n_timesteps === 2 &&
            trajectories.last_timestep === nothing
        #Given nine arguments, no state holds the state of a submodel
        @test trajectories.submodel_states === (; cognitive_model=(;), environment=(;))
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
