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
