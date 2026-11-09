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
        @test propertynames(save_history) ==
            (:states, :inputs, :outputs, :output_distributions, :save_initial_values)
        @test save_history.output_distributions === ()
        @test_throws FieldError save_history.parameters
        #The kinds are named as the model's own, not by their role in a simulation
        @test_throws FieldError save_history.observations
        @test_throws FieldError save_history.actions
    end

    @testset "It saves the initial values by default, and can leave them out" begin
        @test SaveHistory().save_initial_values === true
        @test SaveHistory(SaveEverything()).save_initial_values === true
        @test SaveHistory(; states=:expected_reward, save_initial_values=false).save_initial_values ===
            false
        @test SaveHistory(SaveEverything(); save_initial_values=false).states ===
            SaveEverything()
        @test CognitiveModeling.saves_initial_values(SaveHistory()) &&
            !CognitiveModeling.saves_initial_values(
            SaveHistory(; save_initial_values=false)
        )
        @test_throws TypeError SaveHistory(; save_initial_values=1)
        @test SaveHistory(SaveEverything()).output_distributions === SaveEverything()
        @test SaveHistory(; output_distributions=:choice).output_distributions ===
            (:choice,)
        @test CognitiveModeling.saves_nothing(SaveHistory(; save_initial_values=false))
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
        @test_throws MethodError SaveHistory(; states="expected_reward")
        @test_throws MethodError SaveHistory(; states=(:expected_reward, 1))
        @test_throws MethodError SaveHistory(; states=true)
    end

    @testset "The selections are in the type, so constructing it from literals is inferred, and holding it costs nothing" begin
        from_literals() =
            SaveHistory(; outputs=:choice, states=(:expected_reward, :surprise))
        @test Base.return_types(from_literals, Tuple{})[1] ===
            SaveHistory{(:expected_reward, :surprise),(),(:choice,),(),true}
        without_initial_values() =
            SaveHistory(; states=:expected_reward, save_initial_values=false)
        @test Base.return_types(without_initial_values, Tuple{})[1] ===
            SaveHistory{(:expected_reward,),(),(),(),false}
        @test @inferred(SaveHistory(SaveEverything())) isa SaveHistory
        @test @inferred(SaveHistory()) isa SaveHistory
        @test isbitstype(typeof(SaveHistory(; states=:expected_reward)))
        @test sizeof(SaveHistory(; states=(:expected_reward, :surprise))) == 0
    end

    @testset "The REPL shows the selections on one line each" begin
        @test repr(
            "text/plain",
            SaveHistory(; outputs=:choice, states=(:expected_reward, :surprise)),
        ) == """
SaveHistory
States: expected_reward, surprise
Inputs: none
Outputs: choice
Output distributions: none
Initial values: saved"""
        @test repr(
            "text/plain", SaveHistory(SaveEverything(); save_initial_values=false)
        ) == """
SaveHistory
States: everything
Inputs: everything
Outputs: everything
Output distributions: everything
Initial values: not saved"""
    end

    @testset "select_names keeps what is selected" begin
        state = (; expected_reward=0.5, surprise=0.1, n_timesteps=3)
        @test CognitiveModeling.select_names(state, Val(SaveEverything())) === state
        @test CognitiveModeling.select_names(state, Val(())) === (;)
        @test CognitiveModeling.select_names(state, Val((:surprise, :expected_reward))) ===
            (; surprise=0.1, expected_reward=0.5)
        @test CognitiveModeling.select_names(
            (; reward=NoObservation(), cue=1), Val((:reward,))
        ) === (; reward=NoObservation())
        @test CognitiveModeling.select_names(NoObservation(), Val(())) === (;)
        @test CognitiveModeling.select_names([0.5], Val(())) === (;)
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(
            NoObservation(), Val((:reward,))
        )
        @test_throws FieldError CognitiveModeling.select_names(state, Val((:nope,)))
        @test_throws "can only select names from a NamedTuple" CognitiveModeling.select_names(
            [0.5], Val((:expected_reward,))
        )
        #The selections of a SaveHistory come as Vals, so that the names are known at compile time
        save_history = SaveHistory(; inputs=SaveEverything(), states=:expected_reward)
        @test CognitiveModeling.input_selection(save_history) === Val(SaveEverything())
        @test CognitiveModeling.output_selection(save_history) === Val(())
        @test CognitiveModeling.state_selection(save_history) === Val((:expected_reward,))
        @test CognitiveModeling.output_distribution_selection(
            SaveHistory(; output_distributions=:choice)
        ) === Val((:choice,))
    end

    @testset "The sizes of the timesteps are kept with a history, one for each timestep, when anything is saved" begin
        saves_states = SaveHistory(; states=:expected_reward, save_initial_values=false)
        @test CognitiveModeling.saved_timestep_sizes(1, 3, saves_states) == [1, 1, 1]
        @test CognitiveModeling.saved_timestep_sizes(
            0.5, 2, SaveHistory(SaveEverything(); save_initial_values=false)
        ) == [0.5, 0.5]
        #A vector of sizes is kept as it is, in a vector of its own
        sizes = [0.5, 2.0, 0.25]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, saves_states) == sizes
        #With the initial values, the sizes start with BeforeFirstTimestep for timestep 0
        @test isequal(
            CognitiveModeling.saved_timestep_sizes(
                sizes, 3, SaveHistory(; states=:expected_reward)
            ),
            [BeforeFirstTimestep(), 0.5, 2.0, 0.25],
        )
        @test isequal(
            CognitiveModeling.saved_timestep_sizes(1, 2, SaveHistory(SaveEverything())),
            [BeforeFirstTimestep(), 1, 1],
        )
        #Nothing is kept when nothing is saved
        @test CognitiveModeling.saved_timestep_sizes(1, 3, SaveHistory()) == Int[]
        @test CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory()) == Float64[]
        @test isempty(CognitiveModeling.saved_timestep_sizes(sizes, 3, SaveHistory())) &&
            length(sizes) == 3
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, saves_states)) isa
            Vector{Float64}
        @test @inferred(
            CognitiveModeling.saved_timestep_sizes(
                0.5, 2, SaveHistory(; states=:expected_reward)
            )
        ) isa Vector{Union{BeforeFirstTimestep,Float64}}
        @test @inferred(CognitiveModeling.saved_timestep_sizes(0.5, 2, SaveHistory())) isa
            Vector{Float64}
    end

    @testset "saves_nothing is only true when no kind has a selection" begin
        @test CognitiveModeling.saves_nothing(SaveHistory())
        @test !CognitiveModeling.saves_nothing(SaveHistory(; states=:expected_reward))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; inputs=SaveEverything()))
        @test !CognitiveModeling.saves_nothing(SaveHistory(; output_distributions=:choice))
    end
end
