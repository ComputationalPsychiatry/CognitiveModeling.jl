@testset "Values to condition on" begin
    distributions = (; choice=NormalByHand(0.0, 1.0), bet=NoValue())

    @testset "is_value_to_condition_on" begin
        @test CognitiveModeling.is_value_to_condition_on(1.0) &&
            CognitiveModeling.is_value_to_condition_on(true)
        @test !CognitiveModeling.is_value_to_condition_on(MissingValue()) &&
            !CognitiveModeling.is_value_to_condition_on(NoValue())
        @test CognitiveModeling.is_value_to_condition_on(1.0) &&
            !CognitiveModeling.is_value_to_condition_on(NoValue()) &&
            !CognitiveModeling.is_value_to_condition_on(MissingValue()) &&
            !CognitiveModeling.is_value_to_condition_on(UnimputedMissingValue())
    end

    @testset "The names to condition on are those with values in the data, and no observation is without an environment" begin
        session_info = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, false]);
            who_starts=EnvironmentStarts(),
        )
        @test CognitiveModeling.action_names_to_condition_on(session_info) ===
            Val((:choice,))
        @test CognitiveModeling.observation_names_to_condition_on(session_info) === Val(())
        with_environment = SessionInfo(
            ScoringModel(),
            (; reward=[1.0, 0.0]),
            (; choice=[true, MissingValue()]);
            who_starts=EnvironmentStarts(),
            environment=ScoringEnvironment(),
        )
        @test CognitiveModeling.action_names_to_condition_on(with_environment) ===
            Val((:choice,))
        @test CognitiveModeling.observation_names_to_condition_on(with_environment) ===
            Val((:reward,))
        #An action whose column can only hold the marker for a missing value has no values to condition on
        @test CognitiveModeling.action_names_to_condition_on(
            SessionInfo(
                ScoringModel(),
                (; reward=[1.0, 0.0]),
                (; choice=[MissingValue(), MissingValue()]);
                who_starts=EnvironmentStarts(),
                environment=ScoringEnvironment(),
            ),
        ) === Val(())
    end

    @testset "An output is conditioned on at a timestep when the data has a value, and such a value where the model gave nothing is an error" begin
        @test CognitiveModeling.find_outputs_to_condition_on(
            (; choice=0.5, bet=NoValue()), distributions, ScoringModel()
        ) === (; choice=true, bet=false)
        @test CognitiveModeling.find_outputs_to_condition_on(
            (; choice=MissingValue(), bet=MissingValue()), distributions, ScoringModel()
        ) === (; choice=false, bet=false)
        @test CognitiveModeling.find_outputs_to_condition_on((;), (;), ScoringModel()) ===
            (;)
        @test_throws "A cognitive model of type ScoringModel gave no output bet on a timestep where the data has 2.0 for it." CognitiveModeling.find_outputs_to_condition_on(
            (; choice=0.5, bet=2.0), distributions, ScoringModel()
        )
        #The same for the observations an environment gives
        @test CognitiveModeling.find_outputs_to_condition_on(
            (; reward=1.0), (; reward=NormalByHand(0.0, 1.0)), ScoringEnvironment()
        ) === (; reward=true)
        @test_throws "An environment model of type ScoringEnvironment gave no output reward on a timestep where the data has 1.0 for it." CognitiveModeling.find_outputs_to_condition_on(
            (; reward=1.0), (; reward=NoValue()), ScoringEnvironment()
        )
    end

    @testset "The vectors of distributions are typed by the first timestep, and widened when a later one differs" begin
        initial = CognitiveModeling.initialize_output_distributions_for_conditioning((;
            choice=NormalByHand(0.0, 1.0), bet=NoValue()
        ))
        @test initial.choice isa Vector{NormalByHand{Float64}} && isempty(initial.choice)
        @test initial.bet isa Vector{Union{}}
        stored = CognitiveModeling.store_output_distributions_for_conditioning(
            initial,
            (; choice=NormalByHand(0.0, 1.0), bet=NoValue()),
            (; choice=true, bet=false),
        )
        @test stored.choice == [NormalByHand(0.0, 1.0)] && isempty(stored.bet)
        stored = CognitiveModeling.store_output_distributions_for_conditioning(
            stored,
            (; choice=NormalByHand(1.0, 1.0), bet=NormalByHand(2.0, 0.5)),
            (; choice=false, bet=true),
        )
        @test stored.choice == [NormalByHand(0.0, 1.0)] &&
            stored.bet == [NormalByHand(2.0, 0.5)] &&
            stored.bet isa Vector{NormalByHand{Float64}}
        #A distribution of another type widens the vector
        stored = CognitiveModeling.store_output_distributions_for_conditioning(
            stored,
            (; choice=NormalByHand(1.0f0, 1.0f0), bet=NormalByHand(2.0, 0.5)),
            (; choice=true, bet=false),
        )
        @test stored.choice isa
              Vector{Union{NormalByHand{Float64},NormalByHand{Float32}}} &&
            length(stored.choice) == 2
    end
end
