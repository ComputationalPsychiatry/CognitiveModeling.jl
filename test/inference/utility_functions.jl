@testset "Filling in missing observations" begin
    from_environment = (; reward = 1.0, cue = 2)

    @testset "A missing entry is taken from the environment, and the others are kept" begin
        @test CognitiveModeling.fill_missing_observations((; reward = MissingObservation(), cue = 5), from_environment) === (; reward = 1.0, cue = 5)
        @test CognitiveModeling.fill_missing_observations((; reward = 0.0, cue = 5), from_environment) === (; reward = 0.0, cue = 5)
        @test CognitiveModeling.fill_missing_observations((; reward = MissingObservation(), cue = MissingObservation()), from_environment) === from_environment
        #The names of the data decide, so the environment can give more
        @test CognitiveModeling.fill_missing_observations((; reward = MissingObservation()), from_environment) === (; reward = 1.0)
        @test CognitiveModeling.fill_missing_observations((;), from_environment) === (;)
    end

    @testset "An entry the environment doesn't give can't be filled in" begin
        @test_throws FieldError CognitiveModeling.fill_missing_observations((; punishment = MissingObservation()), from_environment)
    end

    @testset "It is inferred" begin
        @test @inferred(CognitiveModeling.fill_missing_observations((; reward = MissingObservation(), cue = 5), from_environment)) === (; reward = 1.0, cue = 5)
        @test @inferred(CognitiveModeling.fill_missing_observations((; reward = 0.5), from_environment)) === (; reward = 0.5)
    end

    @testset "has_missing_observation" begin
        @test CognitiveModeling.has_missing_observation((; reward = MissingObservation(), cue = 5))
        @test !CognitiveModeling.has_missing_observation((; reward = 1.0, cue = 5))
        @test !CognitiveModeling.has_missing_observation((;))
        #NoObservation isn't missing data
        @test !CognitiveModeling.has_missing_observation((; reward = NoObservation()))
    end
end

@testset "Preparing the data of a session" begin
    actions = (; choice = [true, MissingAction(), false, NoAction(), true], bet = Any[MissingAction(), MissingAction(), 1.0, 2.0, MissingAction()])

    @testset "The recorded actions of each name are kept in order, with a narrow type" begin
        recorded = CognitiveModeling.recorded_actions(actions)
        @test recorded.choice == [true, false, true] && recorded.choice isa Vector{Bool}
        @test recorded.bet == [1.0, 2.0] && recorded.bet isa Vector{Float64}
        @test CognitiveModeling.recorded_actions((; choice = [true, false])) == (; choice = [true, false])
        #An action without recorded values is left out
        @test CognitiveModeling.recorded_actions((; choice = [true], skipped = [MissingAction()], never = [NoAction(), NoAction()])) == (; choice = [true])
        @test CognitiveModeling.recorded_actions((;)) === (;)
    end

    @testset "is_recorded" begin
        @test CognitiveModeling.is_recorded(1.0) && CognitiveModeling.is_recorded(true)
        @test !CognitiveModeling.is_recorded(MissingAction()) && !CognitiveModeling.is_recorded(NoAction())
    end

    @testset "A row has the value of each column at one timestep" begin
        @test CognitiveModeling.row((; reward = [1.0, 0.0], cue = [1, 2]), 2) === (; reward = 0.0, cue = 2)
        @test @inferred(CognitiveModeling.row((; reward = [1.0, 0.0]), 1)) === (; reward = 1.0)
        @test CognitiveModeling.row((;), 1) === (;)
    end
end
