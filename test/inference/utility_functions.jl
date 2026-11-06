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

    @testset "Missing actions can be replaced by NoAction, which leaves the recorded ones as they are" begin
        replaced = CognitiveModeling.without_missing_actions(actions)
        @test replaced.choice == [true, NoAction(), false, NoAction(), true]
        @test replaced.bet == [NoAction(), NoAction(), 1.0, 2.0, NoAction()]
        @test CognitiveModeling.recorded_actions(replaced) == CognitiveModeling.recorded_actions(actions)
        #The columns of the data aren't changed, and columns without missing actions keep their type
        @test actions.choice[2] === MissingAction()
        @test CognitiveModeling.without_missing_actions((; choice = [true, false])).choice isa Vector{Bool}
        @test CognitiveModeling.without_missing_actions((;)) === (;)
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
