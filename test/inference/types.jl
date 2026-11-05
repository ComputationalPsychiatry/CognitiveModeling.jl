@testset "Types for data" begin
    @testset "The markers for missing data are their own types, apart from those for nothing happening" begin
        @test MissingAction() === MissingAction()
        @test MissingObservation() === MissingObservation()
        @test MissingAction() !== NoAction()
        @test MissingObservation() !== NoObservation()
        @test !(MissingAction() isa NoAction) && !(MissingObservation() isa NoObservation)
        @test isbitstype(MissingAction) && isbitstype(MissingObservation)
    end

    @testset "They can stand in for one entry of the actions and observations" begin
        action = (; choice = MissingAction(), bet = 1.0)
        observation = (; reward = MissingObservation(), cue = 1)
        @test action.choice isa MissingAction && action.bet == 1.0
        @test observation.reward isa MissingObservation && observation.cue == 1
        #Unlike missing, they don't pass through arithmetic without an error
        @test_throws MethodError MissingAction() + 1
        @test_throws MethodError MissingObservation() * 2.0
    end
end
