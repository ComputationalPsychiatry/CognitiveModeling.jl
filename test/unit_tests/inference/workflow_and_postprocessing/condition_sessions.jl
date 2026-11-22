@testset "What condition_sessions is made of" begin
    @testset "with_data prepares a session again with other data and its own settings" begin
        session_info = SessionInfo(
            SessionCounter(),
            COUNTER_REWARDS,
            (; choice=fill(SimulatedValue(), 3));
            who_starts=AgentStarts(),
            timestep_sizes=[1.0, 2.0, 1.0],
            check_parameter_domains=false,
        )
        with_choices = CognitiveModeling.with_data(
            session_info,
            session_info.observations,
            (; choice=[true, false, true]);
            check_stopping_criterion=nothing,
        )
        @test with_choices.actions.choice == [true, false, true]
        @test with_choices.actions_to_condition_on.choice == [true, false, true]
        @test with_choices.timestep_sizes == [1.0, 2.0, 1.0]
        @test with_choices.who_starts === AgentStarts()
        @test with_choices.check_parameter_domains === false
        #The stopping criterion is checked from the data the copy has, after its last timestep with data
        @test with_choices.first_timestep_to_check_stopping == 4
    end

    @testset "with_common_element_types gives the columns of a name the types of their entries in every session" begin
        columns_of_sessions = [
            (; choice=Any[true, SimulatedValue()], bet=Any[1.0, 2.0]),
            (; choice=Any[false, true], bet=Any[3, 4.0]),
        ]
        typed = CognitiveModeling.with_common_element_types(columns_of_sessions)
        @test all(columns -> columns.choice isa Vector{Union{SimulatedValue,Bool}}, typed)
        @test all(columns -> columns.bet isa Vector{Union{Int,Float64}}, typed)
        @test typed[1].choice == [true, SimulatedValue()] && typed[2].bet == [3, 4.0]
        #Columns whose entries have one type in every session get that type
        @test only(
            CognitiveModeling.with_common_element_types([(; choice=Any[true, false])])
        ).choice isa Vector{Bool}
    end
end
