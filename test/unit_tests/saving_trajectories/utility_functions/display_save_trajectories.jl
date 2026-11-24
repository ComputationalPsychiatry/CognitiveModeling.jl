@testset "Displaying SaveTrajectories" begin
    @testset "The REPL shows the selections on one line each" begin
        @test repr(
            "text/plain",
            SaveTrajectories(; outputs=:choice, states=(:expected_reward, :surprise)),
        ) == """
SaveTrajectories
States: expected_reward, surprise
Inputs: none
Outputs: choice
Output distributions: none
Initial timestep values: saved"""
        @test repr(
            "text/plain",
            SaveTrajectories(SaveEverything(); save_initial_timestep_values=false),
        ) == """
SaveTrajectories
States: everything
Inputs: everything
Outputs: everything
Output distributions: everything
Initial timestep values: not saved"""
    end

    @testset "The states of a submodel are shown with the name of the submodel and a dot before them" begin
        @test repr(
            "text/plain",
            SaveTrajectories(;
                states=(:surprise, :perception => (:expected_reward, :learning => :mean))
            ),
        ) == """
SaveTrajectories
States: surprise, perception.expected_reward, perception.learning.mean
Inputs: none
Outputs: none
Output distributions: none
Initial timestep values: saved"""
    end
end
