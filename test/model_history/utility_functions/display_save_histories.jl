@testset "Displaying save histories" begin
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
Initial timestep values: saved"""
        @test repr(
            "text/plain", SaveHistory(SaveEverything(); save_initial_timestep_values=false)
        ) == """
SaveHistory
States: everything
Inputs: everything
Outputs: everything
Output distributions: everything
Initial timestep values: not saved"""
    end
end
