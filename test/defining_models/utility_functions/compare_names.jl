@testset "Comparing names" begin
    @testset "A name is among the names of a tuple or not, and so is every name of another tuple" begin
        @test CognitiveModeling.has_name(:reward, (:cue, :reward))
        @test !CognitiveModeling.has_name(:choice, (:cue, :reward)) &&
            !CognitiveModeling.has_name(:choice, ())
        @test CognitiveModeling.has_every_name((:reward,), (:cue, :reward))
        @test !CognitiveModeling.has_every_name((:reward, :choice), (:cue, :reward))
        #No names are among any names
        @test CognitiveModeling.has_every_name((), (:cue, :reward)) &&
            CognitiveModeling.has_every_name((), ())
    end

    @testset "Two tuples have the same names when each has every name of the other, in any order" begin
        @test CognitiveModeling.same_names((:cue, :reward), (:reward, :cue))
        @test CognitiveModeling.same_names((), ())
        @test !CognitiveModeling.same_names((:cue,), (:cue, :reward)) &&
            !CognitiveModeling.same_names((:cue, :reward), (:cue,))
    end
end
