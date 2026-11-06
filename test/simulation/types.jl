@testset "NameMap" begin
    @testset "It keeps the names it takes and the names it gives in the type" begin
        @test NameMap(:payoff => :reward, :cue => :stimulus) === NameMap{(:payoff, :cue), (:reward, :stimulus)}()
        #A single name keeps its name
        @test NameMap(:payoff => :reward, :cue) === NameMap{(:payoff, :cue), (:reward, :cue)}()
        @test NameMap(:reward) === NameMap{(:reward,), (:reward,)}()
        #A map can pass nothing on
        @test NameMap() === NameMap{(), ()}()
        @test isbitstype(typeof(NameMap(:payoff => :reward))) && sizeof(NameMap(:payoff => :reward)) == 0
    end

    @testset "Each name can only be there once, and only names are taken" begin
        @test_throws "A NameMap can only give each name once, but it gives (:reward, :reward)." NameMap(:payoff => :reward, :cue => :reward)
        @test_throws "A NameMap can only take each name once, but it takes (:payoff, :payoff)." NameMap(:payoff => :reward, :payoff => :bonus)
        @test_throws MethodError NameMap("payoff" => "reward")
        @test_throws MethodError NameMap(:payoff => 1)
        @test_throws MethodError NameMap((:payoff => :reward,))
    end

    @testset "Constructing it from literals is inferred" begin
        from_literals() = NameMap(:payoff => :reward, :cue)
        @test Base.return_types(from_literals, Tuple{})[1] === NameMap{(:payoff, :cue), (:reward, :cue)}
    end

    @testset "It shows as it is written" begin
        @test repr(NameMap(:payoff => :reward, :cue)) == "NameMap(:payoff => :reward, :cue)"
        @test repr(NameMap()) == "NameMap()"
    end

    @testset "map_names selects the outputs the map takes, and gives them their new names" begin
        outputs = (; payoff = 1.0, cue = 2, block = 3)
        @test CognitiveModeling.map_names(outputs, nothing) === outputs
        @test CognitiveModeling.map_names(outputs, NameMap(:payoff => :reward)) === (; reward = 1.0)
        @test CognitiveModeling.map_names(outputs, NameMap(:cue, :payoff => :reward)) === (; cue = 2, reward = 1.0)
        @test CognitiveModeling.map_names(outputs, NameMap()) === (;)
        #Markers are passed on like any value
        @test CognitiveModeling.map_names((; payoff = NoObservation()), NameMap(:payoff => :reward)) === (; reward = NoObservation())
        @test_throws FieldError CognitiveModeling.map_names(outputs, NameMap(:nope => :reward))
        @test @inferred(CognitiveModeling.map_names(outputs, NameMap(:cue, :payoff => :reward))) isa NamedTuple
    end
end
