@testset "Storing what is saved" begin
    @testset "The storage has the type of the first value, with room for timestep 0 when there is an initial value" begin
        storage = CognitiveModeling.initialize_storage(0.5, 3)
        @test storage isa Vector{Float64} && length(storage) == 3 && storage[1] == 0.5
        with_initial = CognitiveModeling.initialize_storage(BeforeFirstTimestep(), 0.5, 3)
        @test with_initial isa Vector{Union{BeforeFirstTimestep,Float64}} &&
            length(with_initial) == 4
        @test with_initial[1] === BeforeFirstTimestep() && with_initial[2] == 0.5
    end

    @testset "A value of the storage's type is stored in place, and another type moves the values to a wider vector" begin
        storage = CognitiveModeling.initialize_storage(0.5, 3)
        @test CognitiveModeling.store_value!(storage, 0.25, 2) === storage
        @test storage[2] == 0.25
        wider = CognitiveModeling.store_value!(storage, NoValue(), 3)
        @test wider isa Vector{Union{Float64,NoValue}} && wider !== storage
        @test wider[1] == 0.5 && wider[2] == 0.25 && wider[3] === NoValue()
    end

    @testset "A value is added at the end, and the first one gives an empty storage its type" begin
        storage = CognitiveModeling.push_value!(Vector{Union{}}(), 0.5)
        @test storage isa Vector{Float64} && storage == [0.5]
        @test CognitiveModeling.push_value!(storage, 0.25) === storage
        wider = CognitiveModeling.push_value!(storage, NoValue())
        @test wider isa Vector{Union{Float64,NoValue}} && length(wider) == 3
        @test wider[3] === NoValue()
    end
end
