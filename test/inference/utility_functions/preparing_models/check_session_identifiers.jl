@testset "Checking the identifiers of the sessions" begin
    @testset "The identifier of every session must give a symbol name of its own" begin
        @test CognitiveModeling.check_unique_session_symbol_names(["A", "B"]) === nothing
        @test CognitiveModeling.check_unique_session_symbol_names([
            (; id="A", block=1), (; id="A", block=2)
        ]) === nothing
        @test_throws "The session identifier \"A\" is given for more than one session. Every session must have an identifier of its own." CognitiveModeling.check_unique_session_symbol_names([
            "A", "B", "A"
        ])
        #Two identifiers which aren't equal can still give the same symbol name
        @test_throws "The session identifiers 1 and \"1\" both give the symbol name 1, which the chains of a fit and FixedSessionParametersBySession find a session by. The identifier of every session must give a symbol name of its own." CognitiveModeling.check_unique_session_symbol_names(
            Any[1, "1"]
        )
    end
end
