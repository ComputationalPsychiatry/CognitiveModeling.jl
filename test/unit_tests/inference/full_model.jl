#The functions that every inference backend has throw an error which names the package to load when the backend's
#methods aren't there, as here, where Turing.jl isn't loaded
@testset "Models of an inference backend without the package of the backend" begin
    session_info = SessionInfo(
        SessionCounter(),
        (; reward=[1.0, 0.0]),
        (; choice=[true, false]);
        who_starts=EnvironmentStarts(),
    )
    sessions_info = MultipleSessionsInfo([session_info], ["S1"])
    @test_throws "There is no full_model method for TuringBackend() with arguments of the types" full_model(
        TuringBackend(), FixedParameters((; probability=0.5)), sessions_info
    )
    #A single session is passed on as a set of one session, so the same error names the package
    @test_throws "There is no full_model method for TuringBackend() with arguments of the types" full_model(
        TuringBackend(), session_info, FixedParameters((; probability=0.5))
    )
    @test_throws "There is no session_model method for TuringBackend() with arguments of the types" session_model(
        TuringBackend(), session_info, (; probability=0.5)
    )
    @test_throws "There is no multiple_sessions_model method for TuringBackend() with arguments of the types" multiple_sessions_model(
        TuringBackend(),
        sessions_info,
        (; probability=[0.5]);
        parallelization=SerialSessions(),
    )
    @test_throws ArgumentError session_model(
        TuringBackend(), session_info, (; probability=0.5)
    )
end
