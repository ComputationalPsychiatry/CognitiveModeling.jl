#An environment whose parameter has a default value
struct GivesRewardByDefault <: AbstractEnvironmentModel end
function CognitiveModeling.attribute_names(::GivesRewardByDefault)
    return ModelAttributes(; parameters=:scale, states=(), inputs=:choice, outputs=:reward)
end
CognitiveModeling.default_parameters(::GivesRewardByDefault) = (; scale=1.0)

#The sessions of the grouped table, with an environment or without one
function sessions_to_prepare(; keywords...)
    return MultipleSessionsInfo(
        SessionCounter(),
        GROUPS_TABLE;
        observation_columns=:outcome,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

@testset "prepare_full_model" begin
    sampled = IndependentSessionParameters((; probability=1))

    @testset "It returns the parts of a full model, with each session parameter model prepared for the sessions" begin
        sessions_info = sessions_to_prepare()
        by_session = FixedParametersBySession((;
            S1=(; probability=0.1),
            S2=(; probability=0.2),
            S3=(; probability=0.3),
            S4=(; probability=0.4),
        ))
        prepared = CognitiveModeling.prepare_full_model(by_session, sessions_info)
        @test keys(prepared) == (
            :cognitive_parameters_model,
            :environment_parameters_model,
            :downstream_model,
            :sessions_info,
        )
        @test prepared.cognitive_parameters_model.parameters_by_session ==
            (; probability=[0.1, 0.2, 0.3, 0.4])
        @test prepared.environment_parameters_model === NoParameters()
        @test prepared.downstream_model === NoDownstreamModel()
        @test prepared.sessions_info === sessions_info
    end

    @testset "With sampled environment parameters, every session runs its environment, and keeps the table" begin
        sessions_info = sessions_to_prepare(; environment=GivesReward())
        @test !any(session -> session.includes_environment, sessions_info.sessions)
        prepared = CognitiveModeling.prepare_full_model(
            sampled,
            sessions_info;
            environment_parameters_model=IdenticalSessionParameters((; scale=1)),
        )
        @test all(session -> session.includes_environment, prepared.sessions_info.sessions)
        @test prepared.sessions_info.table === sessions_info.table
        @test prepared.sessions_info.session_rows == sessions_info.session_rows
    end

    @testset "The parameters that a session parameter model gives are checked against those of the model" begin
        sessions_info = sessions_to_prepare()
        @test_throws "The session parameter model, a IndependentSessionParameters, gives the parameters (:bias,), which a cognitive model of type SessionCounter doesn't list. It lists the parameters (:probability,)." CognitiveModeling.prepare_full_model(
            IndependentSessionParameters((; probability=1, bias=2)), sessions_info
        )
        @test_throws "The session parameter model, a NoParameters, doesn't give the parameters (:probability,) of a cognitive model of type SessionCounter, which have no default values." CognitiveModeling.prepare_full_model(
            NoParameters(), sessions_info
        )
        #One of one's own doesn't tell its parameters, so nothing is checked
        @test CognitiveModeling.prepare_full_model(
            ParametersModelOfOwn(), sessions_info
        ).cognitive_parameters_model === ParametersModelOfOwn()
        #Sessions without an environment take no environment parameters
        @test_throws "The sessions have no environment model, but the session parameter model of the environment, a FixedSessionParameters, gives the parameters (:scale,)." CognitiveModeling.prepare_full_model(
            sampled,
            sessions_info;
            environment_parameters_model=FixedSessionParameters((; scale=2.0)),
        )
    end

    @testset "An environment needs its parameters only when a session runs it, and a default value is enough" begin
        #No session runs the environment, since nothing is missing, so its parameter isn't needed
        not_running = sessions_to_prepare(; environment=GivesReward())
        @test CognitiveModeling.prepare_full_model(sampled, not_running).environment_parameters_model ===
            NoParameters()
        #A session with something missing runs it, and then the parameter must be given
        running = sessions_to_prepare(;
            environment=GivesReward(),
            environment_save_history=SaveHistory(; outputs=:reward),
        )
        @test_throws "The session parameter model, a NoParameters, doesn't give the parameters (:scale,) of an environment model of type GivesReward, which have no default values." CognitiveModeling.prepare_full_model(
            sampled, running
        )
        with_default = sessions_to_prepare(;
            environment=GivesRewardByDefault(),
            environment_save_history=SaveHistory(; outputs=:reward),
        )
        @test CognitiveModeling.prepare_full_model(sampled, with_default).environment_parameters_model ===
            NoParameters()
    end

    @testset "The parts must be of the right kind" begin
        sessions_info = sessions_to_prepare()
        @test_throws "A session parameter model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it has type @NamedTuple{probability::Int64}." CognitiveModeling.prepare_full_model(
            (; probability=1), sessions_info
        )
        @test_throws "A session parameter model must be a subtype of AbstractSessionParametersModel" CognitiveModeling.prepare_full_model(
            sampled, sessions_info; environment_parameters_model=(; scale=2.0)
        )
        @test_throws "The downstream model must be a subtype of AbstractDownstreamModel, or NoDownstreamModel() for none, but it has type Symbol." CognitiveModeling.prepare_full_model(
            sampled, sessions_info; downstream_model=:none
        )
    end
    @testset "Fixed values join the session parameter model in a composite, and are checked against the model" begin
        sessions_info = sessions_to_prepare()
        own = ParametersModelOfOwn()
        fixed = FixedSessionParameters((; probability=0.3))
        #Without parameters of its own, the fixed values are all there is
        prepared = CognitiveModeling.prepare_full_model(
            NoParameters(), sessions_info; fixed_cognitive_parameters=(; probability=0.3)
        )
        @test prepared.cognitive_parameters_model === fixed
        #A model of one's own keeps its place next to the fixed values, and the parts of a composite stay side by side
        prepared = CognitiveModeling.prepare_full_model(
            own, sessions_info; fixed_cognitive_parameters=(; probability=0.3)
        )
        @test prepared.cognitive_parameters_model.parts === (own, fixed)
        @test CognitiveModeling.with_fixed_parameters(
            CompositeSessionParameters(own, own),
            (; probability=0.3),
            SessionCounter(),
            "fixed_cognitive_parameters",
        ).parts === (own, own, fixed)
        #Nothing fixed leaves the model as it is
        @test CognitiveModeling.with_fixed_parameters(
            own, (;), SessionCounter(), "fixed_cognitive_parameters"
        ) === own
        #The environment's parameters are fixed in the same way, which doesn't make the sessions include it
        with_environment = sessions_to_prepare(; environment=GivesReward())
        prepared = CognitiveModeling.prepare_full_model(
            sampled, with_environment; fixed_environment_parameters=(; scale=2.0)
        )
        @test prepared.environment_parameters_model ===
            FixedSessionParameters((; scale=2.0))
        @test !any(session -> session.includes_environment, prepared.sessions_info.sessions)
        #A parameter is either fixed or given, a fixed one must be listed, and sessions without an environment take none for it
        @test_throws "fixed_cognitive_parameters fixes the parameters (:probability,), which the session parameter model, a IndependentSessionParameters, gives too. A parameter is either fixed or given by the session parameter model." CognitiveModeling.prepare_full_model(
            sampled, sessions_info; fixed_cognitive_parameters=(; probability=0.3)
        )
        @test_throws "fixed_cognitive_parameters fixes the parameters (:bias,), which a cognitive model of type SessionCounter doesn't list. It lists the parameters (:probability,)." CognitiveModeling.prepare_full_model(
            own, sessions_info; fixed_cognitive_parameters=(; bias=1.0)
        )
        @test_throws "fixed_environment_parameters fixes the parameters (:scale,), but the sessions have no environment model." CognitiveModeling.prepare_full_model(
            sampled, sessions_info; fixed_environment_parameters=(; scale=1.0)
        )
    end
end
