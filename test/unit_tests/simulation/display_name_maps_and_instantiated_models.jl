#Has more parameters than a display lists, and a large array in its state
struct HasManyParameters <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::HasManyParameters, parameters)
    return (; values=collect(1.0:100.0))
end
function CognitiveModeling.attribute_names(model::HasManyParameters)
    return ModelAttributes(;
        parameters=(:a, :b, :c, :d, :e, :f), states=:values, inputs=(), outputs=()
    )
end

@testset "Displaying name maps and instantiated models" begin
    @testset "A name map shows as it is written" begin
        @test repr(NameMap(:payoff => :reward, :cue)) == "NameMap(:payoff => :reward, :cue)"
        @test repr(NameMap()) == "NameMap()"
    end

    @testset "The REPL shows its model's kind and type, the timesteps, and the values of the parameters and the state" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        simulate!(agent, (; reward=1.0))
        @test repr("text/plain", agent) == """
        -- Instantiated cognitive model --
        Type: Baseline
        Timesteps simulated: 1
        Parameters: learning_rate = 0.5, inverse_temperature = 2.0
        State: expected_reward = 0.75"""
        #An environment without parameters or a state
        @test repr("text/plain", InstantiatedModel(OnlyCues(), (;))) == """
        -- Instantiated environment model --
        Type: OnlyCues
        Timesteps simulated: 0
        Parameters: none
        State: none"""
    end

    @testset "The REPL shows only how many values there are when there are more than five, and shortens large arrays" begin
        agent = InstantiatedModel(HasManyParameters(), (; a=1, b=2, c=3, d=4, e=5, f=6))
        shown = repr("text/plain", agent; context=:limit => true)
        @test occursin("\nParameters: 6 (see .parameters)\n", shown)
        @test occursin("State: values = [1.0, 2.0, 3.0", shown) && occursin("…", shown)
        #Without a limit, as when printing to a file, everything is listed
        unlimited = repr("text/plain", agent)
        @test occursin("Parameters: a = 1, b = 2, c = 3, d = 4, e = 5, f = 6", unlimited)
        @test occursin("100.0]", unlimited) && !occursin("…", unlimited)
    end

    @testset "The REPL says so when the stopping criterion is met" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        environment = InstantiatedModel(
            WithStoppingCriterionEnvironment(ByHand()),
            (; reward_probability=0.8, last_timestep=2),
        )
        @test !occursin("Stopping criterion", repr("text/plain", environment))
        simulate!(agent, environment; who_starts=EnvironmentStarts(), n_steps=10)
        @test endswith(
            repr("text/plain", environment),
            "\nStopping criterion met: reset! to simulate it again",
        )
        @test !occursin("Stopping criterion", repr("text/plain", reset!(environment)))
    end

    @testset "On one line it shows its model's type and kind, and how many timesteps it has run" begin
        agent = InstantiatedModel(Baseline(ByHand()), BASELINE_PARAMETERS)
        @test repr(agent) ==
            "Baseline (instantiated cognitive model, 0 timesteps simulated)"
        simulate!(agent, (; reward=1.0))
        @test repr(agent) == "Baseline (instantiated cognitive model, 1 timestep simulated)"
        @test occursin(
            "OnlyCues (instantiated environment model, 0 timesteps simulated)",
            repr("text/plain", [InstantiatedModel(OnlyCues(), (;))]),
        )
    end
end
