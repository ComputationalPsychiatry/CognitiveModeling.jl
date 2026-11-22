# The benchmarks of the package: simulations, and for the models of each inference backend, the log density and its
# gradient with each automatic differentiation backend, on the baseline of the example models that the test suites
# share. The simulations and two of the models are small, so that they run quickly, and one full model has the size of a
# study, with simulated data. SUITE is the name that PkgBenchmark.jl and AirspeedVelocity.jl look for, and
# run_benchmarks.jl runs it.
using CognitiveModeling
using ADTypes: AutoEnzyme, AutoForwardDiff, AutoMooncake, AutoReverseDiff
using BenchmarkTools: BenchmarkGroup, @benchmarkable
using Enzyme: Const, Reverse, set_runtime_activity
using ForwardDiff: ForwardDiff
using Mooncake: Mooncake
using Random: Xoshiro
using ReverseDiff: ReverseDiff
using Turing
using Turing.DynamicPPL: LogDensityFunction, LogDensityProblems

## The models ##

# The example models of the test suites, with the distributions of Distributions.jl. The baseline learns the reward it
# expects for a choice of true, and chooses true by a softmax of it, and its environment rewards a choice of true with a
# probability, and a choice of false with half of it
const TEST_DIRECTORY = joinpath(@__DIR__, "..", "test")
include(joinpath(TEST_DIRECTORY, "example_models", "ExampleModels.jl"))
using .ExampleModels
include(joinpath(TEST_DIRECTORY, "example_models", "turing_methods.jl"))

const COGNITIVE_MODEL = Baseline(DistributionsJl())
const ENVIRONMENT = BaselineEnvironment(DistributionsJl())

## The simulations and the data ##

# A simulation of one session
function simulate_baseline_session(
    rng,
    save_history;
    n_timesteps=100,
    cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
)
    return simulate_session(
        COGNITIVE_MODEL,
        ENVIRONMENT;
        who_starts=EnvironmentStarts(),
        n_steps=n_timesteps,
        cognitive_parameters,
        environment_parameters=(; reward_probability=0.7),
        rng,
        cognitive_save_history=save_history,
        environment_save_history=save_history,
    )
end

# Simulated sessions as a table, with a row for each timestep. The first half of the sessions are of group a, and the
# second half of group b, which learns faster
function simulate_table(n_sessions, n_timesteps)
    save_history = SaveHistory(SaveEverything(); save_initial_timestep_values=false)
    group_of_each_session = [
        session <= n_sessions ÷ 2 ? "a" : "b" for session in 1:n_sessions
    ]
    simulations = [
        simulate_baseline_session(
            Xoshiro(session),
            save_history;
            n_timesteps,
            cognitive_parameters=(;
                learning_rate=group_of_each_session[session] == "a" ? 0.2 : 0.4,
                inverse_temperature=2.0,
            ),
        ) for session in 1:n_sessions
    ]
    return (;
        id=repeat(1:n_sessions; inner=n_timesteps),
        group=repeat(group_of_each_session; inner=n_timesteps),
        reward=[
            observation.reward for simulation in simulations for
            observation in simulation.observations
        ],
        choice=[
            action.choice for simulation in simulations for action in simulation.actions
        ],
    )
end

# The sessions of a table, with the baseline as their model
function sessions_of(table)
    return MultipleSessionsInfo(
        COGNITIVE_MODEL,
        table;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    )
end

## The Turing.jl models ##

# A model of one's own around a single session
Turing.@model function fit_single_session(session_info)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ LogNormal(0, 1)
    session ~ to_submodel(
        session_model(
            TuringBackend(), session_info, (; learning_rate, inverse_temperature)
        ),
        false,
    )
    return nothing
end

# The automatic differentiation backends that the gradient is benchmarked with. Enzyme has runtime activity, which Turing.jl's own tests run it with
const AD_BACKENDS = [
    "ForwardDiff" => AutoForwardDiff(),
    "ReverseDiff" => AutoReverseDiff(),
    "Mooncake" => AutoMooncake(),
    "Enzyme" => AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const),
]

# The benchmarks of a Turing.jl model: its log density, and the log density with its gradient for each backend, as a sampler
# evaluates them, at a point where every parameter is within the support of its prior
function benchmark_turing_model(model)
    group = BenchmarkGroup()
    log_density_function = LogDensityFunction(model)
    x = collect(range(0.3, 0.7; length=LogDensityProblems.dimension(log_density_function)))
    group["log density"] = @benchmarkable LogDensityProblems.logdensity(
        $log_density_function, $x
    )
    for (backend_name, adtype) in AD_BACKENDS
        with_gradient = LogDensityFunction(model; adtype)
        group["gradient, $backend_name"] = @benchmarkable LogDensityProblems.logdensity_and_gradient(
            $with_gradient, $x
        )
    end
    return group
end

## The suite ##

const SUITE = BenchmarkGroup()

SUITE["simulate"] = BenchmarkGroup()
SUITE["simulate"]["saving nothing"] = @benchmarkable simulate_baseline_session(
    rng, $(SaveHistory())
) setup = (rng = Xoshiro(1))
SUITE["simulate"]["saving everything"] = @benchmarkable simulate_baseline_session(
    rng, $(SaveHistory(SaveEverything()))
) setup = (rng = Xoshiro(1))

const BENCHMARK_SESSIONS = sessions_of(simulate_table(10, 100))

SUITE["turing"] = BenchmarkGroup()
SUITE["turing"]["single session"] = benchmark_turing_model(
    fit_single_session(first(BENCHMARK_SESSIONS.sessions))
)
SUITE["turing"]["full model, independent sessions"] = benchmark_turing_model(
    full_model(
        TuringBackend(),
        IndependentParameters((;
            learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 1)
        )),
        BENCHMARK_SESSIONS,
    ),
)

# A study of 50 sessions of 200 timesteps, whose parameters come from a distribution for each group, with a learning rate
# that a logistic function maps to between 0 and 1, and an inverse temperature that is positive
to_unit_interval(x) = 1 / (1 + exp(-x))
const STUDY_SESSIONS = sessions_of(simulate_table(50, 200))
SUITE["turing"]["study of 50 sessions of 200 timesteps, parameters by group"] = benchmark_turing_model(
    full_model(
        TuringBackend(),
        HierarchicalGroupedParameters((;
            learning_rate=HierarchicalParameter(
                (; mean=Normal(0, 1), sd=Exponential(1));
                transformation=to_unit_interval,
                group_by=:group,
            ),
            inverse_temperature=HierarchicalParameter(
                (; mean=Normal(0, 1), sd=Exponential(1));
                transformation=exp,
                group_by=:group,
            ),
        )),
        STUDY_SESSIONS,
    ),
)
