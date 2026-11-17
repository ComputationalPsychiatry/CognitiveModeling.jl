# The benchmarks of the package: simulations, and for the models of each inference backend, the log density and its
# gradient with each automatic differentiation backend. They are kept small, so that they run on every push.
# SUITE is the name that PkgBenchmark.jl and AirspeedVelocity.jl look for, and run_benchmarks.jl runs it.
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

# A Rescorla-Wagner model which takes or leaves an option, by a softmax of the reward it expects
struct BenchmarkRescorlaWagner <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::BenchmarkRescorlaWagner, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.run_timestep(
    model::BenchmarkRescorlaWagner, context::ModelContext
)
    V₋₁ = context.previous_state.expected_reward
    r = context.input.reward
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + context.parameters.learning_rate * (r - V₋₁)
    return ModelOutput(
        (; choice=BernoulliLogit(context.parameters.inverse_temperature * (Vₜ - 0.5))),
        (; expected_reward=Vₜ),
    )
end
function CognitiveModeling.attribute_names(model::BenchmarkRescorlaWagner)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# An environment which gives a reward with a fixed probability, whatever the choice
struct BenchmarkRewards <: AbstractEnvironmentModel end
CognitiveModeling.initialize_state(environment::BenchmarkRewards, parameters) = (;)
function CognitiveModeling.run_timestep(
    environment::BenchmarkRewards, context::ModelContext
)
    return ModelOutput((; reward=Bernoulli(context.parameters.reward_probability)))
end
function CognitiveModeling.attribute_names(environment::BenchmarkRewards)
    return ModelAttributes(;
        parameters=:reward_probability, states=(), inputs=:choice, outputs=:reward
    )
end

## The simulations and the data ##

const N_TIMESTEPS = 100
const N_SESSIONS = 10

# A simulation of one session
function simulate_session(rng, save_history)
    return simulate(
        BenchmarkRescorlaWagner(),
        BenchmarkRewards();
        who_starts=EnvironmentStarts(),
        n_steps=N_TIMESTEPS,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.7),
        rng,
        cognitive_save_history=save_history,
        environment_save_history=save_history,
    )
end

# The simulated sessions as a table, with a row for each timestep
function simulate_table()
    save_history = SaveHistory(SaveEverything(); save_initial_timestep_values=false)
    simulations = [simulate_session(Xoshiro(i), save_history) for i in 1:N_SESSIONS]
    return (;
        id=repeat(1:N_SESSIONS; inner=N_TIMESTEPS),
        reward=[
            observation.reward for simulation in simulations for
            observation in simulation.observations
        ],
        choice=[
            action.choice for simulation in simulations for action in simulation.actions
        ],
    )
end

## The Turing.jl models ##

# A model of one's own around a single session
Turing.@model function fit_single_session(session_info)
    learning_rate ~ Beta(2, 2)
    inverse_temperature ~ LogNormal(0, 1)
    session ~ to_submodel(
        turing_behavioral_session(session_info, (; learning_rate, inverse_temperature)),
        false,
    )
    return nothing
end

# The automatic differentiation backends that the gradient is benchmarked with. Enzyme has the settings that Turing.jl documents for it
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
SUITE["simulate"]["saving nothing"] = @benchmarkable simulate_session(rng, $(SaveHistory())) setup = (
    rng = Xoshiro(1)
)
SUITE["simulate"]["saving everything"] = @benchmarkable simulate_session(
    rng, $(SaveHistory(SaveEverything()))
) setup = (rng = Xoshiro(1))

const BENCHMARK_SESSIONS = MultipleSessionsInfo(
    BenchmarkRescorlaWagner(),
    simulate_table();
    observation_columns=:reward,
    action_columns=:choice,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)

SUITE["turing"] = BenchmarkGroup()
SUITE["turing"]["single session"] = benchmark_turing_model(
    fit_single_session(first(BENCHMARK_SESSIONS.sessions))
)
SUITE["turing"]["full model, independent sessions"] = benchmark_turing_model(
    turing_full_model(
        IndependentParameters((;
            learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 1)
        )),
        BENCHMARK_SESSIONS,
    ),
)
