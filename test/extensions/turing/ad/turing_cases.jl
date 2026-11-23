using Turing
using Turing.DynamicPPL:
    LogDensityFunction,
    LogDensityProblems,
    getlogjoint_internal,
    LinkAll,
    InitFromParams,
    InitFromPrior,
    VarInfo
using Random: Xoshiro

#The code that the suites share, the package in test/common, whose extension for Turing.jl loads with Turing.jl above
#and gives the distributions of Distributions.jl. The simulation cases give real_numbers and with_real_numbers, which make
#the parameters of a case variables
using CommonTestFunctionality

# Test cases for the log density of Turing.jl models of sessions and of full models, which only this suite runs, since
# Turing.jl isn't loaded by the AD and Reactant suites of the package itself, in test/ad_and_reactant/ad and
# test/ad_and_reactant/reactant. Each case differentiates the log density the way a sampler does: through a
# LogDensityFunction made once, on the unconstrained values that samplers move in, with the parameters and the latent
# values in one vector in the order of the model. The cases are the shared inference cases whose variables are all
# continuous, since a gradient can't go through a discrete one, such as an imputed choice: every model with its
# environment, the imputed values of each continuous kind, sessions with continuous imputed values, and every parameter
# model, on sessions without missing values

#A model of the sessions of a case, whose real parameters are variables with flat priors, so that its log density is the
#sessions' own, and the gradient is that of the parameters and of the latent values. The environment's parameters are
#those of the case
Turing.@model function session_with_flat_priors(case)
    real_parameters ~
    Turing.filldist(Turing.Flat(), length(real_numbers(case.cognitive_parameters)))
    cognitive_parameters, _ = with_real_numbers(
        case.cognitive_parameters, real_parameters, 1
    )
    session ~ to_submodel(
        prefix(
            session_model(
                TuringBackend(),
                case.session_info,
                cognitive_parameters,
                case.environment_parameters,
            ),
            @varname(session),
        ),
        false,
    )
    return nothing
end
Turing.@model function sessions_with_flat_priors(case)
    real_parameters ~
    Turing.filldist(Turing.Flat(), length(real_numbers(case.cognitive_parameters)))
    cognitive_parameters, _ = with_real_numbers(
        case.cognitive_parameters, real_parameters, 1
    )
    sessions ~ to_submodel(
        multiple_sessions_model(
            TuringBackend(),
            case.sessions_info,
            cognitive_parameters,
            case.environment_parameters,
        ),
        false,
    )
    return nothing
end

#Whether every variable of a model is continuous, from values drawn from it: a floating-point number, or an array, a
#tuple or a NamedTuple of them, such as the coefficients of a regression
all_continuous(values) = all(name -> is_continuous(values[name]), keys(values))
is_continuous(value::AbstractFloat) = true
is_continuous(values::Union{AbstractArray,Tuple,NamedTuple}) = all(is_continuous, values)
is_continuous(value) = false

#A case of the log density of a model as a function of the vector of its variables, at the point given by an
#initialization strategy, on the unconstrained values. The point holds the parameters of the case, or values drawn from
#the priors of a full model, and latent values drawn from their distributions. Whether every variable is continuous is
#read from values drawn from `model_with_values`, the model itself or, for one with flat priors, which can't be drawn
#from, the model of its sessions with the case's parameters. The LogDensityFunction is made from the same point
function log_density_case(name, model, initialization, model_with_values=model)
    all_continuous(rand(Xoshiro(2), model_with_values)) || return nothing
    varinfo = VarInfo(Xoshiro(2), model, initialization, LinkAll())
    log_density_function = LogDensityFunction(model, getlogjoint_internal, varinfo)
    x = rand(Xoshiro(2), log_density_function, initialization)
    return (; name, f=x -> LogDensityProblems.logdensity(log_density_function, x), x)
end

#The sessions of the cases of one session: every model with its environment, also when the agent starts for the
#models where who starts makes a difference to what is differentiated, and the imputed actions and observations of each
#continuous kind
const SESSION_CASE_NAMES = [
    ["$(pair.name) with its environment, environment starts" for pair in MODEL_PAIRS];
    [
        "$name with its environment, agent starts" for name in
        ("Baseline", "EveryFeature", "Baseline with OutputNotEveryTimestepEnvironment")
    ];
    [
        "ContinuousOutput with a missing rating, environment starts",
        "ContinuousOutput with a missing reward observed, environment starts",
        "ManyAttributes with a missing confidence, environment starts",
        "ManyAttributes with a missing cue observed, environment starts",
        "VectorAttributes with a missing ratings, environment starts",
        "VectorAttributes with a missing rewards observed, environment starts",
        "EveryFeature with a missing rating, environment starts",
        "Baseline with SamplingWithinTimestepEnvironment with a missing reward observed, environment starts",
    ]
]

#The cases of full models with continuous missing values, beside every full model without missing values
const FULL_MODEL_CASE_NAMES_WITH_MISSING_VALUES = (
    "A parameter that is a vector", "Every feature with a regression"
)

#Every case whose variables are continuous, and which has variables
function turing_cases()
    cases = []
    for case in session_cases(DistributionsJl())
        #A model without real parameters has none to differentiate
        case.name in SESSION_CASE_NAMES || continue
        isempty(real_numbers(case.cognitive_parameters)) && continue
        model = session_with_flat_priors(case)
        initialization = InitFromParams(
            (; real_parameters=real_numbers(case.cognitive_parameters)), InitFromPrior()
        )
        with_values = session_model(
            TuringBackend(),
            case.session_info,
            case.cognitive_parameters,
            case.environment_parameters,
        )
        push!(cases, log_density_case(case.name, model, initialization, with_values))
    end
    for case in sessions_cases(DistributionsJl())
        isempty(real_numbers(case.cognitive_parameters)) && continue
        model = sessions_with_flat_priors(case)
        initialization = InitFromParams(
            (; real_parameters=real_numbers(case.cognitive_parameters)), InitFromPrior()
        )
        with_values = multiple_sessions_model(
            TuringBackend(),
            case.sessions_info,
            case.cognitive_parameters,
            case.environment_parameters,
        )
        push!(cases, log_density_case(case.name, model, initialization, with_values))
    end
    #Every full model without missing values, but the one that has none anyway
    full_models = [
        [
            merge(case, (; name="$(case.name), without missing values")) for
            case in full_model_cases(DistributionsJl(); missing_values=false) if
            case.name != "Independent parameters without missing values"
        ];
        filter(
            case -> case.name in FULL_MODEL_CASE_NAMES_WITH_MISSING_VALUES,
            full_model_cases(DistributionsJl()),
        )
    ]
    for case in full_models
        model = full_model(
            TuringBackend(),
            case.parameters_model,
            case.sessions_info;
            case.full_model_keywords...,
        )
        push!(cases, log_density_case(case.name, model, InitFromPrior()))
    end
    return filter(case -> case !== nothing && !isempty(case.x), cases)
end

const TURING_CASES = turing_cases()
