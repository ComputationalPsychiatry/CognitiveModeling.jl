# # Models from SequentialSamplingModels.jl as cognitive models
#
# SequentialSamplingModels.jl has distributions over a choice and the time it took, such as
# the drift diffusion model (DDM). A value of one is a `NamedTuple` with both, such as
# `(choice = 1, rt = 0.53)`. This page uses a DDM as the distribution that a cognitive model
# samples its response from, simulates the model on a task with two choices, and fits it to
# the responses of three simulated participants with Turing.jl. It was run with
# SequentialSamplingModels.jl 0.13.3. Both packages export a function named `simulate`, so
# only `DDM` is taken from SequentialSamplingModels.jl.

using CognitiveModeling
using SequentialSamplingModels: DDM
using Turing
using Random: Xoshiro
using Statistics: mean

# ## The cognitive model
#
# On each trial, the model sees a stimulus, 1 or 2, and responds with a choice and a
# response time from a DDM. The evidence drifts toward the upper boundary, choice 1, when
# the stimulus is 1, and toward the lower boundary, choice 2, when it is 2. The parameters
# are those of the DDM: the drift rate, the separation of the boundaries, the starting point
# as a share of the separation, and the non-decision time. The model keeps no state between
# trials.

struct DriftDiffusion <: AbstractCognitiveModel end

#No state carries over from one trial to the next
CognitiveModeling.initialize_state(model::DriftDiffusion, parameters) = (;)

function CognitiveModeling.run_timestep(model::DriftDiffusion, context::ModelContext)
    #The drift points toward the boundary of the choice that matches the stimulus
    parameters = context.parameters
    drift_rate = if context.input.stimulus == 1
        parameters.drift_rate
    else
        -parameters.drift_rate
    end

    #The choice and the response time are sampled together from the DDM
    response = DDM(
        drift_rate,
        parameters.boundary_separation,
        parameters.starting_point,
        parameters.non_decision_time,
    )
    return ModelOutput((; response))
end

function CognitiveModeling.attribute_names(model::DriftDiffusion)
    return ModelAttributes(;
        parameters=(:drift_rate, :boundary_separation, :starting_point, :non_decision_time),
        states=(),
        inputs=:stimulus,
        outputs=:response,
    )
end

# The DDM takes a separation and a non-decision time of zero or more, and a starting point
# between 0 and 1, which the domains of the parameters say:

function CognitiveModeling.parameter_domains(model::DriftDiffusion)
    return (;
        boundary_separation=ParameterDomain(; lower=0),
        starting_point=ParameterDomain(; lower=0, upper=1),
        non_decision_time=ParameterDomain(; lower=0),
    )
end

# ## The task
#
# To simulate the model, the environment shows stimulus 1 or 2 at random on each trial. It
# takes the response, but does nothing with it:

struct RandomStimuli <: AbstractEnvironmentModel end

#The environment keeps no state either
CognitiveModeling.initialize_state(model::RandomStimuli, parameters) = (;)

function CognitiveModeling.run_timestep(model::RandomStimuli, context::ModelContext)
    #Each stimulus is as likely as the other
    return ModelOutput((; stimulus=Categorical(2)))
end

function CognitiveModeling.attribute_names(model::RandomStimuli)
    return ModelAttributes(; parameters=(), states=(), inputs=:response, outputs=:stimulus)
end

# ## Simulating
#
# Each response is a `NamedTuple` with the choice and the response time, as the DDM gives
# it:

#The parameters of one participant
parameters = (;
    drift_rate=1.5, boundary_separation=1.2, starting_point=0.5, non_decision_time=0.3
)

#Five trials, where the environment shows the stimulus before the model responds
trajectories = simulate(
    DriftDiffusion(),
    RandomStimuli();
    who_starts=EnvironmentStarts(),
    n_steps=5,
    cognitive_parameters=parameters,
    environment_parameters=(;),
    rng=Xoshiro(1),
    cognitive_save_history=SaveHistory(
        SaveEverything(); save_initial_timestep_values=false
    ),
)
trajectories.actions

# ## Data
#
# The data are 150 trials of each of three participants, with drift rates of 0.5, 1.5 and
# 2.5, in a table with a row for each trial and the choice and the response time in columns
# of their own:

#The trials of one participant, simulated with their drift rate
function simulate_participant(drift_rate, n_trials, seed)
    parameters = (;
        drift_rate, boundary_separation=1.2, starting_point=0.5, non_decision_time=0.3
    )
    trajectories = simulate(
        DriftDiffusion(),
        RandomStimuli();
        who_starts=EnvironmentStarts(),
        n_steps=n_trials,
        cognitive_parameters=parameters,
        environment_parameters=(;),
        rng=Xoshiro(seed),
        cognitive_save_history=SaveHistory(
            SaveEverything(); save_initial_timestep_values=false
        ),
    )
    stimuli = [observation.stimulus for observation in trajectories.observations]
    responses = [action.response for action in trajectories.actions]
    return stimuli, responses
end

#The table, with a row for each trial of each participant
true_drift_rates = [0.5, 1.5, 2.5]
n_trials = 150
trials = [
    simulate_participant(drift_rate, n_trials, seed) for
    (seed, drift_rate) in enumerate(true_drift_rates)
]
data = (;
    participant=repeat(["P1", "P2", "P3"]; inner=n_trials),
    stimulus=reduce(vcat, [stimuli for (stimuli, responses) in trials]),
    choice=reduce(vcat, [[r.choice for r in responses] for (stimuli, responses) in trials]),
    rt=reduce(vcat, [[r.rt for r in responses] for (stimuli, responses) in trials]),
)

# `action_columns = (; choice = :choice, rt = :rt)` builds the model's one action, the
# response, from the two columns, as a `NamedTuple` with those names on each row:

sessions_info = MultipleSessionsInfo(
    DriftDiffusion(),
    data;
    observation_columns=:stimulus,
    action_columns=(; choice=:choice, rt=:rt),
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
)
first(sessions_info.sessions).actions.response[1:3]

# ## Fitting
#
# Each participant gets their own parameters. The DDM throws an error, rather than giving a
# density of zero, for a response time shorter than the non-decision time. So the prior of
# the non-decision time stays below the shortest response time in the data, or the sampler
# stops at the first value above it:

#The priors, the same for every participant
priors = (;
    drift_rate=Normal(1, 1),
    boundary_separation=LogNormal(0, 0.5),
    starting_point=Beta(5, 5),
    non_decision_time=Uniform(0, minimum(data.rt)),
)

#The full model, and the samples of its posterior
model = full_model(TuringBackend(), IndependentParameters(priors), sessions_info)
chain = sample(Xoshiro(1), model, NUTS(), 500)

# The fitted parameters of each participant, as averages over the samples:

session_parameters = get_session_parameters(model, chain)
for (participant, true_drift_rate) in zip((:P1, :P2, :P3), true_drift_rates)
    drift_rate = mean(session_parameters[@varname($participant.drift_rate)])
    non_decision_time = mean(session_parameters[@varname($participant.non_decision_time)])
    println(
        participant,
        ": drift rate ",
        round(drift_rate; digits=2),
        " (",
        true_drift_rate,
        "), non-decision time ",
        round(non_decision_time; digits=3),
        " (0.3)",
    )
end

# ## Missing responses
#
# A trial without a response, or with only one of its two columns, has a missing response.
# A missing response can't be imputed: a DDM can't be a variable that a sampler draws, since
# its choice is discrete and its response time continuous. So such trials are left out of
# the fit, with `impute_missing_actions = false`:

#The fifth trial of the first participant has a choice but no response time
data_with_missing = merge(data, (; rt=Vector{Union{Missing,Float64}}(data.rt)))
data_with_missing.rt[5] = missing

#The response of that trial is left out
sessions_info_with_missing = MultipleSessionsInfo(
    DriftDiffusion(),
    data_with_missing;
    observation_columns=:stimulus,
    action_columns=(; choice=:choice, rt=:rt),
    session_columns=:participant,
    who_starts=EnvironmentStarts(),
    impute_missing_actions=false,
)
first(sessions_info_with_missing.sessions).actions.response[4:6]

# ## Automatic differentiation
#
# NUTS uses ForwardDiff.jl for its gradients by default. Mooncake.jl, and Enzyme.jl with
# runtime activity, give the same gradients on this model. ReverseDiff.jl doesn't run on
# it: a function of SequentialSamplingModels.jl that computes the density takes two numbers
# that must be of the same type, which ReverseDiff.jl's tracked numbers aren't.
