# # Models from control theory as cognitive models
#
# Control theory describes how a system is steered towards a goal from what is observed of
# it, which also makes it a model of behavior, such as of tracking a moving target with a
# hand. The packages of the JuliaControl organization give the parts: DiscretePIDs.jl a PID
# controller, LowLevelParticleFilters.jl Kalman filters and particle filters, which can
# model perception, and ControlSystemsBase.jl linear systems, which can model what is
# controlled.
#
# This page shows a PID controller as a cognitive model, which controls a simulated plant as
# its environment, and fits its gains with Turing.jl. It then shows a Kalman filter as the
# perception of a cognitive model, and ends with how a model with a controller of
# ModelPredictiveControl.jl is fitted. Its code isn't run when the documentation is built,
# to keep the documentation's environment small. The package's tests run it instead, with
# DiscretePIDs.jl 1.0, ControlSystemsBase.jl 1.25 and LowLevelParticleFilters.jl 3.38.

using CognitiveModeling
using ControlSystemsBase: ss, c2d
using DiscretePIDs: DiscretePID
using Distributions: Normal, LogNormal
using LowLevelParticleFilters: KalmanFilter, SimpleMvNormal, correct!, predict!
using Random: Xoshiro
using Turing

# ## The plant as the environment
#
# The environment is what the agent controls: a first-order lag from a force to a position,
# made with ControlSystemsBase.jl as a system in continuous time and discretized with the
# size of a timestep. Its state is the plant's state and the number of the timestep. On each
# timestep it applies the agent's force, and gives the target, which moves along two sines,
# and the position, observed with noise.

struct FirstOrderPlant{Tsystem} <: AbstractEnvironmentModel
    system::Tsystem
end

function FirstOrderPlant(; time_constant=0.5, timestep_size=0.1)
    return FirstOrderPlant(
        c2d(ss(-1 / time_constant, 1 / time_constant, 1, 0), timestep_size)
    )
end

function target_at(timestep, timestep_size)
    return sin(0.5 * timestep * timestep_size) + 0.5 * sin(1.3 * timestep * timestep_size)
end

function CognitiveModeling.initialize_state(environment::FirstOrderPlant, parameters)
    return (; plant_state=zeros(environment.system.nx), timestep=0)
end

# On the first timestep the agent hasn't acted yet, which [`AbstractNoValue`](@ref) covers,
# and no force is applied.

function CognitiveModeling.apply_timestep_model(
    environment::FirstOrderPlant, context::ModelContext
)

    ## Unpack the context ##
    (; plant_state, timestep) = context.previous_state
    force = context.input.force isa AbstractNoValue ? 0.0 : context.input.force
    (; A, B, C) = environment.system

    ## Update state ##
    #The plant's state after the force has acted for one timestep
    xₜ = A * plant_state + B * [force]

    ## Compute observations ##
    #The position, observed with noise, and the target of the next timestep
    position = only(C * xₜ)
    target = target_at(timestep + 1, environment.system.Ts)

    return ModelOutput(
        (;
            target=Deterministic(target),
            position=Normal(position, context.parameters.observation_noise),
        ),
        (; plant_state=xₜ, timestep=timestep + 1),
    )
end

function CognitiveModeling.attribute_names(environment::FirstOrderPlant)
    return ModelAttributes(;
        parameters=:observation_noise,
        states=(:plant_state, :timestep),
        inputs=:force,
        outputs=(:target, :position),
    )
end

# ## A PID controller as the agent
#
# The agent sees the target and the position, and pushes with the force its PID controller
# gives, with motor noise. The controller's gains are the agent's parameters: the
# proportional gain `K`, the integral time `Ti` and the derivative time `Td`. Its state is
# the controller of DiscretePIDs.jl, made from the parameters when the state is initialized,
# so that its numbers are of the type the parameters have when a model is fitted.

struct PIDController <: AbstractCognitiveModel
    timestep_size::Float64
end

# The limits of the force are given, since the controller's default limits need a number
# type that ReverseDiff.jl's tracked numbers don't have.

function CognitiveModeling.initialize_state(model::PIDController, parameters)
    (; K, Ti, Td) = parameters
    return (; pid=DiscretePID(; K, Ti, Td, Ts=model.timestep_size, umin=(-Inf), umax=Inf))
end

function CognitiveModeling.apply_timestep_model(model::PIDController, context::ModelContext)

    ## Unpack the context ##
    pid = context.previous_state.pid
    σ = context.parameters.motor_noise

    ## Update state and compute the action ##
    #The controller's force for the target and the position, which updates the integral and the derivative it keeps
    uₜ = pid(context.input.target, context.input.position)

    return ModelOutput((; force=Normal(uₜ, σ)), (; pid))
end

function CognitiveModeling.attribute_names(model::PIDController)
    return ModelAttributes(;
        parameters=(:K, :Ti, :Td, :motor_noise),
        states=:pid,
        inputs=(:target, :position),
        outputs=:force,
    )
end

# The controller is changed in place on each timestep, so each state that is saved must be a
# copy.

function CognitiveModeling.prepare_state_for_saving(model::PIDController, state)
    return (; pid=deepcopy(state.pid))
end

# The controller could also be kept as its three numbers, its integral, its derivative and
# its previous error, and made again on each timestep from them and the parameters, with
# `DiscretePID(; K, Ti, Td, Ts, I, D, yold)`. Then nothing is changed in place, and nothing
# needs copying.

# ## Simulating
#
# A session of the agent controlling the plant, with the observations and the actions saved:

timestep_size = 0.1
simulation = simulate(
    PIDController(timestep_size),
    FirstOrderPlant(; timestep_size);
    who_starts=EnvironmentStarts(),
    n_steps=150,
    cognitive_parameters=(; K=2.0, Ti=1.0, Td=0.1, motor_noise=0.2),
    environment_parameters=(; observation_noise=0.05),
    cognitive_save_history=SaveHistory(;
        inputs=(:target, :position), outputs=:force, save_initial_timestep_values=false
    ),
    rng=Xoshiro(1),
)

# ## Fitting
#
# Three sessions with different gains, simulated the same way, make a table with one row for
# each timestep:

true_parameters = [
    (; K=2.0, Ti=1.0, Td=0.1, motor_noise=0.2),
    (; K=1.0, Ti=0.5, Td=0.05, motor_noise=0.2),
    (; K=1.5, Ti=2.0, Td=0.05, motor_noise=0.2),
]
simulations = map(enumerate(true_parameters)) do (session, parameters)
    return simulate(
        PIDController(timestep_size),
        FirstOrderPlant(; timestep_size);
        who_starts=EnvironmentStarts(),
        n_steps=150,
        cognitive_parameters=parameters,
        environment_parameters=(; observation_noise=0.05),
        cognitive_save_history=SaveHistory(;
            inputs=(:target, :position), outputs=:force, save_initial_timestep_values=false
        ),
        rng=Xoshiro(20 + session),
    )
end
table = (;
    id=repeat(["S1", "S2", "S3"]; inner=150),
    target=[observation.target for s in simulations for observation in s.observations],
    position=[observation.position for s in simulations for observation in s.observations],
    force=[action.force for s in simulations for action in s.actions],
)

# The gains of each session are fitted with a prior of their own, and NUTS samples them:

sessions_info = MultipleSessionsInfo(
    PIDController(timestep_size),
    table;
    observation_columns=(:target, :position),
    action_columns=:force,
    session_columns=:id,
    who_starts=EnvironmentStarts(),
)
full_model = turing_full_model(
    IndependentSessionParameters((;
        K=LogNormal(0, 1),
        Ti=LogNormal(0, 1),
        Td=LogNormal(-2, 1),
        motor_noise=LogNormal(-1, 1),
    )),
    sessions_info,
)
chain = sample(Xoshiro(11), full_model, NUTS(300, 0.65), 300; progress=false)

# The chain has the gains of each session under `K.session`, `Ti.session` and `Td.session`.
#
# NUTS needs the gradient of the log density. ForwardDiff.jl, which is Turing.jl's default,
# ReverseDiff.jl and Mooncake.jl 0.5 gave the right gradient for this model, when checked
# against finite differences. Enzyme.jl 0.13 failed: with runtime activity it stopped Julia
# with an error from its type analysis, and without it, it threw an error for the controller
# that is changed in place.

# ## A Kalman filter as perception
#
# A cognitive model can also perceive with a filter. In this task a target drifts as a
# random walk, the agent sees it only through noisy cues, and it reports where it believes
# the target is. The agent's belief is a Kalman filter of LowLevelParticleFilters.jl, which
# assumes a random walk with the agent's own drift and cue noise, its parameters.

struct DriftingTarget <: AbstractEnvironmentModel end

function CognitiveModeling.initialize_state(environment::DriftingTarget, parameters)
    return (; target_position=0.0)
end

function CognitiveModeling.apply_timestep_model(
    environment::DriftingTarget, context::ModelContext
)
    #The target drifts, which is sampled with the generator of the context
    target_position =
        context.previous_state.target_position +
        context.parameters.drift * randn(context.rng)
    return ModelOutput(
        (; cue=Normal(target_position, context.parameters.cue_noise)), (; target_position)
    )
end

function CognitiveModeling.attribute_names(environment::DriftingTarget)
    return ModelAttributes(;
        parameters=(:drift, :cue_noise),
        states=:target_position,
        inputs=:report,
        outputs=:cue,
    )
end

# The filter is made from the parameters, with a starting belief of the parameters' number
# type, since a filter keeps the types it starts with. Its matrices are fixed and already in
# discrete time, so its check of them, which computes their eigenvalues, is turned off.

struct KalmanTracker <: AbstractCognitiveModel end

function CognitiveModeling.initialize_state(model::KalmanTracker, parameters)
    (; assumed_drift, assumed_cue_noise) = parameters
    T = typeof(assumed_drift * assumed_cue_noise)
    starting_belief = SimpleMvNormal(zeros(T, 1), ones(T, 1, 1))
    kalman_filter = KalmanFilter(
        [1.0;;],
        [0.0;;],
        [1.0;;],
        [0.0;;],
        fill(assumed_drift^2, 1, 1),
        fill(assumed_cue_noise^2, 1, 1),
        starting_belief;
        check=false,
    )
    return (; kalman_filter)
end

function CognitiveModeling.apply_timestep_model(model::KalmanTracker, context::ModelContext)

    ## Unpack the context ##
    kalman_filter = context.previous_state.kalman_filter
    no_input = zero(kalman_filter.x)

    ## Update state ##
    #Correct the belief with the cue, as a vector of the belief's number type, keep its mean, and predict the next timestep
    correct!(kalman_filter, no_input, [context.input.cue] .+ zero(kalman_filter.x))
    believed_position = kalman_filter.x[1]
    predict!(kalman_filter, no_input)

    ## Compute the action ##
    #The report is where the agent believes the target is, with motor noise
    return ModelOutput(
        (; report=Normal(believed_position, context.parameters.motor_noise)),
        (; kalman_filter),
    )
end

function CognitiveModeling.attribute_names(model::KalmanTracker)
    return ModelAttributes(;
        parameters=(:assumed_drift, :assumed_cue_noise, :motor_noise),
        states=:kalman_filter,
        inputs=:cue,
        outputs=:report,
    )
end

function CognitiveModeling.prepare_state_for_saving(model::KalmanTracker, state)
    return (; kalman_filter=deepcopy(state.kalman_filter))
end

# A session of the tracker:

tracking = simulate(
    KalmanTracker(),
    DriftingTarget();
    who_starts=EnvironmentStarts(),
    n_steps=100,
    cognitive_parameters=(; assumed_drift=0.3, assumed_cue_noise=0.5, motor_noise=0.1),
    environment_parameters=(; drift=0.3, cue_noise=0.5),
    cognitive_save_history=SaveHistory(;
        inputs=:cue, outputs=:report, save_initial_timestep_values=false
    ),
    rng=Xoshiro(2),
)

# For a filter of one dimension, as here, the filter written by hand is faster. The package
# earns its place for the extended and unscented Kalman filters and the particle filters,
# which are more work to write.

# ## Model predictive control
#
# ModelPredictiveControl.jl gives controllers that plan the next actions by optimizing them
# over a horizon, together with state estimators and plant models. Its controllers run
# inside a cognitive model in the same way, but their gradients can't be computed yet, so a
# model with one is fitted with a sampler that needs no gradient, such as a slice sampler
# from SliceSampling.jl, or Turing.jl's `MH` with a covariance matrix, for a random walk.
# Turing.jl computes one gradient when it looks for initial values for such a sampler, so
# give it an `adtype` of finite differences, which only evaluates the model with ordinary
# numbers:
#
# ```julia
# using FiniteDifferences: central_fdm
# using SliceSampling: RandPermGibbs, SliceSteppingOut
#
# sampler = externalsampler(
#     RandPermGibbs(SliceSteppingOut(2.0));
#     adtype = AutoFiniteDifferences(; fdm = central_fdm(3, 1)),
# )
# ```
#
# Each step of these samplers evaluates the whole model, so they suit a single session, or
# sessions fitted one at a time, better than a hierarchical model of many sessions. A
# `LinMPC` with soft constraints warns about the conditioning of its problem every time it
# is made, which is on every evaluation of the model. When only the actions are bounded,
# hard bounds, with `Cwt = Inf`, give the same controller without the warning.
