# # Models from POMDPs.jl as cognitive models
#
# POMDPs.jl describes a task as a problem object, and has solvers which give a policy for
# it, and updaters which keep a belief about the hidden state. A cognitive model can keep
# such a belief as its state, update it with their updater, and choose from the values their
# policy gives each action.
#
# This page shows such a cognitive model. It doesn't simulate or fit it yet, and its code
# isn't run when the documentation is built, to keep the documentation's environment small.
# The package's tests run it instead, with POMDPs.jl 1.0, POMDPTools.jl 1.1 and QMDP.jl 0.1.

using CognitiveModeling
using Distributions: DiscreteNonParametric
using LogExpFunctions: softmax
using POMDPs: POMDPs
using POMDPTools: DiscreteUpdater, actionvalues
using QMDP: QMDPSolver

# ## A Bayesian observer with a solved policy
#
# The cognitive model holds the problem, a policy for it and a belief updater. The problem
# is solved once, when the cognitive model is created, so nothing is solved on a timestep.

struct BayesianObserver{Tpomdp,Tpomdp_policy,Tpomdp_updater} <: AbstractCognitiveModel
    pomdp::Tpomdp
    pomdp_policy::Tpomdp_policy
    pomdp_updater::Tpomdp_updater
end

function BayesianObserver(pomdp)
    return BayesianObserver(
        pomdp, POMDPs.solve(QMDPSolver(), pomdp), DiscreteUpdater(pomdp)
    )
end

# Its state is the belief, which starts as the problem's distribution over initial states.

function CognitiveModeling.initialize_state(model::BayesianObserver, parameters)
    return (;
        belief=POMDPs.initialize_belief(
            model.pomdp_updater, POMDPs.initialstate(model.pomdp)
        )
    )
end

# On each timestep, the belief is updated from the previous action and the observation that
# followed it, and the action probabilities are a softmax over the values the policy gives
# each action under that belief, with the softmax from LogExpFunctions.jl. On the first
# timestep there is no previous action, which [`AbstractNoValue`](@ref) covers, and the
# belief is left as it is.

function CognitiveModeling.apply_model(model::BayesianObserver, context::ModelContext)

    ## Unpack the context ##
    belief = context.previous_state.belief
    previous_choice = context.previous_output.choice
    β = context.parameters.inverse_temperature

    ## Update state ##
    #Update the belief from the previous action and what was observed after it
    if !(previous_choice isa AbstractNoValue)
        belief = POMDPs.update(
            model.pomdp_updater, belief, previous_choice, context.input.observation
        )
    end

    ## Compute action probabilities ##
    #The value the policy gives each action under the belief
    Qₜ = actionvalues(model.pomdp_policy, belief)
    #Softmax over the action values, as a distribution over the problem's own actions
    Pₜ = DiscreteNonParametric(collect(POMDPs.actions(model.pomdp)), softmax(β .* Qₜ))

    return ModelOutput((; choice=Pₜ), (; belief))
end

function CognitiveModeling.attribute_names(model::BayesianObserver)
    return ModelAttributes(;
        parameters=:inverse_temperature,
        states=:belief,
        inputs=:observation,
        outputs=:choice,
    )
end

# The updater returns a new belief on each timestep, so the saved states need no copying.
#
# A problem from POMDPModels.jl gives a cognitive model for its task, such as
# `BayesianObserver(TigerPOMDP())`. The problem can be used as the environment too, through
# [`CommonRLEnvironment`](@ref), as in
# `CommonRLEnvironment(POMDPCommonRLEnv{Any}(TigerPOMDP()); action_name = :choice)`. The
# `{Any}` makes POMDPTools.jl give the observations as they are, which the updater needs.
# Without it, as with `convert`, each observation is put in an array, and the updater throws
# an error. That environment also gives `reward` and `terminated`, which this cognitive
# model doesn't list, so a [`NameMap`](@ref) must select the observation for it:
# `observation_map = NameMap(:observation)`.
#
# The learners in TabularTDLearning.jl can't be used this way. They are solvers which run
# every episode themselves, so there is no single timestep to call.

# ## Fitting
#
# This page doesn't fit the cognitive model yet. When it is fitted with a sampler that uses
# gradients, use Enzyme.jl in reverse mode with runtime activity as the automatic
# differentiation backend, which for Turing.jl is
# `AutoEnzyme(; mode = set_runtime_activity(Reverse), function_annotation = Const)`. It gave
# the right gradient of the log density of a session for this cognitive model, when checked
# against finite differences with Enzyme.jl 0.13. So did ForwardDiff.jl and ReverseDiff.jl,
# while Mooncake.jl and Enzyme's forward mode threw an error.
#
# Use the softmax from LogExpFunctions.jl, as above, and not one written by hand. Written
# in one line, a softmax got a wrong gradient for this cognitive model from Enzyme's reverse
# mode, with no error, which the page [Automatic differentiation and Reactant](@ref)
# describes under its pitfalls.
