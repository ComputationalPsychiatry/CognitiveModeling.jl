# Sessions run by hand: what the model of a session in an inference backend does, without the backend. The log
# density of the values in the data and of the imputed values is added up directly, and everything else is done by
# the functions that the package has for use within such a model: those in
# src/inference/utility_functions/within_models, and those that they and a session call.
#
# The Reactant suite compiles these sessions, since the Turing.jl models can't be compiled with Reactant, so that
# the functions are known to work with an inference backend which can. The steps follow turing_behavioral_session,
# turing_run_all_timesteps, turing_run_timestep_both_models and turing_run_timestep_one_model in
# ext/CognitiveModelingTuringExt, so a change in what those call belongs here too.
using CognitiveModeling:
    AbstractTimestepModel,
    timestep_model,
    initialize_state,
    stopping_criterion_of_either_model,
    merge_default_parameters,
    bundle_parameters,
    check_match_attribute_names,
    check_parameters_in_domains,
    check_model_output,
    parameters_are_valid,
    combine_parameters_valid,
    masks_flag,
    masks_bool,
    map_names,
    get_timestep_size
using CognitiveModeling:
    has_environment,
    check_environment_parameters,
    last_timestep_with_data,
    remake_session_without_environment,
    select_row,
    build_named_entry,
    has_missing_value,
    action_names_to_condition_on,
    observation_names_to_condition_on,
    find_outputs_to_condition_on,
    initialize_output_distributions_for_conditioning,
    store_output_distributions_for_conditioning,
    select_timestep_values_to_save,
    select_initial_timestep_values_to_save,
    with_empty_environment_trajectories,
    timestep_sizes_until,
    data_continues_after,
    check_stopping_criterion_not_met,
    session_parameters_valid,
    masking_fitted_session_after_first_timestep,
    masking_fitted_session_before_timestep,
    stop_masked_fitted_session,
    count_masked_fitted_timestep,
    log_density_of_masked_ending
using CognitiveModeling:
    select_names,
    saves_initial_timestep_values,
    map_saved,
    initialize_all_storage,
    store_timestep_values!,
    saved_trajectories,
    initialize_storage,
    store_value!
using Random: default_rng

## Distributions with their log densities written out, since Distributions.jl can't be compiled with Reactant ##

# A Rescorla-Wagner model which gives its action as a Taction of its probability. Its state starts as a Float64, whatever
# number type the parameters have
struct LearnsRewards{Taction} <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::LearnsRewards, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::LearnsRewards{Taction}, context::ModelContext
) where {Taction}
    V₋₁ = context.previous_state.expected_reward
    α = context.parameters.learning_rate
    β = context.parameters.inverse_temperature
    Vₜ = if context.input.reward isa AbstractNoValue
        V₋₁
    else
        V₋₁ + α * (context.input.reward - V₋₁)
    end
    return ModelOutput(
        (; choice=Taction(inv(1 + exp(-β * (Vₜ - 0.5))))), (; expected_reward=Vₜ)
    )
end
function CognitiveModeling.attribute_names(model::LearnsRewards)
    return ModelAttributes(;
        parameters=(:learning_rate, :inverse_temperature),
        states=:expected_reward,
        inputs=:reward,
        outputs=:choice,
    )
end

# A choice that is made with a probability
struct ChoiceByHand{T}
    probability::T
end
function log_density(distribution::ChoiceByHand, value::Bool)
    return value ? log(distribution.probability) : log(1 - distribution.probability)
end

# A number around a mean, with a standard deviation of a half
struct AroundMean{T}
    mean::T
end
function log_density(distribution::AroundMean, value)
    return -(2 * (value - distribution.mean))^2 / 2 - log(0.5) - log(2π) / 2
end

## Models whose outputs are those distributions ##

# An environment whose reward is around the size of the action, and which counts the timesteps in its state
struct NoisyRewardsFollowActions <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::NoisyRewardsFollowActions, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::NoisyRewardsFollowActions, context::ModelContext
)
    action = context.input.choice isa AbstractNoValue ? 0.5 : context.input.choice
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; reward=AroundMean(context.parameters.reward_size * action)), (; n_timesteps)
    )
end
function CognitiveModeling.attribute_names(environment::NoisyRewardsFollowActions)
    return ModelAttributes(;
        parameters=:reward_size, states=:n_timesteps, inputs=:choice, outputs=:reward
    )
end

# The same environment under other names than the model's, which maps of names pass on
struct NoisyPayoffsFollowLevers <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(
    environment::NoisyPayoffsFollowLevers, parameters
)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::NoisyPayoffsFollowLevers, context::ModelContext
)
    action = context.input.lever isa AbstractNoValue ? 0.5 : context.input.lever
    n_timesteps = context.previous_state.n_timesteps + 1
    return ModelOutput(
        (; payoff=AroundMean(context.parameters.reward_size * action)), (; n_timesteps)
    )
end
function CognitiveModeling.attribute_names(environment::NoisyPayoffsFollowLevers)
    return ModelAttributes(;
        parameters=:reward_size, states=:n_timesteps, inputs=:lever, outputs=:payoff
    )
end

# The same environment as NoisyRewardsFollowActions, which ends after four timesteps
struct NoisyRewardsUntilEnd <: AbstractEnvironmentModel end
function CognitiveModeling.initialize_state(environment::NoisyRewardsUntilEnd, parameters)
    return (; n_timesteps=0)
end
function CognitiveModeling.timestep_model(
    environment::NoisyRewardsUntilEnd, context::ModelContext
)
    return timestep_model(NoisyRewardsFollowActions(), context)
end
function CognitiveModeling.attribute_names(environment::NoisyRewardsUntilEnd)
    return CognitiveModeling.attribute_names(NoisyRewardsFollowActions())
end
function CognitiveModeling.stopping_criterion(
    environment::NoisyRewardsUntilEnd, state, parameters
)
    return state.n_timesteps >= 4
end

# A model which takes no action before it has received a reward, which is on the first timestep when the agent starts
struct ActsAfterFirstReward <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::ActsAfterFirstReward, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::ActsAfterFirstReward, context::ModelContext
)
    if context.input.reward isa AbstractNoValue
        return ModelOutput((; choice=NoValue()), context.previous_state)
    end
    return timestep_model(LearnsRewards{ChoiceByHand}(), context)
end
function CognitiveModeling.attribute_names(model::ActsAfterFirstReward)
    return CognitiveModeling.attribute_names(LearnsRewards{ChoiceByHand}())
end

## Models whose parameters can turn invalid, or whose stopping criterion depends on their state ##

# A Rescorla-Wagner model which sets parameters_valid to false once its expected reward is above 0.95, which a compiler
# traces when the parameters are traced
struct TurnsInvalidByHand <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::TurnsInvalidByHand, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::TurnsInvalidByHand, context::ModelContext)
    output = timestep_model(LearnsRewards{ChoiceByHand}(), context)
    return ModelOutput(
        output.output_distribution,
        output.new_state;
        parameters_valid=output.new_state.expected_reward <= 0.95,
    )
end
function CognitiveModeling.attribute_names(model::TurnsInvalidByHand)
    return CognitiveModeling.attribute_names(LearnsRewards{ChoiceByHand}())
end

# A Rescorla-Wagner model whose stopping criterion is met once its expected reward is above 0.8
struct StopsAboveByHand <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::StopsAboveByHand, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(model::StopsAboveByHand, context::ModelContext)
    return timestep_model(LearnsRewards{ChoiceByHand}(), context)
end
function CognitiveModeling.attribute_names(model::StopsAboveByHand)
    return CognitiveModeling.attribute_names(LearnsRewards{ChoiceByHand}())
end
function CognitiveModeling.stopping_criterion(model::StopsAboveByHand, state, parameters)
    return state.expected_reward > 0.8
end

# A stand-in for a Bool that a compiler traces, which ifelse takes as a condition, and which holds a Bool in plain Julia and
# a traced Bool under Reactant, so that a session is masked in both, and the plain one is the reference of the compiled one
struct MasksByHand{T}
    value::T
end
function Base.ifelse(condition::MasksByHand, value_if_true, value_if_false)
    return ifelse(condition.value, value_if_true, value_if_false)
end
Base.:!(flag::MasksByHand) = MasksByHand(!flag.value)
Base.:&(first::MasksByHand, second::MasksByHand) = MasksByHand(first.value & second.value)
Base.:&(first::MasksByHand, second::Bool) = MasksByHand(first.value & second)
Base.:&(first::Bool, second::MasksByHand) = MasksByHand(first & second.value)
Base.:|(first::MasksByHand, second::MasksByHand) = MasksByHand(first.value | second.value)
Base.:|(first::MasksByHand, second::Bool) = MasksByHand(first.value | second)
Base.:|(first::Bool, second::MasksByHand) = MasksByHand(first | second.value)

# A Rescorla-Wagner model with a choice around its expected reward, whose stopping criterion is met once its expected reward
# is above 0.8, as the stand-in, so that a session of it is masked from its first check of the criterion on
struct StopsAboveMaskedByHand <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::StopsAboveMaskedByHand, parameters)
    return (; expected_reward=0.5)
end
function CognitiveModeling.timestep_model(
    model::StopsAboveMaskedByHand, context::ModelContext
)
    return timestep_model(LearnsRewards{AroundMean}(), context)
end
function CognitiveModeling.attribute_names(model::StopsAboveMaskedByHand)
    return CognitiveModeling.attribute_names(LearnsRewards{AroundMean}())
end
function CognitiveModeling.stopping_criterion(
    model::StopsAboveMaskedByHand, state, parameters
)
    return MasksByHand(state.expected_reward > 0.8)
end

## What an inference backend keeps while it evaluates a model ##

# The imputed values, which a backend samples as latent variables and which are taken from a vector here, in the order
# that the session imputes them, and the log density, which every imputed value and every value in the data adds to
mutable struct EvaluationByHand{Tlatents,Tlog_density}
    latents::Tlatents
    n_latents_taken::Int
    log_density::Tlog_density
end
function take_latent!(evaluation::EvaluationByHand)
    evaluation.n_latents_taken += 1
    return evaluation.latents[evaluation.n_latents_taken]
end

## A session ##

# A session without an environment: every observation comes from the data
function session_by_hand(
    session_info::SessionInfo{<:Any,NoEnvironment},
    cognitive_parameters,
    environment_parameters,
    evaluation,
)
    check_environment_parameters(session_info, environment_parameters)
    cognitive_model = session_info.cognitive_model

    #Fill in the default parameters, and initialize the state with them, bundled as a model with submodels reads them
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    bundled_cognitive_parameters = bundle_parameters(cognitive_model, cognitive_parameters)
    initial_cognitive_state = initialize_state(
        cognitive_model, bundled_cognitive_parameters
    )

    #Check the names against attribute_names, and that each parameter is within its domain
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters,
        initial_cognitive_state,
        session_info.previous_action,
    )
    if session_info.check_parameter_domains
        check_parameters_in_domains(cognitive_model, cognitive_parameters)
    end

    #Run the timesteps from where the session starts, with the values arranged as those of a timestep
    previous = (;
        state=initial_cognitive_state,
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        environment=(; state=nothing),
    )
    return timesteps_by_hand(
        session_info, bundled_cognitive_parameters, (;), previous, evaluation
    )
end

# The same with an environment, which fills in the observations that are missing in the data
function session_by_hand(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel},
    cognitive_parameters,
    environment_parameters,
    evaluation,
)
    #A session with nothing missing runs without its environment, and has an empty record of it
    if !session_info.includes_environment
        return with_empty_environment_trajectories(
            session_by_hand(
                remake_session_without_environment(session_info),
                cognitive_parameters,
                (;),
                evaluation,
            ),
        )
    end
    cognitive_model = session_info.cognitive_model
    environment = session_info.environment

    #Fill in the default parameters of both models, and initialize both states with them
    cognitive_parameters = merge_default_parameters(cognitive_model, cognitive_parameters)
    environment_parameters = merge_default_parameters(environment, environment_parameters)
    bundled_cognitive_parameters = bundle_parameters(cognitive_model, cognitive_parameters)
    bundled_environment_parameters = bundle_parameters(environment, environment_parameters)
    initial_cognitive_state = initialize_state(
        cognitive_model, bundled_cognitive_parameters
    )
    initial_environment_state = initialize_state(
        environment, bundled_environment_parameters
    )

    #Check the names of both models against attribute_names, and that each parameter is within its domain
    check_match_attribute_names(
        cognitive_model,
        cognitive_parameters,
        initial_cognitive_state,
        session_info.previous_action,
    )
    check_match_attribute_names(
        environment,
        environment_parameters,
        initial_environment_state,
        session_info.previous_environment_observation,
    )
    if session_info.check_parameter_domains
        check_parameters_in_domains(cognitive_model, cognitive_parameters)
        check_parameters_in_domains(environment, environment_parameters)
    end

    #Run the timesteps from where the session starts: the environment's values are under environment, in its own names
    previous = (;
        state=initial_cognitive_state,
        observation=session_info.previous_observation,
        action=session_info.previous_action,
        environment=(;
            state=initial_environment_state,
            observation=session_info.previous_environment_observation,
            action=map_names(session_info.previous_action, session_info.action_map),
        ),
    )
    return timesteps_by_hand(
        session_info,
        bundled_cognitive_parameters,
        bundled_environment_parameters,
        previous,
        evaluation,
    )
end

# Several sessions, each with its own row of the parameters, which are given as a vector for each parameter
function sessions_by_hand(
    sessions_info::MultipleSessionsInfo,
    cognitive_parameters,
    environment_parameters,
    evaluation,
)
    check_environment_parameters(sessions_info, environment_parameters)
    sessions = sessions_info.sessions

    #The first session runs outside the loop, so that the vector of what the sessions saved has its type
    saved_of_sessions = initialize_storage(
        session_by_hand(
            sessions[1],
            select_row(cognitive_parameters, 1),
            select_row(environment_parameters, 1),
            evaluation,
        ),
        length(sessions),
    )
    for i in 2:length(sessions)
        saved_of_sessions = store_value!(
            saved_of_sessions,
            session_by_hand(
                sessions[i],
                select_row(cognitive_parameters, i),
                select_row(environment_parameters, i),
                evaluation,
            ),
            i,
        )
    end
    return saved_of_sessions
end

## The timesteps of a session ##

# Runs the timesteps of a session from where it starts, saves what its save histories select, and scores the values in
# the data against the distributions of their timesteps at the end. Returns the trajectories the session saved. A session
# whose parameters turn invalid, or whose stopping criterion is met while its data continues, gets a log density of -Inf
# and ends there, and a criterion met before the first timestep is an error, as in the Turing.jl session. From the first
# timestep whose flag or criterion a compiler traces, every timestep runs instead, and the session is masked, as an
# inference backend that Reactant.jl compiles would mask it
function timesteps_by_hand(
    session_info, cognitive_parameters, environment_parameters, previous, evaluation
)
    n_timesteps = session_info.n_timesteps

    #A criterion that is met already would end the session before its first timestep, which is an error, as in the Turing.jl session. A traced criterion can't be checked, and is masked
    stopped_before_first =
        check_stopping_criterion_not_met(
            session_info.environment, previous.environment.state, environment_parameters
        ) | check_stopping_criterion_not_met(
            session_info.cognitive_model, previous.state, cognitive_parameters
        )

    #Save what is kept from before the first timestep
    initial_saved = select_initial_timestep_values_to_save(
        Val(saves_initial_timestep_values(session_info.cognitive_save_history)),
        session_info,
        previous,
    )

    #Run the first timestep on its own, since its inputs can have other types than all the rest
    timestep_values = run_timestep_both_models_by_hand(
        session_info, 1, cognitive_parameters, environment_parameters, previous, evaluation
    )

    #Initialize the storage for what is saved, which starts with timestep 0 when it is kept, and the distributions of the values to condition on
    storage = initialize_all_storage(
        initial_saved,
        select_timestep_values_to_save(session_info, timestep_values),
        n_timesteps,
    )
    offset = initial_saved === nothing ? 0 : 1
    distributions = store_distributions_by_hand(
        session_info,
        initialize_distributions_by_hand(session_info, timestep_values),
        timestep_values,
    )

    #A flag of the first timestep, or a criterion before it, that a compiler traces masks the remaining timesteps
    if masks_flag(timestep_values.parameters_valid) | masks_bool(stopped_before_first)
        return masked_timesteps_by_hand(
            session_info,
            cognitive_parameters,
            environment_parameters,
            storage,
            distributions,
            masking_fitted_session_after_first_timestep(
                timestep_values, stopped_before_first, data_continues_after(session_info, 0)
            ),
            2,
            offset,
            evaluation,
        )
    end

    #A session whose parameters turned invalid on the first timestep gets a log density of -Inf and ends, keeping nothing of the timestep
    if !parameters_are_valid(timestep_values.parameters_valid)
        evaluation.log_density += -Inf
        map_saved(vector -> resize!(vector, offset), storage)
        return saved_trajectories(
            storage,
            timestep_sizes_until(session_info.saved_timestep_sizes, offset),
            false,
            0,
            nothing,
        )
    end

    #Run the remaining timesteps, which are in a function of their own so that it is compiled for the types of the first one's values
    return remaining_timesteps_by_hand(
        session_info,
        cognitive_parameters,
        environment_parameters,
        timestep_values,
        storage,
        distributions,
        offset,
        evaluation,
    )
end

function remaining_timesteps_by_hand(
    session_info,
    cognitive_parameters,
    environment_parameters,
    timestep_values,
    storage,
    distributions,
    offset,
    evaluation,
)
    n_timesteps = session_info.n_timesteps
    for t in 2:n_timesteps

        #The stopping criterion of either model, as the Turing.jl session checks it. A traced one masks this timestep and the remaining ones
        criterion_met = stopping_criterion_of_either_model(
            session_info.cognitive_model,
            session_info.environment,
            timestep_values,
            cognitive_parameters,
            environment_parameters,
        )
        if masks_bool(criterion_met)
            return masked_timesteps_by_hand(
                session_info,
                cognitive_parameters,
                environment_parameters,
                storage,
                distributions,
                masking_fitted_session_before_timestep(timestep_values, t),
                t,
                offset,
                evaluation,
            )
        end

        #Met while the data continues, the parameters can't give it, as when they turn invalid. Otherwise, the session ends here and keeps the timesteps that ran
        if criterion_met
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            if data_continues_after(session_info, t - 1)
                evaluation.log_density += -Inf
                return saved_trajectories(
                    storage,
                    timestep_sizes_until(session_info.saved_timestep_sizes, t - 1 + offset),
                    session_parameters_valid(timestep_values.parameters_valid, false),
                    t - 1,
                    nothing,
                )
            end
            return finish_session_by_hand(
                session_info,
                storage,
                distributions,
                offset,
                t - 1,
                session_parameters_valid(timestep_values.parameters_valid, true),
                t - 1,
                evaluation,
            )
        end

        #Run the timestep from the values of the one before, and store what it saves as element t + offset
        previous_timestep_values = timestep_values
        timestep_values = run_timestep_both_models_by_hand(
            session_info,
            t,
            cognitive_parameters,
            environment_parameters,
            previous_timestep_values,
            evaluation,
        )
        storage = store_timestep_values!(
            storage,
            select_timestep_values_to_save(session_info, timestep_values),
            t + offset,
        )
        distributions = store_distributions_by_hand(
            session_info, distributions, timestep_values
        )

        #A traced flag is counted as the masking counts it, and masks the remaining timesteps
        if masks_flag(timestep_values.parameters_valid)
            return masked_timesteps_by_hand(
                session_info,
                cognitive_parameters,
                environment_parameters,
                storage,
                distributions,
                count_masked_fitted_timestep(
                    masking_fitted_session_before_timestep(previous_timestep_values, t),
                    timestep_values,
                ),
                t + 1,
                offset,
                evaluation,
            )
        end

        #A session whose parameters turned invalid on this timestep gets a log density of -Inf and ends before it
        if !parameters_are_valid(timestep_values.parameters_valid)
            evaluation.log_density += -Inf
            map_saved(vector -> resize!(vector, t - 1 + offset), storage)
            return saved_trajectories(
                storage,
                timestep_sizes_until(session_info.saved_timestep_sizes, t - 1 + offset),
                session_parameters_valid(timestep_values.parameters_valid, true),
                t - 1,
                nothing,
            )
        end
    end
    return finish_session_by_hand(
        session_info,
        storage,
        distributions,
        offset,
        n_timesteps,
        session_parameters_valid(timestep_values.parameters_valid, true),
        n_timesteps,
        evaluation,
    )
end

# The timesteps from first_timestep on, masked: every one runs, from the values of the last one kept, and is stored and
# scored, and masking carries where the session ended. The log density gets -Inf at the end when the parameters turned
# invalid, also when the criterion was met while the data continues
function masked_timesteps_by_hand(
    session_info,
    cognitive_parameters,
    environment_parameters,
    storage,
    distributions,
    masking,
    first_timestep,
    offset,
    evaluation,
)
    n_timesteps = session_info.n_timesteps
    for t in first_timestep:n_timesteps
        criterion_met = stopping_criterion_of_either_model(
            session_info.cognitive_model,
            session_info.environment,
            masking.previous,
            cognitive_parameters,
            environment_parameters,
        )
        masking = stop_masked_fitted_session(
            masking, criterion_met, data_continues_after(session_info, t - 1)
        )
        timestep_values = run_timestep_both_models_by_hand(
            session_info,
            t,
            cognitive_parameters,
            environment_parameters,
            masking.previous,
            evaluation,
        )
        storage = store_timestep_values!(
            storage,
            select_timestep_values_to_save(session_info, timestep_values),
            t + offset,
        )
        distributions = store_distributions_by_hand(
            session_info, distributions, timestep_values
        )
        masking = count_masked_fitted_timestep(masking, timestep_values)
    end
    evaluation.log_density += log_density_of_masked_ending(masking)
    return finish_session_by_hand(
        session_info,
        storage,
        distributions,
        offset,
        n_timesteps,
        masking.parameters_valid,
        masking.n_timesteps,
        evaluation,
    )
end

# Scores the actions in the data, and the observations in the data with an environment, against the distributions of
# their timesteps, and returns the trajectories, whose vectors have n_stored timesteps
function finish_session_by_hand(
    session_info,
    storage,
    distributions,
    offset,
    n_stored,
    parameters_valid,
    n_timesteps,
    evaluation,
)
    score_by_hand(distributions.actions, session_info.actions_to_condition_on, evaluation)
    if has_environment(session_info)
        score_by_hand(
            distributions.observations,
            session_info.observations_to_condition_on,
            evaluation,
        )
    end
    return saved_trajectories(
        storage,
        timestep_sizes_until(session_info.saved_timestep_sizes, n_stored + offset),
        parameters_valid,
        n_timesteps,
        nothing,
    )
end

## Scoring the values in the data ##

# The distributions that the values to condition on are scored against: the cognitive model's for the actions, and the
# environment's for the observations when it runs, as a vector for each name
function initialize_distributions_by_hand(session_info, timestep_values)
    return (;
        actions=initialize_output_distributions_for_conditioning(
            select_names(
                timestep_values.action_distribution,
                action_names_to_condition_on(session_info),
            ),
        ),
        observations=initialize_output_distributions_for_conditioning(
            select_names(
                observation_distribution_by_hand(session_info, timestep_values),
                observation_names_to_condition_on(session_info),
            ),
        ),
    )
end
function store_distributions_by_hand(session_info, distributions, timestep_values)
    return (;
        actions=store_output_distributions_for_conditioning(
            distributions.actions,
            select_names(
                timestep_values.action_distribution,
                action_names_to_condition_on(session_info),
            ),
            timestep_values.actions_to_condition_on_at_timestep,
        ),
        observations=store_output_distributions_for_conditioning(
            distributions.observations,
            select_names(
                observation_distribution_by_hand(session_info, timestep_values),
                observation_names_to_condition_on(session_info),
            ),
            observations_to_condition_on_by_hand(session_info, timestep_values),
        ),
    )
end
observation_distribution_by_hand(::SessionInfo{<:Any,NoEnvironment}, timestep_values) = (;)
function observation_distribution_by_hand(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observation_distribution
end
function observations_to_condition_on_by_hand(
    ::SessionInfo{<:Any,NoEnvironment}, timestep_values
)
    return (;)
end
function observations_to_condition_on_by_hand(
    ::SessionInfo{<:Any,<:AbstractTimestepModel}, timestep_values
)
    return timestep_values.environment.observations_to_condition_on_at_timestep
end

# Adds the log density of the values in the data of each name, against the distributions of their timesteps
function score_by_hand(
    distributions::NamedTuple{names}, values_to_condition_on::NamedTuple, evaluation
) where {names}
    map(
        (distributions_of_name, values_of_name) ->
            score_each_by_hand(distributions_of_name, values_of_name, evaluation),
        Tuple(distributions),
        Tuple(NamedTuple{names}(values_to_condition_on)),
    )
    return nothing
end
function score_each_by_hand(distributions_of_name, values_of_name, evaluation)
    for i in eachindex(values_of_name)
        evaluation.log_density += log_density(distributions_of_name[i], values_of_name[i])
    end
    return nothing
end

## A single timestep ##

# Without an environment, the observation comes from the data: the model acts on this timestep's observation when the
# environment starts, and on the previous one when the agent starts
function run_timestep_both_models_by_hand(
    session_info::SessionInfo{<:Any,NoEnvironment},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
    evaluation,
)
    data_observation = select_row(session_info.observations, t)
    observation = if session_info.who_starts isa EnvironmentStarts
        data_observation
    else
        previous.observation
    end
    cognitive = run_timestep_one_model_by_hand(
        session_info.cognitive_model,
        ModelContext(
            observation,
            previous.state,
            previous.action,
            cognitive_parameters,
            get_timestep_size(session_info.timestep_sizes, t),
            default_rng(),
            session_info.who_starts,
        ),
        session_info.actions,
        t,
        evaluation,
    )

    #The next timestep starts from this timestep's observation in the data
    return merge(
        (; state=cognitive.new_state),
        build_named_entry(Val(:observation), data_observation),
        build_named_entry(Val(:action), cognitive.output),
        (;
            action_distribution=cognitive.output_distribution,
            actions_to_condition_on_at_timestep=cognitive.outputs_to_condition_on,
            parameters_valid=cognitive.parameters_valid,
            environment=(; state=nothing),
        ),
    )
end

# With an environment, the environment gives the observation in response to the action, before the model acts when the
# environment starts, and after when the agent starts
function run_timestep_both_models_by_hand(
    session_info::SessionInfo{<:Any,<:AbstractTimestepModel},
    t,
    cognitive_parameters,
    environment_parameters,
    previous,
    evaluation,
)
    if session_info.who_starts isa EnvironmentStarts
        environment = environment_timestep_by_hand(
            session_info,
            t,
            environment_parameters,
            map_names(previous.action, session_info.action_map),
            previous,
            evaluation,
        )
        observation = map_names(environment.output, session_info.observation_map)
        cognitive = cognitive_timestep_by_hand(
            session_info, t, cognitive_parameters, observation, previous, evaluation
        )
        environment_action = map_names(cognitive.output, session_info.action_map)
    else
        cognitive = cognitive_timestep_by_hand(
            session_info,
            t,
            cognitive_parameters,
            previous.observation,
            previous,
            evaluation,
        )
        environment_action = map_names(cognitive.output, session_info.action_map)
        environment = environment_timestep_by_hand(
            session_info,
            t,
            environment_parameters,
            environment_action,
            previous,
            evaluation,
        )
        observation = map_names(environment.output, session_info.observation_map)
    end

    #Collect the environment's values, in its own names, and the cognitive model's at the top
    environment_timestep_values = merge(
        (; state=environment.new_state),
        build_named_entry(Val(:observation), environment.output),
        build_named_entry(Val(:action), environment_action),
        (;
            observation_distribution=environment.output_distribution,
            observations_to_condition_on_at_timestep=environment.outputs_to_condition_on,
        ),
    )
    return merge(
        (; state=cognitive.new_state),
        build_named_entry(Val(:observation), observation),
        build_named_entry(Val(:action), cognitive.output),
        (;
            action_distribution=cognitive.output_distribution,
            actions_to_condition_on_at_timestep=cognitive.outputs_to_condition_on,
            parameters_valid=combine_parameters_valid(
                cognitive.parameters_valid, environment.parameters_valid
            ),
        ),
        build_named_entry(Val(:environment), environment_timestep_values),
    )
end

function cognitive_timestep_by_hand(
    session_info, t, cognitive_parameters, observation, previous, evaluation
)
    return run_timestep_one_model_by_hand(
        session_info.cognitive_model,
        ModelContext(
            observation,
            previous.state,
            previous.action,
            cognitive_parameters,
            get_timestep_size(session_info.timestep_sizes, t),
            default_rng(),
            session_info.who_starts,
        ),
        session_info.actions,
        t,
        evaluation,
    )
end
function environment_timestep_by_hand(
    session_info, t, environment_parameters, environment_action, previous, evaluation
)
    return run_timestep_one_model_by_hand(
        session_info.environment,
        ModelContext(
            environment_action,
            previous.environment.state,
            previous.environment.observation,
            environment_parameters,
            get_timestep_size(session_info.timestep_sizes, t),
            default_rng(),
            session_info.who_starts,
        ),
        session_info.environment_observations,
        t,
        evaluation,
    )
end

# One model's part of a timestep, the same for a cognitive model and an environment model: the model runs, and its
# outputs are then taken from the data where it has values, and imputed from what the model gave where they are missing
function run_timestep_one_model_by_hand(model, context, data_columns, t, evaluation)
    data_output = select_row(data_columns, t)
    model_output = check_model_output(timestep_model(model, context), model)
    output = impute_by_hand(data_output, model_output.output_distribution, evaluation)
    outputs_to_condition_on = find_outputs_to_condition_on(
        data_output, model_output.output_distribution, model
    )
    return merge(
        build_named_entry(Val(:output), output),
        (;
            new_state=model_output.new_state,
            output_distribution=model_output.output_distribution,
            outputs_to_condition_on,
            parameters_valid=model_output.parameters_valid,
        ),
    )
end

## Imputing the values that are missing in the data ##

function impute_by_hand(data, distributions, evaluation)
    has_missing_value(data) || return data
    return impute_each_by_hand(data, distributions, evaluation)
end
impute_each_by_hand(data::NamedTuple{()}, distributions, evaluation) = (;)
function impute_each_by_hand(
    data::NamedTuple{names}, distributions, evaluation
) where {names}
    name = first(names)
    value = impute_value_by_hand(first(data), getproperty(distributions, name), evaluation)
    return merge(
        build_named_entry(Val(name), value),
        impute_each_by_hand(Base.tail(data), distributions, evaluation),
    )
end

# A value that is there is taken as it is, and so is any marker but ImputedMissingValue
impute_value_by_hand(data_value, distribution, evaluation) = data_value

# A missing value is a latent, with the distribution the model gives for it
function impute_value_by_hand(::ImputedMissingValue, distribution, evaluation)
    value = take_latent!(evaluation)
    evaluation.log_density += log_density(distribution, value)
    return value
end

# A missing value that the model gives without randomness is that value, and one where the model gives nothing stays as the model gives it
function impute_value_by_hand(
    ::ImputedMissingValue, deterministic::Deterministic, evaluation
)
    return deterministic.value
end
function impute_value_by_hand(::ImputedMissingValue, no_value::AbstractNoValue, evaluation)
    return no_value
end

## The cases ##

# The log density of a session, or of several, run by hand, with what it saved of the states of the cognitive model
# added, so that saving is part of what is compiled and differentiated. zero_log_density has the number type of the
# parameters
function log_density_by_hand(
    run_by_hand,
    sessions,
    cognitive_parameters,
    environment_parameters,
    latents,
    zero_log_density,
)
    evaluation = EvaluationByHand(latents, 0, zero_log_density)
    saved = run_by_hand(sessions, cognitive_parameters, environment_parameters, evaluation)
    return evaluation.log_density + total_of_saved_states(saved)
end
function total_of_saved_states(saved::SessionTrajectories)
    total = 0.0
    for state in saved.states
        total += sum(Tuple(state); init=0.0)
    end
    return total
end
function total_of_saved_states(saved_of_sessions::AbstractVector)
    return sum(total_of_saved_states, saved_of_sessions)
end

const REWARDS_BY_HAND = (; reward=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
const CHOICES_BY_HAND = (;
    choice=[true, false, true, true, false, false, true, true, false, true]
)
const CONTINUOUS_CHOICES_BY_HAND = (;
    choice=[
        0.3, 0.6, ImputedMissingValue(), 0.4, 0.7, ImputedMissingValue(), 0.5, 0.2, 0.8, 0.6
    ]
)
const MISSING_REWARDS_BY_HAND = (;
    reward=[
        ImputedMissingValue(),
        0.0,
        ImputedMissingValue(),
        1.0,
        ImputedMissingValue(),
        1.0,
        0.0,
        ImputedMissingValue(),
        1.0,
        1.0,
    ]
)
const REWARDS_NOT_ALWAYS_GIVEN_BY_HAND = (;
    reward=[1.0, 0.0, NoValue(), 1.0, 0.0, NoValue(), 0.0, 0.0, NoValue(), 1.0]
)

# Each of the functions below prepares a session once, as an inference backend's model takes it, and returns the log
# density of the session as a function of a vector, with the parameters first and then the imputed values

# A session whose choices are made with a probability
function plain_session_by_hand(who_starts, rewards; keywords...)
    session_info = SessionInfo(
        LearnsRewards{ChoiceByHand}(), rewards, CHOICES_BY_HAND; who_starts, keywords...
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (;),
        (),
        zero(x[1]),
    )
end

# A session with two actions that are missing in the data, whose imputed values come after the parameters in the vector
function imputed_actions_session_by_hand()
    session_info = SessionInfo(
        LearnsRewards{AroundMean}(),
        REWARDS_BY_HAND,
        CONTINUOUS_CHOICES_BY_HAND;
        who_starts=EnvironmentStarts(),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (;),
        (x[3], x[4]),
        zero(x[1]),
    )
end

# A session with an environment, whose parameter is the third in the vector, and whose imputed rewards come after it
function environment_session_by_hand(who_starts, rewards; keywords...)
    session_info = SessionInfo(
        LearnsRewards{ChoiceByHand}(),
        rewards,
        CHOICES_BY_HAND;
        who_starts,
        environment=NoisyRewardsFollowActions(),
        keywords...,
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (; reward_size=x[3]),
        (x[4], x[5], x[6], x[7]),
        zero(x[1]),
    )
end

# The same with an environment under other names, which maps of names pass on
function name_maps_session_by_hand()
    session_info = SessionInfo(
        LearnsRewards{ChoiceByHand}(),
        MISSING_REWARDS_BY_HAND,
        CHOICES_BY_HAND;
        who_starts=EnvironmentStarts(),
        environment=NoisyPayoffsFollowLevers(),
        observation_map=NameMap(:payoff => :reward),
        action_map=NameMap(:choice => :lever),
        environment_save_history=SaveHistory(; outputs=:payoff),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (; reward_size=x[3]),
        (x[4], x[5], x[6], x[7]),
        zero(x[1]),
    )
end

# A session whose environment ends after four timesteps, while its data has ten, of which the last six have nothing
# recorded: the first four choices are in the data, and every reward is imputed up to where the session ends
function ending_session_by_hand()
    session_info = SessionInfo(
        LearnsRewards{AroundMean}(),
        (; reward=fill(ImputedMissingValue(), 10)),
        (; choice=[0.3, 0.6, 0.4, 0.7, fill(ImputedMissingValue(), 6)...]);
        who_starts=EnvironmentStarts(),
        environment=NoisyRewardsUntilEnd(),
        cognitive_save_history=SaveHistory(; states=:expected_reward),
        timestep_sizes=collect(range(0.5, 1.5; length=10)),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (; reward_size=x[3]),
        (x[4], x[5], x[6], x[7]),
        zero(x[1]),
    )
end

# A session whose model gives no action on the first timestep, where the data has none either, so that the distributions
# of the actions to condition on get their type from the second timestep
function no_first_action_session_by_hand()
    session_info = SessionInfo(
        ActsAfterFirstReward(),
        REWARDS_BY_HAND,
        (; choice=[NoValue(), false, true, true, false, false, true, true, false, true]);
        who_starts=AgentStarts(),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (;),
        (),
        zero(x[1]),
    )
end

# Three sessions, each with its own parameters, which are given as a vector for each parameter
const TABLE_BY_HAND = (;
    id=repeat([1, 2, 3]; inner=10),
    reward=[
        REWARDS_BY_HAND.reward
        reverse(REWARDS_BY_HAND.reward)
        REWARDS_BY_HAND.reward
    ],
    choice=[
        CHOICES_BY_HAND.choice
        CHOICES_BY_HAND.choice
        reverse(CHOICES_BY_HAND.choice)
    ],
)
function several_sessions_by_hand(; keywords...)
    sessions_info = MultipleSessionsInfo(
        LearnsRewards{ChoiceByHand}(),
        TABLE_BY_HAND;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
    return x -> log_density_by_hand(
        sessions_by_hand,
        sessions_info,
        (; learning_rate=x[1:3], inverse_temperature=x[4:6]),
        (;),
        (),
        zero(x[1]),
    )
end

# The same sessions with an environment which runs in each, whose parameter is the same for all of them
function several_sessions_with_environment_by_hand()
    sessions_info = MultipleSessionsInfo(
        LearnsRewards{ChoiceByHand}(),
        TABLE_BY_HAND;
        observation_columns=:reward,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        environment=NoisyRewardsFollowActions(),
        include_environment=true,
    )
    return x -> log_density_by_hand(
        sessions_by_hand,
        sessions_info,
        (; learning_rate=x[1:3], inverse_temperature=x[4:6]),
        (; reward_size=[x[7], x[7], x[7]]),
        (),
        zero(x[1]),
    )
end

# A session of a model whose parameters can turn invalid, which a compiler traces, so that the session is masked
function invalid_parameters_session_by_hand()
    session_info = SessionInfo(
        TurnsInvalidByHand(),
        REWARDS_BY_HAND,
        CHOICES_BY_HAND;
        who_starts=EnvironmentStarts(),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (;),
        (),
        zero(x[1]),
    )
end

# A session whose stopping criterion is met while its data continues, which counts as parameters that turned invalid
function criterion_with_data_session_by_hand()
    session_info = SessionInfo(
        StopsAboveByHand(), REWARDS_BY_HAND, CHOICES_BY_HAND; who_starts=EnvironmentStarts()
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (;),
        (),
        zero(x[1]),
    )
end

# A masked session whose stopping criterion is met after its data, which has four timesteps, so that the imputed rewards and
# choices after the end keep their scores
function masked_ending_session_by_hand()
    session_info = SessionInfo(
        StopsAboveMaskedByHand(),
        (; reward=[1.0, 0.0, 1.0, 1.0, fill(ImputedMissingValue(), 6)...]),
        (; choice=[0.3, 0.6, 0.4, 0.7, fill(ImputedMissingValue(), 6)...]);
        who_starts=EnvironmentStarts(),
        environment=NoisyRewardsFollowActions(),
    )
    return x -> log_density_by_hand(
        session_by_hand,
        session_info,
        (; learning_rate=x[1], inverse_temperature=x[2]),
        (; reward_size=x[3]),
        Tuple(x[4:15]),
        zero(x[1]),
    )
end

# What the Reactant tests check of the functions for use within the model of an inference backend. In each entry, f takes
# a vector x, with the parameters first and then the imputed values, and returns the log density of a session run by hand
const SESSION_BY_HAND_CASES = [
    (;
        name="session by hand, environment starts",
        f=plain_session_by_hand(EnvironmentStarts(), REWARDS_BY_HAND),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand, agent starts",
        f=plain_session_by_hand(AgentStarts(), REWARDS_BY_HAND),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand with a reward that isn't always given, environment starts",
        f=plain_session_by_hand(EnvironmentStarts(), REWARDS_NOT_ALWAYS_GIVEN_BY_HAND),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand with a reward that isn't always given, agent starts",
        f=plain_session_by_hand(AgentStarts(), REWARDS_NOT_ALWAYS_GIVEN_BY_HAND),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand which saves everything, with a size for each timestep",
        f=plain_session_by_hand(
            EnvironmentStarts(),
            REWARDS_BY_HAND;
            cognitive_save_history=SaveHistory(SaveEverything()),
            timestep_sizes=collect(range(0.5, 1.5; length=10)),
        ),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand which saves its states without timestep 0",
        f=plain_session_by_hand(
            AgentStarts(),
            REWARDS_BY_HAND;
            cognitive_save_history=SaveHistory(;
                states=:expected_reward, save_initial_timestep_values=false
            ),
        ),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand with imputed actions",
        f=imputed_actions_session_by_hand(),
        x=[0.3, 2.0, 0.2, -0.1],
    ),
    (;
        name="session by hand with an environment and imputed observations, environment starts",
        f=environment_session_by_hand(EnvironmentStarts(), MISSING_REWARDS_BY_HAND),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand with an environment and imputed observations, agent starts",
        f=environment_session_by_hand(AgentStarts(), MISSING_REWARDS_BY_HAND),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand with an environment, saving the histories of both models",
        f=environment_session_by_hand(
            EnvironmentStarts(),
            MISSING_REWARDS_BY_HAND;
            cognitive_save_history=SaveHistory(SaveEverything()),
            environment_save_history=SaveHistory(SaveEverything()),
        ),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand with an environment which doesn't run, since nothing in the data is missing",
        f=environment_session_by_hand(
            EnvironmentStarts(),
            REWARDS_BY_HAND;
            cognitive_save_history=SaveHistory(; states=:expected_reward),
        ),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand with an environment that runs with nothing missing, so that every reward is scored",
        f=environment_session_by_hand(
            EnvironmentStarts(), REWARDS_BY_HAND; include_environment=true
        ),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand with maps of names",
        f=name_maps_session_by_hand(),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand which ends when the environment's stopping criterion is met",
        f=ending_session_by_hand(),
        x=[0.3, 2.0, 0.9, 0.6, 0.2, 0.8, -0.1],
    ),
    (;
        name="session by hand whose model gives no action on the first timestep",
        f=no_first_action_session_by_hand(),
        x=[0.3, 2.0],
    ),
    (;
        name="three sessions by hand",
        f=several_sessions_by_hand(),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
    ),
    (;
        name="three sessions by hand which save their states",
        f=several_sessions_by_hand(;
            cognitive_save_history=SaveHistory(; states=:expected_reward)
        ),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0],
    ),
    (;
        name="three sessions by hand with an environment whose parameter they share",
        f=several_sessions_with_environment_by_hand(),
        x=[0.2, 0.3, 0.4, 1.0, 2.0, 3.0, 0.9],
    ),
    (;
        name="session by hand whose parameters can turn invalid and stay valid, masked when traced",
        f=invalid_parameters_session_by_hand(),
        x=[0.3, 2.0],
    ),
    (;
        name="session by hand whose parameters turn invalid on its fourth timestep",
        f=invalid_parameters_session_by_hand(),
        x=[0.8, 2.0],
        gradient=false,
    ),
    (;
        name="session by hand whose stopping criterion is met while its data continues",
        f=criterion_with_data_session_by_hand(),
        x=[0.6, 2.0],
        gradient=false,
    ),
    (;
        name="masked session by hand whose stopping criterion is met after its data, with imputed values after the end",
        f=masked_ending_session_by_hand(),
        x=[0.5, 2.0, 0.9, 0.5, 0.6, 0.4, 0.7, 0.6, 0.5, 0.8, 0.3, 0.5, 0.6, 0.7, 0.4],
    ),
]
