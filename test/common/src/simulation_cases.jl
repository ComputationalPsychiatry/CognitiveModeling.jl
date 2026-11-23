#The simulation cases that the suites share. Each simulates a model of the tests in one of the ways that the package
#offers, with its real numbers, usually those of its parameters, taken from a vector x, and returns what was saved, so
#that a suite can check what the simulation saved, differentiate it by x, or compile it. simulation_cases(distributions)
#gives every case with a kind of distributions of the models: with their means in the AD and Reactant suites, where a
#simulation is deterministic and the gradients go all the way round its loop, and with the distributions written by hand
#in the main suite, where a case samples with a random number generator of its own, seeded on every run
using Random: Xoshiro

## The real numbers of the parameters ##
#The real numbers among the values of a NamedTuple of parameters, in order, which a case is differentiated by: numbers
#that aren't whole numbers or Bools, and the elements of vectors of them. Symbols, Bools and whole numbers are kept as
#they are
function real_numbers(parameters::NamedTuple)
    return reduce(vcat, map(real_number_vector, values(parameters)); init=Float64[])
end
real_number_vector(value::AbstractFloat) = [value]
real_number_vector(value::AbstractVector{<:AbstractFloat}) = collect(value)
real_number_vector(value) = Float64[]

#The parameters with their real numbers taken from x, from element `index` on, in the order of real_numbers, and the
#element after the last one taken. This runs within the functions that are differentiated and compiled
function with_real_numbers(parameters::NamedTuple{names}, x, index) where {names}
    new_values, next_index = replace_real_numbers(values(parameters), x, index)
    return NamedTuple{names}(new_values), next_index
end
replace_real_numbers(::Tuple{}, x, index) = ((), index)
function replace_real_numbers(values::Tuple, x, index)
    head, next_index = replace_real_number(first(values), x, index)
    tail, last_index = replace_real_numbers(Base.tail(values), x, next_index)
    return (head, tail...), last_index
end
replace_real_number(value::AbstractFloat, x, index) = (x[index], index + 1)
function replace_real_number(value::AbstractVector{<:AbstractFloat}, x, index)
    last_index = index + length(value) - 1
    return x[index:last_index], last_index + 1
end
replace_real_number(value, x, index) = (value, index)

#The parameters of the cognitive model and of the environment, with their real numbers from x, in that order
function both_with_real_numbers(cognitive_parameters, environment_parameters, x)
    cognitive_values, next_index = with_real_numbers(cognitive_parameters, x, 1)
    environment_values, _ = with_real_numbers(environment_parameters, x, next_index)
    return cognitive_values, environment_values
end

## The sum that is differentiated ##
#The sum of every number in what a case returned, which the AD and Reactant suites differentiate, a number of an
#automatic differentiation package and a traced number of Reactant included. Markers and Symbols add nothing, a
#Deterministic adds its value, and other distributions add nothing. The elements of a tuple are added one by one, those
#of an array of numbers in one sum, which Reactant compiles as one reduction, and those of other arrays in a loop, which
#every backend differentiates and Reactant compiles. The trajectories of a session add every one of their fields, which
#needs a method of its own, since anything without one adds nothing
sum_of_reals(value::Number) = value
sum_of_reals(values::NamedTuple) = sum_of_reals(Tuple(values))
sum_of_reals(trajectories::SessionTrajectories) = sum_of_reals(NamedTuple(trajectories))
sum_of_reals(::Tuple{}) = 0.0
sum_of_reals(values::Tuple) = sum_of_reals(first(values)) + sum_of_reals(Base.tail(values))
sum_of_reals(values::AbstractArray{<:Number}) = isempty(values) ? 0.0 : sum(values)
function sum_of_reals(values::AbstractArray)
    total = 0.0
    for value in values
        total += sum_of_reals(value)
    end
    return total
end
sum_of_reals(deterministic::Deterministic) = sum_of_reals(deterministic.value)
sum_of_reals(value) = 0.0

#The sum of every number in what a simulation kept, which offset says whether it starts with timestep 0: the elements of
#its vectors up to its last timestep kept, with its other fields. A simulation that Reactant masks instead of stopping it
#has every timestep in its vectors, and the elements after the last one kept add nothing, through ifelse, since the
#number kept is traced too. The vectors, which have different types, are gone through one at a time, as sum_of_reals
#goes through a tuple, so that the type of the sum is inferred
function sum_of_kept_reals(trajectories::SessionTrajectories, offset)
    vectors = (
        trajectories.states,
        trajectories.observations,
        trajectories.actions,
        trajectories.action_distributions,
        values(trajectories.environment)...,
    )
    how_it_ended = (
        trajectories.parameters_valid, trajectories.n_timesteps, trajectories.last_timestep
    )
    return sum_of_reals(how_it_ended) +
           sum_of_kept_elements(vectors, trajectories.n_timesteps + offset)
end
sum_of_kept_elements(::Tuple{}, n_kept) = 0.0
function sum_of_kept_elements(vectors::Tuple, n_kept)
    return sum_of_kept_elements(first(vectors), n_kept) +
           sum_of_kept_elements(Base.tail(vectors), n_kept)
end
function sum_of_kept_elements(vector::AbstractVector, n_kept)
    total = 0.0
    for (index, element) in enumerate(vector)
        total += ifelse(index <= n_kept, sum_of_reals(element), 0.0)
    end
    return total
end

## The kinds of cases ##
#Every case is a NamedTuple with its name, the function that runs it at x, x itself, whether Reactant compiles it,
#whether the type of what it returns is inferred, whether a second run gives the same, and whether it has a gradient to
#compare, which it doesn't when x is empty or nothing that the case saves depends on x
#The random number generator of a run of a case: a new one with a seed on every run, so that two runs give the same,
#with the distributions written by hand, which sample, and none with the means, which sample nothing, so that the
#default is passed on untouched, which Reactant can compile
random_number_generator(distributions) = Xoshiro(1)
random_number_generator(::Means) = nothing
function with_random_number_generator(keywords, distributions)
    rng = random_number_generator(distributions)
    return rng === nothing ? keywords : merge(keywords, (; rng))
end

#A case of simulate_session with a cognitive model and an environment, run with the real numbers of both models'
#parameters
function simulate_case(
    name,
    distributions;
    model,
    environment,
    cognitive_parameters,
    environment_parameters,
    reactant=true,
    inferred=true,
    differentiated=true,
    keywords...,
)
    function run(x)
        cognitive_values, environment_values = both_with_real_numbers(
            cognitive_parameters, environment_parameters, x
        )
        return simulate_session(
            model,
            environment;
            cognitive_parameters=cognitive_values,
            environment_parameters=environment_values,
            with_random_number_generator(NamedTuple(keywords), distributions)...,
        )
    end
    x = vcat(real_numbers(cognitive_parameters), real_numbers(environment_parameters))
    return (;
        name,
        run,
        x,
        reactant,
        inferred,
        deterministic=true,
        differentiated=differentiated && !isempty(x),
    )
end

#A case of simulate_session whose simulation ends before its last timestep, by a stopping criterion or by parameters
#that turn invalid, at a timestep that depends on the parameters, which returns the sum of what it kept
function ending_case(name, distributions; keywords...)
    case = simulate_case(name, distributions; keywords...)
    offset =
        if CognitiveModeling.saves_initial_timestep_values(
            keywords[:cognitive_save_history]
        )
            1
        else
            0
        end
    return merge(case, (; run=x -> sum_of_kept_reals(case.run(x), offset)))
end

#A case of an agent and an environment, instantiated with the real numbers of their parameters, and simulated with
#simulate_session! once, or twice, with both models reset between the two calls or not, which returns the trajectories
#of every call and the agent's history after them, read the way a user reads it, as vectors. The calls are written out,
#without a loop, which Reactant would turn into a traced broadcast
function instantiated_case(
    name,
    distributions;
    model,
    environment,
    cognitive_parameters,
    environment_parameters,
    save_history=SaveHistory(SaveEverything()),
    environment_save_history=SaveHistory(),
    twice=false,
    reset_between=false,
    reactant=true,
    keywords...,
)
    function run(x)
        cognitive_values, environment_values = both_with_real_numbers(
            cognitive_parameters, environment_parameters, x
        )
        agent = InstantiatedModel(model, cognitive_values; save_history)
        environment_instance = InstantiatedModel(
            environment, environment_values; save_history=environment_save_history
        )
        call_keywords = with_random_number_generator(NamedTuple(keywords), distributions)
        first_trajectories = simulate_session!(
            agent, environment_instance; call_keywords...
        )
        trajectories = if twice
            if reset_between
                reset!(agent)
                reset!(environment_instance)
            end
            (
                first_trajectories,
                simulate_session!(agent, environment_instance; call_keywords...),
            )
        else
            (first_trajectories,)
        end
        history = agent.history
        return (;
            trajectories,
            history=(;
                history.states,
                history.inputs,
                history.outputs,
                history.output_distributions,
                history.timestep_sizes,
            ),
        )
    end
    x = vcat(real_numbers(cognitive_parameters), real_numbers(environment_parameters))
    return (;
        name,
        run,
        x,
        reactant,
        inferred=false,
        deterministic=true,
        differentiated=(!isempty(x)),
    )
end

#A case of a cognitive model simulated on one input at a time with simulate_timestep, which passes its output and its
#state on to the next, and returns the outputs and the states
function single_inputs_case(
    name, distributions; model, parameters, inputs, reactant=true, keywords...
)
    function run(x)
        parameter_values, _ = with_real_numbers(parameters, x, 1)
        rng_keywords = with_random_number_generator(NamedTuple(keywords), distributions)
        first_output, first_state = simulate_timestep(
            model, first(inputs); parameters=parameter_values, rng_keywords...
        )
        outputs = [first_output]
        states = [first_state]
        for input in inputs[2:end]
            output, state = simulate_timestep(
                model,
                input;
                parameters=parameter_values,
                previous_state=last(states),
                previous_output=last(outputs),
                rng_keywords...,
            )
            push!(outputs, output)
            push!(states, CognitiveModeling.prepare_state_for_saving(model, state))
        end
        return (; outputs, states)
    end
    x = real_numbers(parameters)
    return (;
        name,
        run,
        x,
        reactant,
        inferred=false,
        deterministic=true,
        differentiated=(!isempty(x)),
    )
end

#A case of an agent simulated on one input at a time with simulate_timestep!, which returns its outputs, in a tuple, and
#the states of its history. The tuple of the inputs is made once, outside the function, since Enzyme's forward mode
#gives a wrong gradient, with no error, when a function maps over a tuple that it makes from a vector, whose length is
#then only known when it runs
function instantiated_single_inputs_case(
    name,
    distributions;
    model,
    parameters,
    inputs,
    save_history=SaveHistory(SaveEverything()),
    reactant=true,
    keywords...,
)
    inputs_tuple = Tuple(inputs)
    function run(x)
        parameter_values, _ = with_real_numbers(parameters, x, 1)
        agent = InstantiatedModel(model, parameter_values; save_history)
        rng_keywords = with_random_number_generator(NamedTuple(keywords), distributions)
        outputs = map(
            input -> simulate_timestep!(agent, input; rng_keywords...), inputs_tuple
        )
        return (; outputs, states=agent.history.states)
    end
    x = real_numbers(parameters)
    return (;
        name,
        run,
        x,
        reactant,
        inferred=false,
        deterministic=true,
        differentiated=(!isempty(x)),
    )
end

#A case of simulate_session where x is the sizes of the timesteps, which the gradient is of, with the parameters as they
#are
function timestep_sizes_case(
    name,
    distributions;
    model,
    environment,
    cognitive_parameters,
    environment_parameters,
    timestep_sizes,
    reactant=true,
    keywords...,
)
    function run(x)
        return simulate_session(
            model,
            environment;
            cognitive_parameters,
            environment_parameters,
            timestep_sizes=x,
            n_steps=length(timestep_sizes),
            with_random_number_generator(NamedTuple(keywords), distributions)...,
        )
    end
    return (;
        name,
        run,
        x=timestep_sizes,
        reactant,
        inferred=true,
        deterministic=true,
        differentiated=true,
    )
end

## The cases ##
#Saving everything of both models
const SAVE_BOTH = (;
    cognitive_save_history=SaveHistory(SaveEverything()),
    environment_save_history=SaveHistory(SaveEverything()),
)

#The models that Reactant can't compile, whose values change type between timesteps, as for NOT_INFERRED below: Reactant
#throws a FieldError from within its tracing. A stopping criterion or a parameters_valid on a traced value is masked, so
#models with them are compiled
const NOT_COMPILED_BY_REACTANT = ("EveryFeature", "EveryFeatureWithoutSampling")

#The pairs whose simulation the compiler can't infer the type of, because an output is NoValue on some timesteps, or a
#state changes type, which widens the vectors of saved values within the loop
const NOT_INFERRED = (
    "StateChangesType",
    "OutputNotEveryTimestep",
    "Baseline with OutputNotEveryTimestepEnvironment",
    "Baseline with NoOutputBeforeFirstInputEnvironment",
    "EveryFeature",
    "EveryFeatureWithoutSampling",
)

#A choice and a reward of the kind of distributions, which a simulation can be given as the action and the observation
#before its first timestep: a Bool with the distributions written by hand, and a number with their means
a_choice_or_reward(distributions) = true
a_choice_or_reward(::Means) = 1.0

#The rewards that the cases on single observations receive, which the agents start without
const SINGLE_REWARDS = [(; reward=reward) for reward in (1.0, 0.0, 1.0, 1.0, 0.0)]

#Every simulation case with a kind of distributions of the models
function simulation_cases(distributions)
    cases = []
    baseline = pair_named("Baseline")
    baseline_parts = (;
        model=Baseline(distributions),
        environment=BaselineEnvironment(distributions),
        baseline.cognitive_parameters,
        baseline.environment_parameters,
    )
    baseline_with(; replacements...) = merge(baseline_parts, NamedTuple(replacements))

    #Every model with its environment, saving everything, as the models' own tests run them
    pairs = distributions isa Means ? PAIRS_WITH_MEANS : MODEL_PAIRS
    for pair in pairs, (who_starts, starts) in WHO_STARTS
        push!(
            cases,
            simulate_case(
                "$(pair.name), $starts",
                distributions;
                model=pair.model(distributions),
                environment=pair.environment(distributions),
                pair.cognitive_parameters,
                pair.environment_parameters,
                reactant=(!(pair.name in NOT_COMPILED_BY_REACTANT)),
                inferred=(!(pair.name in NOT_INFERRED)),
                who_starts,
                n_steps=10,
                SAVE_BOTH...,
                get(pair, :keywords, (;))...,
            ),
        )
    end

    #Simulations that end before their last timestep, at a timestep that depends on the parameters, by a stopping
    #criterion, which a learning rate of 0.9 meets after a few rewards, or by parameters that turn invalid, from an
    #expected reward above the highest valid one. Reactant masks them instead of stopping them. When the agent starts
    #from such an expected reward, its first timestep, without a reward, leaves it as it is, so nothing kept depends on
    #the parameters
    for (who_starts, starts) in WHO_STARTS
        push!(
            cases,
            ending_case(
                "WithStoppingCriterion until its criterion is met, $starts",
                distributions;
                model=WithStoppingCriterion(distributions),
                environment=BaselineEnvironment(distributions),
                cognitive_parameters=(; learning_rate=0.9, inverse_temperature=2.0),
                environment_parameters=(; reward_probability=1.0),
                who_starts,
                n_steps=10,
                SAVE_BOTH...,
            ),
        )
    end
    push!(
        cases,
        ending_case(
            "WithInvalidParameters from an expected reward above the highest valid one, environment starts",
            distributions;
            model=WithInvalidParameters(distributions),
            environment=BaselineEnvironment(distributions),
            cognitive_parameters=(; learning_rate=0.001, inverse_temperature=2.0),
            environment_parameters=(; reward_probability=0.8),
            who_starts=EnvironmentStarts(),
            n_steps=10,
            previous_cognitive_state=(; expected_reward=0.995),
            warn_on_invalid_parameters=false,
            SAVE_BOTH...,
        ),
    )

    #The baseline with each setting of a simulation
    for (who_starts, starts) in WHO_STARTS
        append!(
            cases,
            [
                simulate_case(
                    "Baseline with one size for every timestep, $starts",
                    distributions;
                    baseline_with(; model=UsesTimestepSize(distributions))...,
                    who_starts,
                    n_steps=10,
                    timestep_sizes=0.5,
                    SAVE_BOTH...,
                ),
                timestep_sizes_case(
                    "The sizes of the timesteps as the point, $starts",
                    distributions;
                    baseline_with(; model=UsesTimestepSize(distributions))...,
                    timestep_sizes=collect(range(0.5, 1.5; length=5)),
                    who_starts,
                    SAVE_BOTH...,
                ),
                simulate_case(
                    "Baseline saving nothing, $starts",
                    distributions;
                    baseline_parts...,
                    differentiated=false,
                    who_starts,
                    n_steps=10,
                ),
                simulate_case(
                    "Baseline saving the states only, $starts",
                    distributions;
                    baseline_parts...,
                    who_starts,
                    n_steps=10,
                    cognitive_save_history=SaveHistory(; states=:expected_reward),
                    environment_save_history=SaveHistory(),
                ),
                simulate_case(
                    "Baseline saving everything but timestep 0, $starts",
                    distributions;
                    baseline_parts...,
                    who_starts,
                    n_steps=10,
                    cognitive_save_history=SaveHistory(
                        SaveEverything(); save_initial_timestep_values=false
                    ),
                    environment_save_history=SaveHistory(SaveEverything()),
                ),
                simulate_case(
                    "Baseline continuing from given states, action and observation, $starts",
                    distributions;
                    baseline_parts...,
                    who_starts,
                    n_steps=10,
                    previous_cognitive_state=(; expected_reward=0.7),
                    previous_action=(; choice=a_choice_or_reward(distributions)),
                    previous_environment_state=(; n_timesteps=5),
                    previous_observation=(; reward=a_choice_or_reward(distributions)),
                    SAVE_BOTH...,
                ),
                simulate_case(
                    "Baseline with fixed observations, $starts",
                    distributions;
                    baseline_with(;
                        environment=FixedObservationsEnvironment((;
                            reward=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
                        )),
                        environment_parameters=(;),
                    )...,
                    who_starts,
                    n_steps=20,
                    action_map=NameMap(),
                    SAVE_BOTH...,
                ),
                simulate_case(
                    "Baseline with fixed observations that aren't always given, $starts",
                    distributions;
                    inferred=false,
                    baseline_with(;
                        environment=FixedObservationsEnvironment((;
                            reward=Union{NoValue,Float64}[
                                1.0,
                                0.0,
                                NoValue(),
                                1.0,
                                0.0,
                                NoValue(),
                                0.0,
                                0.0,
                                NoValue(),
                                1.0,
                            ]
                        )),
                        environment_parameters=(;),
                    )...,
                    who_starts,
                    n_steps=20,
                    action_map=NameMap(),
                    SAVE_BOTH...,
                ),
                simulate_case(
                    "WithDomains without checking the parameter domains, $starts",
                    distributions;
                    baseline_with(;
                        model=WithDomains(distributions),
                        cognitive_parameters=(;
                            learning_rate=0.3, inverse_temperature=-1.0
                        ),
                    )...,
                    who_starts,
                    n_steps=10,
                    check_parameter_domains=false,
                    SAVE_BOTH...,
                ),
                instantiated_case(
                    "An agent and an environment, $starts",
                    distributions;
                    baseline_parts...,
                    who_starts,
                    n_steps=10,
                ),
                instantiated_case(
                    "An agent and an environment saving the states only, $starts",
                    distributions;
                    baseline_parts...,
                    save_history=SaveHistory(; states=:expected_reward),
                    who_starts,
                    n_steps=10,
                ),
                instantiated_case(
                    "An agent and an environment simulated twice, $starts",
                    distributions;
                    baseline_parts...,
                    environment_save_history=SaveHistory(SaveEverything()),
                    twice=true,
                    who_starts,
                    n_steps=5,
                ),
                instantiated_case(
                    "An agent and an environment reset before the second simulation, $starts",
                    distributions;
                    baseline_parts...,
                    twice=true,
                    reset_between=true,
                    who_starts,
                    n_steps=5,
                ),
                instantiated_case(
                    "An agent which changes its state in place and an environment, $starts",
                    distributions;
                    baseline_with(; model=StateChangedInPlace(distributions))...,
                    who_starts,
                    n_steps=10,
                ),
                instantiated_case(
                    "An agent which saves its state with deepcopy and an environment, $starts",
                    distributions;
                    baseline_with(;
                        model=StateChangedInPlaceSavedWithDeepcopy(distributions)
                    )...,
                    who_starts,
                    n_steps=10,
                ),
                simulate_case(
                    "StateChangedInPlace saving the states only, $starts",
                    distributions;
                    baseline_with(; model=StateChangedInPlace(distributions))...,
                    who_starts,
                    n_steps=10,
                    cognitive_save_history=SaveHistory(; states=:expected_reward),
                ),
                simulate_case(
                    "Baseline with OutputNotEveryTimestepEnvironment saving the states only, $starts",
                    distributions;
                    inferred=false,
                    baseline_with(;
                        environment=OutputNotEveryTimestepEnvironment(distributions)
                    )...,
                    who_starts,
                    n_steps=10,
                    cognitive_save_history=SaveHistory(; states=:expected_reward),
                ),
            ],
        )
    end

    #The models simulated on one observation at a time
    append!(
        cases,
        [
            single_inputs_case(
                "Baseline on single observations",
                distributions;
                model=Baseline(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
            ),
            single_inputs_case(
                "WithDomains on single observations",
                distributions;
                model=WithDomains(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
            ),
            single_inputs_case(
                "StateChangedInPlace on single observations",
                distributions;
                model=StateChangedInPlace(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
            ),
            instantiated_single_inputs_case(
                "An agent on single observations, with a history",
                distributions;
                model=Baseline(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
            ),
            instantiated_single_inputs_case(
                "An agent on single observations, without a history",
                distributions;
                model=Baseline(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
                save_history=SaveHistory(),
            ),
            instantiated_single_inputs_case(
                "An agent with domains on single observations",
                distributions;
                model=WithDomains(distributions),
                parameters=baseline.cognitive_parameters,
                inputs=SINGLE_REWARDS,
            ),
        ],
    )

    #The models with every feature, in the other ways of simulating them
    every_feature_models = if distributions isa Means
        [pair_named("EveryFeatureWithoutSampling")]
    else
        [pair_named("EveryFeature"), pair_named("EveryFeatureWithoutSampling")]
    end
    for pair in every_feature_models, (who_starts, starts) in WHO_STARTS
        parts = (;
            model=pair.model(distributions),
            environment=pair.environment(distributions),
            pair.cognitive_parameters,
            pair.environment_parameters,
        )
        maps = haskey(pair.keywords, :action_map) ? (; pair.keywords.action_map) : (;)
        append!(
            cases,
            [
                instantiated_case(
                    "$(pair.name) as an agent with an environment, $starts",
                    distributions;
                    parts...,
                    environment_save_history=SaveHistory(SaveEverything()),
                    reactant=false,
                    who_starts,
                    n_steps=10,
                    maps...,
                ),
                simulate_case(
                    "$(pair.name) saving the states only, $starts",
                    distributions;
                    parts...,
                    reactant=false,
                    inferred=false,
                    who_starts,
                    n_steps=10,
                    cognitive_save_history=SaveHistory(; states=:expected_rewards),
                    maps...,
                ),
                simulate_case(
                    "$(pair.name) continuing from a given state and action, $starts",
                    distributions;
                    parts...,
                    reactant=false,
                    inferred=false,
                    who_starts,
                    n_steps=10,
                    previous_action=merge(
                        CognitiveModeling.output_before_first_timestep(parts.model),
                        (; choice=a_choice_or_reward(distributions)),
                    ),
                    SAVE_BOTH...,
                    pair.keywords...,
                ),
            ],
        )
    end
    return [case for case in cases]
end
