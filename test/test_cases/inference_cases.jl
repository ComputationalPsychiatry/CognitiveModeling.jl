#The inference cases that the suites share. Each is a session of an example model, several sessions, or a full model of
#them, with data from a simulation with the distributions written by hand, in which some values can be missing.
#session_log_density_by_hand gives the log density of a session by hand, at the values of its variables, which are the
#imputed values and what the models sample of their own, so that a suite can compare the log density of an inference
#backend's model with it. inference_cases(distributions) gives every case with a kind of distributions of the example
#models, those of Distributions.jl in the Turing.jl suites. A runner includes this file after ExampleModels.jl and
#turing_methods.jl
using Random: Xoshiro

## The data of the sessions ##
#The data of a simulation of a pair with the distributions written by hand, as the columns of a session, with the number
#of timesteps that ran
function simulated_data(pair, who_starts; n_steps=10, seed=1)
    simulation = simulate_pair(pair, ByHand(), who_starts; n_steps, rng=Xoshiro(seed))
    observations = columns(simulation.observations[2:end])
    actions = columns(simulation.actions[2:end])
    return observations, actions, length(simulation.actions) - 1
end

#The columns with ImputedMissingValue() on the timesteps given for each name, in a NamedTuple of names and timesteps
function with_missing(columns::NamedTuple{names}, timesteps_by_name) where {names}
    return NamedTuple{names}(
        map(names) do name
            column = Vector{Any}(columns[name])
            for t in get(timesteps_by_name, name, ())
                column[t] = ImputedMissingValue()
            end
            return identity.(column)
        end,
    )
end

#The first timestep from the second on whose value is given, which a case can make missing
function first_given_timestep(column)
    return findfirst(t -> t >= 2 && !(column[t] isa AbstractNoValue), eachindex(column))
end

## The names of each model ##
#The names that a map of names passes on, from the names of one model to those of the other, or the same names
names_passed_on(map::NameMap{from,to}, names) where {from,to} = (from, to)
names_passed_on(map::Nothing, names) = (names, names)

#A NamedTuple of the values of one model under the names of the other, as a map passes them on
function renamed(values::NamedTuple, (from, to))
    return NamedTuple{to}(map(name -> values[name], from))
end

#A NamedTuple of BeforeFirstTimestep() for each name
before_first_timestep(names) = NamedTuple{names}(map(_ -> BeforeFirstTimestep(), names))

## The log density of a session by hand ##
#The values of a row of the data, with the value of the variable of each missing value that is imputed, named by
#`imputed_name`, and UnimputedMissingValue() for each that isn't
function filled_row(
    columns::NamedTuple{names}, t, imputes, imputed_name, variable
) where {names}
    return NamedTuple{names}(
        map(names) do name
            value = columns[name][t]
            value isa ImputedMissingValue || return value
            return imputes ? variable(imputed_name(name)) : UnimputedMissingValue()
        end,
    )
end

#The log density of a session by hand at the values of its variables, which `variable` gives by their names within the
#session: the log likelihood of its actions, with those that are imputed, and of what the cognitive model sampled of its
#own, and of the observations and of what the environment sampled, when the session includes its environment
function session_log_density_by_hand(
    case, cognitive_parameters, environment_parameters, variable
)
    (; model, environment, observations, actions, who_starts, n_timesteps) = case
    session_info = case.session_info
    cognitive_parameters = merge(
        CognitiveModeling.default_parameters(model), cognitive_parameters
    )

    #The data with the imputed values, in the cognitive model's names
    observation_names = names_passed_on(case.observation_map, keys(observations))
    filled_actions = [
        filled_row(
            actions,
            t,
            case.impute_missing_actions,
            name -> "timesteps[$t].$name.imputed",
            variable,
        ) for t in 1:n_timesteps
    ]
    filled_observations = [
        filled_row(
            observations,
            t,
            case.impute_missing_observations,
            name ->
                "timesteps[$t].environment.$(environment_name(observation_names, name)).imputed",
            variable,
        ) for t in 1:n_timesteps
    ]

    #The cognitive model receives the observation of the timestep when the environment starts, and that of the timestep
    #before when the agent starts
    cognitive_inputs = if who_starts isa EnvironmentStarts
        filled_observations
    else
        [
            before_first_timestep(keys(observations));
            filled_observations[1:(n_timesteps - 1)]
        ]
    end
    cognitive_timesteps = [
        (;
            input=cognitive_inputs[t],
            output=filled_actions[t],
            timestep_size=case.timestep_sizes[t],
        ) for t in 1:n_timesteps
    ]
    cognitive_latents = latents_from_variables(model, variable, n_timesteps)
    log_density =
        log_likelihood_by_hand(
            model, cognitive_parameters, cognitive_timesteps, cognitive_latents
        ) + log_density_of_latents_by_hand(model, cognitive_parameters, cognitive_latents)
    session_info.includes_environment || return log_density

    #The environment receives the actions and gives the observations in its own names, the action of the timestep before
    #when it starts, and that of the timestep when the agent starts
    environment_parameters = merge(
        CognitiveModeling.default_parameters(environment), environment_parameters
    )
    action_names = names_passed_on(case.action_map, keys(actions))
    environment_actions = [renamed(action, action_names) for action in filled_actions]
    environment_inputs = if who_starts isa AgentStarts
        environment_actions
    else
        [
            before_first_timestep(last(action_names));
            environment_actions[1:(n_timesteps - 1)]
        ]
    end
    environment_outputs = [
        renamed(observation, reverse(observation_names)) for
        observation in filled_observations
    ]
    environment_timesteps = [
        (;
            input=environment_inputs[t],
            output=environment_outputs[t],
            timestep_size=case.timestep_sizes[t],
        ) for t in 1:n_timesteps
    ]
    environment_latents = latents_from_variables(environment, variable, n_timesteps)
    return log_density +
           log_likelihood_by_hand(
               environment,
               environment_parameters,
               environment_timesteps,
               environment_latents,
           ) +
           log_density_of_latents_by_hand(
               environment, environment_parameters, environment_latents
           )
end

#The environment's name of an observation, from the cognitive model's name
function environment_name((environment_names, cognitive_names), name)
    return environment_names[findfirst(==(name), cognitive_names)]
end

## Sessions ##
#A case of one session of a pair, with data from a simulation, where the values on the timesteps given are missing, and
#the keywords of SessionInfo
function session_case(
    name,
    distributions;
    pair,
    who_starts,
    n_steps=10,
    seed=1,
    missing_actions=(;),
    missing_observations=(;),
    session_keywords...,
)
    observations, actions, n_timesteps = simulated_data(pair, who_starts; n_steps, seed)
    observations = with_missing(observations, missing_observations)
    actions = with_missing(actions, missing_actions)
    keywords = get(pair, :keywords, (;))
    observation_map = get(keywords, :observation_map, nothing)
    action_map = get(keywords, :action_map, nothing)
    timestep_sizes = sizes_of_timesteps(keywords, n_timesteps)
    model = pair.model(distributions)
    environment = pair.environment(distributions)
    session_info = SessionInfo(
        model,
        observations,
        actions;
        who_starts,
        environment,
        timestep_sizes=if get(keywords, :timestep_sizes, 1) isa AbstractVector
            timestep_sizes
        else
            1
        end,
        observation_map,
        action_map,
        session_keywords...,
    )
    return (;
        name,
        session_info,
        model,
        environment,
        observations,
        actions,
        who_starts,
        n_timesteps,
        timestep_sizes,
        observation_map,
        action_map,
        pair.cognitive_parameters,
        pair.environment_parameters,
        impute_missing_actions=get(session_keywords, :impute_missing_actions, true),
        impute_missing_observations=get(
            session_keywords, :impute_missing_observations, true
        ),
    )
end

#Every case of a single session with a kind of distributions of the example models
function session_cases(distributions)
    cases = []
    for pair in EXAMPLE_PAIRS, (who_starts, starts) in WHO_STARTS
        observations, actions, _ = simulated_data(pair, who_starts)

        #The session as it is, which leaves the environment out, and with the environment, which scores the observations
        push!(cases, session_case("$(pair.name), $starts", distributions; pair, who_starts))
        push!(
            cases,
            session_case(
                "$(pair.name) with its environment, $starts",
                distributions;
                pair,
                who_starts,
                include_environment=true,
            ),
        )

        #An action of each kind missing, and imputed, which includes the environment
        for name in keys(actions)
            push!(
                cases,
                session_case(
                    "$(pair.name) with a missing $name, $starts",
                    distributions;
                    pair,
                    who_starts,
                    missing_actions=NamedTuple{(name,)}(((
                        first_given_timestep(actions[name]),
                    ),)),
                ),
            )
        end

        #An observation of each kind missing, and imputed by the environment
        for name in keys(observations)
            push!(
                cases,
                session_case(
                    "$(pair.name) with a missing $name observed, $starts",
                    distributions;
                    pair,
                    who_starts,
                    missing_observations=NamedTuple{(name,)}(((
                        first_given_timestep(observations[name]),
                    ),)),
                ),
            )
        end
    end

    #The baseline with missing values in each way
    baseline = pair_named("Baseline")
    for (who_starts, starts) in WHO_STARTS
        append!(
            cases,
            [
                session_case(
                    "Baseline with a missing choice not imputed, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_actions=(; choice=(3,)),
                    impute_missing_actions=false,
                ),
                session_case(
                    "Baseline with a missing choice imputed without the environment, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_actions=(; choice=(3,)),
                    include_environment=false,
                ),
                session_case(
                    "Baseline with a missing reward not imputed, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_observations=(; reward=(3,)),
                    impute_missing_observations=false,
                    include_environment=false,
                ),
                session_case(
                    "Baseline with a missing choice and reward on the same timestep, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_actions=(; choice=(3,)),
                    missing_observations=(; reward=(3,)),
                ),
                session_case(
                    "Baseline with the first choice and the last reward missing, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_actions=(; choice=(1,)),
                    missing_observations=(; reward=(10,)),
                ),
                session_case(
                    "Baseline with every choice missing, $starts",
                    distributions;
                    pair=baseline,
                    who_starts,
                    missing_actions=(; choice=Tuple(1:10)),
                ),
            ],
        )
    end
    return [case for case in cases]
end

## Several sessions ##
#The real numbers of parameters made different for each of several sessions, so that a session run with another
#session's parameters has another log density
for_session(value::AbstractFloat, i) = value * (1 + (i - 1) / 10)
for_session(value::AbstractVector{<:AbstractFloat}, i) = value .* (1 + (i - 1) / 10)
for_session(value, i) = value
function parameters_of_sessions(parameters::NamedTuple, n_sessions)
    return map(value -> [for_session(value, i) for i in 1:n_sessions], parameters)
end

#The parameters of session i, from a NamedTuple with a vector for each parameter
parameters_of_session(parameters::NamedTuple, i) = map(values -> values[i], parameters)

#Julia's missing for ImputedMissingValue(), as a table has it
as_table_value(value::ImputedMissingValue) = missing
as_table_value(value) = value

#A case of several sessions of a pair, in one table with a session column and a column for each observation and action,
#each session simulated with its own seed and length, and with the values on the timesteps given missing in each,
#prepared with the keywords of MultipleSessionsInfo. Each session is also a case of one session, which gives its log
#density by hand
function sessions_case(
    name,
    distributions;
    pair,
    who_starts,
    lengths=(10, 6, 8),
    missing_actions=((;), (;), (;)),
    missing_observations=((;), (;), (;)),
    columns_of_sessions=(;),
    sessions_keywords...,
)
    keywords = get(pair, :keywords, (;))
    observation_map = get(keywords, :observation_map, nothing)
    action_map = get(keywords, :action_map, nothing)
    model = pair.model(distributions)
    environment = pair.environment(distributions)
    n_sessions = length(lengths)
    session_ids = ["S$i" for i in 1:n_sessions]

    #The data of each session, and the rows of all of them in one table
    sessions = map(1:n_sessions) do i
        observations, actions, n_timesteps = simulated_data(
            pair, who_starts; n_steps=lengths[i], seed=i
        )
        return (;
            observations=with_missing(observations, missing_observations[i]),
            actions=with_missing(actions, missing_actions[i]),
            n_timesteps,
            timestep_sizes=sizes_of_timesteps(keywords, n_timesteps),
        )
    end
    observation_names = keys(first(sessions).observations)
    action_names = keys(first(sessions).actions)
    data_columns = map((observation_names..., action_names...)) do name
        values = [
            as_table_value(merge(session.observations, session.actions)[name][t]) for
            session in sessions for t in 1:(session.n_timesteps)
        ]
        return identity.(values)
    end
    table = merge(
        (;
            id=[session_ids[i] for i in 1:n_sessions for _ in 1:(sessions[i].n_timesteps)],
            timestep_size=[size for session in sessions for size in session.timestep_sizes],
        ),
        NamedTuple{(observation_names..., action_names...)}(data_columns),
        map(
            values_of_sessions -> [
                values_of_sessions[i] for i in 1:n_sessions for
                _ in 1:(sessions[i].n_timesteps)
            ],
            columns_of_sessions,
        ),
    )
    sessions_info = MultipleSessionsInfo(
        model,
        table;
        observation_columns=observation_names,
        action_columns=action_names,
        session_columns=:id,
        who_starts,
        environment,
        timestep_sizes=:timestep_size,
        observation_map,
        action_map,
        sessions_keywords...,
    )

    #Each session as a case of one session, for its log density by hand
    session_cases = map(1:n_sessions) do i
        return (;
            name="$name, session $i",
            session_info=sessions_info.sessions[i],
            model,
            environment,
            sessions[i].observations,
            sessions[i].actions,
            who_starts,
            sessions[i].n_timesteps,
            sessions[i].timestep_sizes,
            observation_map,
            action_map,
            impute_missing_actions=get(sessions_keywords, :impute_missing_actions, true),
            impute_missing_observations=get(
                sessions_keywords, :impute_missing_observations, true
            ),
        )
    end
    return (;
        name,
        sessions_info,
        session_ids,
        sessions=session_cases,
        cognitive_parameters=parameters_of_sessions(pair.cognitive_parameters, n_sessions),
        environment_parameters=parameters_of_sessions(
            pair.environment_parameters, n_sessions
        ),
    )
end

#The log density of several sessions by hand, the sum of each session's, at the values of the variables of the model of
#them, where those of session i have the prefix sessions[i]. Whether a session includes its environment is read from
#`sessions_info`, which a full model prepares from the sessions of the case
function sessions_log_density_by_hand(
    case,
    cognitive_parameters,
    environment_parameters,
    values;
    sessions_info=case.sessions_info,
)
    return sum(eachindex(case.sessions)) do i
        return session_log_density_by_hand(
            merge(case.sessions[i], (; session_info=sessions_info.sessions[i])),
            parameters_of_session(cognitive_parameters, i),
            parameters_of_session(environment_parameters, i),
            variables_of(values; prefix="sessions[$i]."),
        )
    end
end

#Every case of several sessions with a kind of distributions of the example models: sessions of different lengths, of
#which the first and the third have missing values, so that only they include their environment
function sessions_cases(distributions)
    cases = []
    for name in (
            "Baseline",
            "ContinuousOutput",
            "VectorAttributes",
            "NamedTupleAttributes",
            "SamplingWithinTimestep",
            "UsesTimestepSize",
            "Baseline with OtherNamesEnvironment",
            "EveryFeature",
        ),
        (who_starts, starts) in WHO_STARTS

        pair = pair_named(name)
        observations, actions, _ = simulated_data(pair, who_starts; n_steps=10, seed=1)
        _, third_actions, _ = simulated_data(pair, who_starts; n_steps=8, seed=3)
        first_action = first(keys(actions))
        first_observation = first(keys(observations))
        missing_actions = (
            NamedTuple{(first_action,)}(((first_given_timestep(actions[first_action]),),)),
            (;),
            NamedTuple{(first_action,)}(((
                first_given_timestep(third_actions[first_action][3:end]) + 2,
            ),)),
        )
        missing_observations = ((;), (;), NamedTuple{(first_observation,)}(((5,),)))
        push!(
            cases,
            sessions_case(
                "$name, sessions, $starts",
                distributions;
                pair,
                who_starts,
                missing_actions,
                missing_observations,
            ),
        )
    end
    return [case for case in cases]
end

## Full models ##
#The columns of a regression and of the groups of the sessions, one value for each of three sessions: an age, a group that
#puts the first and the third together, and a participant with two of them
const COLUMNS_OF_SESSIONS = (;
    age=[-1.0, 0.5, 1.0], group=["a", "b", "a"], participant=["P1", "P2", "P1"]
)

#Maps the real line to the values between 0 and 1, as a transformation of a learning rate
to_unit_interval(x) = 1 / (1 + exp(-x))

#A case of a full model of several sessions of a pair, with the parameter models and the keywords of full_model
function full_model_case(
    name,
    distributions;
    pair,
    parameters_model,
    who_starts=EnvironmentStarts(),
    missing_actions=((; choice=(3,)), (;), (;)),
    missing_observations=((;), (;), (;)),
    missing_values=true,
    sessions_keywords=(;),
    full_model_keywords...,
)
    #Without missing values when they are left out, such as for the gradients, which a discrete imputed value has none of
    if !missing_values
        missing_actions = ((;), (;), (;))
        missing_observations = ((;), (;), (;))
    end
    sessions = sessions_case(
        name,
        distributions;
        pair,
        who_starts,
        missing_actions,
        missing_observations,
        columns_of_sessions=COLUMNS_OF_SESSIONS,
        sessions_keywords...,
    )
    #The environment's parameters are fixed at the pair's, unless a parameter model gives them
    keywords = NamedTuple(full_model_keywords)
    if !haskey(keywords, :environment_parameters_model)
        keywords = merge(
            (; fixed_environment_parameters=pair.environment_parameters), keywords
        )
    end
    return merge(sessions, (; parameters_model, full_model_keywords=keywords))
end

#Every case of a full model with a kind of distributions of the example models, which are those of Distributions.jl, for
#the priors of the parameter models: each parameter model, with the features it has, on three sessions of the baseline,
#where the first has a missing choice, so that only it includes its environment. With missing_values = false, no value is
#missing in any case
function full_model_cases(distributions; missing_values=true)
    baseline = pair_named("Baseline")
    case(arguments...; keywords...) =
        full_model_case(arguments...; keywords..., missing_values)
    group_priors = (; mean=Normal(0, 1), sd=Exponential(1))
    return [
        case(
            "Independent parameters without missing values",
            distributions;
            pair=baseline,
            missing_actions=((;), (;), (;)),
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
            )),
        ),
        case(
            "Independent parameters",
            distributions;
            pair=baseline,
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
            )),
        ),
        case(
            "Independent parameters, agent starts",
            distributions;
            pair=baseline,
            who_starts=AgentStarts(),
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
            )),
        ),
        case(
            "Identical parameters",
            distributions;
            pair=baseline,
            parameters_model=IdenticalParameters((;
                learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
            )),
        ),
        case(
            "Fixed parameters",
            distributions;
            pair=baseline,
            parameters_model=FixedParameters((;
                learning_rate=0.3, inverse_temperature=2.0
            )),
        ),
        case(
            "Fixed parameters for each session",
            distributions;
            pair=baseline,
            parameters_model=SessionwiseFixedParameters((;
                S1=(; learning_rate=0.2, inverse_temperature=1.0),
                S2=(; learning_rate=0.3, inverse_temperature=2.0),
                S3=(; learning_rate=0.4, inverse_temperature=3.0),
            )),
        ),
        case(
            "A parameter fixed by keyword",
            distributions;
            pair=baseline,
            parameters_model=IndependentParameters((; learning_rate=Beta(2, 2))),
            fixed_cognitive_parameters=(; inverse_temperature=2.0),
        ),
        case(
            "Hierarchical parameters, centered and non-centered",
            distributions;
            pair=baseline,
            parameters_model=HierarchicalGroupedParameters((;
                learning_rate=HierarchicalParameter(
                    group_priors;
                    transformation=to_unit_interval,
                    group_by=:group,
                    centered=true,
                ),
                inverse_temperature=HierarchicalParameter(group_priors; transformation=exp),
            )),
        ),
        case(
            "A regression with every kind of term",
            distributions;
            pair=baseline,
            parameters_model=HierarchicalRegressionParameters((;
                learning_rate=RegressionParameter(
                    Intercept(Normal(0, 1)),
                    ContinuousPredictor(:age, Normal(0, 1)),
                    CategoricalPredictor(:group, Normal(0, 1)),
                    Interaction((:age, :group), Normal(0, 1)),
                    VaryingEffects(
                        :participant,
                        Intercept(Exponential(1)),
                        ContinuousPredictor(:age, Exponential(1)),
                    ),
                    VaryingEffects(
                        :id, Intercept(Exponential(1)); group_by=:group, name=:session
                    );
                    transformation=to_unit_interval,
                ),
                inverse_temperature=RegressionParameter(
                    Intercept(Normal(0, 1));
                    family=RegressionFamily(
                        (; mean=RegressionOutcome(), sd=Exponential(1));
                        distribution=LogNormal,
                    ),
                ),
            )),
        ),
        case(
            "Composite parameters",
            distributions;
            pair=baseline,
            parameters_model=CompositeParameters(
                IdenticalParameters((; learning_rate=Beta(2, 2)));
                temperatures=IndependentParameters((;
                    inverse_temperature=LogNormal(0, 0.5)
                )),
            ),
        ),
        case(
            "Environment parameters estimated",
            distributions;
            pair=baseline,
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 2), inverse_temperature=LogNormal(0, 0.5)
            )),
            environment_parameters_model=IdenticalParameters((;
                reward_probability=Beta(2, 2)
            )),
        ),
        case(
            "A discrete parameter",
            distributions;
            pair=pair_named("WithDiscreteParameter, learning"),
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 2),
                inverse_temperature=LogNormal(0, 0.5),
                learns=Bernoulli(0.7),
            )),
        ),
        case(
            "A parameter that is a vector",
            distributions;
            pair=pair_named("VectorAttributes"),
            missing_actions=((; ratings=(3,)), (;), (;)),
            parameters_model=IndependentParameters((;
                learning_rates=product_distribution(fill(Beta(2, 2), 3)),
                noise=LogNormal(-1, 0.5),
            )),
        ),
        case(
            "Every feature with a regression",
            distributions;
            pair=pair_named("EveryFeature"),
            missing_actions=((; rating=(3,)), (;), (; rating=(2,))),
            parameters_model=CompositeParameters(
                HierarchicalRegressionParameters((;
                    stickiness=RegressionParameter(
                        Intercept(Normal(1, 0.5)), ContinuousPredictor(:age, Normal(0, 0.5))
                    ),
                )),
                FixedParameters((;
                    learning_rate=0.3,
                    initial_value=0.4,
                    learning_rates=[0.2, 0.3, 0.4],
                    rating_noise=0.2,
                    initial_noise=0.1,
                    update_noise=0.05,
                    rate=2.0,
                )),
            ),
        ),
        case(
            "Parameters that can turn invalid",
            distributions;
            pair=pair_named("WithInvalidParameters"),
            parameters_model=IndependentParameters((;
                learning_rate=Beta(2, 8), inverse_temperature=LogNormal(0, 0.5)
            )),
        ),
    ]
end
