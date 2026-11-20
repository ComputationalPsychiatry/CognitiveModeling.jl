## Each cognitive model with the environment it runs with ##
#The pairs that the example models' own tests simulate, and that the test cases are made from, with every kind of
#distributions, and helpers to simulate them and to read what each model received

#Each cognitive model with the environment it runs with, as functions of a kind of distributions, and their parameters
const EXAMPLE_PAIRS = [
    (;
        name="Baseline",
        model=Baseline,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithoutParameters",
        model=WithoutParameters,
        environment=WithoutParametersEnvironment,
        cognitive_parameters=(;),
        environment_parameters=(;),
    ),
    (;
        name="WithoutStates",
        model=WithoutStates,
        environment=WithoutStatesEnvironment,
        cognitive_parameters=(; inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithoutInputs",
        model=WithoutInputs,
        environment=_ -> WithoutOutputsEnvironment(),
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(;),
    ),
    (;
        name="WithoutOutputs",
        model=_ -> WithoutOutputs(),
        environment=WithoutInputsEnvironment,
        cognitive_parameters=(; learning_rate=0.3),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="ManyAttributes",
        model=ManyAttributes,
        environment=ManyAttributesEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0, noise=0.2),
        environment_parameters=(; reward_probability=0.8, noise=0.2, bonus_size=0.5),
    ),
    (;
        name="VectorAttributes",
        model=VectorAttributes,
        environment=VectorAttributesEnvironment,
        cognitive_parameters=(; learning_rates=[0.2, 0.3, 0.4], noise=0.2),
        environment_parameters=(; payoffs=[0.2, 0.5, 0.8], noise=0.1),
    ),
    (;
        name="NamedTupleAttributes",
        model=NamedTupleAttributes,
        environment=NamedTupleAttributesEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0, rate=2.0),
        environment_parameters=(; reward_probability=0.8, cue_mean=0.6, cue_noise=0.1),
    ),
    (;
        name="ContinuousOutput",
        model=ContinuousOutput,
        environment=ContinuousInputEnvironment,
        cognitive_parameters=(; learning_rate=0.3, noise=0.2),
        environment_parameters=(; reward_noise=0.2),
    ),
    (;
        name="SymbolOutput",
        model=SymbolOutput,
        environment=SymbolInputEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithSymbolParameter, learning",
        model=WithSymbolParameter,
        environment=BaselineEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3, inverse_temperature=2.0, strategy=:learning
        ),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithSymbolParameter, guessing",
        model=WithSymbolParameter,
        environment=BaselineEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3, inverse_temperature=2.0, strategy=:guessing
        ),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithDiscreteParameter, learning",
        model=WithDiscreteParameter,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0, learns=true),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithDiscreteParameter, not learning",
        model=WithDiscreteParameter,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0, learns=false),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithDefaultParameters",
        model=WithDefaultParameters,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithDomains",
        model=WithDomains,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="WithStoppingCriterion",
        model=WithStoppingCriterion,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.95),
    ),
    (;
        name="DependsOnPreviousOutput",
        model=DependsOnPreviousOutput,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0, stickiness=1.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="UsesTimestepSize",
        model=UsesTimestepSize,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
        keywords=(; timestep_sizes=collect(range(0.5, 1.5; length=10))),
    ),
    (;
        name="InitialStateFromParameter",
        model=InitialStateFromParameter,
        environment=BaselineEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3, inverse_temperature=2.0, initial_value=0.2
        ),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="SampledInitialState",
        model=SampledInitialState,
        environment=BaselineEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3, inverse_temperature=2.0, initial_value=0.2, initial_noise=0.1
        ),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="SamplingWithinTimestep",
        model=SamplingWithinTimestep,
        environment=BaselineEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3, inverse_temperature=2.0, update_noise=0.1
        ),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="OutputNotEveryTimestep",
        model=OutputNotEveryTimestep,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="ThrowsInvalidParameters",
        model=ThrowsInvalidParameters,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="StateChangedInPlace",
        model=StateChangedInPlace,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="StateChangedInPlaceSavedWithDeepcopy",
        model=StateChangedInPlaceSavedWithDeepcopy,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="StateChangesType",
        model=StateChangesType,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="MethodsFromSupertype",
        model=MethodsFromSupertype,
        environment=BaselineEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="Baseline with WithoutStatesEnvironment",
        model=Baseline,
        environment=WithoutStatesEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="Baseline with OutputNotEveryTimestepEnvironment",
        model=Baseline,
        environment=OutputNotEveryTimestepEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="Baseline with WithStoppingCriterionEnvironment",
        model=Baseline,
        environment=WithStoppingCriterionEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8, last_timestep=6),
    ),
    (;
        name="Baseline with NoOutputBeforeFirstInputEnvironment",
        model=Baseline,
        environment=NoOutputBeforeFirstInputEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
    ),
    (;
        name="Baseline with SamplingWithinTimestepEnvironment",
        model=Baseline,
        environment=SamplingWithinTimestepEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; drift_noise=0.1, reward_noise=0.2),
    ),
    (;
        name="Baseline with OtherNamesEnvironment",
        model=Baseline,
        environment=OtherNamesEnvironment,
        cognitive_parameters=(; learning_rate=0.3, inverse_temperature=2.0),
        environment_parameters=(; reward_probability=0.8),
        keywords=(;
            observation_map=NameMap(:payoff => :reward),
            action_map=NameMap(:choice => :lever),
        ),
    ),
    (;
        name="EveryFeature",
        model=EveryFeature,
        environment=EveryFeatureEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3,
            stickiness=1.0,
            initial_value=0.4,
            learning_rates=[0.2, 0.3, 0.4],
            rating_noise=0.2,
            initial_noise=0.1,
            update_noise=0.05,
            rate=2.0,
        ),
        environment_parameters=(;
            reward_probability=0.8,
            payoffs=[0.2, 0.5, 0.8],
            noise=0.2,
            cue_mean=0.5,
            drift_noise=0.05,
        ),
        keywords=(;
            timestep_sizes=collect(range(0.5, 1.5; length=10)),
            action_map=NameMap(:choice => :choice, :rating => :rating, :option => :option),
        ),
    ),
    (;
        name="EveryFeatureWithoutSampling",
        model=EveryFeatureWithoutSampling,
        environment=EveryFeatureEnvironment,
        cognitive_parameters=(;
            learning_rate=0.3,
            stickiness=1.0,
            initial_value=0.4,
            learning_rates=[0.2, 0.3, 0.4],
            rating_noise=0.2,
        ),
        environment_parameters=(;
            reward_probability=0.8,
            payoffs=[0.2, 0.5, 0.8],
            noise=0.2,
            cue_mean=0.5,
            drift_noise=0.05,
        ),
        keywords=(; timestep_sizes=collect(range(0.5, 1.5; length=10))),
    ),
]

#The pair of a name
pair_named(name) = only(filter(pair -> pair.name == name, EXAMPLE_PAIRS))

#The pairs whose distributions all have means, which simulate without randomness with them
const PAIRS_WITH_MEANS = filter(
    pair -> !(pair.name in ("NamedTupleAttributes", "SymbolOutput", "EveryFeature")),
    EXAMPLE_PAIRS,
)

#The input that a model received on each timestep, from what the other model gave, saved with timestep 0 first. The model
#that moves second on a timestep receives what the other gave on the same timestep, and the model that moves first what
#the other gave on the timestep before, which on the first timestep is what was given before it
function inputs_received(saved, moves_first::Bool)
    n_timesteps = length(saved) - 1
    return moves_first ? saved[1:n_timesteps] : saved[2:end]
end

#The timesteps of a simulation as the log likelihoods by hand take them, from the inputs received on each timestep, and
#the outputs saved with timestep 0 first
function timesteps_of(inputs, outputs, timestep_sizes)
    return [
        (; input=inputs[t], output=outputs[t + 1], timestep_size=timestep_sizes[t]) for
        t in eachindex(inputs)
    ]
end

#The size of each timestep that ran, from the keywords of a simulation
function sizes_of_timesteps(keywords, n_timesteps)
    sizes = get(keywords, :timestep_sizes, 1)
    return sizes isa AbstractVector ? sizes[1:n_timesteps] : fill(sizes, n_timesteps)
end

#Simulates a pair with a kind of distributions, saving everything of both models
function simulate_pair(pair, distributions, who_starts; n_steps=10, rng=Xoshiro(1))
    return simulate(
        pair.model(distributions),
        pair.environment(distributions);
        who_starts,
        n_steps,
        cognitive_parameters=pair.cognitive_parameters,
        environment_parameters=pair.environment_parameters,
        rng,
        cognitive_save_history=SaveHistory(SaveEverything()),
        environment_save_history=SaveHistory(SaveEverything()),
        get(pair, :keywords, (;))...,
    )
end

#The values of each name on every timestep, from a vector of NamedTuples, as the columns of a session's data
function columns(values)
    names = keys(first(values))
    return NamedTuple{names}(map(name -> [value[name] for value in values], names))
end
