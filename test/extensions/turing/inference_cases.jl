#What holds for every inference case that the suites share, with the distributions of Distributions.jl. The log density of
#the Turing.jl model of each case, at values drawn from it, is the log density by hand: that of a session, the sum of
#those of several sessions, and for a full model, that of its parameter models on their own plus those of the sessions,
#with the parameters that the full model gives them. Each full model is then drawn from with Prior(), which needs no
#gradients, and the functions that read a fit read the draws. NUTS fits two of them shortly, since it adds variables of its
#own to a chain: the baseline without missing values, and every feature with a regression, whose missing values are
#continuous, as NUTS needs

#The log density of the parameter models of a full model on their own, at the values of the full model's variables
function parameter_models_log_density(case, values)
    keywords_of_every_backend = (
        :environment_parameters_model,
        :downstream_model,
        :fixed_cognitive_parameters,
        :fixed_environment_parameters,
    )
    prepared = CognitiveModeling.prepare_full_model(
        case.parameters_model,
        case.sessions_info;
        filter(
            ((name, _),) -> name in keywords_of_every_backend,
            pairs(case.full_model_keywords),
        )...,
    )
    return sum((
        prepared.cognitive_parameters_model, prepared.environment_parameters_model
    )) do parameters_model
        model_on_its_own = CognitiveModeling.turing_parameters_model(
            parameters_model, prepared.sessions_info
        )
        return logjoint(model_on_its_own, values)
    end
end

#The cases of a full model that NUTS fits, whose latent variables are all continuous
const CASES_FITTED_WITH_NUTS = (
    "Independent parameters without missing values", "Every feature with a regression"
)

#The cases of a full model with an action that is a Symbol, which sample can't check before it samples: DynamicPPL's
#check_model looks for NaN in every observed value with isnan, which has no method for a Symbol. They sample without the
#check, and the check is marked broken, so that a fix shows up as an unexpected pass
const CASES_WITH_SYMBOL_ACTIONS = ("Every feature with a regression",)

#The name of the variable of a chain that is written as `name`, such as "S1.learning_rate", or nothing
function chain_variable(chain, name)
    names = Turing.FlexiChains.get_name.(keys(chain))
    index = findfirst(variable -> string(variable) == name, names)
    return index === nothing ? nothing : names[index]
end

#Checks the chains that the functions which read a fit make from a chain of a full model: the parameters of each session
#on each draw are those the model gave it, and the saved trajectories of each session have its timesteps, with the
#actions in its data where they are recorded
function test_reading_the_fit(case, model, chain)
    returned_values = returned(model, chain)
    parameters = get_session_parameters(model, chain)
    save_everything = SaveTrajectories(SaveEverything())
    trajectories = get_session_trajectories(
        model, chain; save_cognitive_trajectories=save_everything
    )
    predicted = predict_session_trajectories(
        model, chain; save_cognitive_trajectories=save_everything
    )
    for (i, session_id) in enumerate(case.session_ids)
        session = case.sessions[i]
        for name in keys(case.cognitive_parameters)
            variable = chain_variable(parameters, "$session_id.$name")
            variable === nothing && continue
            @test vec(parameters[variable]) ==
                vec(map(values -> values.cognitive_parameters[name][i], returned_values))
        end
        for name in keys(session.actions)
            variable = chain_variable(trajectories, "$session_id.$name")
            saved = trajectories[variable][1]
            @test length(saved) == session.n_timesteps + 1
            @test all(
                t ->
                    session.actions[name][t] isa ImputedMissingValue ||
                    isequal(saved[t + 1], session.actions[name][t]),
                1:(session.n_timesteps),
            )
            @test length(predicted[variable][1]) == session.n_timesteps + 1
        end
    end
end

@testset "Inference cases" begin
    @testset "Every session case: $(case.name)" for case in session_cases(DistributionsJl())
        model = session_model(
            TuringBackend(),
            case.session_info,
            case.cognitive_parameters,
            case.environment_parameters,
        )
        values = rand(Xoshiro(2), model)
        @test logjoint(model, values) ≈ session_log_density_by_hand(
            case,
            case.cognitive_parameters,
            case.environment_parameters,
            variables_of(values),
        )
    end

    @testset "Every case of several sessions: $(case.name)" for case in sessions_cases(
        DistributionsJl()
    )
        model = multiple_sessions_model(
            TuringBackend(),
            case.sessions_info,
            case.cognitive_parameters,
            case.environment_parameters,
        )
        values = rand(Xoshiro(2), model)
        @test logjoint(model, values) ≈ sessions_log_density_by_hand(
            case, case.cognitive_parameters, case.environment_parameters, values
        )
    end

    @testset "Every full model case: $(case.name)" for case in
                                                       full_model_cases(DistributionsJl())
        model = full_model(
            TuringBackend(),
            case.sessions_info,
            case.parameters_model;
            case.full_model_keywords...,
        )
        values = rand(Xoshiro(2), model)
        returned_values = returned(model, values)
        @test logjoint(model, values) ≈
            parameter_models_log_density(case, values) + sessions_log_density_by_hand(
            case,
            returned_values.cognitive_parameters,
            returned_values.environment_parameters,
            values;
            sessions_info=model.args.sessions_info,
        )

        #Draws from the prior, read as a fit is
        check_model = !(case.name in CASES_WITH_SYMBOL_ACTIONS)
        if !check_model
            @test_broken try
                sample(Xoshiro(3), model, Prior(), 10; progress=false)
                true
            catch error
                error isa MethodError || rethrow()
                false
            end
        end
        test_reading_the_fit(
            case, model, sample(Xoshiro(3), model, Prior(), 10; progress=false, check_model)
        )

        #A short fit with NUTS, of 20 draws after 20 steps of adaptation
        if case.name in CASES_FITTED_WITH_NUTS
            chain = sample(
                Xoshiro(4), model, NUTS(20, 0.65), 20; progress=false, check_model
            )
            test_reading_the_fit(case, model, chain)
            parameters = get_session_parameters(model, chain)
            @test all(
                variable -> string(Turing.FlexiChains.get_name(variable))[1] == 'S',
                keys(parameters),
            )
        end
    end
end
