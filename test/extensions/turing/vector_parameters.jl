## Models for the tests ##
#A cognitive model which chooses one of three options with probabilities that are a parameter, a vector, mixed with a lapse rate, a number, which makes every option equally likely and is zero unless it is given
struct ChoosesWithProbabilities <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ChoosesWithProbabilities, parameters) = (;)
function CognitiveModeling.run_timestep(
    model::ChoosesWithProbabilities, context::ModelContext
)
    choice_probabilities = context.parameters.choice_probabilities
    lapse_rate = context.parameters.lapse_rate
    return ModelOutput((;
        choice=Categorical(
            (1 - lapse_rate) .* choice_probabilities .+
            lapse_rate / length(choice_probabilities),
        ),
    ))
end
function CognitiveModeling.attribute_names(model::ChoosesWithProbabilities)
    return ModelAttributes(;
        parameters=(:choice_probabilities, :lapse_rate),
        states=(),
        inputs=:trial,
        outputs=:choice,
    )
end
function CognitiveModeling.parameter_domains(model::ChoosesWithProbabilities)
    return (;
        choice_probabilities=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        lapse_rate=Domain(; lower=0, upper=1),
    )
end
CognitiveModeling.default_parameters(model::ChoosesWithProbabilities) = (; lapse_rate=0.0)

#The choices of three sessions among three options, with the trial as the observation, and a column that groups them: S1 and S3 in one group, and S2 in another
const CHOICES_TABLE = (;
    id=["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", "S3", "S3"],
    trial=[1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5],
    choice=[1, 1, 2, 3, 3, 3, 2, 2, 1, 2, 2, 3],
    group=["a", "a", "a", "a", "b", "b", "b", "a", "a", "a", "a", "a"],
)
function choice_sessions(table=CHOICES_TABLE; keywords...)
    return MultipleSessionsInfo(
        ChoosesWithProbabilities(),
        table;
        observation_columns=:trial,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

#The log likelihood of the choices of each session by hand, with the choice probabilities of each session
function choices_by_hand(probabilities_of_sessions; lapse_rate=0.0)
    log_likelihood = 0.0
    for (session, id) in enumerate(unique(CHOICES_TABLE.id))
        mixed = (1 - lapse_rate) .* probabilities_of_sessions[session] .+ lapse_rate / 3
        for row in findall(==(id), CHOICES_TABLE.id)
            log_likelihood += log(mixed[CHOICES_TABLE.choice[row]])
        end
    end
    return log_likelihood
end

const CHOICE_PROBABILITIES = [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6], [0.3, 0.5, 0.2]]
const CONCENTRATION_PRIOR = filldist(Gamma(2, 2), 3)

@testset "Parameters that are vectors, with Dirichlet priors" begin
    @testset "A vector shared by every session: IdenticalParameters" begin
        model = full_model(
            TuringBackend(),
            IdenticalParameters((; choice_probabilities=Dirichlet(ones(3)))),
            choice_sessions(),
        )
        @test Set(keys(rand(Xoshiro(1), model))) ==
            Set([@varname(choice_probabilities.shared)])
        values = Dict(@varname(choice_probabilities.shared) => [0.2, 0.3, 0.5])
        #Every session has the same vector, and the lapse rate takes its default in the sessions
        parameters = returned(model, values).cognitive_parameters
        @test parameters.choice_probabilities == fill([0.2, 0.3, 0.5], 3)
        @test logjoint(model, values) ≈
            logpdf(Dirichlet(ones(3)), [0.2, 0.3, 0.5]) +
              choices_by_hand(fill([0.2, 0.3, 0.5], 3))
    end

    @testset "A vector for each session: IndependentParameters gives each session its column, as a vector of its own" begin
        model = full_model(
            TuringBackend(),
            IndependentParameters((; choice_probabilities=Dirichlet(ones(3)))),
            choice_sessions(),
        )
        #The vectors of the sessions are the columns of one variable
        sampled = rand(Xoshiro(1), model)
        @test Set(keys(sampled)) == Set([@varname(choice_probabilities.session)])
        @test size(sampled[@varname(choice_probabilities.session)]) == (3, 3)
        values = Dict(
            @varname(choice_probabilities.session) => reduce(hcat, CHOICE_PROBABILITIES)
        )
        parameters = returned(model, values).cognitive_parameters
        @test parameters.choice_probabilities == CHOICE_PROBABILITIES
        @test all(vector -> vector isa Vector{Float64}, parameters.choice_probabilities)
        @test logjoint(model, values) ≈
            sum(p -> logpdf(Dirichlet(ones(3)), p), CHOICE_PROBABILITIES) +
              choices_by_hand(CHOICE_PROBABILITIES)
    end

    @testset "A vector for each session from a hierarchical Dirichlet: HierarchicalGroupedParameters, with one group and with two" begin
        for (group_by, concentrations, concentration_of_session) in (
            (nothing, [2.0, 1.0, 0.5], [1, 1, 1]),
            (:group, [[2.0, 1.0, 0.5], [0.5, 1.0, 3.0]], [1, 2, 1]),
        )
            parameters_model = HierarchicalGroupedParameters((;
                choice_probabilities=HierarchicalParameter(
                    (; concentration=CONCENTRATION_PRIOR); distribution=Dirichlet, group_by
                ),
            ))
            model = full_model(TuringBackend(), parameters_model, choice_sessions())
            #Without groups the concentrations are one vector, which isn't taken for values of groups, and with groups one vector for each group, under its name
            if group_by === nothing
                concentration_values = Dict(
                    @varname(choice_probabilities.concentration.group) => concentrations
                )
                concentration_of_each_session = fill(concentrations, 3)
                prior = logpdf(CONCENTRATION_PRIOR, concentrations)
            else
                concentration_values = Dict(
                    @varname(choice_probabilities.concentration.group) =>
                        (; a=concentrations[1], b=concentrations[2]),
                )
                concentration_of_each_session = concentrations[concentration_of_session]
                prior = sum(c -> logpdf(CONCENTRATION_PRIOR, c), concentrations)
            end
            values = merge(
                concentration_values,
                Dict(
                    @varname(choice_probabilities.session) =>
                        reduce(hcat, CHOICE_PROBABILITIES),
                ),
            )
            @test Set(keys(rand(Xoshiro(1), model))) == Set(keys(values))
            parameters = returned(model, values).cognitive_parameters
            @test parameters.choice_probabilities == CHOICE_PROBABILITIES
            @test logjoint(model, values) ≈
                prior +
                  sum(
                      logpdf(Dirichlet(concentration), p) for (concentration, p) in
                      zip(concentration_of_each_session, CHOICE_PROBABILITIES)
                  ) +
                  choices_by_hand(CHOICE_PROBABILITIES)
        end
    end

    @testset "A vector parameter beside a number, each with its own parameter model: CompositeParameters" begin
        parameters_model = CompositeParameters(
            IndependentParameters((; choice_probabilities=Dirichlet(ones(3)))),
            IdenticalParameters((; lapse_rate=Beta(1, 9))),
        )
        model = full_model(TuringBackend(), parameters_model, choice_sessions())
        values = Dict(
            @varname(choice_probabilities.session) => reduce(hcat, CHOICE_PROBABILITIES),
            @varname(lapse_rate.shared) => 0.1,
        )
        parameters = returned(model, values).cognitive_parameters
        @test parameters.choice_probabilities == CHOICE_PROBABILITIES
        @test parameters.lapse_rate == fill(0.1, 3)
        @test logjoint(model, values) ≈
            sum(p -> logpdf(Dirichlet(ones(3)), p), CHOICE_PROBABILITIES) +
              logpdf(Beta(1, 9), 0.1) +
              choices_by_hand(CHOICE_PROBABILITIES; lapse_rate=0.1)
    end

    @testset "Fixed vectors: FixedParameters for every session, and SessionwiseFixedParameters for each" begin
        shared = full_model(
            TuringBackend(),
            FixedParameters((; choice_probabilities=[0.2, 0.3, 0.5])),
            choice_sessions(),
        )
        @test logjoint(shared, (;)) ≈ choices_by_hand(fill([0.2, 0.3, 0.5], 3))
        each = full_model(
            TuringBackend(),
            SessionwiseFixedParameters(
                Dict(
                    id => (; choice_probabilities=p) for
                    (id, p) in zip(["S1", "S2", "S3"], CHOICE_PROBABILITIES)
                ),
            ),
            choice_sessions(),
        )
        @test logjoint(each, (;)) ≈ choices_by_hand(CHOICE_PROBABILITIES)
    end

    @testset "The domain of a vector is checked inside the model, and an element outside it is named by its position" begin
        model = full_model(
            TuringBackend(),
            IdenticalParameters((; choice_probabilities=filldist(Normal(0.3, 0.3), 3))),
            choice_sessions(),
        )
        @test_throws ParameterDomainError(
            "The parameter choice_probabilities of a cognitive model of type ChoosesWithProbabilities has choice_probabilities[2] = -0.1, which is outside its domain, from 0 to 1.",
        ) logjoint(model, Dict(@varname(choice_probabilities.shared) => [0.5, -0.1, 0.6]))
        #A number where the domain takes a vector is named too
        number_model = full_model(
            TuringBackend(),
            IdenticalParameters((; choice_probabilities=Beta(2, 2))),
            choice_sessions(),
        )
        @test_throws ParameterDomainError(
            "The parameter choice_probabilities of a cognitive model of type ChoosesWithProbabilities is 0.5, of type Float64, but its domain takes values of type AbstractVector{Real}.",
        ) logjoint(number_model, Dict(@varname(choice_probabilities.shared) => 0.5))
    end

    @testset "A fit with NUTS gives each session its vector, which get_session_parameters reads" begin
        model = full_model(
            TuringBackend(),
            IndependentParameters((; choice_probabilities=Dirichlet(ones(3)))),
            choice_sessions(),
        )
        chain = sample(Xoshiro(1), model, NUTS(), 50; progress=false)
        session_parameters = get_session_parameters(model, chain)
        for id in (:S1, :S2, :S3)
            draws = vec(session_parameters[@varname($id.choice_probabilities)])
            @test length(draws) == 50
            @test all(draw -> length(draw) == 3 && sum(draw) ≈ 1, draws)
        end
    end
end
