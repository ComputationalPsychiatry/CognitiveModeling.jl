using Random: AbstractRNG, randexp

## Models for the tests ##
#A stand-in for the distributions of SequentialSamplingModels.jl, whose values are NamedTuples of a choice and a response time: the choice is 1 with a probability and 2 otherwise, and the time is exponential with a rate, independent of the choice
struct ChoiceAndTime{T} <: Turing.Distributions.Distribution{
    Turing.Distributions.Multivariate,Turing.Distributions.ValueSupport
}
    probability::T
    rate::T
end
function Turing.Distributions.logpdf(distribution::ChoiceAndTime, value::NamedTuple)
    choice_density = if value.choice == 1
        log(distribution.probability)
    else
        log1p(-distribution.probability)
    end
    return choice_density + log(distribution.rate) - distribution.rate * value.rt
end
function Base.rand(rng::AbstractRNG, distribution::ChoiceAndTime)
    choice = rand(rng) < distribution.probability ? 1 : 2
    return (; choice, rt=randexp(rng) / distribution.rate)
end

#A cognitive model which responds with a choice and a time, and with a confidence, a number, around its choice probability
struct RespondsWithChoiceAndTime <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::RespondsWithChoiceAndTime, parameters) = (;)
function CognitiveModeling.timestep_model(
    model::RespondsWithChoiceAndTime, context::ModelContext
)
    probability = context.parameters.choice_probability
    return ModelOutput((;
        response=ChoiceAndTime(probability, context.parameters.rate),
        confidence=Normal(probability, 0.1),
    ))
end
function CognitiveModeling.attribute_names(model::RespondsWithChoiceAndTime)
    return ModelAttributes(;
        parameters=(:choice_probability, :rate),
        states=(),
        inputs=:trial,
        outputs=(:response, :confidence),
    )
end
function CognitiveModeling.parameter_domains(model::RespondsWithChoiceAndTime)
    return (; choice_probability=Domain(; lower=0, upper=1), rate=Domain(; lower=0))
end

#The responses of three sessions, with the choice and the response time in two columns, and a confidence
const RESPONSES_TABLE = (;
    id=["S1", "S1", "S1", "S2", "S2", "S3", "S3", "S3"],
    trial=[1, 2, 3, 1, 2, 1, 2, 3],
    choice=[1, 2, 1, 1, 1, 2, 2, 1],
    rt=[0.4, 0.7, 0.3, 0.5, 0.2, 0.9, 1.1, 0.6],
    confidence=[0.7, 0.4, 0.8, 0.9, 0.8, 0.3, 0.2, 0.6],
)
const RESPONSE_PRIORS = (; choice_probability=Beta(2, 2), rate=LogNormal(0, 0.5))
function response_sessions(table=RESPONSES_TABLE; keywords...)
    return MultipleSessionsInfo(
        RespondsWithChoiceAndTime(),
        table;
        observation_columns=:trial,
        action_columns=((; choice=:choice, rt=:rt) => :response, :confidence),
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

#The log likelihood of the rows of the table by hand, of the responses and of the confidences, leaving out the rows given
function responses_by_hand(probabilities, rates; left_out=())
    log_likelihood = 0.0
    for (session, id) in enumerate(unique(RESPONSES_TABLE.id))
        for row in findall(==(id), RESPONSES_TABLE.id)
            row in left_out && continue
            response = (; choice=RESPONSES_TABLE.choice[row], rt=RESPONSES_TABLE.rt[row])
            log_likelihood +=
                logpdf(ChoiceAndTime(probabilities[session], rates[session]), response) +
                logpdf(Normal(probabilities[session], 0.1), RESPONSES_TABLE.confidence[row])
        end
    end
    return log_likelihood
end

#A cognitive model which reports two numbers at once, from a normal distribution over vectors around a mean that is a parameter
struct ReportsTwoNumbers <: AbstractCognitiveModel end
CognitiveModeling.initialize_state(model::ReportsTwoNumbers, parameters) = (;)
function CognitiveModeling.timestep_model(model::ReportsTwoNumbers, context::ModelContext)
    return ModelOutput((; report=MvNormal(fill(context.parameters.mean, 2), IDENTITY_2)))
end
function CognitiveModeling.attribute_names(model::ReportsTwoNumbers)
    return ModelAttributes(; parameters=:mean, states=(), inputs=:trial, outputs=:report)
end
const IDENTITY_2 = [1.0 0.0; 0.0 1.0]

const PROBABILITIES = [0.6, 0.7, 0.3]
const RATES = [2.0, 3.0, 1.5]

@testset "Outputs whose values aren't numbers: NamedTuples, such as a choice and a response time, and vectors" begin
    @testset "Two columns make one output, which is scored by the sum of the log densities of its values, next to an output of numbers" begin
        sessions_info = response_sessions()
        @test eltype(first(sessions_info.sessions).actions.response) ==
            @NamedTuple{choice::Int64, rt::Float64}
        model = full_model(
            TuringBackend(), IndependentParameters(RESPONSE_PRIORS), sessions_info
        )
        values = Dict(
            @varname(choice_probability.sessions) => PROBABILITIES,
            @varname(rate.sessions) => RATES,
        )
        @test logjoint(model, values) ≈
            sum(p -> logpdf(Beta(2, 2), p), PROBABILITIES) +
              sum(r -> logpdf(LogNormal(0, 0.5), r), RATES) +
              responses_by_hand(PROBABILITIES, RATES)
    end

    @testset "A response with a value missing in one of its columns is left out, when missing actions aren't imputed" begin
        table = (; RESPONSES_TABLE..., rt=Union{Missing,Float64}[RESPONSES_TABLE.rt...])
        table.rt[2] = missing
        model = full_model(
            TuringBackend(),
            IndependentParameters(RESPONSE_PRIORS),
            response_sessions(table; impute_missing_actions=false),
        )
        values = Dict(
            @varname(choice_probability.sessions) => PROBABILITIES,
            @varname(rate.sessions) => RATES,
        )
        #The confidence of that row still counts
        @test logjoint(model, values) ≈
            sum(p -> logpdf(Beta(2, 2), p), PROBABILITIES) +
              sum(r -> logpdf(LogNormal(0, 0.5), r), RATES) +
              responses_by_hand(PROBABILITIES, RATES; left_out=(2,)) +
              logpdf(Normal(PROBABILITIES[1], 0.1), RESPONSES_TABLE.confidence[2])
    end

    @testset "NUTS fits the model" begin
        model = full_model(
            TuringBackend(), IndependentParameters(RESPONSE_PRIORS), response_sessions()
        )
        chain = sample(Xoshiro(1), model, NUTS(), 50; progress=false)
        @test all(isfinite, vec(chain[:logjoint]))
        @test all(
            draw -> all(0 .< draw .< 1), vec(chain[@varname(choice_probability.sessions)])
        )
    end

    @testset "An output whose values are vectors, made from two columns, is scored by the sum of the log densities of its values" begin
        table = (;
            id=["S1", "S1", "S2"],
            trial=[1, 2, 1],
            first=[0.1, 0.3, -0.2],
            second=[0.5, 0.2, 0.0],
        )
        sessions_info = MultipleSessionsInfo(
            ReportsTwoNumbers(),
            table;
            observation_columns=:trial,
            action_columns=(:first, :second) => ((first, second) -> [first, second]),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        @test eltype(first(sessions_info.sessions).actions.report) == Vector{Float64}
        model = full_model(
            TuringBackend(), IndependentParameters((; mean=Normal(0, 1))), sessions_info
        )
        values = Dict(@varname(mean.sessions) => [0.2, -0.1])
        @test logjoint(model, values) ≈
            logpdf(Normal(0, 1), 0.2) +
              logpdf(Normal(0, 1), -0.1) +
              logpdf(MvNormal(fill(0.2, 2), IDENTITY_2), [0.1, 0.5]) +
              logpdf(MvNormal(fill(0.2, 2), IDENTITY_2), [0.3, 0.2]) +
              logpdf(MvNormal(fill(-0.1, 2), IDENTITY_2), [-0.2, 0.0])
    end

    @testset "pointwise_loglikelihoods with factorize = true gives the log-likelihood of each response and each confidence" begin
        model = full_model(
            TuringBackend(), IndependentParameters(RESPONSE_PRIORS), response_sessions()
        )
        chain = sample(Xoshiro(1), model, Prior(), 3; progress=false)
        pointwise = pointwise_loglikelihoods(model, chain; factorize=true)
        whole = pointwise_loglikelihoods(model, chain)
        for (session, id) in enumerate(unique(RESPONSES_TABLE.id)), draw in 1:3
            name = @varname(sessions[session].session_info.actions_to_condition_on)
            probability = chain[@varname(choice_probability.sessions)][draw, 1][session]
            rate = chain[@varname(rate.sessions)][draw, 1][session]
            rows = findall(==(id), RESPONSES_TABLE.id)
            each_value = pointwise[name][draw, 1]
            #A value for each row of the session, in the order of its timesteps
            @test each_value.response ≈ [
                logpdf(
                    ChoiceAndTime(probability, rate),
                    (; choice=RESPONSES_TABLE.choice[row], rt=RESPONSES_TABLE.rt[row]),
                ) for row in rows
            ]
            @test each_value.confidence ≈ [
                logpdf(Normal(probability, 0.1), RESPONSES_TABLE.confidence[row]) for
                row in rows
            ]
            #Together they are the log-likelihood of the session
            @test sum(each_value.response) + sum(each_value.confidence) ≈
                whole[name][draw, 1]
        end
    end

    @testset "PosteriorStats.jl's PSIS-LOO takes the log-likelihood of each response and each confidence" begin
        model = full_model(
            TuringBackend(), IndependentParameters(RESPONSE_PRIORS), response_sessions()
        )
        chain = sample(Xoshiro(1), model, NUTS(), 100; progress=false)
        pointwise = pointwise_loglikelihoods(model, chain; factorize=true)
        #An array of draws by chains by values, with the responses and then the confidences of each session in turn
        names = [
            @varname(sessions[session].session_info.actions_to_condition_on) for
            session in 1:3
        ]
        log_likelihood = stack(
            [getproperty(pointwise[name][draw, 1], output)[i] for draw in 1:100, _ in 1:1]
            for name in names for output in (:response, :confidence) for
            i in eachindex(getproperty(pointwise[name][1, 1], output))
        )
        @test size(log_likelihood) == (100, 1, 2 * length(RESPONSES_TABLE.id))
        #Together they are the log-likelihood of the sessions in each draw
        whole = pointwise_loglikelihoods(model, chain)
        @test vec(sum(log_likelihood; dims=3)) ≈
            [sum(whole[name][draw, 1] for name in names) for draw in 1:100]
        #An estimate of the expected log predictive density of each value, which is finite, and of all of them
        result = PosteriorStats.loo(log_likelihood)
        @test length(result.pointwise.elpd) == 2 * length(RESPONSES_TABLE.id)
        @test all(isfinite, result.pointwise.elpd) &&
            all(isfinite, result.pointwise.pareto_shape)
        @test isfinite(result.estimates.elpd)
        @test result.estimates.elpd ≈ sum(result.pointwise.elpd)
    end

    @testset "pointwise_loglikelihoods with factorize = true gives the log-likelihood of each vector of an output whose values are vectors" begin
        table = (;
            id=["S1", "S1", "S2"],
            trial=[1, 2, 1],
            first=[0.1, 0.3, -0.2],
            second=[0.5, 0.2, 0.0],
        )
        sessions_info = MultipleSessionsInfo(
            ReportsTwoNumbers(),
            table;
            observation_columns=:trial,
            action_columns=(:first, :second) => ((first, second) -> [first, second]),
            session_columns=:id,
            who_starts=EnvironmentStarts(),
        )
        model = full_model(
            TuringBackend(), IndependentParameters((; mean=Normal(0, 1))), sessions_info
        )
        chain = sample(Xoshiro(1), model, Prior(), 2; progress=false)
        pointwise = pointwise_loglikelihoods(model, chain; factorize=true)
        means = chain[@varname(mean.sessions)][1, 1]
        @test pointwise[@varname(sessions[1].session_info.actions_to_condition_on)][1, 1].report ≈
            [
            logpdf(MvNormal(fill(means[1], 2), IDENTITY_2), [0.1, 0.5]),
            logpdf(MvNormal(fill(means[1], 2), IDENTITY_2), [0.3, 0.2]),
        ]
        @test pointwise[@varname(sessions[2].session_info.actions_to_condition_on)][1, 1].report ≈
            [logpdf(MvNormal(fill(means[2], 2), IDENTITY_2), [-0.2, 0.0])]
    end

    @testset "The product of the outputs of a session scores each output, and draws a value for each" begin
        extension = Base.get_extension(CognitiveModeling, :CognitiveModelingTuringExt)
        product = extension.ProductOfOutputs((;
            choice=product_distribution([Bernoulli(0.3), Bernoulli(0.6)]),
            response=extension.ProductOfValuesOfAnyType([ChoiceAndTime(0.6, 2.0)]),
        ))
        values = (; choice=[true, false], response=[(; choice=1, rt=0.4)])
        @test logpdf(product, values) ≈
            logpdf(Bernoulli(0.3), true) +
              logpdf(Bernoulli(0.6), false) +
              logpdf(ChoiceAndTime(0.6, 2.0), (; choice=1, rt=0.4))
        @test loglikelihood(product, values) == logpdf(product, values)
        draw = rand(Xoshiro(1), product)
        @test keys(draw) == (:choice, :response) &&
            length(draw.choice) == 2 &&
            only(draw.response) isa NamedTuple
    end
end
