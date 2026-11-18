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
function CognitiveModeling.run_timestep(
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
    return (;
        choice_probability=ParameterDomain(; lower=0, upper=1),
        rate=ParameterDomain(; lower=0),
    )
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
function CognitiveModeling.run_timestep(model::ReportsTwoNumbers, context::ModelContext)
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
            @varname(choice_probability.session) => PROBABILITIES,
            @varname(rate.session) => RATES,
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
            @varname(choice_probability.session) => PROBABILITIES,
            @varname(rate.session) => RATES,
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
            draw -> all(0 .< draw .< 1), vec(chain[@varname(choice_probability.session)])
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
        values = Dict(@varname(mean.session) => [0.2, -0.1])
        @test logjoint(model, values) ≈
            logpdf(Normal(0, 1), 0.2) +
              logpdf(Normal(0, 1), -0.1) +
              logpdf(MvNormal(fill(0.2, 2), IDENTITY_2), [0.1, 0.5]) +
              logpdf(MvNormal(fill(0.2, 2), IDENTITY_2), [0.3, 0.2]) +
              logpdf(MvNormal(fill(-0.1, 2), IDENTITY_2), [-0.2, 0.0])
    end
end
