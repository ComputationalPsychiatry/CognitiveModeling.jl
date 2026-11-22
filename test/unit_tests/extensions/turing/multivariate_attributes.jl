## Models for the tests ##
#A cognitive model which learns the expected payoff of each option, a vector, with a learning rate for each option, a vector, from the payoffs of every option, a vector, beside the number of the trial, a number. It rates each option, a vector from a normal distribution over vectors, chooses an option with a probability in proportion to its expected payoff, a number, and counts the timesteps it learned on, a number
struct RatesAndChoosesOptions <: AbstractCognitiveModel end
function CognitiveModeling.initialize_state(model::RatesAndChoosesOptions, parameters)
    return (; expected_payoffs=0.5 .* one.(parameters.learning_rates), n_updates=0)
end
function CognitiveModeling.timestep_model(
    model::RatesAndChoosesOptions, context::ModelContext
)
    (; expected_payoffs, n_updates) = context.previous_state
    payoffs = context.input.payoffs
    if !(payoffs isa AbstractNoValue)
        expected_payoffs =
            expected_payoffs .+
            context.parameters.learning_rates .* (payoffs .- expected_payoffs)
        n_updates += 1
    end
    return ModelOutput(
        (;
            ratings=MvNormal(
                expected_payoffs, context.parameters.rating_noise^2 .* IDENTITY_3
            ),
            choice=Categorical(expected_payoffs ./ sum(expected_payoffs)),
        ),
        (; expected_payoffs, n_updates),
    )
end
function CognitiveModeling.attribute_names(model::RatesAndChoosesOptions)
    return ModelAttributes(;
        parameters=(:learning_rates, :rating_noise),
        states=(:expected_payoffs, :n_updates),
        inputs=(:payoffs, :trial),
        outputs=(:ratings, :choice),
    )
end
function CognitiveModeling.parameter_domains(model::RatesAndChoosesOptions)
    return (;
        learning_rates=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        rating_noise=Domain(; lower=0),
    )
end
const IDENTITY_3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

#The payoffs of three options, the ratings of them and the choices in three sessions of different lengths, with the payoffs and the ratings in a column for each option
const OPTIONS_TABLE = (;
    id=["S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", "S3"],
    trial=[1, 2, 1, 2, 3, 1, 2, 3, 4],
    payoff_1=[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    payoff_2=[0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0],
    payoff_3=[3.0, 3.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 3.0],
    rating_1=[0.7, 0.6, 0.4, 0.8, 0.9, 0.3, 0.2, 0.7, 0.5],
    rating_2=[0.2, 1.1, 1.4, 0.9, 1.5, 1.6, 0.8, 0.6, 1.3],
    rating_3=[2.4, 2.8, 0.3, 1.9, 0.9, 2.5, 2.9, 1.2, 2.2],
    choice=[3, 3, 2, 1, 2, 2, 3, 3, 3],
)
const OPTIONS_PRIORS = (;
    learning_rates=filldist(Beta(2, 2), 3), rating_noise=LogNormal(-1, 0.5)
)

#Builds the vector of an observation or an action from a column for each option
vector_of_three(first, second, third) = [first, second, third]

#The sessions of a table of options, with the payoffs and the ratings each built from a column for each option
function options_sessions(table=OPTIONS_TABLE; keywords...)
    return MultipleSessionsInfo(
        RatesAndChoosesOptions(),
        table;
        observation_columns=(
            (:payoff_1, :payoff_2, :payoff_3) => vector_of_three => :payoffs, :trial
        ),
        action_columns=(
            (:rating_1, :rating_2, :rating_3) => vector_of_three => :ratings, :choice
        ),
        session_columns=:id,
        who_starts=EnvironmentStarts(),
        keywords...,
    )
end

#The log likelihood of the rows of OPTIONS_TABLE by hand, with the learning rates of each session in a column of a matrix. The ratings of the rows in ratings_left_out don't count, and a rating given in imputed_ratings counts for its row
function options_by_hand(
    learning_rates, rating_noises; ratings_left_out=(), imputed_ratings=Dict()
)
    log_likelihood = 0.0
    for (session, id) in enumerate(unique(OPTIONS_TABLE.id))
        expected_payoffs = fill(0.5, 3)
        for row in findall(==(id), OPTIONS_TABLE.id)
            payoffs = [
                OPTIONS_TABLE.payoff_1[row],
                OPTIONS_TABLE.payoff_2[row],
                OPTIONS_TABLE.payoff_3[row],
            ]
            expected_payoffs =
                expected_payoffs .+
                learning_rates[:, session] .* (payoffs .- expected_payoffs)
            ratings = get(imputed_ratings, row) do
                return [
                    OPTIONS_TABLE.rating_1[row],
                    OPTIONS_TABLE.rating_2[row],
                    OPTIONS_TABLE.rating_3[row],
                ]
            end
            if !(row in ratings_left_out)
                log_likelihood += logpdf(
                    MvNormal(expected_payoffs, rating_noises[session]^2 .* IDENTITY_3),
                    ratings,
                )
            end
            log_likelihood += log(
                expected_payoffs[OPTIONS_TABLE.choice[row]] / sum(expected_payoffs)
            )
        end
    end
    return log_likelihood
end

#The prior of the learning rates of each session, a column of the matrix, and of their rating noises
function options_prior(learning_rates, rating_noises)
    return sum(
        logpdf(OPTIONS_PRIORS.learning_rates, learning_rates[:, session]) for
        session in axes(learning_rates, 2)
    ) + sum(noise -> logpdf(OPTIONS_PRIORS.rating_noise, noise), rating_noises)
end

const OPTIONS_LEARNING_RATES = [0.2 0.5 0.8; 0.6 0.3 0.4; 0.9 0.7 0.1]
const OPTIONS_RATING_NOISES = [0.3, 0.5, 0.4]
const OPTIONS_VALUES = Dict(
    @varname(learning_rates.sessions) => OPTIONS_LEARNING_RATES,
    @varname(rating_noise.sessions) => OPTIONS_RATING_NOISES,
)

@testset "Observations, actions, states and parameters that are vectors, beside ones that are numbers" begin
    @testset "The log density is that by hand, with a vector of learning rates for each session, as the column of a matrix" begin
        sessions_info = options_sessions()
        #Each session's observations and actions are vectors beside numbers
        @test sessions_info.sessions[1].observations ==
            (; payoffs=[[1.0, 0.0, 3.0], [0.0, 2.0, 3.0]], trial=[1, 2])
        @test sessions_info.sessions[1].actions ==
            (; ratings=[[0.7, 0.2, 2.4], [0.6, 1.1, 2.8]], choice=[3, 3])
        model = full_model(
            TuringBackend(), IndependentParameters(OPTIONS_PRIORS), sessions_info
        )
        sampled = rand(Xoshiro(1), model)
        @test size(sampled[@varname(learning_rates.sessions)]) == (3, 3)
        @test size(sampled[@varname(rating_noise.sessions)]) == (3,)
        @test logjoint(model, OPTIONS_VALUES) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES)
        #Each session has its own vector, beside its number
        parameters = returned(model, OPTIONS_VALUES).cognitive_parameters
        @test parameters.learning_rates ==
            [OPTIONS_LEARNING_RATES[:, session] for session in 1:3]
        @test parameters.rating_noise == OPTIONS_RATING_NOISES
    end

    @testset "Observations and actions read from a column of vectors give the same as those built from a column for each option" begin
        rows = eachindex(OPTIONS_TABLE.id)
        table_of_vectors = (;
            id=OPTIONS_TABLE.id,
            trial=OPTIONS_TABLE.trial,
            payoffs=[
                [
                    OPTIONS_TABLE.payoff_1[row],
                    OPTIONS_TABLE.payoff_2[row],
                    OPTIONS_TABLE.payoff_3[row],
                ] for row in rows
            ],
            ratings=[
                [
                    OPTIONS_TABLE.rating_1[row],
                    OPTIONS_TABLE.rating_2[row],
                    OPTIONS_TABLE.rating_3[row],
                ] for row in rows
            ],
            choice=OPTIONS_TABLE.choice,
        )
        model = full_model(
            TuringBackend(),
            IndependentParameters(OPTIONS_PRIORS),
            MultipleSessionsInfo(
                RatesAndChoosesOptions(),
                table_of_vectors;
                observation_columns=(:payoffs, :trial),
                action_columns=(:ratings, :choice),
                session_columns=:id,
                who_starts=EnvironmentStarts(),
            ),
        )
        @test logjoint(model, OPTIONS_VALUES) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES)
    end

    @testset "A missing action that is a vector is imputed as a vector, or left out while the rest of its row counts" begin
        #The second rating of the fourth row, the second timestep of S2, is missing, which makes its whole vector missing
        table = (;
            OPTIONS_TABLE..., rating_2=Union{Missing,Float64}[OPTIONS_TABLE.rating_2...]
        )
        table.rating_2[4] = missing
        imputed = full_model(
            TuringBackend(), IndependentParameters(OPTIONS_PRIORS), options_sessions(table)
        )
        imputed_name = @varname(sessions[2].timesteps[2].ratings.imputed)
        @test Set(keys(rand(Xoshiro(1), imputed))) ==
            Set([keys(OPTIONS_VALUES)..., imputed_name])
        @test size(rand(Xoshiro(1), imputed)[imputed_name]) == (3,)
        values_with_imputed = merge(OPTIONS_VALUES, Dict(imputed_name => [0.1, 0.2, 0.3]))
        @test logjoint(imputed, values_with_imputed) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(
            OPTIONS_LEARNING_RATES,
            OPTIONS_RATING_NOISES;
            imputed_ratings=Dict(4 => [0.1, 0.2, 0.3]),
        )
        #Left out, the choice of the row still counts
        left_out = full_model(
            TuringBackend(),
            IndependentParameters(OPTIONS_PRIORS),
            options_sessions(table; impute_missing_actions=false),
        )
        @test logjoint(left_out, OPTIONS_VALUES) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(
            OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES; ratings_left_out=(4,)
        )
    end

    @testset "A vector with a missing element, in a column of vectors, is missing as a whole when a function among the missing values says so: imputed as a vector, or left out" begin
        #Every vector of ratings can hold a missing value by its type, and the second rating of the fourth row is missing
        ratings = Vector{Union{Missing,Float64}}[
            [
                OPTIONS_TABLE.rating_1[row],
                OPTIONS_TABLE.rating_2[row],
                OPTIONS_TABLE.rating_3[row],
            ] for row in eachindex(OPTIONS_TABLE.id)
        ]
        ratings[4][2] = missing
        table = (; OPTIONS_TABLE..., ratings)
        has_missing_element(value) = value isa AbstractVector && any(ismissing, value)
        function sessions_with_a_column_of_ratings(; keywords...)
            return MultipleSessionsInfo(
                RatesAndChoosesOptions(),
                table;
                observation_columns=(
                    (:payoff_1, :payoff_2, :payoff_3) => vector_of_three => :payoffs, :trial
                ),
                action_columns=(:ratings, :choice),
                session_columns=:id,
                who_starts=EnvironmentStarts(),
                missing_action_values=(missing, has_missing_element),
                keywords...,
            )
        end
        #The vectors of the other rows hold only their numbers
        sessions_info = sessions_with_a_column_of_ratings()
        @test sessions_info.sessions[2].actions.ratings ==
            [[0.4, 1.4, 0.3], ImputedMissingValue(), [0.9, 1.5, 0.9]]
        @test eltype(sessions_info.sessions[1].actions.ratings) ==
            Union{ImputedMissingValue,Vector{Float64}}
        imputed = full_model(
            TuringBackend(), IndependentParameters(OPTIONS_PRIORS), sessions_info
        )
        imputed_name = @varname(sessions[2].timesteps[2].ratings.imputed)
        values_with_imputed = merge(OPTIONS_VALUES, Dict(imputed_name => [0.1, 0.2, 0.3]))
        @test logjoint(imputed, values_with_imputed) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(
            OPTIONS_LEARNING_RATES,
            OPTIONS_RATING_NOISES;
            imputed_ratings=Dict(4 => [0.1, 0.2, 0.3]),
        )
        left_out = full_model(
            TuringBackend(),
            IndependentParameters(OPTIONS_PRIORS),
            sessions_with_a_column_of_ratings(; impute_missing_actions=false),
        )
        @test logjoint(left_out, OPTIONS_VALUES) ≈
            options_prior(OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES) +
              options_by_hand(
            OPTIONS_LEARNING_RATES, OPTIONS_RATING_NOISES; ratings_left_out=(4,)
        )
    end

    @testset "A fit with NUTS gives the trajectories of the vector states and actions of each session, which follow from each draw" begin
        model = full_model(
            TuringBackend(), IndependentParameters(OPTIONS_PRIORS), options_sessions()
        )
        chain = sample(Xoshiro(1), model, NUTS(), 30; progress=false)
        trajectories = get_session_trajectories(
            model,
            chain;
            cognitive_save_history=SaveHistory(;
                outputs=(:ratings, :choice), states=:expected_payoffs
            ),
        )
        @test Turing.FlexiChains.get_name.(keys(trajectories)) == [
            @varname(S1.ratings),
            @varname(S1.choice),
            @varname(S1.expected_payoffs),
            @varname(S2.ratings),
            @varname(S2.choice),
            @varname(S2.expected_payoffs),
            @varname(S3.ratings),
            @varname(S3.choice),
            @varname(S3.expected_payoffs),
        ]
        session_parameters = get_session_parameters(model, chain)
        for draw in (1, 30)
            #The expected payoffs of S2 follow from the payoffs of its three rows and the learning rates of the draw, from timestep 0 on
            learning_rates = chain[@varname(learning_rates.sessions)][draw][:, 2]
            @test session_parameters[@varname(S2.learning_rates)][draw] == learning_rates
            expected_payoffs = trajectories[@varname(S2.expected_payoffs)][draw]
            @test expected_payoffs[1] == [0.5, 0.5, 0.5]
            by_hand = [0.5, 0.5, 0.5]
            for (timestep, row) in enumerate(3:5)
                payoffs = [
                    OPTIONS_TABLE.payoff_1[row],
                    OPTIONS_TABLE.payoff_2[row],
                    OPTIONS_TABLE.payoff_3[row],
                ]
                by_hand = by_hand .+ learning_rates .* (payoffs .- by_hand)
                @test expected_payoffs[timestep + 1] ≈ by_hand
            end
            #The ratings are those in the data, a vector on each timestep
            @test trajectories[@varname(S2.ratings)][draw][2:end] ==
                [[0.4, 1.4, 0.3], [0.8, 0.9, 1.9], [0.9, 1.5, 0.9]]
        end
    end
end
