#A small example of fitting a cognitive model to several participants with Turing.jl, and plotting the results with Makie.
#It runs in the docs environment. Run it line by line, or as a whole from the root of the repository with
#   julia --project=docs docs/examples/fit_three_participants.jl

using CognitiveModeling
using Turing
using CairoMakie
import Turing.FlexiChains.Makie as FM


## The cognitive model ##

#A Rescorla-Wagner model which predicts the next outcome, with Gaussian noise on its prediction
struct PredictingRescorlaWagner <: AbstractCognitiveModel end

#The expected value starts at a parameter, which is 0 unless it is given
function CognitiveModeling.initialize_state(model::PredictingRescorlaWagner, parameters)
    return (; expected_value = parameters.initial_value)
end

function CognitiveModeling.apply_model(model::PredictingRescorlaWagner, input::ModelInput)

    ## Unpack input ##
    #The outcome observed on the previous trial, which is BeforeFirstTimestep() on the first
    r = input.observation.outcome
    #Expected value from the previous state
    V₋₁ = input.previous_state.expected_value
    #Learning rate and noise parameters
    α = input.parameters.learning_rate
    σ = input.parameters.noise

    ## Update state ##
    #Learn from the prediction error, when there is an outcome to learn from
    Vₜ = r isa AbstractNoValue ? V₋₁ : V₋₁ + α * (r - V₋₁)

    ## Predict ##
    #The prediction of the next outcome is the expected value, with Gaussian noise
    return ModelOutput((; prediction = Normal(Vₜ, σ)), (; expected_value = Vₜ))
end

function CognitiveModeling.attribute_names(model::PredictingRescorlaWagner)
    return ModelAttributes(;
        parameters = (:learning_rate, :noise, :initial_value),
        states = :expected_value,
        observations = :outcome,
        actions = :prediction,
    )
end

CognitiveModeling.default_parameters(model::PredictingRescorlaWagner) = (; initial_value = 0.0)


## The data ##

#Three participants with five trials each. On each trial, the participant predicts the outcome,
#and then sees it. The data was simulated with learning rates of 0.2, 0.5 and 0.8, and a noise of 0.1.
#A DataFrame works the same way after Tables.columntable(dataframe)
data = (;
    participant = ["P1", "P1", "P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2", "P3", "P3", "P3", "P3", "P3"],
    prediction = [-0.07, 0.34, 0.43, 0.36, 0.58, -0.16, 0.09, 0.42, 0.66, 0.23, -0.07, 0.76, 0.07, -0.02, 0.86],
    outcome = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
)

#One session for each participant. The participant acts before seeing the outcome, so the agent starts each trial
sessions_info = MultipleSessionsInfo(
    PredictingRescorlaWagner(),
    data;
    observation_columns = :outcome,
    action_columns = :prediction,
    session_columns = :participant,
    who_starts = AgentStarts(),
)


## The Turing model ##

#Each participant has their own parameters, all from the same priors. The initial value isn't fitted, so it takes its default.
#The parameters are named after the population model's names, so the learning rate of the third participant is learning_rate.session[3]
population_model = IndependentSessions((; learning_rate = Beta(2, 2), noise = LogNormal(-2, 1)))

model = turing_full_model(population_model, sessions_info)


## Fitting ##

chain = sample(model, NUTS(), 1000)


## The results ##

#The model was fitted without saving the expected values, which is fastest. session_values_chain reruns it on every sample
#with sessions that save them, and gives a chain with the parameters and the expected values of each participant,
#named after them, as P1.learning_rate and P1.expected_value
session_values = session_values_chain(model, chain; save_history = SaveHistory(; states = :expected_value))

#The participants, as the first part of those names
participants = Symbol.(sessions_info.session_ids)

#The fitted parameters of each participant, as averages over the samples
for participant in participants
    learning_rate = mean(session_values[@varname($participant.learning_rate)])
    noise = mean(session_values[@varname($participant.noise)])
    println(participant, ": learning rate ", round(learning_rate; digits = 2), ", noise ", round(noise; digits = 2))
end


## Plots ##

#Trace and density of every parameter, to check the sampling
Makie.plot(chain)

#The learning rates of the participants side by side, with their medians and intervals
FM.forestplot(chain, @varname(learning_rate.session))

#The same as densities
FM.ridgeline(chain, @varname(learning_rate.session))

#The learning rates again, named after the participants
FM.forestplot(session_values, [@varname($participant.learning_rate) for participant in participants])

#The expected values of the first participant over the trials. The first is from before the first trial
FM.forestplot(session_values, @varname(P1.expected_value))

#The expected value of each participant over the trials, with the median and a 90 % interval over the samples, against their predictions and the outcomes
figure = Figure(; size = (900, 330))
for (i, participant) in enumerate(participants)
    axis = Axis(figure[1, i]; title = string(participant), xlabel = "Trial", ylabel = i == 1 ? "Expected value" : "")

    #The expected values with a row for each sample and a column for each trial, starting with trial 0, before the first
    expected_values = stack(vec(session_values[@varname($participant.expected_value)]); dims = 1)
    timesteps = 0:5
    band!(axis, timesteps, [quantile(column, 0.05) for column in eachcol(expected_values)], [quantile(column, 0.95) for column in eachcol(expected_values)]; color = (:steelblue, 0.3))
    lines!(axis, timesteps, [median(column) for column in eachcol(expected_values)]; color = :steelblue, label = "Expected value")

    #The data of the participant
    session = sessions_info.sessions[i]
    scatter!(axis, 1:5, session.actions.prediction; color = :black, label = "Prediction")
    scatter!(axis, 1:5, session.observations.outcome; color = :orange, marker = :diamond, label = "Outcome")
end
Legend(figure[2, :], content(figure[1, 1]); orientation = :horizontal, framevisible = false)
figure
