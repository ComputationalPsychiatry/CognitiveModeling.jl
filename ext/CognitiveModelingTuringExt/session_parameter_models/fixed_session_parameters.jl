### THE SAME GIVEN VALUES FOR EVERY SESSION ###

#The same given values for every session
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::FixedSessionParameters, sessions_info::MultipleSessionsInfo
)
    return map(value -> fill(value, n_sessions(sessions_info)), parameters_model.parameters)
end
