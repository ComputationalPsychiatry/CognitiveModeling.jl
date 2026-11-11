### NO PARAMETERS ###

#No parameters, so that each takes the value the model gives it by default
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    ::NoParameters, sessions_info::MultipleSessionsInfo
)
    return (;)
end
