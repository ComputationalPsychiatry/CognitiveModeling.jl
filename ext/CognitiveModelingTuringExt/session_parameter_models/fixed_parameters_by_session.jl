### GIVEN VALUES FOR EACH SESSION ###

#Given values for each session, found by the session's symbol name or identifier, unless they were looked up when the full model was prepared
Turing.@model function CognitiveModeling.turing_session_parameters_model(
    parameters_model::FixedParametersBySession, sessions_info::MultipleSessionsInfo
)
    return collect_parameters_in_session_order(
        parameters_model.parameters_by_session, sessions_info.session_ids
    )
end
