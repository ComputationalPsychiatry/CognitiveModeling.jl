### REJECTING INVALID PARAMETERS ###

#Turing model which runs any Turing model inside a try block, and gives the parameters a log density of -Inf if a cognitive model throws an InvalidParametersError
Turing.@model function CognitiveModeling.turing_catch_invalid_parameters(model)
    returned_value = nothing
    try
        returned_value ~ to_submodel(model, false)
    catch error
        error isa InvalidParametersError || rethrow()
        Turing.@addlogprob! -Inf
    end
    return returned_value
end
