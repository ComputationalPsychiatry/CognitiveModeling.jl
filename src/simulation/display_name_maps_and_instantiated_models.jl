## Displaying name maps and instantiated models ##

#Shows the map as it is written
function Base.show(io::IO, ::NameMap{Tgiven,Treceived}) where {Tgiven,Treceived}
    print(io, "NameMap(")
    join(
        io,
        (
            given === received ? repr(given) : string(repr(given), " => ", repr(received))
            for (given, received) in zip(Tgiven, Treceived)
        ),
        ", ",
    )
    print(io, ")")
    return nothing
end

#Shows the kind and the type of the model in the REPL and notebooks, how many timesteps it has simulated, its parameters and its state, and whether its stopping criterion is met
function Base.show(io::IO, ::MIME"text/plain", instantiated_model::InstantiatedModel)
    model = instantiated_model.model

    #Print the kind and the type of the model, and how many timesteps it has simulated
    print(io, "-- Instantiated ", model_kind(model), " --\nType: ", nameof(typeof(model)))
    print(io, "\nTimesteps simulated: ", instantiated_model.n_timesteps)

    #Print the values of the parameters and of the state
    print_current_values(io, "Parameters", instantiated_model.parameters, ".parameters")
    print_current_values(io, "State", instantiated_model.state, ".state")

    #Say that the model's stopping criterion is met, if it is
    if stopping_criterion(model, instantiated_model.state, instantiated_model.parameters)
        print(io, "\nStopping criterion met: reset! to simulate it again")
    end

    return nothing
end

#Shows the type and the kind of the model on one line, as inside a vector, and how many timesteps it has simulated
function Base.show(io::IO, instantiated_model::InstantiatedModel)
    model = instantiated_model.model
    n_timesteps = instantiated_model.n_timesteps
    print(
        io,
        nameof(typeof(model)),
        " (instantiated ",
        model_kind(model),
        ", ",
        n_timesteps,
        n_timesteps == 1 ? " timestep simulated)" : " timesteps simulated)",
    )
    return nothing
end
