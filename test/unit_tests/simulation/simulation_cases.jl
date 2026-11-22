#What holds for every simulation case that the suites share, with the distributions written by hand, which sample with a
#seeded random number generator, and with their means, which sample nothing: it runs, a second run gives the same, and
#the vectors that one call of simulate_session saves have the same length. With the distributions written by hand, the
#type of what it returns is inferred where the case says it can be. With the means, the choices and the rewards are
#numbers instead of Bools, so the states of some models that start with a Bool change type, and their types aren't
#checked

#What a case returned without the distributions it saved, whose values a second run compares, since the distributions
#written by hand hold vectors, which isequal compares by identity
const DISTRIBUTION_NAMES = (
    :action_distributions, :observation_distributions, :output_distributions
)
function without_distributions(values::NamedTuple{names}) where {names}
    kept = filter(name -> !(name in DISTRIBUTION_NAMES), names)
    return NamedTuple{kept}(map(name -> without_distributions(values[name]), kept))
end
function without_distributions(values::Union{Tuple,AbstractVector})
    return map(without_distributions, values)
end
function without_distributions(trajectories::SessionTrajectories)
    return without_distributions(NamedTuple(trajectories))
end
without_distributions(value) = value

#The lengths of the vectors that a call of simulate_session saved, of both models, leaving out the empty ones, which a
#model that saves nothing of a kind returns
function saved_lengths(trajectories)
    lengths = Int[]
    for name in (:states, :observations, :actions, :action_distributions)
        cognitive_values = getproperty(trajectories, name)
        isempty(cognitive_values) || push!(lengths, length(cognitive_values))
    end
    for name in (:states, :observations, :actions, :observation_distributions)
        environment_values = trajectories.environment[name]
        isempty(environment_values) || push!(lengths, length(environment_values))
    end
    return lengths
end

@testset "Every simulation case, $(nameof(typeof(distributions)))" for distributions in (
    ExampleModels.ByHand(), Means()
)
    @testset "$(case.name)" for case in simulation_cases(distributions)
        result = case.run(case.x)
        @test isequal(
            without_distributions(result), without_distributions(case.run(case.x))
        )
        if result isa SessionTrajectories
            @test allequal(saved_lengths(result))
        end
        if case.inferred && distributions isa ExampleModels.ByHand
            @inferred case.run(case.x)
        end
    end
end
