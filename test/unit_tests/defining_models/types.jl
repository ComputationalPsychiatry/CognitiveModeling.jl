using Random: Xoshiro, default_rng

@testset "Types" begin
    observation = (; reward=1.0)
    state = (; expected_reward=0.5)
    parameters = (; learning_rate=0.1)
    previous_action = (; choice=NoValue())

    @testset "ModelContext sets the timestep size to 1 if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).timestep_size ===
            1
        @test ModelContext(observation, state, previous_action, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "ModelContext for an environment model has the action as its input and the observation as its previous output" begin
        environment_context = ModelContext(previous_action, state, observation, parameters)
        @test environment_context.input === previous_action
        @test environment_context.previous_output === observation
        @test environment_context.timestep_size === 1
        @test ModelContext(previous_action, state, observation, parameters, 0.5).timestep_size ===
            0.5
    end

    @testset "ModelContext uses the default generator if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).rng ===
            default_rng()
        @test ModelContext(previous_action, state, observation, parameters).rng ===
            default_rng()
        rng = Xoshiro(1)
        @test ModelContext(observation, state, previous_action, parameters, 1, rng).rng ===
            rng
        @test ModelContext(previous_action, state, observation, parameters, 1, rng).rng ===
            rng
    end

    @testset "ModelContext says who starts, and unknown if it isn't specified" begin
        @test ModelContext(observation, state, previous_action, parameters).who_starts ===
            UnknownWhoStarts()
        @test ModelContext(
            observation, state, previous_action, parameters, 1, default_rng(), AgentStarts()
        ).who_starts === AgentStarts()
        @test ModelContext(
            observation,
            state,
            previous_action,
            parameters,
            1,
            default_rng(),
            EnvironmentStarts(),
        ).who_starts === EnvironmentStarts()
        #It is in the type, so a model can choose by it at compile time
        @test typeof(
            ModelContext(
                observation,
                state,
                previous_action,
                parameters,
                1,
                default_rng(),
                AgentStarts(),
            ),
        ).parameters[end] === AgentStarts
        @test isbitstype(UnknownWhoStarts)
        @test ModelContext(previous_action, state, observation, parameters).who_starts ===
            UnknownWhoStarts()
        @test ModelContext(
            previous_action,
            state,
            observation,
            parameters,
            1,
            default_rng(),
            EnvironmentStarts(),
        ).who_starts === EnvironmentStarts()
    end

    @testset "ModelContext takes its arguments by name too, with the same defaults" begin
        rng = Xoshiro(1)
        @test ModelContext(;
            input=observation,
            previous_state=state,
            previous_output=previous_action,
            parameters,
            timestep_size=0.5,
            rng,
            who_starts=AgentStarts(),
        ) === ModelContext(
            observation, state, previous_action, parameters, 0.5, rng, AgentStarts()
        )
        by_name = ModelContext(;
            input=observation,
            previous_state=state,
            previous_output=previous_action,
            parameters,
        )
        @test by_name.timestep_size === 1
        @test by_name.rng === default_rng()
        @test by_name.who_starts === UnknownWhoStarts()
        @test_throws UndefKeywordError ModelContext(;
            input=observation, previous_state=state, parameters
        )
    end

    @testset "ModelOutput only takes named output distributions" begin
        @test ModelOutput((; choice=0.7)).output_distribution === (; choice=0.7)
        @test ModelOutput((;)).output_distribution === (;)
        @test ModelOutput((; choice=NoValue())).output_distribution === (; choice=NoValue())
        @test_throws MethodError ModelOutput(NoValue())
        @test_throws MethodError ModelOutput(0.7)
        @test_throws MethodError ModelOutput((0.7, 0.5))
    end

    @testset "ModelOutput sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput((; choice=0.7)).new_state === (;)
        @test ModelOutput((; choice=0.7), state).new_state === state
    end

    @testset "ModelOutput only takes a named state, also for a single object" begin
        @test ModelOutput((; choice=0.7), (; values=[0.5, 0.5])).new_state.values ==
            [0.5, 0.5]
        @test_throws MethodError ModelOutput((; choice=0.7), 0.5)
        @test_throws MethodError ModelOutput((; choice=0.7), [0.5, 0.5])
        @test_throws MethodError ModelOutput((; choice=0.7), nothing)
    end

    @testset "ModelOutput says the parameters are valid by default, and takes whether they are" begin
        @test ModelOutput((; choice=0.7)).parameters_valid === ValidByDefault()
        @test ModelOutput((; choice=0.7), state).parameters_valid === ValidByDefault()
        @test ModelOutput((; choice=0.7), state; parameters_valid=false).parameters_valid ===
            false
        #The flag's type is part of the type, so that the default is known when the code is compiled
        @test typeof(ModelOutput((; choice=0.7))).parameters[3] === ValidByDefault
        @test typeof(ModelOutput((; choice=0.7); parameters_valid=true)).parameters[3] ===
            Bool
        #The flags of two models combine, with the default as the neutral one
        @test CognitiveModeling.combine_parameters_valid(
            ValidByDefault(), ValidByDefault()
        ) === ValidByDefault()
        @test CognitiveModeling.combine_parameters_valid(ValidByDefault(), false) === false
        @test CognitiveModeling.combine_parameters_valid(true, ValidByDefault()) === true
        @test CognitiveModeling.combine_parameters_valid(true, false) === false
        @test CognitiveModeling.parameters_are_valid(ValidByDefault()) === true
        @test CognitiveModeling.parameters_are_valid(false) === false
    end

    @testset "ModelOutput for an environment model only takes named observation distributions" begin
        @test ModelOutput((; reward=Deterministic(1.0))).output_distribution ===
            (; reward=Deterministic(1.0))
        @test ModelOutput((; reward=NoValue())).output_distribution === (; reward=NoValue())
        @test_throws MethodError ModelOutput(NoValue())
        @test_throws MethodError ModelOutput(Deterministic(1.0))
    end

    @testset "ModelOutput for an environment model sets the new state to an empty NamedTuple if it isn't specified" begin
        @test ModelOutput(observation).new_state === (;)
        @test ModelOutput(observation, state).new_state === state
    end

    @testset "The constructors are type stable" begin
        @test @inferred(ModelContext(observation, state, previous_action, parameters)) isa
            ModelContext
        @test @inferred(
            ModelContext(observation, state, previous_action, parameters, 0.5)
        ) isa ModelContext
        @test @inferred(ModelContext(previous_action, state, observation, parameters)) isa
            ModelContext
        @test @inferred(
            ModelContext(previous_action, state, observation, parameters, 0.5)
        ) isa ModelContext
        @test @inferred(
            ModelContext(;
                input=observation,
                previous_state=state,
                previous_output=previous_action,
                parameters,
                timestep_size=0.5,
            )
        ) isa ModelContext
        @test @inferred(ModelOutput((; choice=0.7))) isa ModelOutput
        @test @inferred(ModelOutput((; choice=0.7), state)) isa ModelOutput
        @test @inferred(ModelOutput((;))) isa ModelOutput
        @test @inferred(ModelOutput(observation)) isa ModelOutput
        @test @inferred(ModelOutput(observation, state)) isa ModelOutput
    end

    @testset "The types hold their values directly, so they don't allocate" begin
        @test isbitstype(
            typeof(ModelContext(observation, state, previous_action, parameters))
        )
        @test isbitstype(typeof(ModelOutput((; choice=0.7))))
        @test isbitstype(typeof(ModelOutput((; reward=Deterministic(1.0)))))
    end

    @testset "Sampling from a Deterministic gives its value" begin
        rng = Xoshiro(1)
        @test rand(rng, Deterministic(1.0)) === 1.0
        @test rand(rng, Deterministic(:left)) === :left
        @test @inferred(rand(rng, Deterministic(2))) === 2
        #The generator isn't used, and the default one works too
        @test rng == Xoshiro(1)
        @test rand(Deterministic(true)) === true
        #The value is kept as it is, without a copy
        values = [0.2, 0.7]
        @test Deterministic(values).value === values
        @test rand(rng, Deterministic(values)) === values
    end

    @testset "Cognitive models and environment models share a supertype" begin
        @test AbstractCognitiveModel <: CognitiveModeling.AbstractTimestepModel
        @test AbstractEnvironmentModel <: CognitiveModeling.AbstractTimestepModel
    end

    @testset "ModelAttributes keeps the names of each kind in a tuple" begin
        attributes = ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=(:expected_reward,),
            inputs=(:reward,),
            outputs=(:choice,),
        )
        @test attributes.parameters === (:learning_rate, :noise)
        @test attributes.states === (:expected_reward,)
        @test attributes.inputs === (:reward,)
        @test attributes.outputs === (:choice,)
        @test propertynames(attributes) == (:parameters, :states, :inputs, :outputs)
        @test_throws FieldError attributes.names
    end

    @testset "ModelAttributes puts a single name or a vector of names in a tuple" begin
        attributes = ModelAttributes(;
            parameters=:learning_rate,
            states=[:expected_reward, :surprise],
            inputs=(),
            outputs=(),
        )
        @test attributes.parameters === (:learning_rate,)
        @test attributes.states === (:expected_reward, :surprise)
        @test attributes.inputs === ()
    end

    @testset "ModelAttributes needs all four kinds, and only takes names" begin
        @test_throws UndefKeywordError ModelAttributes(;
            parameters=(:learning_rate,), states=(), inputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters=("learning_rate",), states=(), inputs=(), outputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters=(:learning_rate, 1), states=(), inputs=(), outputs=()
        )
        @test_throws TypeError ModelAttributes(;
            parameters="learning_rate", states=(), inputs=(), outputs=()
        )
    end

    @testset "ModelAttributes only takes a name once, within a kind and across the kinds" begin
        #A name used for two kinds says which two
        @test_throws "The name choice is used for both a state and an output, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=:learning_rate, states=:choice, inputs=:reward, outputs=:choice
        )
        @test_throws "The name reward is used for both a parameter and an input, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=:reward, states=(), inputs=:reward, outputs=:choice
        )
        @test_throws "The name arm is used for both an input and an output, but a model can only use a name for one attribute." ModelAttributes(;
            parameters=(), states=(), inputs=:arm, outputs=:arm
        )
        #A name used twice within a kind shows the names of that kind
        @test_throws "The states are named (:a, :a), but a model can only use a name for one attribute." ModelAttributes(;
            parameters=(), states=(:a, :a), inputs=(), outputs=()
        )
        @test_throws "The parameters are named (:a, :b, :a), but a model can only use a name for one attribute." ModelAttributes(;
            parameters=[:a, :b, :a], states=(), inputs=(), outputs=()
        )
        @test_throws ArgumentError ModelAttributes(;
            parameters=(), states=(), inputs=(:reward, :reward), outputs=()
        )
        #Distinct names pass, however many kinds have names
        @test ModelAttributes(;
            parameters=(:learning_rate, :noise),
            states=:expected_reward,
            inputs=:reward,
            outputs=:choice,
        ) isa ModelAttributes
        @test ModelAttributes(; parameters=(), states=(), inputs=(), outputs=()) isa
            ModelAttributes
        @test CognitiveModeling.check_unique_attribute_names(
            ModelAttributes(; parameters=:a, states=:b, inputs=:c, outputs=:d)
        ) === nothing
    end

    @testset "ModelAttributes keeps the names in the type, so constructing it from literals is inferred" begin
        from_literals() = ModelAttributes(;
            parameters=(:learning_rate,),
            states=:expected_reward,
            inputs=(),
            outputs=(:choice,),
        )
        @test Base.return_types(from_literals, Tuple{})[1] ===
            ModelAttributes{(:learning_rate,),(:expected_reward,),(),(:choice,)}
        @test sizeof(from_literals()) == 0
    end

    @testset "The kinds of attributes are types of their own, with those that the data of a session has under a supertype" begin
        observable_kinds = (
            CognitiveModeling.ObservationKind(), CognitiveModeling.ActionKind()
        )
        other_kinds = (CognitiveModeling.ParameterKind(), CognitiveModeling.StateKind())
        @test all(
            kind -> kind isa CognitiveModeling.AbstractObservableAttributeKind,
            observable_kinds,
        )
        @test !any(
            kind -> kind isa CognitiveModeling.AbstractObservableAttributeKind, other_kinds
        )
        @test all(
            kind -> kind isa CognitiveModeling.AbstractAttributeKind,
            (observable_kinds..., other_kinds...),
        )
        @test CognitiveModeling.AbstractObservableAttributeKind <:
            CognitiveModeling.AbstractAttributeKind
        @test allunique((observable_kinds..., other_kinds...))
    end

    @testset "Domain takes every real number by default, and keeps its type and its bounds" begin
        domain = Domain()
        @test domain.type === Real
        @test domain.lower === -Inf
        @test domain.upper === Inf
        @test domain.possible_values === nothing
        @test domain.constraint === nothing
        @test propertynames(domain) ===
            (:type, :lower, :upper, :possible_values, :constraint)
        bounded = Domain(; lower=0, upper=1)
        @test (bounded.type, bounded.lower, bounded.upper) === (Real, 0, 1)
        @test Domain(; lower=0.5).lower === 0.5
        whole = Domain(; type=Integer, lower=1)
        @test (whole.type, whole.lower, whole.upper) === (Integer, 1, Inf)
        @test Domain(; type=Symbol).type === Symbol
        #The bounds can be equal, for a parameter with one value
        @test Domain(; lower=1, upper=1).upper === 1
    end

    @testset "Domain keeps the type of its values in its own type, and holds its bounds directly" begin
        @test Domain(; type=Integer, lower=1) isa Domain{Integer,Integer,Int,Float64}
        @test Domain(; type=AbstractVector{Real}, upper=1) isa
            Domain{AbstractVector{Real},Real,Float64,Int}
        @test Domain(; type=AbstractString) isa Domain{AbstractString}
        @test isbitstype(typeof(Domain(; lower=0, upper=1)))
        @test isbitstype(typeof(Domain(; type=Symbol)))
        from_literals() = Domain(; type=Integer, lower=1)
        @test @inferred(from_literals()) isa Domain
    end

    @testset "Domain only takes types and bounds that fit together" begin
        @test_throws ArgumentError(
            "A Domain was given type = Float64, but a type of floating-point numbers leaves out the others, and the numbers of automatic differentiation packages. Give Real, the default, for a parameter that is a real number.",
        ) Domain(; type=Float64)
        @test_throws ArgumentError Domain(; type=AbstractFloat, lower=0)
        @test_throws ArgumentError(
            "A Domain with type = Symbol was given the bounds lower = 0 and upper = Inf, but only a parameter that is a real number, or an array or a tuple of real numbers, has bounds.",
        ) Domain(; type=Symbol, lower=0)
        @test_throws ArgumentError Domain(; type=AbstractString, upper=3)
        @test_throws ArgumentError(
            "The lower bound of a Domain is at most its upper bound, but it was given lower = 1 and upper = 0.",
        ) Domain(; lower=1, upper=0)
        @test_throws ArgumentError Domain(; lower=NaN)
        @test_throws ArgumentError Domain(; upper=NaN)
        #A misspelled keyword, the old discrete flag and bounds without their names are errors
        @test_throws MethodError Domain(; lowest=0)
        @test_throws MethodError Domain(; lower=1, discrete=true)
        @test_throws MethodError Domain(0, 1)
        @test_throws TypeError Domain(; type=:Integer)
        @test_throws TypeError Domain(; lower="0")
    end

    @testset "Domain takes array types and tuple types, with the bounds for each element" begin
        vector = Domain(; type=AbstractVector{Real}, lower=0, upper=1)
        @test (vector.type, vector.lower, vector.upper) === (AbstractVector{Real}, 0, 1)
        @test Domain(; type=AbstractMatrix{Integer}, lower=1).type ===
            AbstractMatrix{Integer}
        #A type of the elements that is left open is an error, since it is made anew on every call
        @test_throws ArgumentError(
            "A Domain was given type = AbstractVector{<:Real}, which leaves the type of its elements open. Give the type of the elements, as in AbstractVector{Real}, which takes arrays with elements of every real number type, NTuple{3,Real}, or AbstractVector{Any} for elements of any type.",
        ) Domain(; type=AbstractVector{<:Real}, lower=0, upper=1)
        @test_throws ArgumentError Domain(; type=AbstractVector)
        @test_throws ArgumentError Domain(; type=Vector{<:Integer})
        @test_throws ArgumentError Domain(; type=NTuple{3})
        @test Domain(; type=AbstractVector{Any}).type === AbstractVector{Any}
        @test Domain(; type=NTuple{3,Real}, lower=0).type === NTuple{3,Real}
        @test Domain(; type=Tuple{Vararg{Real}}).type === Tuple{Vararg{Real}}
        @test Domain(; type=AbstractVector{Symbol}).type === AbstractVector{Symbol}
        #The type of the elements is read from the type, also when it is left open
        type_of_elements = CognitiveModeling.type_of_elements
        @test type_of_elements(Real) === Real
        @test type_of_elements(Symbol) === Symbol
        @test type_of_elements(AbstractVector{Real}) === Real
        @test type_of_elements(Vector{Integer}) === Integer
        @test type_of_elements(AbstractMatrix{Bool}) === Bool
        @test type_of_elements(AbstractArray{Real}) === Real
        @test type_of_elements(AbstractVector{Any}) === Any
        @test type_of_elements(NTuple{3,Real}) === Real
        @test type_of_elements(Tuple{Vararg{Integer}}) === Integer
        #Elements that are floating-point numbers leave out the others, as a single one does
        @test_throws ArgumentError(
            "A Domain was given type = Vector{Float64}, whose elements are floating-point numbers, but these leave out the others, and the numbers of automatic differentiation packages. Give elements of type Real, as in AbstractVector{Real} or NTuple{3,Real}.",
        ) Domain(; type=Vector{Float64})
        @test_throws ArgumentError Domain(; type=AbstractVector{<:AbstractFloat})
        @test_throws ArgumentError Domain(; type=NTuple{2,Float64})
        #Only elements that are real numbers have bounds
        @test_throws ArgumentError Domain(; type=AbstractVector{Symbol}, lower=0)
        @test_throws ArgumentError Domain(;
            type=AbstractVector{<:AbstractVector{<:Real}}, upper=1
        )
        @test_throws ArgumentError Domain(; type=AbstractVector, lower=0)
    end

    @testset "Domain takes possible values in a tuple or a range, and has their type unless it is given one" begin
        strategy = Domain(; possible_values=(:greedy, :random))
        @test (strategy.type, strategy.possible_values) === (Symbol, (:greedy, :random))
        @test (strategy.lower, strategy.upper) === (-Inf, Inf)
        #Numbers have the abstract type of their kind, so that any number type is taken
        @test Domain(; possible_values=(2, 4, 8)).type === Integer
        @test Domain(; possible_values=1:4).type === Integer
        @test Domain(; possible_values=0.0:0.25:1.0).type === Real
        @test Domain(; possible_values=(0.5, 1)).type === Real
        @test Domain(; possible_values=(true, false)).type === Bool
        @test Domain(; possible_values=("low", "high")).type === AbstractString
        @test Domain(; possible_values=(:low, "high")).type === Any
        #A type that is given is kept, and an array type or a tuple type has the possible values of each element
        @test Domain(; type=Real, possible_values=(1, 2)).type === Real
        levels = Domain(; type=AbstractVector{Integer}, possible_values=1:3)
        @test (levels.type, levels.possible_values) === (AbstractVector{Integer}, 1:3)
        @test Domain(; type=NTuple{2,Symbol}, possible_values=(:a, :b)).type ===
            NTuple{2,Symbol}
        #The possible values are kept in the type of the domain, so the check is compiled for them
        @test typeof(strategy) ===
            Domain{Symbol,Symbol,Float64,Float64,Tuple{Symbol,Symbol},Nothing}
    end

    @testset "Domain only takes possible values in a tuple or a range, without bounds, and of its type" begin
        @test_throws ArgumentError(
            "A Domain takes its possible values in a tuple, such as (:greedy, :random), or in a range, such as 1:4, but it was given possible_values = [1, 2].",
        ) Domain(; possible_values=[1, 2])
        @test_throws ArgumentError Domain(; possible_values=:greedy)
        @test_throws ArgumentError Domain(; possible_values=Set([1, 2]))
        @test_throws ArgumentError(
            "A Domain takes at least one possible value, but it was given possible_values = ().",
        ) Domain(; possible_values=())
        @test_throws ArgumentError Domain(; possible_values=1:0)
        @test_throws ArgumentError(
            "A Domain takes possible values or bounds, but it was given possible_values = (1, 2) and the bounds lower = 0 and upper = Inf.",
        ) Domain(; possible_values=(1, 2), lower=0)
        @test_throws ArgumentError Domain(; possible_values=1:3, upper=3)
        @test_throws ArgumentError(
            "A Domain with type = Symbol was given possible_values = (1, 2), but each of them must be a value of type Symbol.",
        ) Domain(; type=Symbol, possible_values=(1, 2))
        @test_throws ArgumentError Domain(; type=Integer, possible_values=(1, 2.5))
        @test_throws ArgumentError Domain(;
            type=AbstractVector{Integer}, possible_values=(:a,)
        )
        #A whole number in another number type is a possible value of a type of whole numbers
        @test Domain(; type=Integer, possible_values=(1.0, 2.0)).type === Integer
    end

    @testset "Domain keeps a constraint, which is any function of the whole value" begin
        domain = Domain(; type=AbstractVector{Real}, constraint=issorted)
        @test domain.constraint === issorted
        @test (domain.type, domain.lower, domain.upper) ===
            (AbstractVector{Real}, -Inf, Inf)
        #A constraint goes with bounds and with possible values alike
        @test Domain(; lower=0, constraint=isinteger).constraint === isinteger
        @test Domain(;
            type=NTuple{2,Integer}, possible_values=1:3, constraint=allunique
        ).constraint === allunique
    end

    @testset "Domain is shown as it is written, without the defaults" begin
        @test repr(Domain()) == "Domain()"
        @test repr(Domain(; lower=0, upper=1)) == "Domain(; lower = 0, upper = 1)"
        @test repr(Domain(; upper=0.5)) == "Domain(; upper = 0.5)"
        @test repr(Domain(; type=Integer, lower=1)) == "Domain(; type = Integer, lower = 1)"
        @test repr(Domain(; type=Symbol)) == "Domain(; type = Symbol)"
        @test repr("text/plain", Domain(; type=Bool)) == "Domain(; type = Bool)"
        @test repr(Domain(; type=AbstractVector{Real}, lower=0, upper=1)) ==
            "Domain(; type = AbstractVector{Real}, lower = 0, upper = 1)"
        #The type of the possible values isn't shown, a type that differs from it is, and a constraint is shown by its name
        @test repr(Domain(; possible_values=(:greedy, :random))) ==
            "Domain(; possible_values = (:greedy, :random))"
        @test repr(Domain(; type=Real, possible_values=1:3)) ==
            "Domain(; type = Real, possible_values = 1:3)"
        @test repr(Domain(; type=AbstractVector{Real}, constraint=issorted)) ==
            "Domain(; type = AbstractVector{Real}, constraint = issorted)"
        @test repr(Domain(; constraint=value -> value != 0)) ==
            "Domain(; constraint = an anonymous function)"
    end

    @testset "A parameter is discrete when it has possible values, or when its type, or the type of its elements, is one of whole numbers, or isn't one of real numbers" begin
        @test !CognitiveModeling.is_discrete(Domain())
        @test !CognitiveModeling.is_discrete(Domain(; lower=0, upper=1))
        @test !CognitiveModeling.is_discrete(Domain(; type=Rational))
        @test CognitiveModeling.is_discrete(Domain(; type=Integer))
        @test CognitiveModeling.is_discrete(Domain(; type=Int, lower=1))
        @test CognitiveModeling.is_discrete(Domain(; type=Bool))
        @test CognitiveModeling.is_discrete(Domain(; type=Symbol))
        @test CognitiveModeling.is_discrete(Domain(; type=AbstractString))
        #An array or a tuple is discrete when its elements are
        @test !CognitiveModeling.is_discrete(Domain(; type=AbstractVector{Real}))
        @test !CognitiveModeling.is_discrete(Domain(; type=NTuple{2,Real}))
        @test CognitiveModeling.is_discrete(Domain(; type=AbstractVector{Integer}))
        @test CognitiveModeling.is_discrete(Domain(; type=AbstractMatrix{Bool}))
        @test CognitiveModeling.is_discrete(Domain(; type=AbstractVector{Symbol}))
        #Possible values make a parameter discrete, also when they are real numbers, and a constraint doesn't change it
        @test CognitiveModeling.is_discrete(Domain(; possible_values=(0.5, 1.0)))
        @test CognitiveModeling.is_discrete(
            Domain(; type=AbstractVector{Real}, possible_values=0.0:0.5:1.0)
        )
        @test !CognitiveModeling.is_discrete(Domain(; constraint=isfinite))
        @test CognitiveModeling.is_discrete(Domain(; type=Integer, constraint=iseven))
    end

    @testset "ParameterDomainError has the message it is given" begin
        error = ParameterDomainError("The parameter learning_rate is 1.5.")
        @test error isa Exception
        @test error.message == "The parameter learning_rate is 1.5."
        @test sprint(showerror, error) ==
            "ParameterDomainError: The parameter learning_rate is 1.5."
        @test_throws ParameterDomainError throw(error)
        #It needs a message, since only the check of a domain knows what was wrong
        @test_throws MethodError ParameterDomainError()
    end
end
