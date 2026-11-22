#A model whose parameters have domains: one within bounds, one of whole numbers with a lower bound, one that is a name, one that is true or false, and one without a domain
struct ModelWithDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithDomains)
    return ModelAttributes(;
        parameters=(:learning_rate, :n_options, :strategy, :is_greedy, :bias),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithDomains)
    return (;
        learning_rate=Domain(; lower=0, upper=1),
        n_options=Domain(; type=Integer, lower=1),
        strategy=Domain(; type=Symbol),
        is_greedy=Domain(; type=Bool),
    )
end

#A model whose parameters are arrays and tuples: probabilities in a vector, counts in a vector of whole numbers, weights in a matrix, a pair of positive numbers, rates in a tuple of any length, names in a vector, and values in an array of any number of dimensions
struct ModelWithArrayDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithArrayDomains)
    return ModelAttributes(;
        parameters=(:probabilities, :counts, :weights, :pair, :rates, :names, :values),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithArrayDomains)
    return (;
        probabilities=Domain(; type=AbstractVector{Real}, lower=0, upper=1),
        counts=Domain(; type=AbstractVector{Integer}, lower=0),
        weights=Domain(; type=AbstractMatrix{Real}, upper=1),
        pair=Domain(; type=NTuple{2,Real}, lower=0),
        rates=Domain(; type=Tuple{Vararg{Real}}, lower=0, upper=1),
        names=Domain(; type=AbstractVector{Symbol}),
        values=Domain(; type=AbstractArray{Real}, lower=-1, upper=1),
    )
end

#A model whose parameter is a vector of vectors of names, whose elements are checked against the type of the elements as a whole
struct ModelWithNestedArrays <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithNestedArrays)
    return ModelAttributes(; parameters=:vectors, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(model::ModelWithNestedArrays)
    return (; vectors=Domain(; type=AbstractVector{Vector{Symbol}}))
end

#Constraints on the whole value of a parameter: probabilities that sum to one, and a pair whose first number is below its second, and a constraint that returns a number and not true or false
sums_to_one(values) = sum(values) ≈ 1
is_increasing(pair) = pair[1] < pair[2]
total_of(values) = sum(values)

#A model whose parameters have possible values or constraints: a strategy among two names, a number of options among three whole numbers, a step size in a range of numbers, levels in a vector which are each in a range of whole numbers, a pair of sides which are each one of three names, probabilities in a vector which sum to one, a pair of numbers in increasing order, and thresholds in a vector whose constraint is an anonymous function
struct ModelWithPossibleValues <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithPossibleValues)
    return ModelAttributes(;
        parameters=(
            :strategy,
            :n_options,
            :step_size,
            :levels,
            :sides,
            :probabilities,
            :pair,
            :thresholds,
        ),
        states=(),
        inputs=(),
        outputs=(),
    )
end
function CognitiveModeling.parameter_domains(model::ModelWithPossibleValues)
    return (;
        strategy=Domain(; possible_values=(:greedy, :random)),
        n_options=Domain(; possible_values=(2, 4, 8)),
        step_size=Domain(; possible_values=0.0:0.25:1.0),
        levels=Domain(; type=AbstractVector{Integer}, possible_values=1:3),
        sides=Domain(; type=NTuple{2,Symbol}, possible_values=(:left, :middle, :right)),
        probabilities=Domain(;
            type=AbstractVector{Real}, lower=0, upper=1, constraint=sums_to_one
        ),
        pair=Domain(; type=NTuple{2,Real}, constraint=is_increasing),
        thresholds=Domain(;
            type=AbstractVector{Real}, constraint=values -> issorted(values)
        ),
    )
end

#A model whose constraint returns a number, and not true or false
struct ModelWithConstraintOfNumbers <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithConstraintOfNumbers)
    return ModelAttributes(; parameters=:weights, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(model::ModelWithConstraintOfNumbers)
    return (; weights=Domain(; type=AbstractVector{Real}, constraint=total_of))
end

#Models whose domains don't fit them: a parameter that isn't in the model's attributes, a domain that isn't a Domain, and domains that aren't in a NamedTuple
struct ModelWithUnknownDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithUnknownDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithUnknownDomain)
    return (; noise=Domain(; lower=0),)
end
struct ModelWithNamedTupleDomain <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithNamedTupleDomain)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithNamedTupleDomain)
    return (; learning_rate=(; lower=0, upper=1),)
end
struct ModelWithUnnamedDomains <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(model::ModelWithUnnamedDomains)
    return ModelAttributes(; parameters=:learning_rate, states=(), inputs=(), outputs=())
end
function CognitiveModeling.parameter_domains(::ModelWithUnnamedDomains)
    return [Domain(; lower=0, upper=1)]
end

#A number that isn't a real number, as the numbers that Reactant.jl traces aren't, which can't be compared with anything
struct NumberWithoutValue <: Number end

#A stand-in for a number of automatic differentiation which orders numbers of the same value by their derivatives, as those of ForwardDiff.jl are ordered
struct NumberWithDerivative <: Real
    value::Float64
    derivative::Float64
end
function Base.:<=(number::NumberWithDerivative, bound::Real)
    return number.value < bound || (number.value == bound && number.derivative <= 0)
end
function Base.:<=(bound::Real, number::NumberWithDerivative)
    return bound < number.value || (bound == number.value && number.derivative >= 0)
end
Base.isnan(number::NumberWithDerivative) = isnan(number.value)

#Checks the same parameters many times, to measure what a check costs when a model runs
function check_many_times(model, parameters, n_times)
    for _ in 1:n_times
        CognitiveModeling.check_parameters_in_domains(model, parameters)
    end
    return nothing
end

@testset "Checking parameters against their domains" begin
    #Parameters of ModelWithDomains which are all within their domains
    valid_parameters = (;
        learning_rate=0.5, n_options=3, strategy=:greedy, is_greedy=true, bias=-10.0
    )
    #Parameters of ModelWithArrayDomains which are all within their domains
    valid_array_parameters = (;
        probabilities=[0.2, 0.3, 0.5],
        counts=[0, 3],
        weights=[0.1 0.2; 0.3 1.0],
        pair=(0.5, 2),
        rates=(0.1, 0.2, 0.3),
        names=[:a, :b],
        values=zeros(2, 2, 2),
    )
    #Parameters of ModelWithPossibleValues which are all within their domains
    valid_parameters_with_possible_values = (;
        strategy=:random,
        n_options=4,
        step_size=0.75,
        levels=[1, 3, 2],
        sides=(:left, :right),
        probabilities=[0.2, 0.3, 0.5],
        pair=(0.1, 0.4),
        thresholds=[-1.0, 0.0, 2.0],
    )
    check_possible_values(changes) = CognitiveModeling.check_parameters_in_domains(
        ModelWithPossibleValues(), merge(valid_parameters_with_possible_values, changes)
    )

    @testset "A model has no domains by default, and its parameters can then take any value" begin
        @test CognitiveModeling.parameter_domains(ModelWithAttributes()) === (;)
        @test CognitiveModeling.check_parameters_in_domains(
            ModelWithAttributes(), (; learning_rate=-5.0, action_noise="loud")
        ) === nothing
    end

    @testset "Parameters within their domains pass, with the bounds included" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(model, valid_parameters) ===
            nothing
        @test CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=0))
        ) === nothing
        @test CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=1, n_options=1))
        ) === nothing
    end

    @testset "A domain of real numbers takes every type of them" begin
        model = ModelWithDomains()
        for learning_rate in (0.5f0, 0.5, big"0.5", 1//2, true, Float16(0.5))
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; learning_rate))
            ) === nothing
        end
    end

    @testset "A domain of whole numbers takes them in any number type, and one of Bool takes 0 and 1" begin
        model = ModelWithDomains()
        for n_options in (3, 3.0, 3.0f0, Int8(3), big(3), 6//2, true)
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; n_options))
            ) === nothing
        end
        for is_greedy in (true, false, 0, 1, 0.0, 1.0)
            @test CognitiveModeling.check_parameters_in_domains(
                model, merge(valid_parameters, (; is_greedy))
            ) === nothing
        end
    end

    @testset "A value outside the bounds is a ParameterDomainError, and so is a NaN" begin
        model = ModelWithDomains()
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is 1.5, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=1.5))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is -0.1, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=-0.1))
        )
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 0, which is outside its domain, from 1 to Inf.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options=0))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is NaN, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=NaN))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=Inf))
        )
    end

    @testset "A number of automatic differentiation that equals a bound is within it, whatever its derivative" begin
        model = ModelWithDomains()
        check(learning_rate) = CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate))
        )
        #The stand-in compares as beyond a bound that it equals when its derivative points beyond it
        @test !(NumberWithDerivative(1.0, 2.0) <= 1)
        @test !(0 <= NumberWithDerivative(0.0, -2.0))
        @test check(NumberWithDerivative(1.0, 2.0)) === nothing
        @test check(NumberWithDerivative(0.0, -2.0)) === nothing
        @test check(NumberWithDerivative(0.5, 2.0)) === nothing
        #A value beyond a bound is outside it, also by the smallest step there is, and so is a NaN
        @test_throws ParameterDomainError check(NumberWithDerivative(1.5, 0.0))
        @test_throws ParameterDomainError check(NumberWithDerivative(nextfloat(1.0), 2.0))
        @test_throws ParameterDomainError check(NumberWithDerivative(-0.5, 2.0))
        @test_throws ParameterDomainError check(NumberWithDerivative(prevfloat(0.0), -2.0))
        @test_throws ParameterDomainError check(NumberWithDerivative(NaN, 0.0))
        #A side without a bound has nothing beyond it
        @test !CognitiveModeling.is_beyond_bounds(
            NumberWithDerivative(Inf, 2.0), Domain(; lower=0)
        )
        @test !CognitiveModeling.is_beyond_bounds(
            NumberWithDerivative(-Inf, -2.0), Domain(; upper=0)
        )
        #Julia's own numbers are compared with the bounds as they are, however close they are
        @test_throws ParameterDomainError check(big"1.0" + big"1e-30")
        @test_throws ParameterDomainError check(nextfloat(1.0))
    end

    @testset "A value of another type than the domain's is a ParameterDomainError" begin
        model = ModelWithDomains()
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is 2.5, of type Float64, but its domain takes whole numbers, as its type is Integer.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options=2.5))
        )
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithDomains is \"three\", of type String, but its domain takes values of type Integer.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; n_options="three"))
        )
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithDomains is \"greedy\", of type String, but its domain takes values of type Symbol.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy="greedy"))
        )
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithDomains is 1, of type Int64, but its domain takes values of type Symbol.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy=1))
        )
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is :high, of type Symbol, but its domain takes values of type Real.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=:high))
        )
        @test_throws ParameterDomainError(
            "The parameter is_greedy of a cognitive model of type ModelWithDomains is 2, of type Int64, but its domain takes true and false, or 0 and 1.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; is_greedy=2))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; is_greedy=0.5))
        )
    end

    @testset "A domain of single values doesn't take an array or a tuple" begin
        model = ModelWithDomains()
        @test_throws ParameterDomainError(
            "The parameter learning_rate of a cognitive model of type ModelWithDomains is [0.2, 0.8], of type Vector{Float64}, but its domain takes values of type Real.",
        ) CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=[0.2, 0.8]))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; learning_rate=(0.2, 0.8)))
        )
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_parameters, (; strategy=[:greedy]))
        )
    end

    @testset "A domain of arrays takes every array with its number of dimensions, and checks each element" begin
        model = ModelWithArrayDomains()
        check(changes) = CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_array_parameters, changes)
        )
        @test check((;)) === nothing
        #Arrays of other kinds and other number types are taken as they are, also numbers of automatic differentiation at a bound, and whole numbers in another number type
        @test check((; probabilities=view([0.2, 0.3, 0.5, 9.0], 1:3))) === nothing
        @test check((; probabilities=Float32[0.2, 0.8], counts=[0.0, 3.0])) === nothing
        @test check((;
            probabilities=[NumberWithDerivative(1.0, 2.0), NumberWithDerivative(0.0, -2.0)]
        )) === nothing
        @test check((; weights=[true false; false true], counts=Int8[2])) === nothing
        #An array type without a number of dimensions takes any number of them
        @test check((; values=[0.5])) === nothing
        @test check((; values=fill(0.5, 1, 1, 1))) === nothing
        #An element outside the bounds is named by its position
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithArrayDomains has probabilities[3] = -0.2, which is outside its domain, from 0 to 1.",
        ) check((; probabilities=[0.5, 0.7, -0.2]))
        @test_throws ParameterDomainError(
            "The parameter weights of a cognitive model of type ModelWithArrayDomains has weights[2, 1] = 1.3, which is outside its domain, from -Inf to 1.",
        ) check((; weights=[0.1 0.2; 1.3 0.4]))
        @test_throws ParameterDomainError check((; probabilities=[0.5, NaN]))
        @test_throws ParameterDomainError check((; values=fill(2.0, 1, 1, 1)))
        @test_throws ParameterDomainError check((;
            probabilities=[NumberWithDerivative(nextfloat(1.0), 2.0)]
        ))
        #An element of another type is named by its position too
        @test_throws ParameterDomainError(
            "The parameter counts of a cognitive model of type ModelWithArrayDomains has counts[2] = 2.5, of type Float64, but its domain takes whole numbers there, as the type of its elements is Integer.",
        ) check((; counts=[1, 2.5]))
        @test_throws ParameterDomainError(
            "The parameter names of a cognitive model of type ModelWithArrayDomains has names[2] = \"b\", of type String, but its domain takes values of type Symbol there.",
        ) check((; names=Any[:a, "b"]))
        #A value that isn't an array, or an array with another number of dimensions, isn't taken
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithArrayDomains is 0.1, of type Float64, but its domain takes values of type AbstractVector{Real}.",
        ) check((; probabilities=0.1))
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithArrayDomains is [0.5 0.5], of type Matrix{Float64}, but its domain takes values of type AbstractVector{Real}.",
        ) check((; probabilities=[0.5 0.5]))
        @test_throws ParameterDomainError check((; weights=[0.1, 0.2]))
        @test_throws ParameterDomainError check((; probabilities=(0.2, 0.8)))
        #The elements of an array of arrays are checked against the type of the elements as a whole
        @test CognitiveModeling.check_parameters_in_domains(
            ModelWithNestedArrays(), (; vectors=[[:a], [:b, :c]])
        ) === nothing
        @test_throws ParameterDomainError(
            "The parameter vectors of a cognitive model of type ModelWithNestedArrays has vectors[2] = [\"b\"], of type Vector{String}, but its domain takes values of type Vector{Symbol} there.",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithNestedArrays(), (; vectors=[[:a], ["b"]])
        )
    end

    @testset "A domain of tuples takes tuples of the length of its type, and checks each element" begin
        model = ModelWithArrayDomains()
        check(changes) = CognitiveModeling.check_parameters_in_domains(
            model, merge(valid_array_parameters, changes)
        )
        @test check((; pair=(0, 1.5f0), rates=(0.5,))) === nothing
        #A tuple type of any length takes the empty tuple
        @test check((; rates=())) === nothing
        @test_throws ParameterDomainError(
            "The parameter pair of a cognitive model of type ModelWithArrayDomains has pair[2] = -1, which is outside its domain, from 0 to Inf.",
        ) check((; pair=(0.5, -1)))
        @test_throws ParameterDomainError(
            "The parameter rates of a cognitive model of type ModelWithArrayDomains has rates[3] = 1.5, which is outside its domain, from 0 to 1.",
        ) check((; rates=(0.1, 0.2, 1.5)))
        @test_throws ParameterDomainError(
            "The parameter pair of a cognitive model of type ModelWithArrayDomains is (0.5, 1, 2), of type Tuple{Float64, Int64, Int64}, but its domain takes values of type Tuple{Real, Real}.",
        ) check((; pair=(0.5, 1, 2)))
        @test_throws ParameterDomainError check((; pair=[0.5, 1.0]))
        @test_throws ParameterDomainError check((; rates=(0.1, :fast)))
    end

    @testset "A value among the possible values passes, compared by its value, and any other value is a ParameterDomainError" begin
        @test check_possible_values((;)) === nothing
        @test check_possible_values((; strategy=:greedy, n_options=2, step_size=0)) ===
            nothing
        #A number is compared by its value, in any number type
        @test check_possible_values((;
            n_options=4.0, step_size=1, levels=[1.0, 2.0, 3.0]
        )) === nothing
        @test check_possible_values((; n_options=Int32(8), step_size=1//4)) === nothing
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithPossibleValues is :lazy, which isn't one of its possible values, (:greedy, :random).",
        ) check_possible_values((; strategy=:lazy))
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithPossibleValues is 3, which isn't one of its possible values, (2, 4, 8).",
        ) check_possible_values((; n_options=3))
        @test_throws ParameterDomainError(
            "The parameter step_size of a cognitive model of type ModelWithPossibleValues is 0.3, which isn't one of its possible values, 0.0:0.25:1.0.",
        ) check_possible_values((; step_size=0.3))
        @test_throws ParameterDomainError check_possible_values((; step_size=1.25))
        #A value of another type than the possible values is told so, before they are compared
        @test_throws ParameterDomainError(
            "The parameter strategy of a cognitive model of type ModelWithPossibleValues is \"greedy\", of type String, but its domain takes values of type Symbol.",
        ) check_possible_values((; strategy="greedy"))
        @test_throws ParameterDomainError(
            "The parameter n_options of a cognitive model of type ModelWithPossibleValues is 2.5, of type Float64, but its domain takes whole numbers, as its type is Integer.",
        ) check_possible_values((; n_options=2.5))
    end

    @testset "Each element of an array or a tuple is one of the possible values, and an element that isn't is named by its position" begin
        @test_throws ParameterDomainError(
            "The parameter levels of a cognitive model of type ModelWithPossibleValues has levels[2] = 4, which isn't one of its possible values, 1:3.",
        ) check_possible_values((; levels=[1, 4, 2]))
        @test_throws ParameterDomainError check_possible_values((; levels=[0]))
        @test check_possible_values((; levels=Int[])) === nothing
        @test_throws ParameterDomainError(
            "The parameter sides of a cognitive model of type ModelWithPossibleValues has sides[2] = :up, which isn't one of its possible values, (:left, :middle, :right).",
        ) check_possible_values((; sides=(:left, :up)))
        @test_throws ParameterDomainError check_possible_values((; sides=(:left,)))
    end

    @testset "A value that doesn't satisfy its domain's constraint is a ParameterDomainError, which names the constraint unless it is an anonymous function" begin
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithPossibleValues is [0.2, 0.3, 0.4], which doesn't satisfy the constraint sums_to_one of its domain.",
        ) check_possible_values((; probabilities=[0.2, 0.3, 0.4]))
        @test_throws ParameterDomainError(
            "The parameter pair of a cognitive model of type ModelWithPossibleValues is (0.4, 0.1), which doesn't satisfy the constraint is_increasing of its domain.",
        ) check_possible_values((; pair=(0.4, 0.1)))
        @test_throws ParameterDomainError(
            "The parameter thresholds of a cognitive model of type ModelWithPossibleValues is [1.0, 0.0], which doesn't satisfy the constraint of its domain.",
        ) check_possible_values((; thresholds=[1.0, 0.0]))
        #The constraint is called after the type and the bounds are checked, which are told first
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithPossibleValues has probabilities[2] = -0.5, which is outside its domain, from 0 to 1.",
        ) check_possible_values((; probabilities=[0.5, -0.5, 1.0]))
        @test_throws ParameterDomainError(
            "The parameter probabilities of a cognitive model of type ModelWithPossibleValues is 0.5, of type Float64, but its domain takes values of type AbstractVector{Real}.",
        ) check_possible_values((; probabilities=0.5))
        #Numbers that aren't real numbers are left alone, also by the constraint
        @test check_possible_values((;
            probabilities=[NumberWithoutValue()],
            pair=(NumberWithoutValue(), NumberWithoutValue()),
            thresholds=[2.5 + 0.0im, 1.0 + 0.0im],
        )) === nothing
    end

    @testset "A constraint that returns something other than true or false is an ArgumentError" begin
        @test_throws ArgumentError(
            "The constraint total_of of the domain of the parameter weights of a cognitive model of type ModelWithConstraintOfNumbers returned 1.0 for [0.5, 0.5], but a constraint returns true or false.",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithConstraintOfNumbers(), (; weights=[0.5, 0.5])
        )
    end

    @testset "Numbers that aren't real numbers are left alone, also in arrays and tuples" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            merge(
                valid_parameters,
                (; learning_rate=NumberWithoutValue(), n_options=2.5 + 0.0im),
            ),
        ) === nothing
        @test CognitiveModeling.check_parameters_in_domains(
            ModelWithArrayDomains(),
            merge(
                valid_array_parameters,
                (;
                    probabilities=[NumberWithoutValue()],
                    counts=[2.5 + 0.0im],
                    pair=(NumberWithoutValue(), NumberWithoutValue()),
                    values=NumberWithoutValue(),
                ),
            ),
        ) === nothing
    end

    @testset "The order of the parameters doesn't matter, and those without a domain can be anything" begin
        model = ModelWithDomains()
        @test CognitiveModeling.check_parameters_in_domains(
            model,
            (;
                bias="any value",
                is_greedy=false,
                strategy=:random,
                n_options=2,
                learning_rate=0.1,
            ),
        ) === nothing
        @test_throws ParameterDomainError CognitiveModeling.check_parameters_in_domains(
            model,
            (;
                bias="any value",
                is_greedy=false,
                strategy=:random,
                n_options=0,
                learning_rate=0.1,
            ),
        )
    end

    @testset "The domains must be Domains under the names of parameters in the model's attributes" begin
        @test_throws ArgumentError(
            "parameter_domains of a cognitive model of type ModelWithUnknownDomain has domains for (:noise,), but the parameters in its attributes are (:learning_rate,).",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithUnknownDomain(), (; learning_rate=0.5)
        )
        @test_throws ArgumentError(
            "parameter_domains of a cognitive model of type ModelWithNamedTupleDomain returns (learning_rate = (lower = 0, upper = 1),), but it must return a NamedTuple with a Domain under the name of each parameter that has a domain, such as (; learning_rate = Domain(; lower = 0, upper = 1)).",
        ) CognitiveModeling.check_parameters_in_domains(
            ModelWithNamedTupleDomain(), (; learning_rate=0.5)
        )
        @test_throws "parameter_domains of a cognitive model of type ModelWithUnnamedDomains returns" CognitiveModeling.check_parameters_in_domains(
            ModelWithUnnamedDomains(), (; learning_rate=0.5)
        )
    end

    @testset "A value in the data is checked as a parameter is, with an ArgumentError that names its timestep or its row in the table" begin
        in_session(timestep) = CognitiveModeling.DataValue(:action, timestep, false)
        in_table(row) = CognitiveModeling.DataValue(:observation, row, true)
        choice_domain = Domain(; possible_values=(1, 2))
        @test_throws ArgumentError(
            "The action choice at timestep 3 is 3, which isn't one of its possible values, (1, 2).",
        ) CognitiveModeling.check_value_in_domain(in_session(3), :choice, 3, choice_domain)
        @test_throws ArgumentError(
            "The observation reward in row 7 of the table is -0.5, which is outside its domain, from 0 to 1.",
        ) CognitiveModeling.check_value_in_domain(
            in_table(7), :reward, -0.5, Domain(; lower=0, upper=1)
        )
        #An element of an array is named by its position, a value of another type by its type, and a constraint by its name
        @test_throws ArgumentError(
            "The action ratings at timestep 2 has ratings[2] = -0.5, which is outside its domain, from 0 to Inf.",
        ) CognitiveModeling.check_value_in_domain(
            in_session(2),
            :ratings,
            [0.5, -0.5],
            Domain(; type=AbstractVector{Real}, lower=0),
        )
        @test_throws ArgumentError(
            "The action choice at timestep 1 is \"left\", of type String, but its domain takes values of type Integer.",
        ) CognitiveModeling.check_value_in_domain(
            in_session(1), :choice, "left", Domain(; type=Integer)
        )
        @test_throws ArgumentError(
            "The action ratings at timestep 1 is [0.5, 0.6], which doesn't satisfy the constraint sums_to_one of its domain.",
        ) CognitiveModeling.check_value_in_domain(
            in_session(1),
            :ratings,
            [0.5, 0.6],
            Domain(; type=AbstractVector{Real}, constraint=sums_to_one),
        )
        @test_throws ArgumentError(
            "The constraint total_of of the domain of the action ratings at timestep 1 returned 1.1 for [0.5, 0.6], but a constraint returns true or false.",
        ) CognitiveModeling.check_value_in_domain(
            in_session(1),
            :ratings,
            [0.5, 0.6],
            Domain(; type=AbstractVector{Real}, constraint=total_of),
        )
        #A value within its domain passes
        @test CognitiveModeling.check_value_in_domain(
            in_session(1), :choice, 2, choice_domain
        ) === nothing
    end

    @testset "The check is inferred and doesn't allocate, with and without domains" begin
        model = ModelWithDomains()
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(model, valid_parameters)
        ) === nothing
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(
                ModelWithAttributes(), (; learning_rate=0.5, action_noise=1.0)
            )
        ) === nothing
        #Run once first, so that compiling isn't measured
        check_many_times(model, valid_parameters, 1)
        @test @allocated(check_many_times(model, valid_parameters, 100)) == 0
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(
                ModelWithArrayDomains(), valid_array_parameters
            )
        ) === nothing
        check_many_times(ModelWithArrayDomains(), valid_array_parameters, 1)
        @test @allocated(
            check_many_times(ModelWithArrayDomains(), valid_array_parameters, 100)
        ) == 0
        without_domains = (; learning_rate=0.5, action_noise=1.0)
        check_many_times(ModelWithAttributes(), without_domains, 1)
        @test @allocated(check_many_times(ModelWithAttributes(), without_domains, 100)) == 0
        #Possible values and constraints are kept in the types of the domains, so they are compiled in too
        model_with_possible_values = ModelWithPossibleValues()
        @test @inferred(
            CognitiveModeling.check_parameters_in_domains(
                model_with_possible_values, valid_parameters_with_possible_values
            )
        ) === nothing
        check_many_times(
            model_with_possible_values, valid_parameters_with_possible_values, 1
        )
        @test @allocated(
            check_many_times(
                model_with_possible_values, valid_parameters_with_possible_values, 100
            )
        ) == 0
    end
end
