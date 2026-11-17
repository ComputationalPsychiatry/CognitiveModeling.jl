### THE REGRESSION OF A PARAMETER WRITTEN AS A FORMULA ###

#Translates a formula into the terms of a RegressionParameter, which is named by the left-hand side, with the columns that are categorical predictors as categorical_predictors names them or gives them, and the priors of the terms by their names. Terms that a formula can't express are given after it
function CognitiveModeling.RegressionParameter(
    formula::FormulaTerm,
    terms...;
    categorical_predictors=(),
    priors::NamedTuple=(;),
    transformation=identity,
    family=nothing,
    default_priors::DefaultRegressionPriors=DefaultRegressionPriors(),
    default_settings::DefaultRegressionSettings=DefaultRegressionSettings(),
)
    #The left-hand side names the parameter
    parameter_name = read_parameter_name(formula.lhs)
    description = "the formula of $parameter_name"

    #The categorical predictors under the names of their columns, which must be predictors of the formula
    right_hand_side = as_tuple_of_terms(formula.rhs)
    predictors_by_column = read_categorical_predictors(categorical_predictors, description)
    check_names_predictors(predictors_by_column, right_hand_side, description)

    #Each term of the right-hand side as a term of the regression, with an intercept unless 0 leaves it out
    formula_terms = translate_terms(right_hand_side, predictors_by_column, description)

    #The terms named in the priors get those priors, and the others keep the default priors of their kinds
    check_prior_names(priors, formula_terms, description)
    formula_terms_with_priors = map(term -> with_named_prior(term, priors), formula_terms)

    return RegressionParameter(
        formula_terms_with_priors...,
        terms...;
        transformation,
        family,
        default_priors,
        default_settings,
        name=parameter_name,
    )
end

### READING THE FORMULA ###

"""
    read_parameter_name(left_hand_side)

Return the name of the parameter that a formula is the regression of, which is its
left-hand side, a single name, such as `learning_rate` in
`@formula(learning_rate ~ 1 + age)`. Anything else is an error. Used by the formula form of
`RegressionParameter`.
"""
read_parameter_name(left_hand_side::Term) = left_hand_side.sym
function read_parameter_name(left_hand_side)
    return throw(
        ArgumentError(
            "The left-hand side of the formula of a RegressionParameter is the name of the parameter, as in @formula(learning_rate ~ 1 + age), but it is $left_hand_side.",
        ),
    )
end

"""
    as_tuple_of_terms(terms)

Return the terms of a sum in a formula as a tuple: the right-hand side of a formula, which
is a tuple of terms or one term, or the terms of varying effects, which are a sum, as in
`1 + condition` of `(1 + condition | participant)`, or one term. Used by the formula form
of `RegressionParameter` and by `translate_term`.
"""
as_tuple_of_terms(terms::Tuple) = terms
as_tuple_of_terms(sum::FunctionTerm{typeof(+)}) = Tuple(sum.args)
as_tuple_of_terms(term) = (term,)

"""
    translate_terms(formula_terms, predictors_by_column, description)

Return the terms of a regression that `formula_terms`, a tuple of the terms of a sum in a
formula, are, with an [`Intercept`](@ref) first unless the sum has `0` or `-1`, as
`translate_term` translates each, with the columns that are terms of their own in the sum,
for its interactions. Used by the formula form of `RegressionParameter`, and by
`translate_term` for the terms of varying effects.
"""
function translate_terms(
    formula_terms::Tuple, predictors_by_column::NamedTuple, description::String
)
    intercept = includes_intercept(formula_terms, description) ? (Intercept(),) : ()
    columns_with_terms = Tuple(term.sym for term in formula_terms if term isa Term)
    others = Tuple(
        translate_term(term, predictors_by_column, description, columns_with_terms) for
        term in formula_terms if !(term isa ConstantTerm)
    )
    return (intercept..., others...)
end

"""
    includes_intercept(formula_terms, description)

Return whether a sum in a formula, a tuple of its terms, includes an intercept: unless it
has the constant `0` or `-1`, as in `0 + age`. A constant other than these and `1` is an
error. Used by `translate_terms`.
"""
function includes_intercept(formula_terms::Tuple, description::String)
    for term in formula_terms
        term isa ConstantTerm || continue
        if !(term.n in (-1, 0, 1))
            throw(
                ArgumentError(
                    "The sum of $description has the constant $(term.n), but a constant in a formula is 1 for an intercept, or 0 or -1 for none, as in @formula(learning_rate ~ 0 + age).",
                ),
            )
        end
    end
    return !any(term -> term isa ConstantTerm && term.n in (-1, 0), formula_terms)
end

"""
    translate_term(term, predictors_by_column, description, columns_with_terms)

Return the term of a regression that `term`, a term of a formula other than a constant, is:
a column is a [`ContinuousPredictor`](@ref), or the [`CategoricalPredictor`](@ref) that
`predictors_by_column` has under its name, as `predictor_of_column` decides, columns joined
by `&` are an [`Interaction`](@ref), whose predictors
`predictor_in_interaction` gives by their names or themselves, and `(terms | columns)` are
[`VaryingEffects`](@ref) over `columns`, with an intercept unless `terms` has `0`. Anything
else, such as a function of a column, is an error. `description` says which formula the
term is in, for the errors, and `columns_with_terms` has the columns that are terms of
their own in the same sum. Used by `translate_terms`.
"""
function translate_term(
    term::Term,
    predictors_by_column::NamedTuple,
    description::String,
    columns_with_terms::Tuple,
)
    return predictor_of_column(term.sym, predictors_by_column)
end

#An interaction is an InteractionTerm, and within a call, as in varying effects, a call of &
function translate_term(
    interaction::InteractionTerm,
    predictors_by_column::NamedTuple,
    description::String,
    columns_with_terms::Tuple,
)
    return Interaction(
        map(
            part -> predictor_in_interaction(
                read_column(part, description), columns_with_terms, predictors_by_column
            ),
            interaction.terms,
        ),
    )
end
function translate_term(
    interaction::FunctionTerm{typeof(&)},
    predictors_by_column::NamedTuple,
    description::String,
    columns_with_terms::Tuple,
)
    return Interaction(
        Tuple(
            predictor_in_interaction(
                read_column(part, description), columns_with_terms, predictors_by_column
            ) for part in interaction.args
        ),
    )
end

#Varying effects are written as in MixedModels.jl, with the terms whose coefficients vary before the bar and the columns whose levels they vary between after it
function translate_term(
    varying::FunctionTerm{typeof(|)},
    predictors_by_column::NamedTuple,
    description::String,
    columns_with_terms::Tuple,
)
    terms_side, columns_side = varying.args
    columns = read_varying_effects_columns(columns_side, description)
    varying_terms = translate_terms(
        as_tuple_of_terms(terms_side),
        predictors_by_column,
        "the varying effects over $(join(columns, " & ")) in $description",
    )
    return VaryingEffects(columns, varying_terms...)
end

#Anything else isn't a term of a regression
function translate_term(
    term, predictors_by_column::NamedTuple, description::String, columns_with_terms::Tuple
)
    return throw(
        ArgumentError(
            "The term $term of $description isn't a term of a RegressionParameter. A formula has columns, interactions of columns as in diagnosis & condition, and varying effects as in (1 + condition | participant). A column whose values are transformed, such as with log(age), is given as a column of the table with the transformed values.",
        ),
    )
end

"""
    predictor_in_interaction(column, columns_with_terms, predictors_by_column)

Return how `column` of an interaction in a formula is given to the [`Interaction`](@ref):
by its name when it is a term of its own in the same sum, whose columns are in
`columns_with_terms`, and otherwise as the predictor itself, as `predictor_of_column` makes
it, so that the interaction comes without the main effect of that column. Used by
`translate_term`.
"""
function predictor_in_interaction(
    column::Symbol, columns_with_terms::Tuple, predictors_by_column::NamedTuple
)
    column in columns_with_terms && return column
    return predictor_of_column(column, predictors_by_column)
end

"""
    read_column(term, description)
    read_varying_effects_columns(term, description)

Return the name of the column of `term`, a part of an interaction in a formula, and the
columns of varying effects, which are one column, as in `(1 | participant)`, or several
joined by `&`, as in `(1 | participant & block)`, whose levels are the combinations of their
values. Anything else is an error. Used by `translate_term`.
"""
read_column(term::Term, description::String) = term.sym
function read_column(term, description::String)
    return throw(
        ArgumentError(
            "An interaction in $description multiplies columns, as in diagnosis & condition, but it has $term.",
        ),
    )
end
read_varying_effects_columns(term::Term, description::String) = (term.sym,)
function read_varying_effects_columns(columns::FunctionTerm{typeof(&)}, description::String)
    return Tuple(read_column(part, description) for part in columns.args)
end
function read_varying_effects_columns(term, description::String)
    return throw(
        ArgumentError(
            "The varying effects of $description are over $term, but varying effects are over a column, as in (1 | participant), or over the combinations of the values of several, as in (1 | participant & block).",
        ),
    )
end

### THE KINDS OF THE PREDICTORS ###

"""
    read_categorical_predictors(categorical_predictors, description)

Return the categorical predictors of a formula as a `NamedTuple` with the
[`CategoricalPredictor`](@ref) of each under the name of its column, from what the keyword
argument `categorical_predictors` was given: the name of a column or a
`CategoricalPredictor`, or a tuple or a vector of them, each read by
`as_categorical_predictor`. A column that is given twice is an error. `description` says
which formula it is, for the errors. Used by the formula form of `RegressionParameter`.
"""
function read_categorical_predictors(categorical_predictors, description::String)
    predictors = map(
        entry -> as_categorical_predictor(entry, description),
        as_tuple(categorical_predictors),
    )
    columns = map(predictor -> only(predictor.columns), predictors)
    for column in columns
        if count(==(column), columns) > 1
            throw(
                ArgumentError(
                    "categorical_predictors of $description gives the column $column more than once.",
                ),
            )
        end
    end
    return NamedTuple{columns}(predictors)
end

#A term of a formula is named after its column, and the same column can be a term of its own, part of an interaction and a varying slope, each with a prior of its own under its name in priors. So a predictor given here says how its column is coded and nothing else
"""
    as_categorical_predictor(entry, description)

Return the [`CategoricalPredictor`](@ref) that `entry` of the keyword argument
`categorical_predictors` of a formula is: for the name of a column, one with the default
coding and reference level, and a `CategoricalPredictor` itself, which gives the coding or
the reference level of its column. A `CategoricalPredictor` of several columns, with
another name than its column or with a prior is an error, and so is anything else. Used by
`read_categorical_predictors`.
"""
as_categorical_predictor(column::Symbol, description::String) = CategoricalPredictor(column)
function as_categorical_predictor(predictor::CategoricalPredictor, description::String)
    columns = predictor.columns
    if length(columns) != 1
        throw(
            ArgumentError(
                "categorical_predictors of $description has a CategoricalPredictor of the columns $columns, but a predictor of a formula reads one column. A predictor whose levels are the combinations of the values of several columns is given after the formula, as a term.",
            ),
        )
    end
    column = only(columns)
    if predictor.name !== column
        throw(
            ArgumentError(
                "categorical_predictors of $description has a CategoricalPredictor of the column $column that is named $(predictor.name), but a term of a formula is named after its column. A predictor with another name is given after the formula, as a term.",
            ),
        )
    end
    if predictor.prior !== nothing
        throw(
            ArgumentError(
                "categorical_predictors of $description has a CategoricalPredictor of the column $column with the prior $(predictor.prior), but the priors of the terms of a formula are given in priors, under the names of the terms, such as priors = (; $column = Normal(0, 1)).",
            ),
        )
    end
    return predictor
end
function as_categorical_predictor(entry, description::String)
    return throw(
        ArgumentError(
            "categorical_predictors of $description takes the names of the columns that are categorical predictors, such as (:diagnosis, :site), with a CategoricalPredictor in the place of a name to give its coding or its reference level, such as CategoricalPredictor(:diagnosis; reference_level = \"control\"), but it was given $(repr(entry)).",
        ),
    )
end

"""
    predictor_of_column(column, predictors_by_column)

Return the predictor that `column` of a formula is: the [`CategoricalPredictor`](@ref) that
`predictors_by_column` has under its name, as `read_categorical_predictors` made it from
the keyword argument `categorical_predictors`, and a [`ContinuousPredictor`](@ref)
otherwise. Used by `translate_term` and `predictor_in_interaction`.
"""
function predictor_of_column(column::Symbol, predictors_by_column::NamedTuple)
    if haskey(predictors_by_column, column)
        return getproperty(predictors_by_column, column)
    end
    return ContinuousPredictor(column)
end

"""
    check_names_predictors(predictors_by_column, formula_terms, description)

Throw an informative error if `predictors_by_column`, the categorical predictors that the
keyword argument `categorical_predictors` gives under the names of their columns, has
something that isn't a column of a predictor in `formula_terms`, the terms of the
right-hand side of a formula, as when the name is misspelled: a column that is a term of
its own, of an interaction or of varying effects. Used by the formula form of
`RegressionParameter`.
"""
function check_names_predictors(
    predictors_by_column::NamedTuple, formula_terms::Tuple, description::String
)
    columns = columns_of_predictors(formula_terms)
    for column in keys(predictors_by_column)
        if !(column in columns)
            throw(
                ArgumentError(
                    "categorical_predictors names $column, which isn't a predictor of $description. Its predictors are $(Tuple(columns)).",
                ),
            )
        end
    end
    return nothing
end

"""
    columns_of_predictors(formula_terms)

Return the columns that are predictors in `formula_terms`, the terms of a sum in a formula,
as a vector of names without repeats: a column that is a term of its own, those of an
interaction, and those in the terms of varying effects. Used by `check_names_predictors`.
"""
function columns_of_predictors(formula_terms::Tuple)
    columns = Symbol[]
    for term in formula_terms
        if term isa Term
            push!(columns, term.sym)
        elseif term isa InteractionTerm
            append!(columns, (part.sym for part in term.terms if part isa Term))
        elseif term isa FunctionTerm{typeof(&)}
            append!(columns, (part.sym for part in term.args if part isa Term))
        elseif term isa FunctionTerm{typeof(|)}
            append!(columns, columns_of_predictors(as_tuple_of_terms(first(term.args))))
        end
    end
    return unique(columns)
end

### THE PRIORS OF THE TERMS BY THEIR NAMES ###

"""
    check_prior_names(priors, formula_terms, description)

Throw an informative error if `priors` names something that isn't a term in `formula_terms`,
the terms of a regression from a formula, or gives varying effects anything but a
`NamedTuple` with priors for their terms. Used by the formula form of `RegressionParameter`.
"""
function check_prior_names(priors::NamedTuple, formula_terms::Tuple, description::String)
    term_names = map(term -> term.name, formula_terms)
    for (name, prior) in pairs(priors)
        index = findfirst(==(name), term_names)
        if index === nothing
            throw(
                ArgumentError(
                    "The priors of $description name $name, which isn't one of its terms, $term_names. A term of a formula is named as the term of its kind is named, such as diagnosis_by_condition for diagnosis & condition, and participant for (1 | participant).",
                ),
            )
        end
        term = formula_terms[index]
        term isa VaryingEffects || continue
        if !(prior isa NamedTuple)
            throw(
                ArgumentError(
                    "The priors of $description give the varying effects $name $(repr(prior)), but varying effects take a NamedTuple with the priors of the standard deviations of their terms, such as (; intercept = Exponential(1)).",
                ),
            )
        end
        for varying_name in keys(prior)
            if !haskey(term.terms, varying_name)
                throw(
                    ArgumentError(
                        "The priors of the varying effects $name of $description name $varying_name, which isn't one of their terms, $(keys(term.terms)).",
                    ),
                )
            end
        end
    end
    return nothing
end

"""
    with_named_prior(term, priors)

Return `term`, a term of a regression from a formula, with the prior that `priors` gives it
under its name, or as it is when `priors` doesn't name it. [`VaryingEffects`](@ref) take a
`NamedTuple` with the priors of the standard deviations of their terms under the names of
their terms. Used by the formula form of `RegressionParameter`.
"""
function with_named_prior(term, priors::NamedTuple)
    haskey(priors, term.name) || return term
    return with_prior(term, getproperty(priors, term.name))
end
function with_named_prior(varying_effects::VaryingEffects, priors::NamedTuple)
    haskey(priors, varying_effects.name) || return varying_effects
    priors_of_terms = getproperty(priors, varying_effects.name)
    terms = map(
        term -> with_named_prior(term, priors_of_terms), values(varying_effects.terms)
    )
    return VaryingEffects(
        varying_effects.columns,
        terms...;
        group_by=varying_effects.group_by,
        centered=varying_effects.centered,
        name=varying_effects.name,
    )
end
