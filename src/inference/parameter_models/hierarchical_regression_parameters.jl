## Each parameter given by a regression on columns of the sessions' table ##

"""
    AbstractRegressionTerm

Supertype of the terms of a [`RegressionParameter`](@ref): [`Intercept`](@ref),
[`ContinuousPredictor`](@ref), [`CategoricalPredictor`](@ref), [`Interaction`](@ref) and
[`VaryingEffects`](@ref). Each has its name in the field `name`, and each but
`VaryingEffects` has its prior in the field `prior`.
"""
abstract type AbstractRegressionTerm end

"""
    Intercept(prior=nothing; name=:intercept)

The intercept of a [`RegressionParameter`](@ref): a coefficient that every session has, with
`prior` as its prior, such as `Normal(0, 1)`, or with `nothing` the regression's default
prior for an intercept, from its [`DefaultRegressionPriors`](@ref). In
[`VaryingEffects`](@ref) it is a varying intercept, a deviation for each level, and `prior`
is the prior of the standard deviation of the deviations, such as `Exponential(1)`. Its
coefficient is named `name`. Each argument is kept in a field of the same name.

# Examples
```jldoctest
julia> using Distributions: Normal

julia> Intercept(Normal(0, 1)).name
:intercept
```
"""
struct Intercept{Tprior} <: AbstractRegressionTerm
    prior::Tprior
    name::Symbol

    #Takes the prior, and the name, which is intercept unless another is given
    function Intercept(prior=nothing; name::Symbol=:intercept)
        return new{typeof(prior)}(prior, name)
    end
end

"""
    ContinuousPredictor(column, prior=nothing; name=column)

A continuous predictor of a [`RegressionParameter`](@ref): a coefficient which is multiplied
by the value of `column` in each session, a column of the table that the sessions were split
from whose values are numbers, with `prior` as its prior, such as `Normal(0, 0.5)`, or with
`nothing` the regression's default prior for a continuous predictor, from its
[`DefaultRegressionPriors`](@ref). In [`VaryingEffects`](@ref) it is a varying slope, a
deviation of the coefficient for each level, and `prior` is the prior of the standard
deviation of the deviations. The predictor and its coefficient are named `name`. Each
argument is kept in a field of the same name.

# Examples
```jldoctest
julia> using Distributions: Normal

julia> ContinuousPredictor(:age, Normal(0, 0.5); name = :years).column
:age
```
"""
struct ContinuousPredictor{Tprior} <: AbstractRegressionTerm
    column::Symbol
    prior::Tprior
    name::Symbol

    #Takes one column and the prior, and is named after the column unless another name is given
    function ContinuousPredictor(column::Symbol, prior=nothing; name::Symbol=column)
        return new{typeof(prior)}(column, prior, name)
    end
end

#A continuous predictor reads one column, and the product of several is an interaction
function ContinuousPredictor(columns, prior=nothing; keyword_arguments...)
    return throw(
        ArgumentError(
            "A ContinuousPredictor reads one column, given as a Symbol such as :age, but it was given $(repr(columns)).",
        ),
    )
end

"""
    CategoricalPredictor(columns, prior=nothing; reference_level=nothing, coding=nothing,
                         name)

A categorical predictor of a [`RegressionParameter`](@ref): a coefficient for each level
other than the reference level, or for every level with the coding `:full_dummy`, with
`prior` as its prior, such as `Normal(0, 1)`, or with `nothing` the regression's default
prior for a categorical predictor, from its [`DefaultRegressionPriors`](@ref). Its levels
are the values of `columns`, a column of the table that the sessions were split from, or the
combinations of the values of several columns given as a tuple. In
[`VaryingEffects`](@ref) it gives varying slopes, a deviation of each coefficient for each
level of the varying effects, and `prior` is the prior of the standard deviation of the
deviations.

# Keyword arguments
- `reference_level=nothing`: the reference level, such as `"control"`, or a `NamedTuple` of
  values for several columns, such as `(; diagnosis = "control", site = 1)`. With `nothing`,
  it is the level of the first session. The coding `:full_dummy` has no reference level.
- `coding=nothing`: how the coefficients are coded, `:dummy`, `:effects` or `:full_dummy`,
  or with `nothing` the regression's default coding, from its
  [`DefaultRegressionSettings`](@ref), which is `:dummy` unless another is given. With
  `:dummy`, each is the difference
  between a level and the reference level, such as the difference between a clinical and a
  non-clinical group. With `:effects`, each is the difference between a level and the mean
  of the levels, and the difference of the reference level is minus the sum of the others,
  which gives the reference level a wider prior than the others when there are three levels
  or more. With `:full_dummy`, every level has a coefficient, which is added for the
  sessions of that level. Without an [`Intercept`](@ref), each is the value of its level,
  and every level has the same prior. With an `Intercept`, or another predictor coded this
  way, only the priors tell their coefficients apart.
- `name`: the name of the predictor, which is the name of its column, or the names of its
  columns joined by two underscores, as in `diagnosis__site`. The coefficient of a level of
  one column is named after the predictor and the level, such as `diagnosis_clinical`, and
  that of a level of several columns after the level alone, by its symbol name, made as
  [`session_id_to_symbol`](@ref) makes that of a session, such as
  `diagnosis_clinical__site_2`.

Each argument is kept in a field of the same name, with `columns` as a tuple of names.

# Examples
```jldoctest
julia> using Distributions: Normal

julia> CategoricalPredictor((:diagnosis, :site), Normal(0, 1); coding = :effects).name
:diagnosis__site
```
"""
struct CategoricalPredictor{Tcolumns<:Tuple{Vararg{Symbol}},Tprior,Treference_level} <:
       AbstractRegressionTerm
    columns::Tcolumns
    prior::Tprior
    reference_level::Treference_level
    coding::Union{Nothing,Symbol}
    name::Symbol

    #Takes one column or several, the prior and the settings, checks the coding when it is given, and is named after the columns unless another name is given
    function CategoricalPredictor(
        columns,
        prior=nothing;
        reference_level=nothing,
        coding::Union{Nothing,Symbol}=nothing,
        name::Union{Nothing,Symbol}=nothing,
    )
        columns_tuple = as_tuple(columns)::Tuple{Vararg{Symbol}}
        check_has_columns(columns_tuple, "A CategoricalPredictor")
        coding === nothing || check_coding(coding, "a CategoricalPredictor")

        #With a coefficient for every level, there is no reference level
        if coding === :full_dummy && reference_level !== nothing
            throw(
                ArgumentError(
                    "A CategoricalPredictor with the coding :full_dummy has a coefficient for every level and no reference level, but it was given the reference level $(repr(reference_level)).",
                ),
            )
        end
        predictor_name = something(name, name_of_columns(columns_tuple))
        return new{typeof(columns_tuple),typeof(prior),typeof(reference_level)}(
            columns_tuple, prior, reference_level, coding, predictor_name
        )
    end
end

"""
    check_coding(coding, description)

Throw an informative error if `coding`, the coding of the levels of a categorical predictor
given to `description`, such as `"a CategoricalPredictor"`, isn't `:dummy`, `:effects` or
`:full_dummy`. Used by the constructors of [`CategoricalPredictor`](@ref) and
[`DefaultRegressionSettings`](@ref).
"""
function check_coding(coding::Symbol, description::String)
    coding in (:dummy, :effects, :full_dummy) && return nothing
    return throw(
        ArgumentError(
            "The coding of $description is :dummy, where each coefficient is the difference between a level and the reference level, :effects, where each is the difference between a level and the mean of the levels, or :full_dummy, where every level has a coefficient, but it was given $(repr(coding)).",
        ),
    )
end

"""
    Interaction(predictors, prior=nothing; name)

An interaction of the predictors in `predictors`, a tuple of two or more
[`ContinuousPredictor`](@ref)s or [`CategoricalPredictor`](@ref)s of the same
[`RegressionParameter`](@ref) or [`VaryingEffects`](@ref), each given by its name or itself.
A predictor given by its name, such as `:age`, is a term there itself, with coefficients of
its own, which are its main effect. A predictor given itself, such as
`ContinuousPredictor(:age)`, is part of the interaction only, which then comes without that
main effect: it has no coefficients of its own, so it is given without a prior, and a
categorical one has a coding and a reference level of its own. With
`Interaction((ContinuousPredictor(:age), CategoricalPredictor(:diagnosis; coding = :full_dummy)))`,
each diagnosis has its own slope of the age.

The interaction has a coefficient for each combination of one coefficient of each
predictor, which is multiplied by the product of their values, as the predictors give them
after coding the levels, with `prior` as its prior, such as `Normal(0, 0.5)`, or with
`nothing` the regression's default prior for an interaction, from its
[`DefaultRegressionPriors`](@ref). Two continuous predictors give one coefficient, a
continuous and a categorical predictor one for each coefficient of the categorical
predictor, and two categorical predictors one for each combination of their coefficients.
In `VaryingEffects` it gives varying slopes, and `prior` is the prior of the standard
deviation of their deviations.

The interaction is named `name`. By default, that is the names of its predictors joined by
`_by_`, as in `diagnosis_by_condition`. Each of its coefficients is named after the
coefficients it multiplies, joined by `_by_`, such as
`diagnosis_clinical_by_condition_punishment`, whatever the interaction itself is named. Each
argument is kept in a field of the same name, with `predictors` as a tuple.

# Examples
```jldoctest
julia> using Distributions: Normal

julia> Interaction((:diagnosis, :condition), Normal(0, 0.5)).name
:diagnosis_by_condition

julia> Interaction((ContinuousPredictor(:age), :diagnosis), Normal(0, 0.5)).name
:age_by_diagnosis
```
"""
struct Interaction{Tpredictors<:Tuple,Tprior} <: AbstractRegressionTerm
    predictors::Tpredictors
    prior::Tprior
    name::Symbol

    #Takes two or more different predictors, each by its name or itself, and the prior, and is named after the predictors unless another name is given
    function Interaction(predictors, prior=nothing; name::Union{Nothing,Symbol}=nothing)
        predictors_tuple = as_tuple(predictors)
        are_predictors = all(
            predictor ->
                predictor isa Union{Symbol,ContinuousPredictor,CategoricalPredictor},
            predictors_tuple,
        )
        if !are_predictors ||
            length(predictors_tuple) < 2 ||
            !allunique(map(name_of_predictor, predictors_tuple))
            throw(
                ArgumentError(
                    "An Interaction multiplies two or more different predictors, each given by its name or itself, such as Interaction((:diagnosis, :condition), Normal(0, 0.5)), but it was given $(repr(predictors)).",
                ),
            )
        end

        #A predictor that is given itself has no coefficients of its own, and so no prior
        foreach(check_has_no_prior_of_its_own, predictors_tuple)
        interaction_name = something(
            name, Symbol(join(map(name_of_predictor, predictors_tuple), "_by_"))
        )
        return new{typeof(predictors_tuple),typeof(prior)}(
            predictors_tuple, prior, interaction_name
        )
    end
end

"""
    name_of_predictor(predictor)

Return the name of `predictor`, a predictor given to an [`Interaction`](@ref): the name
itself for one given by its name, and the name of a [`ContinuousPredictor`](@ref) or a
[`CategoricalPredictor`](@ref) that is given itself. Used by the constructor of
`Interaction`.
"""
name_of_predictor(name::Symbol) = name
function name_of_predictor(predictor::Union{ContinuousPredictor,CategoricalPredictor})
    return predictor.name
end

"""
    check_has_no_prior_of_its_own(predictor)

Throw an informative error if `predictor`, a predictor given to an [`Interaction`](@ref)
itself, has a prior, since it has no coefficients of its own there. A predictor given by its
name passes. Used by the constructor of `Interaction`.
"""
check_has_no_prior_of_its_own(name::Symbol) = nothing
function check_has_no_prior_of_its_own(
    predictor::Union{ContinuousPredictor,CategoricalPredictor}
)
    predictor.prior === nothing && return nothing
    return throw(
        ArgumentError(
            "The predictor $(predictor.name) was given to an Interaction with the prior $(repr(predictor.prior)), but a predictor that is given to an interaction itself has no coefficients of its own, and so no prior. The prior of the interaction comes after its predictors, as in Interaction((ContinuousPredictor(:age), :diagnosis), Normal(0, 0.5)).",
        ),
    )
end

"""
    VaryingEffects(columns, terms...; group_by=nothing, centered=nothing, name)

Effects of a [`RegressionParameter`](@ref) that vary between the levels of `columns`, such
as between participants: for each level, a deviation of each coefficient of its `terms`,
from a normal distribution around zero whose standard deviation has the term's prior as its
prior. Varying effects are also called random effects, and the other terms of a regression
fixed effects. The levels are the values of `columns`, a column of the table that the
sessions were split from, or the combinations of the values of several columns given as a
tuple. The terms are an [`Intercept`](@ref), for a varying intercept, and
[`ContinuousPredictor`](@ref)s, [`CategoricalPredictor`](@ref)s and [`Interaction`](@ref)s,
for varying slopes, as in the examples below. A term without a prior has the regression's
default prior for a varying intercept or a varying slope, from its
[`DefaultRegressionPriors`](@ref). Each coefficient has a standard deviation of its own. A
varying slope needs values that differ between the sessions of a level, such as a condition
that changes between the blocks of a participant, and a predictor that an interaction in
varying effects names is a term of the same varying effects.

# Keyword arguments
- `group_by=nothing`: a column of the table that the sessions were split from, or a tuple of
  columns, whose values group the levels, so that each group has a standard deviation of
  its own, such as `:diagnosis` for a clinical and a non-clinical group of participants. The
  sessions of one level must all be in one group. With `nothing`, the levels share one
  standard deviation.
- `centered=nothing`: whether each deviation is sampled from the normal distribution
  directly, with `true`, or as a standardized value which the standard deviation scales,
  with `false`, or with `nothing` as the regression's [`DefaultRegressionSettings`](@ref)
  say, which is `false` unless they say otherwise. The standardized form usually samples
  better when the data informs each deviation little.
- `name`: the name of the varying effects, which their variables are named under: the name
  of their column, or the names of their columns joined by two underscores, as in
  `id__block`.

Each argument is kept in a field of the same name, with `columns` and `group_by` as tuples
of names, `group_by` being empty without groups, and `terms` as a `NamedTuple` with each
term under its name.

# Examples
```jldoctest
julia> using Distributions: Exponential

julia> by_participant = VaryingEffects(:participant, Intercept(Exponential(1)), ContinuousPredictor(:time, Exponential(0.5)); group_by = :diagnosis);

julia> keys(by_participant.terms), by_participant.group_by
((:intercept, :time), (:diagnosis,))

julia> VaryingEffects((:id, :block), Intercept(Exponential(1))).name
:id__block
```
"""
struct VaryingEffects{
    Tcolumns<:Tuple{Vararg{Symbol}},Tterms<:NamedTuple,Tgroup_by<:Tuple{Vararg{Symbol}}
} <: AbstractRegressionTerm
    columns::Tcolumns
    terms::Tterms
    group_by::Tgroup_by
    centered::Union{Nothing,Bool}
    name::Symbol

    #Checks its terms, which take the priors of their standard deviations, keeps them under their names, and is named after its columns unless another name is given
    function VaryingEffects(
        columns,
        terms...;
        group_by=nothing,
        centered::Union{Nothing,Bool}=nothing,
        name::Union{Nothing,Symbol}=nothing,
    )
        columns_tuple = as_tuple(columns)::Tuple{Vararg{Symbol}}
        check_has_columns(columns_tuple, "VaryingEffects")
        varying_effects_name = something(name, name_of_columns(columns_tuple))
        description = "the varying effects $varying_effects_name"

        #They need terms, which can't be varying effects themselves
        if isempty(terms)
            throw(
                ArgumentError(
                    "The varying effects $varying_effects_name have no terms. VaryingEffects takes the terms whose coefficients deviate for each level, such as VaryingEffects(:participant, Intercept(Exponential(1))).",
                ),
            )
        end
        if any(term -> term isa VaryingEffects, terms)
            throw(
                ArgumentError(
                    "The varying effects $varying_effects_name have VaryingEffects among their terms, but the terms of varying effects are an Intercept, ContinuousPredictors, CategoricalPredictors and Interactions.",
                ),
            )
        end
        terms_by_name = collect_terms_by_name(terms, description)
        columns_to_group_by =
            group_by === nothing ? () : as_tuple(group_by)::Tuple{Vararg{Symbol}}
        return new{typeof(columns_tuple),typeof(terms_by_name),typeof(columns_to_group_by)}(
            columns_tuple,
            terms_by_name,
            columns_to_group_by,
            centered,
            varying_effects_name,
        )
    end
end

"""
    DefaultRegressionPriors(; intercept=nothing, continuous_predictor=nothing,
                            categorical_predictor=nothing, interaction=nothing,
                            varying_intercept=nothing, varying_slope=nothing)

The priors that a [`RegressionParameter`](@ref) gives to the terms that are given without
one, by their kind: `intercept` for an [`Intercept`](@ref), `continuous_predictor` for a
[`ContinuousPredictor`](@ref), `categorical_predictor` for a [`CategoricalPredictor`](@ref)
and `interaction` for an [`Interaction`](@ref), and for the terms of
[`VaryingEffects`](@ref) the priors of the standard deviations of their deviations,
`varying_intercept` for an `Intercept` and `varying_slope` for a predictor or an
interaction. With `nothing`, terms of that kind need a prior of their own. Each prior is
kept in a field of the same name.

# Examples
```jldoctest
julia> using Distributions: Normal, Exponential

julia> default_priors = DefaultRegressionPriors(; continuous_predictor = Normal(0, 0.5), varying_intercept = Exponential(1));

julia> learning_rate = RegressionParameter(Intercept(Normal(0, 1)), ContinuousPredictor(:age), VaryingEffects(:participant, Intercept()); default_priors);

julia> learning_rate.terms.age.prior == Normal(0, 0.5)
true
```
"""
struct DefaultRegressionPriors{
    Tintercept,
    Tcontinuous_predictor,
    Tcategorical_predictor,
    Tinteraction,
    Tvarying_intercept,
    Tvarying_slope,
}
    intercept::Tintercept
    continuous_predictor::Tcontinuous_predictor
    categorical_predictor::Tcategorical_predictor
    interaction::Tinteraction
    varying_intercept::Tvarying_intercept
    varying_slope::Tvarying_slope

    #Takes the prior of each kind of term, with no prior for a kind that isn't given one
    function DefaultRegressionPriors(;
        intercept=nothing,
        continuous_predictor=nothing,
        categorical_predictor=nothing,
        interaction=nothing,
        varying_intercept=nothing,
        varying_slope=nothing,
    )
        return new{
            typeof(intercept),
            typeof(continuous_predictor),
            typeof(categorical_predictor),
            typeof(interaction),
            typeof(varying_intercept),
            typeof(varying_slope),
        }(
            intercept,
            continuous_predictor,
            categorical_predictor,
            interaction,
            varying_intercept,
            varying_slope,
        )
    end
end

"""
    DefaultRegressionSettings(; coding=:dummy, centered=false)

The settings that a [`RegressionParameter`](@ref) gives to the terms that are given without
them: `coding`, the coding of the levels of a [`CategoricalPredictor`](@ref), which is
`:dummy`, `:effects` or `:full_dummy`, and `centered`, whether the deviations of
[`VaryingEffects`](@ref) are sampled from their normal distribution directly. A term's own
setting is used when it has one. Each setting is kept in a field of the same name.

# Examples
```jldoctest
julia> using Distributions: Normal, Exponential

julia> default_settings = DefaultRegressionSettings(; coding = :effects, centered = true);

julia> learning_rate = RegressionParameter(Intercept(Normal(0, 1)), CategoricalPredictor(:diagnosis, Normal(0, 1)), VaryingEffects(:participant, Intercept(Exponential(1))); default_settings);

julia> learning_rate.terms.diagnosis.coding, learning_rate.terms.participant.centered
(:effects, true)
```
"""
struct DefaultRegressionSettings
    coding::Symbol
    centered::Bool

    #Takes the coding of categorical predictors and whether varying effects are centered, and checks the coding
    function DefaultRegressionSettings(; coding::Symbol=:dummy, centered::Bool=false)
        check_coding(coding, "a DefaultRegressionSettings")
        return new(coding, centered)
    end
end

"""
    RegressionParameter(terms...; transformation=identity,
                        default_priors=DefaultRegressionPriors(),
                        default_settings=DefaultRegressionSettings(), name=nothing)
    RegressionParameter(formula, terms...; categorical_coding=(;),
                        categorical_reference_levels=(;), priors=(;),
                        transformation=identity,
                        default_priors=DefaultRegressionPriors(),
                        default_settings=DefaultRegressionSettings())

How one parameter of a [`HierarchicalRegressionParameters`](@ref) is given to every
session: by a regression on columns of the table that the sessions were split from, whose
`terms` are an [`Intercept`](@ref), [`ContinuousPredictor`](@ref)s,
[`CategoricalPredictor`](@ref)s, [`Interaction`](@ref)s and [`VaryingEffects`](@ref). The
value of a session is the sum of each coefficient of the intercept, the predictors and the
interactions times its value in the session, and of the deviations of the session's levels
of the varying effects, each times its value in the session, with `transformation` applied
to the sum. Each session must have one value of every column that is used, which isn't
missing. The second form writes the terms as a formula, see below.

A regression commonly has varying effects whose levels are the sessions, over the column or
columns that identify them, such as `VaryingEffects(:id, Intercept(Exponential(1)))`, which
give each session a value of its own. Without them, sessions with the same values in every
column that is used have the same value of the parameter.

Every term has a name of its own, which its coefficients or its variables are named after.
A term is named after its column or columns, and an interaction after its predictors, unless
it is given another name.

# Keyword arguments
- `transformation=identity`: a function applied to the sum for each session, such as `exp`
  for a parameter that must be positive or `logistic` for one between 0 and 1, as an
  inverse link function does in a regression.
- `default_priors=DefaultRegressionPriors()`: the priors of the terms that are given
  without one, by their kind, as a [`DefaultRegressionPriors`](@ref).
- `default_settings=DefaultRegressionSettings()`: the coding of the categorical predictors
  and whether varying effects are centered, for the terms that are given without them, as a
  [`DefaultRegressionSettings`](@ref).
- `name=nothing`: the name of the parameter, which a
  [`HierarchicalRegressionParameters`](@ref) checks against the name the regression
  is given under, or `nothing` for no check. A formula gives it.

The terms are kept in the field `terms`, as a `NamedTuple` with each term under its name,
those without a prior with the one from `default_priors`, and those without a coding or a
choice of centering with the ones from `default_settings`, and the transformation and the
name in fields of the same name.

# Formulas
With StatsModels.jl loaded, the terms can be written as a `@formula`, in the syntax of
MixedModels.jl, such as
`@formula(learning_rate ~ 1 + age + diagnosis + (1 + condition | participant))`. Its
left-hand side is the name of the parameter. An intercept is included unless the formula has
`0 +`, and the same holds within varying effects. Columns joined by `&` are an
`Interaction`, and `a * b` gives `a`, `b` and their interaction. A column of an interaction
that isn't a term of its own in the same sum is given to the `Interaction` itself, so that
`age & diagnosis` on its own is the interaction without the main effects.
`(terms | columns)` are `VaryingEffects` over `columns`, or over the combinations of the
values of several columns joined by `&`, as in `(1 | participant & block)`, and their
deviations are independent, where MixedModels.jl correlates those within one bar. Terms
that a formula can't express, such as `VaryingEffects` with `group_by`, are given after the
formula.

A column is a `ContinuousPredictor`, unless `categorical_coding` or
`categorical_reference_levels` names it, which makes it a `CategoricalPredictor`. Both are
`NamedTuple`s from the names of columns: `categorical_coding` to the coding of each, such
as `(; diagnosis = :effects, site = :dummy)`, and `categorical_reference_levels` to the
reference level of each, such as `(; diagnosis = "control")`, as
[`CategoricalPredictor`](@ref) takes them. A column that only
`categorical_reference_levels` names has the coding of `default_settings`, which `nothing`
as its coding gives too. A column that only `categorical_coding` names has the level of the
first session as its reference level, where StatsModels.jl takes the first level in sorted
order.

Each term is named as the term of its kind is named, and `priors` gives the priors of terms
by those names, such as
`(; age = Normal(0, 0.05), age_by_diagnosis = Normal(0, 0.5), participant = (; intercept = Exponential(0.5)))`,
with the name of an interaction joining those of its predictors by `_by_`, and with a
`NamedTuple` for the terms of varying effects. The other terms have the default priors.

# Examples
```jldoctest
julia> using Distributions: Normal, Exponential

julia> default_priors = DefaultRegressionPriors(; intercept = Normal(0, 1), continuous_predictor = Normal(0, 0.5), categorical_predictor = Normal(0, 1), interaction = Normal(0, 0.5), varying_intercept = Exponential(1));

julia> learning_rate = RegressionParameter(Intercept(), ContinuousPredictor(:age), CategoricalPredictor(:diagnosis), Interaction((:age, :diagnosis)), VaryingEffects(:participant, Intercept()); default_priors);

julia> keys(learning_rate.terms)
(:intercept, :age, :diagnosis, :age_by_diagnosis, :participant)
```

A regression written as a formula, with varying intercepts of the participants whose
standard deviations differ between diagnoses, which a formula can't express, after it:
```julia
using StatsModels

learning_rate = RegressionParameter(
    @formula(learning_rate ~ 1 + age + diagnosis + age & diagnosis),
    VaryingEffects(:participant, Intercept(); group_by = :diagnosis);
    categorical_reference_levels = (; diagnosis = "control"),
    priors = (; age = Normal(0, 0.05), age_by_diagnosis = Normal(0, 0.5)),
    default_priors,
)
```
"""
struct RegressionParameter{Tterms<:NamedTuple,Ttransformation}
    terms::Tterms
    transformation::Ttransformation
    name::Union{Nothing,Symbol}

    #Gives the default priors and settings to the terms without them, checks the terms and keeps them under their names
    function RegressionParameter(
        terms...;
        transformation=identity,
        default_priors::DefaultRegressionPriors=DefaultRegressionPriors(),
        default_settings::DefaultRegressionSettings=DefaultRegressionSettings(),
        name::Union{Nothing,Symbol}=nothing,
    )

        #A regression needs terms
        if isempty(terms)
            throw(
                ArgumentError(
                    "A RegressionParameter takes the terms of the regression, such as RegressionParameter(Intercept(Normal(0, 1)), VaryingEffects(:participant, Intercept(Exponential(1)))), but it was given none.",
                ),
            )
        end

        #The terms without a prior take the default prior of their kind, and those without a coding or a choice of centering the default ones
        terms_with_priors = map(term -> with_default_prior(term, default_priors), terms)
        terms_with_settings = map(
            term -> with_default_settings(term, default_settings), terms_with_priors
        )

        #Every term needs a name of its own, and every coefficient a prior
        terms_by_name = collect_terms_by_name(terms_with_settings, "a RegressionParameter")
        foreach(check_has_prior, terms_with_settings)

        #The coefficients are named coef under the parameter's name, so varying effects can't take that name
        if haskey(terms_by_name, :coef) && terms_by_name.coef isa VaryingEffects
            throw(
                ArgumentError(
                    "VaryingEffects of a RegressionParameter can't be named coef, which is the name of the coefficients of the regression. Give them another name, as in VaryingEffects(:coef, Intercept(Exponential(1)); name = :by_coef).",
                ),
            )
        end
        return new{typeof(terms_by_name),typeof(transformation)}(
            terms_by_name, transformation, name
        )
    end
end

"""
    HierarchicalRegressionParameters(parameters)

The parameter model of a full model which gives each parameter of every session by
a regression on columns of the table that the sessions were split from, such as the age and
the diagnosis of a participant, with varying effects, which are also called random effects.
`parameters` is a `NamedTuple` with a
[`RegressionParameter`](@ref) for each parameter that is fitted, which has the terms of its
regression and their priors. It is kept in the field `parameters`. Useful for estimating the
effect of a predictor on a parameter, or the difference between groups, together with the
parameters of the sessions.

A full model builds the design of every regression once, before it is made, from the table
that [`MultipleSessionsInfo`](@ref) keeps, so sessions that were prepared one at a time can
only have an intercept. The field `designs` is `nothing` until then, and afterwards has, for
each parameter:
- `coefficient_priors` and `design_columns`: the prior of each coefficient of the intercept,
  the predictors and the interactions, and the values it is multiplied by, one for every
  session, under the name of the coefficient: 1 for the intercept, the values of a
  continuous predictor, and those of the coefficients of a categorical predictor and of an
  interaction, as [`CategoricalPredictor`](@ref) and [`Interaction`](@ref) describe them.
- `reference_levels`: the reference level of each categorical predictor, which is `nothing`
  with the coding `:full_dummy`, and under the name of an interaction those of the
  categorical predictors it was given themselves, in a `NamedTuple`.
- `varying_effects`: for the varying effects of each column, the prior of the standard
  deviation of the deviations of each of their coefficients, `coefficient_sd_priors`, which
  is a `NamedTuple` with the prior under the symbol name of each group when the levels are
  grouped, the values that the deviations are multiplied by, `design_columns`, the
  `reference_levels` of their predictors, whether they are `centered`, and `levels`. These
  have the level of every session, `level_of_each_session`, the number of levels,
  `n_levels`, the symbol name of each level, `level_symbol_names`, and `groups`, which is
  `nothing` without groups and otherwise has the group of every level,
  `group_of_each_level`, the number of groups, `n_groups`, and the symbol name of each
  group, `group_symbol_names`.

# Examples
```julia
using Turing
using LogExpFunctions: logistic

parameters_model = HierarchicalRegressionParameters((;
    learning_rate = RegressionParameter(
        Intercept(Normal(0, 1)),
        ContinuousPredictor(:age, Normal(0, 0.5)),
        CategoricalPredictor(:diagnosis, Normal(0, 1); reference_level = "control"),
        CategoricalPredictor(:condition, Normal(0, 1)),
        Interaction((:diagnosis, :condition), Normal(0, 0.5)),
        VaryingEffects(
            :participant,
            Intercept(Exponential(1)),
            CategoricalPredictor(:condition, Exponential(0.5));
            group_by = :diagnosis,
        ),
        VaryingEffects((:id, :block), Intercept(Exponential(1)); name = :session);
        transformation = logistic,
    ),
    inverse_temperature = RegressionParameter(
        Intercept(Normal(0, 1)),
        VaryingEffects((:id, :block), Intercept(Exponential(1)); name = :session);
        transformation = exp,
    ),
))
```
"""
struct HierarchicalRegressionParameters{
    Tparameters<:NamedTuple,Tdesigns<:Union{Nothing,NamedTuple}
} <: AbstractParametersModel
    parameters::Tparameters
    designs::Tdesigns

    #Takes a RegressionParameter for each parameter. The designs of the regressions aren't built yet
    function HierarchicalRegressionParameters(parameters::NamedTuple)
        foreach(check_is_regression_parameter, keys(parameters), values(parameters))
        return new{typeof(parameters),Nothing}(parameters, nothing)
    end

    #The same model with the designs of its regressions for the sessions it is prepared for
    function HierarchicalRegressionParameters(
        parameters_model::HierarchicalRegressionParameters, designs::NamedTuple
    )
        parameters = parameters_model.parameters
        return new{typeof(parameters),typeof(designs)}(parameters, designs)
    end
end

#It gives the parameters it has a RegressionParameter for
function parameter_names(parameters_model::HierarchicalRegressionParameters)
    return keys(parameters_model.parameters)
end

#The design of every regression is built once, when a full model is prepared, and kept in the field for the designs
function prepare_parameters_model(
    parameters_model::HierarchicalRegressionParameters, sessions_info::MultipleSessionsInfo
)
    names = keys(parameters_model.parameters)
    designs = NamedTuple{names}(
        map(
            (name, regression_parameter) ->
                build_regression_design(sessions_info, regression_parameter, name),
            names,
            values(parameters_model.parameters),
        ),
    )
    return HierarchicalRegressionParameters(parameters_model, designs)
end

"""
    check_is_regression_parameter(name, regression_parameter)

Throw an informative error if what a [`HierarchicalRegressionParameters`](@ref) was
given for the parameter `name` isn't a [`RegressionParameter`](@ref), or is one named after
another parameter, as by the left-hand side of its formula. Used by the constructor of
`HierarchicalRegressionParameters`.
"""
function check_is_regression_parameter(
    name::Symbol, regression_parameter::RegressionParameter
)
    regression_parameter.name === nothing && return nothing
    regression_parameter.name === name && return nothing
    return throw(
        ArgumentError(
            "HierarchicalRegressionParameters was given the regression of $(regression_parameter.name) for the parameter $name. The name of a RegressionParameter, which the left-hand side of its formula gives, must be the name it is given under.",
        ),
    )
end
function check_is_regression_parameter(name::Symbol, regression_parameter)
    return throw(
        ArgumentError(
            "HierarchicalRegressionParameters takes a RegressionParameter for each parameter, such as RegressionParameter(Intercept(Normal(0, 1)), VaryingEffects(:participant, Intercept(Exponential(1)))), but for $name it was given $(repr(regression_parameter)).",
        ),
    )
end

"""
    name_of_columns(columns)

Return the name of a term that reads `columns`, a tuple of names: the name of a single
column, or the names of several joined by two underscores, as the parts of the symbol name
of a session are, such as `id__block`. Used by the constructors of
[`CategoricalPredictor`](@ref) and [`VaryingEffects`](@ref).
"""
name_of_columns(columns::Tuple{Symbol}) = only(columns)
name_of_columns(columns::Tuple{Vararg{Symbol}}) = Symbol(join(columns, "__"))

"""
    check_has_columns(columns, description)

Throw an informative error if `columns`, the tuple of columns given to a term, is empty.
`description` names the kind of term, such as `"A CategoricalPredictor"`. Used by the
constructors of [`CategoricalPredictor`](@ref) and [`VaryingEffects`](@ref).
"""
function check_has_columns(columns::Tuple{Vararg{Symbol}}, description::String)
    isempty(columns) || return nothing
    return throw(
        ArgumentError(
            "$description reads one column or several, such as :diagnosis or (:diagnosis, :site), but it was given none.",
        ),
    )
end

"""
    with_default_prior(term, default_priors)

Return `term`, a term given to a [`RegressionParameter`](@ref), with the prior of its kind
in `default_priors`, a [`DefaultRegressionPriors`](@ref), when it has no prior, and
[`VaryingEffects`](@ref) with the prior of the standard deviation of each of their terms
that has none. Terms with a prior are returned as they are, and so is anything that isn't a
term, for the constructor to report. Used by the constructor of `RegressionParameter`.
"""
with_default_prior(term, ::DefaultRegressionPriors) = term
function with_default_prior(
    term::Union{Intercept,ContinuousPredictor,CategoricalPredictor,Interaction},
    default_priors::DefaultRegressionPriors,
)
    term.prior === nothing || return term
    return with_prior(term, getfield(default_priors, default_prior_field(term)))
end
function with_default_prior(
    varying_effects::VaryingEffects, default_priors::DefaultRegressionPriors
)
    terms = map(values(varying_effects.terms)) do term
        term.prior === nothing || return term
        return with_prior(term, getfield(default_priors, varying_default_prior_field(term)))
    end
    return VaryingEffects(
        varying_effects.columns,
        terms...;
        group_by=varying_effects.group_by,
        centered=varying_effects.centered,
        name=varying_effects.name,
    )
end

"""
    with_default_settings(term, default_settings)

Return `term`, a term given to a [`RegressionParameter`](@ref), with the settings in
`default_settings`, a [`DefaultRegressionSettings`](@ref), that it was given without: the
coding of a [`CategoricalPredictor`](@ref), also of one that an [`Interaction`](@ref) was
given itself, and for [`VaryingEffects`](@ref) whether they are centered and the coding of
their categorical predictors. A term's own settings are kept, and anything else is returned
as it is. A reference level is an error when the default coding is `:full_dummy`, which has
none. Used by the constructor of `RegressionParameter`.
"""
with_default_settings(term, ::DefaultRegressionSettings) = term
function with_default_settings(
    predictor::CategoricalPredictor, default_settings::DefaultRegressionSettings
)
    predictor.coding === nothing || return predictor

    #The coding with a coefficient for every level has no reference level
    if default_settings.coding === :full_dummy && predictor.reference_level !== nothing
        throw(
            ArgumentError(
                "The categorical predictor $(predictor.name) has the reference level $(repr(predictor.reference_level)), but it takes the coding :full_dummy from the DefaultRegressionSettings, which has a coefficient for every level and no reference level. Give the predictor a coding of its own, such as coding = :dummy.",
            ),
        )
    end
    return CategoricalPredictor(
        predictor.columns,
        predictor.prior;
        reference_level=predictor.reference_level,
        coding=default_settings.coding,
        name=predictor.name,
    )
end
function with_default_settings(
    interaction::Interaction, default_settings::DefaultRegressionSettings
)
    predictors = map(
        predictor -> with_default_settings(predictor, default_settings),
        interaction.predictors,
    )
    return Interaction(predictors, interaction.prior; name=interaction.name)
end
function with_default_settings(
    varying_effects::VaryingEffects, default_settings::DefaultRegressionSettings
)
    terms = map(
        term -> with_default_settings(term, default_settings), values(varying_effects.terms)
    )
    return VaryingEffects(
        varying_effects.columns,
        terms...;
        group_by=varying_effects.group_by,
        centered=something(varying_effects.centered, default_settings.centered),
        name=varying_effects.name,
    )
end

"""
    default_prior_field(term)

Return the name of the field of a [`DefaultRegressionPriors`](@ref) with the default prior
of `term`, a term of a [`RegressionParameter`](@ref) other than [`VaryingEffects`](@ref),
such as `:continuous_predictor` for a [`ContinuousPredictor`](@ref). Used by
[`with_default_prior`](@ref) and [`check_has_prior`](@ref).
"""
default_prior_field(::Intercept) = :intercept
default_prior_field(::ContinuousPredictor) = :continuous_predictor
default_prior_field(::CategoricalPredictor) = :categorical_predictor
default_prior_field(::Interaction) = :interaction

"""
    varying_default_prior_field(term)

Return the name of the field of a [`DefaultRegressionPriors`](@ref) with the default prior
of the standard deviation of `term`, a term of [`VaryingEffects`](@ref):
`:varying_intercept` for an [`Intercept`](@ref), and `:varying_slope` for a predictor or an
[`Interaction`](@ref). Used by [`with_default_prior`](@ref) and [`check_has_prior`](@ref).
"""
varying_default_prior_field(::Intercept) = :varying_intercept
function varying_default_prior_field(
    ::Union{ContinuousPredictor,CategoricalPredictor,Interaction}
)
    return :varying_slope
end

"""
    check_has_prior(term)

Throw an informative error if `term`, a term of a [`RegressionParameter`](@ref) after it
was given the default priors, has no prior, or if a term of [`VaryingEffects`](@ref) has
none, naming the field of [`DefaultRegressionPriors`](@ref) that would give it one. Used by
the constructor of `RegressionParameter`.
"""
function check_has_prior(
    term::Union{Intercept,ContinuousPredictor,CategoricalPredictor,Interaction}
)
    term.prior === nothing || return nothing
    field = default_prior_field(term)
    return throw(
        ArgumentError(
            "The term $(term.name) of a RegressionParameter has no prior. Give it one, as in ContinuousPredictor(:age, Normal(0, 1)), or give the RegressionParameter default_priors = DefaultRegressionPriors(; $field = Normal(0, 1)) for every term of its kind without one.",
        ),
    )
end
function check_has_prior(varying_effects::VaryingEffects)
    for term in values(varying_effects.terms)
        term.prior === nothing || continue
        field = varying_default_prior_field(term)
        throw(
            ArgumentError(
                "The term $(term.name) of the varying effects $(varying_effects.name) has no prior. Give it the prior of the standard deviation of its deviations, as in Intercept(Exponential(1)), or give the RegressionParameter default_priors = DefaultRegressionPriors(; $field = Exponential(1)) for every term of its kind without one.",
            ),
        )
    end
    return nothing
end

"""
    with_prior(term, prior)

Return `term` with `prior` as its prior, and its other settings as they are. Used by
[`with_default_prior`](@ref).
"""
with_prior(term::Intercept, prior) = Intercept(prior; name=term.name)
function with_prior(term::ContinuousPredictor, prior)
    return ContinuousPredictor(term.column, prior; name=term.name)
end
function with_prior(term::CategoricalPredictor, prior)
    return CategoricalPredictor(
        term.columns,
        prior;
        reference_level=term.reference_level,
        coding=term.coding,
        name=term.name,
    )
end
function with_prior(term::Interaction, prior)
    return Interaction(term.predictors, prior; name=term.name)
end

"""
    collect_terms_by_name(terms, description)

Return `terms`, the terms given to a [`RegressionParameter`](@ref) or to
[`VaryingEffects`](@ref), as a `NamedTuple` with each term under its name, after checking
that each is an [`AbstractRegressionTerm`](@ref), that their names differ, that there is one
[`Intercept`](@ref) at most, and that each [`Interaction`](@ref) multiplies predictors among
them, or ones it was given themselves. `description` says what was given the terms, such as
`"a RegressionParameter"`, for the errors. Used by the constructors of `RegressionParameter`
and `VaryingEffects`.
"""
function collect_terms_by_name(terms::Tuple, description::String)

    #Each term must be one of the kinds of terms
    for term in terms
        if !(term isa AbstractRegressionTerm)
            throw(
                ArgumentError(
                    "The terms given to $description include $(repr(term)), which isn't a term of a regression. The terms are an Intercept, ContinuousPredictors, CategoricalPredictors, Interactions and VaryingEffects, such as ContinuousPredictor(:age, Normal(0, 1)).",
                ),
            )
        end
    end

    #Every term needs a name of its own, and there is one intercept at most
    names = map(term -> term.name, terms)
    check_unique_term_names(names, description)
    if count(term -> term isa Intercept, terms) > 1
        throw(
            ArgumentError(
                "The terms of $description include more than one Intercept, but a regression has one intercept at most.",
            ),
        )
    end
    terms_by_name = NamedTuple{names}(terms)

    #An interaction multiplies predictors among the terms
    for term in terms
        term isa Interaction && check_interaction_terms(term, terms_by_name, description)
    end
    return terms_by_name
end

"""
    check_unique_term_names(term_names, description)

Throw an informative error if two terms in `term_names`, the names of the terms given to
`description`, such as `"a RegressionParameter"`, are the same. Used by
[`collect_terms_by_name`](@ref).
"""
function check_unique_term_names(term_names, description::String)
    allunique(term_names) && return nothing
    repeated_names = unique(name for name in term_names if count(==(name), term_names) > 1)
    return throw(
        ArgumentError(
            "The terms of $description must each have a name of their own, but more than one is named $(join(repeated_names, ", ")). A term is named after its column or columns, and an interaction after its predictors, unless it is given another name, as in ContinuousPredictor(:age; name = :years).",
        ),
    )
end

"""
    check_interaction_terms(interaction, terms_by_name, description)

Throw an informative error if `interaction`, an [`Interaction`](@ref), multiplies a
predictor given by its name that isn't in `terms_by_name`, the terms given to `description`
under their names, or that isn't a [`ContinuousPredictor`](@ref) or a
[`CategoricalPredictor`](@ref) there. A predictor that the interaction was given itself
needs no term. Used by [`collect_terms_by_name`](@ref).
"""
function check_interaction_terms(
    interaction::Interaction, terms_by_name::NamedTuple, description::String
)
    for name in interaction.predictors
        name isa Symbol || continue
        if !haskey(terms_by_name, name)
            throw(
                ArgumentError(
                    "The interaction $(interaction.name) of $description multiplies $name, which isn't one of its terms. Its terms are $(keys(terms_by_name)). A predictor without a term of its own is given to the interaction itself, as in Interaction((ContinuousPredictor(:age), :diagnosis)).",
                ),
            )
        end
        term = getproperty(terms_by_name, name)
        if !(term isa Union{ContinuousPredictor,CategoricalPredictor})
            throw(
                ArgumentError(
                    "The interaction $(interaction.name) of $description multiplies $name, which is of type $(nameof(typeof(term))), but an interaction multiplies ContinuousPredictors and CategoricalPredictors.",
                ),
            )
        end
    end
    return nothing
end

"""
    describe_columns(columns)

Return how `columns`, a tuple of names, is named in an error message: as `the column age`
for one, and as `the columns (:diagnosis, :site)` for several. Used by the errors of
[`build_contrast_columns`](@ref).
"""
describe_columns(columns::Tuple{Symbol}) = "the column $(only(columns))"
describe_columns(columns::Tuple{Vararg{Symbol}}) = "the columns $columns"

"""
    select_terms(predicate, terms)

Return the terms in `terms`, a `NamedTuple` with each term under its name, for which
`predicate` is `true`, in a `NamedTuple` of the same form. Used by
[`build_regression_design`](@ref).
"""
function select_terms(predicate, terms::NamedTuple)
    names = Tuple(name for (name, term) in pairs(terms) if predicate(term))
    return NamedTuple{names}(map(name -> getproperty(terms, name), names))
end

"""
    build_regression_design(sessions_info, regression_parameter, parameter_name)

Return the design of the regression of the parameter `parameter_name` for the sessions in
`sessions_info`, as [`HierarchicalRegressionParameters`](@ref) keeps it in its field
`designs`: the prior and the values of each coefficient of the intercept, the predictors and
the interactions, from [`build_coefficients`](@ref), and the design of the varying effects
of each column, from [`build_varying_effects_design`](@ref). Used by
[`prepare_parameters_model`](@ref).
"""
function build_regression_design(
    sessions_info::MultipleSessionsInfo,
    regression_parameter::RegressionParameter,
    parameter_name::Symbol,
)

    #The coefficients of the intercept, the predictors and the interactions, with the values that each is multiplied by
    non_varying_terms = select_terms(
        term -> !(term isa VaryingEffects), regression_parameter.terms
    )
    coefficients = build_coefficients(
        sessions_info,
        non_varying_terms,
        parameter_name,
        "the regression of $parameter_name",
    )

    #The level of every session for the varying effects of each column, and the values that their deviations are multiplied by
    varying_effects = select_terms(
        term -> term isa VaryingEffects, regression_parameter.terms
    )
    varying_effects_designs = map(
        varying_effects_of_column -> build_varying_effects_design(
            sessions_info, varying_effects_of_column, parameter_name
        ),
        varying_effects,
    )

    return (;
        coefficient_priors=coefficients.priors,
        design_columns=coefficients.columns,
        reference_levels=coefficients.reference_levels,
        varying_effects=varying_effects_designs,
    )
end

"""
    build_coefficients(sessions_info, terms, parameter_name, description)

Return the coefficients of `terms`, a `NamedTuple` with an intercept, predictors and
interactions of `description`, such as `"the regression of learning_rate"`, for the
sessions in `sessions_info`, as a `NamedTuple` with the prior of each coefficient, `priors`,
and the values it is multiplied by in every session, `columns`, both under the name of the
coefficient and in the order of the terms, and with the reference level of each categorical
predictor, `reference_levels`, under the names of the predictors, and those of the
categorical predictors that an interaction was given themselves in a `NamedTuple` under the
name of the interaction. The coefficients of each term come from
[`build_term_columns`](@ref) and [`build_interaction_columns`](@ref), and must have names of
their own. Used by [`build_regression_design`](@ref) and
[`build_varying_effects_design`](@ref).
"""
function build_coefficients(
    sessions_info::MultipleSessionsInfo,
    terms::NamedTuple,
    parameter_name::Symbol,
    description::String,
)

    #The intercept and the predictors first, since an interaction multiplies the columns of predictors
    columns_of_predictors = Dict{Symbol,Any}()
    for (name, term) in pairs(terms)
        term isa Interaction && continue
        columns_of_predictors[name] = build_term_columns(
            sessions_info, term, parameter_name
        )
    end

    #Then every term's coefficients in the order of the terms, each with the prior of its term
    coefficient_names = Symbol[]
    coefficient_priors = Any[]
    coefficient_columns = Vector{Float64}[]
    reference_levels = Pair{Symbol,Any}[]
    for (name, term) in pairs(terms)
        columns_of_term = if term isa Interaction
            build_interaction_columns(
                sessions_info, term, columns_of_predictors, parameter_name, description
            )
        else
            columns_of_predictors[name]
        end
        append!(coefficient_names, columns_of_term.names)
        append!(coefficient_columns, columns_of_term.columns)
        append!(coefficient_priors, fill(term.prior, length(columns_of_term.names)))
        if term isa CategoricalPredictor
            push!(reference_levels, name => columns_of_term.reference_level)
        elseif term isa Interaction && !isempty(columns_of_term.reference_levels)
            push!(reference_levels, name => columns_of_term.reference_levels)
        end
    end

    #Every coefficient needs a name of its own, which it is named by
    check_unique_coefficient_names(coefficient_names, description)
    names = Tuple(coefficient_names)
    return (;
        priors=NamedTuple{names}(Tuple(coefficient_priors)),
        columns=NamedTuple{names}(Tuple(coefficient_columns)),
        reference_levels=(; reference_levels...),
    )
end

"""
    check_unique_coefficient_names(coefficient_names, description)

Throw an informative error if two coefficients in `coefficient_names`, the names of the
coefficients of `description`, such as `"the regression of learning_rate"`, are the same.
Used by [`build_coefficients`](@ref).
"""
function check_unique_coefficient_names(coefficient_names, description::String)
    allunique(coefficient_names) && return nothing
    repeated_names = unique(
        name for name in coefficient_names if count(==(name), coefficient_names) > 1
    )
    return throw(
        ArgumentError(
            "The coefficients of $description must each have a name of their own, but more than one is named $(join(repeated_names, ", ")). The coefficient of a level of a categorical predictor is named after the predictor and the level, as in diagnosis_clinical.",
        ),
    )
end

"""
    build_term_columns(sessions_info, term, parameter_name)

Return the coefficients of `term`, an [`Intercept`](@ref) or a predictor in the regression
of the parameter `parameter_name`, for the sessions in `sessions_info`, as a `NamedTuple`
with the name of each coefficient, `names`, and the values it is multiplied by in every
session, `columns`: 1 for the intercept, the values of a continuous predictor, and the
columns of the levels of a categorical predictor, from [`build_contrast_columns`](@ref),
with its `reference_level`. Used by [`build_coefficients`](@ref).
"""
function build_term_columns(
    sessions_info::MultipleSessionsInfo, term::Intercept, parameter_name::Symbol
)
    return (; names=[term.name], columns=[ones(n_sessions(sessions_info))])
end
function build_term_columns(
    sessions_info::MultipleSessionsInfo, term::ContinuousPredictor, parameter_name::Symbol
)
    values = read_predictor_values(sessions_info, term.column, parameter_name)
    return (; names=[term.name], columns=[values])
end
function build_term_columns(
    sessions_info::MultipleSessionsInfo, term::CategoricalPredictor, parameter_name::Symbol
)
    return build_contrast_columns(sessions_info, term, parameter_name)
end

"""
    read_predictor_values(sessions_info, column, parameter_name)

Return the value of the column `column` of a continuous predictor in every session, as a
vector of `Float64`, for the regression of the parameter `parameter_name`. A value that
isn't a number is an error. Used by [`build_term_columns`](@ref).
"""
function read_predictor_values(
    sessions_info::MultipleSessionsInfo, column::Symbol, parameter_name::Symbol
)
    values = read_one_value_of_each_session(
        sessions_info,
        column,
        "HierarchicalRegressionParameters predicts $parameter_name from",
    )
    for value in values
        if !(value isa Real)
            throw(
                ArgumentError(
                    "HierarchicalRegressionParameters predicts $parameter_name from the column $column as a continuous predictor, but it has the value $(repr(value)), which isn't a number. Give it as a CategoricalPredictor instead.",
                ),
            )
        end
    end
    return Float64[value for value in values]
end

"""
    build_contrast_columns(sessions_info, predictor, parameter_name)

Return the coefficients of `predictor`, a [`CategoricalPredictor`](@ref) in the regression
of the parameter `parameter_name`, in a `NamedTuple` with `names`, `columns` and
`reference_level`: for each level other than the reference level, or for every level with
the coding `:full_dummy`, in the order of their first session, the name of its coefficient
and a vector with 1 for the sessions of that level and 0 for the others. The levels are the
values of a single column, whose coefficients are named after the predictor and the level,
as in `diagnosis_clinical`, or the `NamedTuple`s of the values of several, whose
coefficients are named by the symbol name of the level alone, as in
`diagnosis_clinical__site_2`. With the coding `:effects`, the sessions of the reference
level have -1 in every column. The reference level is the predictor's, or the level of the
first session, and `nothing` with the coding `:full_dummy`. It is an error when there is one
level only, and when the reference level isn't one of them. Used by
[`build_term_columns`](@ref).
"""
function build_contrast_columns(
    sessions_info::MultipleSessionsInfo,
    predictor::CategoricalPredictor,
    parameter_name::Symbol,
)
    columns = predictor.columns
    level_of_each_session = read_level_of_each_session(
        sessions_info,
        columns,
        "HierarchicalRegressionParameters predicts $parameter_name from",
    )

    #The levels are the values of the columns, in the order of their first session
    levels = unique(level_of_each_session)
    if length(levels) < 2
        throw(
            ArgumentError(
                "HierarchicalRegressionParameters predicts $parameter_name from $(describe_columns(columns)) as a categorical predictor, but every session has the value $(repr(only(levels))), so there is no difference between levels to estimate.",
            ),
        )
    end

    #Every level has a coefficient with the coding full_dummy, and otherwise every level but the reference level, which is the one that is given or the level of the first session
    reference_level = if predictor.coding === :full_dummy
        nothing
    elseif predictor.reference_level === nothing
        first(levels)
    else
        predictor.reference_level
    end
    if reference_level !== nothing && !any(isequal(reference_level), levels)
        throw(
            ArgumentError(
                "The reference level of the categorical predictor $(predictor.name) of $parameter_name is $(repr(reference_level)), which isn't a value of $(describe_columns(columns)). Its values are $levels.",
            ),
        )
    end
    levels_with_coefficients = if reference_level === nothing
        levels
    else
        filter(!isequal(reference_level), levels)
    end

    #Each of these levels has a column with 1 for its sessions, and with -1 for the sessions of the reference level with effects coding, so that the coefficients sum to zero
    value_of_reference_level = predictor.coding === :effects ? -1.0 : 0.0
    contrast_columns = map(levels_with_coefficients) do level
        return map(level_of_each_session) do level_of_session
            if isequal(level_of_session, level)
                1.0
            elseif reference_level !== nothing && isequal(level_of_session, reference_level)
                value_of_reference_level
            else
                0.0
            end
        end
    end

    #A level of one column is named after the predictor and its value, and a level of several columns by its symbol name, which has the names of the columns
    names = map(levels_with_coefficients) do level
        if length(columns) == 1
            return Symbol(predictor.name, "_", session_id_to_symbol(level))
        else
            return session_id_to_symbol(level)
        end
    end
    return (; names, columns=contrast_columns, reference_level)
end

"""
    build_interaction_columns(sessions_info, interaction, columns_of_predictors,
                              parameter_name, description)

Return the coefficients of `interaction`, an [`Interaction`](@ref) of `description`, such as
`"the regression of learning_rate"`, as [`build_term_columns`](@ref) returns those of a
predictor: one for each combination of one coefficient of each of its predictors, named
after the coefficients it multiplies joined by `_by_`, with the product of their columns.
The coefficients of each predictor come from [`columns_of_interaction_predictor`](@ref). The
combinations are in the order of the coefficients of the first predictor, then of the
second, and so on. The reference levels of the categorical predictors that the interaction
was given themselves are returned as `reference_levels`, under their names. A column that
is zero in every session is an error, since no session informs its coefficient. Used by
[`build_coefficients`](@ref).
"""
function build_interaction_columns(
    sessions_info::MultipleSessionsInfo,
    interaction::Interaction,
    columns_of_predictors::AbstractDict,
    parameter_name::Symbol,
    description::String,
)
    columns_of_each_predictor = map(
        predictor -> columns_of_interaction_predictor(
            sessions_info, predictor, columns_of_predictors, parameter_name
        ),
        interaction.predictors,
    )

    #One coefficient for each combination of one coefficient of each predictor, with the coefficients of the first predictor changing slowest
    ranges_of_coefficients = map(
        columns -> eachindex(columns.names), columns_of_each_predictor
    )
    names = Symbol[]
    columns = Vector{Float64}[]
    for reversed_indices in Iterators.product(reverse(ranges_of_coefficients)...)
        indices = reverse(reversed_indices)
        name = Symbol(
            join(
                map(
                    (columns, index) -> columns.names[index],
                    columns_of_each_predictor,
                    indices,
                ),
                "_by_",
            ),
        )
        column = reduce(
            .*,
            map(
                (columns, index) -> columns.columns[index],
                columns_of_each_predictor,
                indices,
            ),
        )

        #A coefficient whose column is zero in every session can't be estimated
        if all(iszero, column)
            throw(
                ArgumentError(
                    "The coefficient $name of the interaction $(interaction.name) of $description is multiplied by zero in every session, such as when no session has that combination of levels, so it can't be estimated.",
                ),
            )
        end
        push!(names, name)
        push!(columns, column)
    end

    #The reference levels of the categorical predictors that the interaction was given themselves, which aren't terms
    reference_levels = Pair{Symbol,Any}[]
    for (predictor, columns_of_predictor) in
        zip(interaction.predictors, columns_of_each_predictor)
        predictor isa CategoricalPredictor || continue
        push!(reference_levels, predictor.name => columns_of_predictor.reference_level)
    end
    return (; names, columns, reference_levels=(; reference_levels...))
end

"""
    columns_of_interaction_predictor(sessions_info, predictor, columns_of_predictors,
                                     parameter_name)

Return the coefficients of `predictor`, a predictor of an [`Interaction`](@ref), as
[`build_term_columns`](@ref) returns them: those in `columns_of_predictors` under its name
for a predictor given by its name, which is a term of its own, and those that
`build_term_columns` builds for a predictor that the interaction was given itself. Used by
[`build_interaction_columns`](@ref).
"""
function columns_of_interaction_predictor(
    sessions_info::MultipleSessionsInfo,
    name::Symbol,
    columns_of_predictors::AbstractDict,
    parameter_name::Symbol,
)
    return columns_of_predictors[name]
end
function columns_of_interaction_predictor(
    sessions_info::MultipleSessionsInfo,
    predictor::Union{ContinuousPredictor,CategoricalPredictor},
    columns_of_predictors::AbstractDict,
    parameter_name::Symbol,
)
    return build_term_columns(sessions_info, predictor, parameter_name)
end

"""
    build_varying_effects_design(sessions_info, varying_effects, parameter_name)

Return the design of `varying_effects`, a [`VaryingEffects`](@ref) of the regression of the
parameter `parameter_name`, for the sessions in `sessions_info`, as
[`HierarchicalRegressionParameters`](@ref) describes it: the prior of the standard
deviation of the deviations of each coefficient of their terms, with the prior under the
symbol name of each group when the levels are grouped, the values that the deviations are
multiplied by, from [`build_coefficients`](@ref), the reference levels of their predictors,
whether they are centered, and the levels of the sessions, from
[`find_varying_effects_levels`](@ref). The values of a coefficient other than the intercept
must differ between the sessions of some level, or it is an error. Used by
[`build_regression_design`](@ref).
"""
function build_varying_effects_design(
    sessions_info::MultipleSessionsInfo,
    varying_effects::VaryingEffects,
    parameter_name::Symbol,
)
    description = "the varying effects $(varying_effects.name) of $parameter_name"

    #The level of every session, and the group of every level when the standard deviations differ between groups
    levels = find_varying_effects_levels(sessions_info, varying_effects, parameter_name)

    #The coefficients of the terms, whose deviations are multiplied by their values in each session
    coefficients = build_coefficients(
        sessions_info, varying_effects.terms, parameter_name, description
    )

    #A varying slope needs values that differ between the sessions of some level
    intercept_names = Tuple(
        term.name for term in values(varying_effects.terms) if term isa Intercept
    )
    for (name, column) in pairs(coefficients.columns)
        name in intercept_names && continue
        check_varies_within_levels(column, levels.level_of_each_session, name, description)
    end

    #With groups, the standard deviation of each coefficient has its prior under the symbol name of each group
    coefficient_sd_priors = map(
        prior -> priors_of_groups(prior, levels.groups), coefficients.priors
    )

    return (;
        coefficient_sd_priors,
        design_columns=coefficients.columns,
        reference_levels=coefficients.reference_levels,
        levels,
        centered=varying_effects.centered,
    )
end

"""
    priors_of_groups(prior, groups)

Return the prior of a parameter of a group: `prior` itself without groups, where `groups`
is `nothing`, and otherwise a `NamedTuple` with `prior` under the symbol name of each group
in `groups`, as [`group_sessions`](@ref) and [`find_groups_of_levels`](@ref) give them.
Used by [`build_varying_effects_design`](@ref), for the standard deviations of varying
effects, and by [`prepare_parameters_model`](@ref) for the group-level parameters of a
[`HierarchicalGroupedParameters`](@ref).
"""
priors_of_groups(prior, ::Nothing) = prior
function priors_of_groups(prior, groups::NamedTuple)
    group_names = Tuple(groups.group_symbol_names)
    return NamedTuple{group_names}(ntuple(_ -> prior, length(group_names)))
end

"""
    check_varies_within_levels(column, level_of_each_session, coefficient_name, description)

Throw an informative error if the values in `column`, which the deviations of the varying
slope `coefficient_name` of `description`, such as
`"the varying effects participant of learning_rate"`, are multiplied by, are the same in all
the sessions of each level, where `level_of_each_session` has the level of every session.
Used by [`build_varying_effects_design`](@ref).
"""
function check_varies_within_levels(
    column::AbstractVector,
    level_of_each_session::AbstractVector,
    coefficient_name::Symbol,
    description::String,
)
    varies = any(unique(level_of_each_session)) do level
        values_of_level = column[level_of_each_session .== level]
        return any(!isequal(first(values_of_level)), values_of_level)
    end
    varies && return nothing
    return throw(
        ArgumentError(
            "The varying slope $coefficient_name of $description has one value in all the sessions of each level, so it can't be told apart from a varying intercept. A varying slope needs values that differ between the sessions of a level, such as a condition that changes between the blocks of a participant.",
        ),
    )
end

"""
    find_varying_effects_levels(sessions_info, varying_effects, parameter_name)

Return the levels of `varying_effects`, a [`VaryingEffects`](@ref) of the regression of the
parameter `parameter_name`, as a `NamedTuple` with the level of every session,
`level_of_each_session`, the number of levels, `n_levels`, the symbol name of each level,
`level_symbol_names`, and `groups`, from [`find_groups_of_levels`](@ref). The levels are the
values of a single column, or the combinations of the values of several, numbered in the
order of their first session. Used by [`build_varying_effects_design`](@ref).
"""
function find_varying_effects_levels(
    sessions_info::MultipleSessionsInfo,
    varying_effects::VaryingEffects,
    parameter_name::Symbol,
)
    levels = group_sessions(
        sessions_info,
        varying_effects.columns,
        "HierarchicalRegressionParameters has varying effects of $parameter_name for each value of",
    )
    groups = find_groups_of_levels(
        sessions_info,
        varying_effects.group_by,
        levels.group_of_each_session,
        levels.group_symbol_names,
        "the varying effects $(varying_effects.name) of $parameter_name",
    )
    return (;
        level_of_each_session=levels.group_of_each_session,
        n_levels=levels.n_groups,
        level_symbol_names=levels.group_symbol_names,
        groups,
    )
end

"""
    find_groups_of_levels(sessions_info, group_by, level_of_each_session,
                          level_symbol_names, description)

Return the groups of the levels of varying effects by the values of the columns in
`group_by`, a tuple of names, so that each group has a standard deviation of its own: a
`NamedTuple` with the group of every level, `group_of_each_level`, the number of groups,
`n_groups`, and the symbol name of each group, `group_symbol_names`, as
[`group_sessions`](@ref) gives them for the sessions. Without columns it returns `nothing`,
for one standard deviation. A level whose sessions are in more than one group is an error.
`description` says what is grouped, for the errors, such as
`"the varying effects participant of learning_rate"`. Used by
[`find_varying_effects_levels`](@ref).
"""
function find_groups_of_levels(
    sessions_info::MultipleSessionsInfo,
    ::Tuple{},
    level_of_each_session,
    level_symbol_names,
    description::String,
)
    return nothing
end
function find_groups_of_levels(
    sessions_info::MultipleSessionsInfo,
    group_by::Tuple{Vararg{Symbol}},
    level_of_each_session,
    level_symbol_names,
    description::String,
)
    groups = group_sessions(
        sessions_info, group_by, "HierarchicalRegressionParameters groups $description by"
    )

    #The group of a level is the group of its sessions, which must all be in one
    group_of_each_level = map(eachindex(level_symbol_names)) do level
        groups_of_level = unique(
            groups.group_of_each_session[level_of_each_session .== level]
        )
        if length(groups_of_level) != 1
            throw(
                ArgumentError(
                    "HierarchicalRegressionParameters groups $description by $group_by, but the sessions of $(level_symbol_names[level]) are in the groups $(join(groups.group_symbol_names[groups_of_level], ", ")). The sessions of one level of varying effects must all be in one group.",
                ),
            )
        end
        return only(groups_of_level)
    end
    return (;
        group_of_each_level,
        n_groups=groups.n_groups,
        group_symbol_names=groups.group_symbol_names,
    )
end
