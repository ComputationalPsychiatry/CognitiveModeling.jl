## Each parameter given by a regression on columns of the sessions' table ##

"""
    RandomIntercept(sd_prior; group_by=nothing, centered=false)

A random intercept of a [`RegressionParameter`](@ref): a deviation for each value of a
column, such as each participant, from a normal distribution around zero whose standard
deviation is sampled from `sd_prior`.

# Arguments
- `sd_prior`: the prior of the standard deviation of the deviations, such as
  `Exponential(1)`.

# Keyword arguments
- `group_by=nothing`: a column of the table that the sessions were split from, or a tuple of
  columns, whose values group the deviations, so that each group has a standard deviation
  of its own, such as `:diagnosis` for a clinical and a non-clinical group of participants.
  The sessions with one value of the random intercept's column must all be in one group.
  Groups are numbered in the order of their first session. With `nothing`, the deviations
  share one standard deviation.
- `centered=false`: whether each deviation is sampled from the normal distribution
  directly, with `true`, or as a standardized value which the standard deviation scales,
  with `false`. The standardized form usually samples better when the data informs each
  deviation little.

Each argument is kept in a field of the same name, with `group_by` as a tuple of column
names, which is empty without groups.

# Examples
```jldoctest
julia> using Distributions: Exponential

julia> by_participant = RandomIntercept(Exponential(1); group_by = :diagnosis);

julia> by_participant.group_by, by_participant.centered
((:diagnosis,), false)
```
"""
struct RandomIntercept{Tsd_prior,Tgroup_by<:Tuple{Vararg{Symbol}}}
    sd_prior::Tsd_prior
    group_by::Tgroup_by
    centered::Bool

    #Checks that the prior is given, and keeps the columns to group by as a tuple of names, which is empty without groups
    function RandomIntercept(sd_prior; group_by=nothing, centered::Bool=false)
        if sd_prior === nothing
            throw(
                ArgumentError(
                    "A RandomIntercept takes the prior of its standard deviation, such as RandomIntercept(Exponential(1)), but it was given nothing.",
                ),
            )
        end
        columns_to_group_by =
            group_by === nothing ? () : as_tuple(group_by)::Tuple{Vararg{Symbol}}
        return new{typeof(sd_prior),typeof(columns_to_group_by)}(
            sd_prior, columns_to_group_by, centered
        )
    end
end

"""
    RegressionParameter(; intercept, session_deviation, predictors=(;),
                        categorical_predictors=(;), random_intercepts=(;),
                        <keyword arguments>)

How one parameter of a [`HierarchicalRegression`](@ref) is given to every session: by a
regression on columns of the table that the sessions were split from. The value of a session
is the sum of the intercept, the effect of each predictor, the random intercept of its value
of each of some columns, and a deviation of its own, with `transformation` applied to the
sum. Each session must have one value of every column that is used, which isn't missing.

# Keyword arguments
- `intercept`: the prior of the intercept, such as `Normal(0, 1)`, or `nothing` for a
  regression without one.
- `session_deviation`: the prior of the standard deviation of the deviations of the
  sessions, such as `Exponential(1)`, or a [`RandomIntercept`](@ref) with that prior and
  other settings. The deviations let sessions with the same value of every column differ.
  With `nothing`, such sessions have the same value of the parameter.
- `predictors=(;)`: the continuous predictors, as a `NamedTuple` from the column to the
  prior of its coefficient, such as `(; age = Normal(0, 0.5))`. The values of the column
  must be numbers. The columns can also be given without priors, as `:age` or
  `(:age, :iq)`, and then take `coefficient_prior`.
- `categorical_predictors=(;)`: the categorical predictors, in the same forms, such as
  `(; diagnosis = Normal(0, 1))`. Each value of the column other than its reference level
  has a coefficient with that prior, which is its difference from the reference level, such
  as the difference between a clinical and a non-clinical group.
- `random_intercepts=(;)`: the columns with a random intercept for each of their values, as
  a `NamedTuple` from the column to the prior of the standard deviation of its random
  intercepts, such as `(; participant = Exponential(1))` for participants with several
  sessions each, or to a [`RandomIntercept`](@ref) with that prior and other settings.
- `transformation=identity`: a function applied to the sum for each session, such as `exp`
  for a parameter that must be positive or `logistic` for one between 0 and 1, as an
  inverse link function does in a regression.
- `coefficient_prior=nothing`: the prior of the coefficient of every predictor that is
  given without one.
- `center_predictors=false`: whether the mean over the sessions is subtracted from each
  continuous predictor, so that the intercept is the value at the mean of each.
- `reference_levels=(;)`: the reference level of categorical predictors, as a `NamedTuple`
  from the column to the level, such as `(; diagnosis = "control")`. Any other has the
  value of its first session as its reference level.
- `sum_to_zero_coding=false`: whether the coefficients of each categorical predictor are
  differences from the mean of its levels, with `true`, instead of differences from the
  reference level. The difference of the reference level is then minus the sum of the
  others.

Each argument is kept in a field of the same name, except `coefficient_prior`: `predictors`
and `categorical_predictors` are `NamedTuple`s with the prior of each column, and each
entry of `random_intercepts`, and `session_deviation`, is a `RandomIntercept`.

# Examples
```jldoctest
julia> using Distributions: Normal, Exponential

julia> learning_rate = RegressionParameter(; intercept = Normal(0, 1), predictors = (; age = Normal(0, 0.5)), categorical_predictors = (; diagnosis = Normal(0, 1)), random_intercepts = (; participant = Exponential(1)), session_deviation = Exponential(1));

julia> keys(learning_rate.predictors), keys(learning_rate.categorical_predictors)
((:age,), (:diagnosis,))

julia> learning_rate.random_intercepts.participant.centered
false
```
"""
struct RegressionParameter{
    Tintercept,
    Tpredictors<:NamedTuple,
    Tcategorical_predictors<:NamedTuple,
    Trandom_intercepts<:NamedTuple,
    Tsession_deviation<:Union{Nothing,RandomIntercept},
    Ttransformation,
    Treference_levels<:NamedTuple,
}
    intercept::Tintercept
    predictors::Tpredictors
    categorical_predictors::Tcategorical_predictors
    random_intercepts::Trandom_intercepts
    session_deviation::Tsession_deviation
    transformation::Ttransformation
    center_predictors::Bool
    reference_levels::Treference_levels
    sum_to_zero_coding::Bool

    #Checks the terms of the regression, and keeps every predictor with the prior of its coefficient and every random intercept as a RandomIntercept
    function RegressionParameter(;
        intercept,
        session_deviation,
        predictors=(;),
        categorical_predictors=(;),
        random_intercepts::NamedTuple=(;),
        transformation=identity,
        coefficient_prior=nothing,
        center_predictors::Bool=false,
        reference_levels::NamedTuple=(;),
        sum_to_zero_coding::Bool=false,
    )

        #The predictors that are given without priors take the prior of every such coefficient
        predictors_with_priors = pair_predictors_with_priors(
            predictors, coefficient_prior, "predictors"
        )
        categorical_predictors_with_priors = pair_predictors_with_priors(
            categorical_predictors, coefficient_prior, "categorical_predictors"
        )
        if coefficient_prior !== nothing &&
            predictors isa NamedTuple &&
            categorical_predictors isa NamedTuple
            throw(
                ArgumentError(
                    "A RegressionParameter was given coefficient_prior, but every predictor has a prior of its own. coefficient_prior is for predictors that are given without priors, as in predictors = (:age, :iq).",
                ),
            )
        end

        #A random intercept that is given as the prior of its standard deviation has the default settings
        random_intercepts_with_settings = map(as_random_intercept, random_intercepts)
        session_deviation_with_settings =
            session_deviation === nothing ? nothing : as_random_intercept(session_deviation)

        #A regression must give the parameter some value
        if intercept === nothing &&
            isempty(predictors_with_priors) &&
            isempty(categorical_predictors_with_priors) &&
            isempty(random_intercepts_with_settings) &&
            session_deviation_with_settings === nothing
            throw(
                ArgumentError(
                    "A RegressionParameter needs an intercept, a predictor, a random intercept or a session deviation, but it was given none of them.",
                ),
            )
        end

        #Every term needs a name of its own, which its variables are named under, and a prior
        term_names = (
            :intercept,
            :session_deviation,
            keys(predictors_with_priors)...,
            keys(categorical_predictors_with_priors)...,
            keys(random_intercepts_with_settings)...,
        )
        check_unique_term_names(term_names)
        check_has_priors(predictors_with_priors, "predictors")
        check_has_priors(categorical_predictors_with_priors, "categorical_predictors")

        #A reference level can only be given for a categorical predictor
        unknown_columns = Tuple(
            column for column in keys(reference_levels) if
            !(column in keys(categorical_predictors_with_priors))
        )
        if !isempty(unknown_columns)
            throw(
                ArgumentError(
                    "reference_levels of a RegressionParameter gives a reference level for $unknown_columns, but its categorical predictors are $(keys(categorical_predictors_with_priors)).",
                ),
            )
        end

        return new{
            typeof(intercept),
            typeof(predictors_with_priors),
            typeof(categorical_predictors_with_priors),
            typeof(random_intercepts_with_settings),
            typeof(session_deviation_with_settings),
            typeof(transformation),
            typeof(reference_levels),
        }(
            intercept,
            predictors_with_priors,
            categorical_predictors_with_priors,
            random_intercepts_with_settings,
            session_deviation_with_settings,
            transformation,
            center_predictors,
            reference_levels,
            sum_to_zero_coding,
        )
    end
end

"""
    HierarchicalRegression(parameters)

The session parameter model of a full model which gives each parameter of every session by
a regression on columns of the table that the sessions were split from, such as the age and
the diagnosis of a participant, with random intercepts. `parameters` is a `NamedTuple` with
a [`RegressionParameter`](@ref) for each parameter that is fitted, which has its intercept,
its predictors, its random intercepts and their priors. It is kept in the field
`parameters`. Useful for estimating the effect of a predictor on a parameter, or the
difference between groups, together with the parameters of the sessions.

A full model builds the design of every regression once, before it is made, from the table
that [`MultipleSessionsInfo`](@ref) keeps, so sessions that were prepared one at a time can
only have an intercept and a session deviation. The field `designs` is `nothing` until then,
and afterwards has, for each parameter:
- `coefficient_priors` and `design_columns`: the prior of each coefficient, and the values
  it is multiplied by, one for every session, under the name of the coefficient. A
  continuous predictor has one coefficient, named after its column, such as `age`. A
  categorical predictor has one for each level other than its reference level, named after
  the column and the symbol name of the level, such as `diagnosis_clinical`.
- `predictor_centers`: the mean that was subtracted from each continuous predictor, when
  they are centered.
- `reference_levels`: the reference level of each categorical predictor.
- `random_intercept_levels`: for the column of each random intercept, the level of every
  session, `level_of_each_session`, the number of levels, `n_levels`, the symbol name of
  each level, `level_symbol_names`, and `groups`, which is `nothing` without groups and
  otherwise has the group of every level, `group_of_each_level`, the number of groups,
  `n_groups`, and the symbol name of each group, `group_symbol_names`.
- `session_deviation_levels`: the same for the deviations of the sessions, where every
  session is a level, or `nothing` without them.

# Examples
```julia
using Turing
using LogExpFunctions: logistic

session_parameter_model = HierarchicalRegression((;
    learning_rate = RegressionParameter(;
        intercept = Normal(0, 1),
        predictors = (; age = Normal(0, 0.5)),
        categorical_predictors = (; diagnosis = Normal(0, 1)),
        random_intercepts = (; participant = Exponential(1)),
        session_deviation = RandomIntercept(Exponential(1); group_by = :diagnosis),
        transformation = logistic,
    ),
    inverse_temperature = RegressionParameter(;
        intercept = Normal(0, 1), session_deviation = Exponential(1), transformation = exp
    ),
))
```
"""
struct HierarchicalRegression{
    Tparameters<:NamedTuple,Tdesigns<:Union{Nothing,NamedTuple}
} <: AbstractSessionParametersModel
    parameters::Tparameters
    designs::Tdesigns

    #Takes a RegressionParameter for each parameter. The designs of the regressions aren't built yet
    function HierarchicalRegression(parameters::NamedTuple)
        foreach(check_is_regression_parameter, keys(parameters), values(parameters))
        return new{typeof(parameters),Nothing}(parameters, nothing)
    end

    #The same model with the designs of its regressions for the sessions it is prepared for
    function HierarchicalRegression(
        parameters_model::HierarchicalRegression, designs::NamedTuple
    )
        parameters = parameters_model.parameters
        return new{typeof(parameters),typeof(designs)}(parameters, designs)
    end
end

#It gives the parameters it has a RegressionParameter for
function parameter_names(parameters_model::HierarchicalRegression)
    return keys(parameters_model.parameters)
end

#The design of every regression is built once, when a full model is prepared, and kept in the field for the designs
function prepare_parameters_model(
    parameters_model::HierarchicalRegression, sessions_info::MultipleSessionsInfo
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
    return HierarchicalRegression(parameters_model, designs)
end

"""
    check_is_regression_parameter(name, regression_parameter)

Throw an informative error if what a [`HierarchicalRegression`](@ref) was given for the
parameter `name` isn't a [`RegressionParameter`](@ref). Used by the constructor of
`HierarchicalRegression`.
"""
check_is_regression_parameter(name::Symbol, ::RegressionParameter) = nothing
function check_is_regression_parameter(name::Symbol, regression_parameter)
    return throw(
        ArgumentError(
            "HierarchicalRegression takes a RegressionParameter for each parameter, such as RegressionParameter(; intercept = Normal(0, 1), session_deviation = Exponential(1)), but for $name it was given $(repr(regression_parameter)).",
        ),
    )
end

"""
    as_random_intercept(random_intercept)

Return `random_intercept` as a [`RandomIntercept`](@ref): itself when it is one, and
otherwise one with the default settings whose standard deviation has it as its prior. Used
by the constructor of [`RegressionParameter`](@ref).
"""
as_random_intercept(random_intercept::RandomIntercept) = random_intercept
as_random_intercept(sd_prior) = RandomIntercept(sd_prior)

"""
    pair_predictors_with_priors(predictors, coefficient_prior, keyword)

Return the predictors of a [`RegressionParameter`](@ref) as a `NamedTuple` from each column
to the prior of its coefficient: a `NamedTuple` as it is, and columns that are given without
priors, as one name or several in a tuple or a vector, each with `coefficient_prior`, which
is an error when it is `nothing`. `keyword` is the name of the keyword argument they were
given with, for the error. Used by the constructor of `RegressionParameter`.
"""
function pair_predictors_with_priors(
    predictors::NamedTuple, coefficient_prior, keyword::String
)
    return predictors
end
function pair_predictors_with_priors(predictors, coefficient_prior, keyword::String)
    columns = as_tuple(predictors)::Tuple{Vararg{Symbol}}
    if coefficient_prior === nothing
        throw(
            ArgumentError(
                "$keyword of a RegressionParameter gives the columns $columns without priors, which needs coefficient_prior, the prior of every coefficient that has none of its own. Give it, or give $keyword as a NamedTuple from each column to the prior of its coefficient, such as (; age = Normal(0, 1)).",
            ),
        )
    end
    return NamedTuple{columns}(map(_ -> coefficient_prior, columns))
end

"""
    check_has_priors(predictors, keyword)

Throw an informative error if a column in `predictors`, a `NamedTuple` from each column to
the prior of its coefficient, has `nothing` as its prior. `keyword` is the name of the
keyword argument they were given with. Used by the constructor of
[`RegressionParameter`](@ref).
"""
function check_has_priors(predictors::NamedTuple, keyword::String)
    without_priors = Tuple(
        column for column in keys(predictors) if getproperty(predictors, column) === nothing
    )
    if !isempty(without_priors)
        throw(
            ArgumentError(
                "$keyword of a RegressionParameter has nothing as the prior of $without_priors, but the coefficient of every predictor needs a prior.",
            ),
        )
    end
    return nothing
end

"""
    check_unique_term_names(term_names)

Throw an informative error if two terms of a regression have the same name in `term_names`,
a tuple or a vector with `intercept`, `session_deviation`, the name of each coefficient and
the column of each random intercept, since the variables of a term are named under its
name. Used by the constructor of [`RegressionParameter`](@ref), for the columns, and by
[`build_regression_design`](@ref), for the coefficients of the levels.
"""
function check_unique_term_names(term_names)
    allunique(term_names) && return nothing
    repeated_names = unique(name for name in term_names if count(==(name), term_names) > 1)
    return throw(
        ArgumentError(
            "The terms of a RegressionParameter must each have a name of their own, but more than one is named $(join(repeated_names, ", ")). A term is named after its column, a coefficient of a categorical predictor after its column and its level, as in diagnosis_clinical, and intercept and session_deviation are the names of those two.",
        ),
    )
end

"""
    build_regression_design(sessions_info, regression_parameter, parameter_name)

Return the design of the regression of the parameter `parameter_name` for the sessions in
`sessions_info`, as [`HierarchicalRegression`](@ref) keeps it in its field `designs`: the
prior and the column of each coefficient, the means that were subtracted, the reference
levels, and the levels of the sessions for each random intercept and for the deviations of
the sessions. Used by [`prepare_parameters_model`](@ref).
"""
function build_regression_design(
    sessions_info::MultipleSessionsInfo,
    regression_parameter::RegressionParameter,
    parameter_name::Symbol,
)
    coefficient_names = Symbol[]
    coefficient_priors = Any[]
    design_columns = Vector{Float64}[]

    #A continuous predictor has one coefficient, which is multiplied by its value in each session
    center_predictors = regression_parameter.center_predictors
    centers = Float64[]
    for (column, prior) in pairs(regression_parameter.predictors)
        values = read_predictor_values(sessions_info, column, parameter_name)
        center = center_predictors ? sum(values) / length(values) : 0.0
        push!(coefficient_names, column)
        push!(coefficient_priors, prior)
        push!(design_columns, values .- center)
        push!(centers, center)
    end
    predictor_centers = if center_predictors
        NamedTuple{keys(regression_parameter.predictors)}(Tuple(centers))
    else
        (;)
    end

    #A categorical predictor has one coefficient for each level other than its reference level
    reference_levels = Any[]
    for (column, prior) in pairs(regression_parameter.categorical_predictors)
        contrasts = build_contrast_columns(
            sessions_info, column, regression_parameter, parameter_name
        )
        for (name, contrast_column) in zip(contrasts.names, contrasts.columns)
            push!(coefficient_names, name)
            push!(coefficient_priors, prior)
            push!(design_columns, contrast_column)
        end
        push!(reference_levels, contrasts.reference_level)
    end

    #The coefficients of the levels must not take the name of another term
    term_names = (
        :intercept,
        :session_deviation,
        coefficient_names...,
        keys(regression_parameter.random_intercepts)...,
    )
    check_unique_term_names(term_names)

    #The level of every session for each random intercept, and for the deviations of the sessions
    random_intercept_columns = keys(regression_parameter.random_intercepts)
    random_intercept_levels = NamedTuple{random_intercept_columns}(
        map(
            (column, random_intercept) -> find_random_intercept_levels(
                sessions_info, column, random_intercept, parameter_name
            ),
            random_intercept_columns,
            values(regression_parameter.random_intercepts),
        ),
    )
    session_deviation_levels = find_session_deviation_levels(
        sessions_info, regression_parameter.session_deviation, parameter_name
    )

    names = Tuple(coefficient_names)
    return (;
        coefficient_priors=NamedTuple{names}(Tuple(coefficient_priors)),
        design_columns=NamedTuple{names}(Tuple(design_columns)),
        predictor_centers,
        reference_levels=NamedTuple{keys(regression_parameter.categorical_predictors)}(
            Tuple(reference_levels)
        ),
        random_intercept_levels,
        session_deviation_levels,
    )
end

"""
    read_predictor_values(sessions_info, column, parameter_name)

Return the value of the continuous predictor `column` in every session, as a vector of
`Float64`, for the regression of the parameter `parameter_name`. A value that isn't a number
is an error. Used by [`build_regression_design`](@ref).
"""
function read_predictor_values(
    sessions_info::MultipleSessionsInfo, column::Symbol, parameter_name::Symbol
)
    values = read_one_value_of_each_session(
        sessions_info, column, "HierarchicalRegression predicts $parameter_name from"
    )
    for value in values
        if !(value isa Real)
            throw(
                ArgumentError(
                    "HierarchicalRegression predicts $parameter_name from the column $column as a continuous predictor, but it has the value $(repr(value)), which isn't a number. Give the column in categorical_predictors instead.",
                ),
            )
        end
    end
    return Float64[value for value in values]
end

"""
    build_contrast_columns(sessions_info, column, regression_parameter, parameter_name)

Return the columns of the categorical predictor `column` in the regression of the parameter
`parameter_name`, in a `NamedTuple` with `names`, `columns` and `reference_level`: for each
level other than the reference level, in the order of their first session, the name of its
coefficient, made from the column and the symbol name of the level, as in
`diagnosis_clinical`, and a vector with 1 for the sessions of that level and 0 for the
others. With `sum_to_zero_coding`, the sessions of the reference level have -1 in every
column. The reference level is the one in `reference_levels` of `regression_parameter`, or
the level of the first session. It is an error when the column has one level only, and when
the reference level isn't a value of the column. Used by [`build_regression_design`](@ref).
"""
function build_contrast_columns(
    sessions_info::MultipleSessionsInfo,
    column::Symbol,
    regression_parameter::RegressionParameter,
    parameter_name::Symbol,
)
    values = read_one_value_of_each_session(
        sessions_info, column, "HierarchicalRegression predicts $parameter_name from"
    )

    #The levels are the values of the column, in the order of their first session
    levels = unique(values)
    if length(levels) < 2
        throw(
            ArgumentError(
                "HierarchicalRegression predicts $parameter_name from the column $column as a categorical predictor, but every session has the value $(repr(only(levels))), so there is no difference between levels to estimate.",
            ),
        )
    end

    #The reference level is the one that is given, or the level of the first session
    given_reference_levels = regression_parameter.reference_levels
    reference_level = if haskey(given_reference_levels, column)
        getproperty(given_reference_levels, column)
    else
        first(levels)
    end
    if !any(isequal(reference_level), levels)
        throw(
            ArgumentError(
                "The reference level of the categorical predictor $column of $parameter_name is $(repr(reference_level)), which isn't a value of the column. Its values are $levels.",
            ),
        )
    end

    #Each other level has a column with 1 for its sessions, and with -1 for the sessions of the reference level when the coefficients sum to zero
    value_of_reference_level = regression_parameter.sum_to_zero_coding ? -1.0 : 0.0
    other_levels = filter(!isequal(reference_level), levels)
    columns = map(other_levels) do level
        return map(values) do value
            if isequal(value, level)
                1.0
            elseif isequal(value, reference_level)
                value_of_reference_level
            else
                0.0
            end
        end
    end
    names = [Symbol(column, "_", session_id_to_symbol(level)) for level in other_levels]
    return (; names, columns, reference_level)
end

"""
    find_random_intercept_levels(sessions_info, column, random_intercept, parameter_name)

Return the levels of the random intercept for `column` in the regression of the parameter
`parameter_name`, as a `NamedTuple` with the level of every session,
`level_of_each_session`, the number of levels, `n_levels`, the symbol name of each level,
`level_symbol_names`, and `groups`, from [`find_groups_of_levels`](@ref). The levels are the
values of the column, numbered in the order of their first session. Used by
[`build_regression_design`](@ref).
"""
function find_random_intercept_levels(
    sessions_info::MultipleSessionsInfo,
    column::Symbol,
    random_intercept::RandomIntercept,
    parameter_name::Symbol,
)
    levels = group_sessions(
        sessions_info,
        (column,),
        "HierarchicalRegression has a random intercept of $parameter_name for each value of",
    )
    groups = find_groups_of_levels(
        sessions_info,
        random_intercept.group_by,
        levels.group_of_each_session,
        levels.group_symbol_names,
        "the random intercepts of $parameter_name for $column",
    )
    return (;
        level_of_each_session=levels.group_of_each_session,
        n_levels=levels.n_groups,
        level_symbol_names=levels.group_symbol_names,
        groups,
    )
end

"""
    find_session_deviation_levels(sessions_info, session_deviation, parameter_name)

Return the levels of the deviations of the sessions in the regression of the parameter
`parameter_name`, in the form of [`find_random_intercept_levels`](@ref), where every
session is a level of its own, named by its symbol name, or `nothing` when
`session_deviation` is `nothing`. Used by [`build_regression_design`](@ref).
"""
function find_session_deviation_levels(
    sessions_info::MultipleSessionsInfo, ::Nothing, parameter_name::Symbol
)
    return nothing
end
function find_session_deviation_levels(
    sessions_info::MultipleSessionsInfo,
    session_deviation::RandomIntercept,
    parameter_name::Symbol,
)
    level_of_each_session = collect(1:n_sessions(sessions_info))
    level_symbol_names = map(session_id_to_symbol, sessions_info.session_ids)
    groups = find_groups_of_levels(
        sessions_info,
        session_deviation.group_by,
        level_of_each_session,
        level_symbol_names,
        "the session deviations of $parameter_name",
    )
    return (;
        level_of_each_session,
        n_levels=n_sessions(sessions_info),
        level_symbol_names,
        groups,
    )
end

"""
    find_groups_of_levels(sessions_info, group_by, level_of_each_session,
                          level_symbol_names, description)

Return the groups of the levels of a random intercept by the values of the columns in
`group_by`, a tuple of names, so that each group has a standard deviation of its own: a
`NamedTuple` with the group of every level, `group_of_each_level`, the number of groups,
`n_groups`, and the symbol name of each group, `group_symbol_names`, as
[`group_sessions`](@ref) gives them for the sessions. Without columns it returns `nothing`,
for one standard deviation. A level whose sessions are in more than one group is an error.
`description` says what is grouped, for the errors, such as
`"the random intercepts of learning_rate for participant"`. Used by
[`find_random_intercept_levels`](@ref) and [`find_session_deviation_levels`](@ref).
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
        sessions_info, group_by, "HierarchicalRegression groups $description by"
    )

    #The group of a level is the group of its sessions, which must all be in one
    group_of_each_level = map(eachindex(level_symbol_names)) do level
        groups_of_level = unique(
            groups.group_of_each_session[level_of_each_session .== level]
        )
        if length(groups_of_level) != 1
            throw(
                ArgumentError(
                    "HierarchicalRegression groups $description by $group_by, but the sessions of $(level_symbol_names[level]) are in the groups $(join(groups.group_symbol_names[groups_of_level], ", ")). The sessions with one value of the random intercept's column must all be in one group.",
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
