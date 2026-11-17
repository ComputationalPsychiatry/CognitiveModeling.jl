#A cognitive model with a learning rate, whose sessions are split from a table with a column for each predictor
struct LearnsWithPredictors <: AbstractCognitiveModel end
function CognitiveModeling.attribute_names(::LearnsWithPredictors)
    return ModelAttributes(;
        parameters=:learning_rate, states=(), inputs=:reward, outputs=:choice
    )
end

#Two participants with a block in each of two conditions, with an age and a diagnosis, and one timestep in each block
const FORMULA_TABLE = (;
    participant=["P1", "P1", "P2", "P2"],
    block=[1, 2, 1, 2],
    condition=["reward", "punishment", "reward", "punishment"],
    diagnosis=["control", "control", "clinical", "clinical"],
    age=[23.0, 23.0, 41.0, 41.0],
    reward=[1.0, 0.0, 1.0, 0.0],
    choice=[true, false, true, true],
)

#A default prior for every kind of term, as numbers, which a regression keeps as they are
const FORMULA_PRIORS = DefaultRegressionPriors(;
    intercept=1,
    continuous_predictor=2,
    categorical_predictor=3,
    interaction=4,
    varying_intercept=5,
    varying_slope=6,
)

#The terms of the regression of a formula, with a default prior for every kind of term
function regression_terms_of(formula; keywords...)
    return RegressionParameter(formula; default_priors=FORMULA_PRIORS, keywords...).terms
end

@testset "RegressionParameter from a formula" begin
    @testset "A formula gives the same regression as its terms, named by its left-hand side" begin
        from_formula = RegressionParameter(
            @formula(
                learning_rate ~
                    1 +
                age +
                diagnosis +
                condition +
                diagnosis & condition +
                (1 + condition | participant)
            );
            categorical_predictors=(
                CategoricalPredictor(
                    :diagnosis; coding=:effects, reference_level="control"
                ),
                CategoricalPredictor(:condition; coding=:dummy, reference_level="reward"),
            ),
            priors=(; age=0.5, participant=(; intercept=0.25)),
            transformation=exp,
            default_priors=FORMULA_PRIORS,
        )
        by_hand = RegressionParameter(
            Intercept(),
            ContinuousPredictor(:age, 0.5),
            CategoricalPredictor(:diagnosis; coding=:effects, reference_level="control"),
            CategoricalPredictor(:condition; coding=:dummy, reference_level="reward"),
            Interaction((:diagnosis, :condition)),
            VaryingEffects(
                :participant,
                Intercept(0.25),
                CategoricalPredictor(:condition; coding=:dummy, reference_level="reward"),
            );
            transformation=exp,
            default_priors=FORMULA_PRIORS,
            name=:learning_rate,
        )
        @test from_formula === by_hand
        #Its design for the sessions has the coefficients of the terms
        sessions_info = MultipleSessionsInfo(
            LearnsWithPredictors(),
            FORMULA_TABLE;
            observation_columns=:reward,
            action_columns=:choice,
            session_columns=(:participant, :block),
            who_starts=EnvironmentStarts(),
        )
        prepared = CognitiveModeling.prepare_parameters_model(
            HierarchicalRegressionParameters((; learning_rate=from_formula)), sessions_info
        )
        design = prepared.designs.learning_rate
        @test keys(design.design_columns) === (
            :intercept,
            :age,
            :diagnosis_clinical,
            :condition_punishment,
            :diagnosis_clinical_by_condition_punishment,
        )
        @test design.design_columns.diagnosis_clinical == [-1.0, -1.0, 1.0, 1.0]
        @test design.design_columns.diagnosis_clinical_by_condition_punishment ==
            [0.0, -1.0, 0.0, 1.0]
        @test keys(design.varying_effects.participant.design_columns) ===
            (:intercept, :condition_punishment)
    end

    @testset "A column is continuous unless categorical_predictors has it, by its name or as a CategoricalPredictor" begin
        @test regression_terms_of(@formula(learning_rate ~ age)).age ===
            ContinuousPredictor(:age, 2)

        #A name gives the default coding, and leaves the reference level to the level of the first session, whether it is given on its own, in a tuple or in a vector
        by_name = CategoricalPredictor(:diagnosis, 3; coding=:dummy)
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis); categorical_predictors=:diagnosis
        ).diagnosis === by_name
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis); categorical_predictors=(:diagnosis,)
        ).diagnosis === by_name
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis); categorical_predictors=[:diagnosis]
        ).diagnosis === by_name
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=:diagnosis,
            default_settings=DefaultRegressionSettings(; coding=:effects),
        ).diagnosis === CategoricalPredictor(:diagnosis, 3; coding=:effects)

        #A CategoricalPredictor gives the coding of its column, its reference level, or both, and has the default of what it doesn't give
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(:diagnosis; coding=:effects),
        ).diagnosis === CategoricalPredictor(:diagnosis, 3; coding=:effects)
        @test regression_terms_of(
            @formula(learning_rate ~ 0 + diagnosis);
            categorical_predictors=CategoricalPredictor(:diagnosis; coding=:full_dummy),
        ).diagnosis === CategoricalPredictor(:diagnosis, 3; coding=:full_dummy)
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(
                :diagnosis; reference_level="control"
            ),
        ).diagnosis ===
            CategoricalPredictor(:diagnosis, 3; coding=:dummy, reference_level="control")
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(
                :diagnosis; reference_level="control"
            ),
            default_settings=DefaultRegressionSettings(; coding=:effects),
        ).diagnosis === CategoricalPredictor(
            :diagnosis, 3; coding=:effects, reference_level="control"
        )
        @test regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(
                :diagnosis; coding=:effects, reference_level="control"
            ),
        ).diagnosis === CategoricalPredictor(
            :diagnosis, 3; coding=:effects, reference_level="control"
        )

        #Names and CategoricalPredictors can be given together, only for some of the columns, and for a column of varying effects
        terms = regression_terms_of(
            @formula(learning_rate ~ age + diagnosis + (1 + condition | participant));
            categorical_predictors=(
                :diagnosis, CategoricalPredictor(:condition; coding=:effects)
            ),
        )
        @test terms.age isa ContinuousPredictor && terms.diagnosis isa CategoricalPredictor
        @test terms.participant.terms.condition.coding === :effects

        #A column is the same predictor as a term of its own and in varying effects
        both_places = regression_terms_of(
            @formula(learning_rate ~ condition + (1 + condition | participant));
            categorical_predictors=CategoricalPredictor(
                :condition; coding=:effects, reference_level="reward"
            ),
        )
        @test both_places.condition.reference_level == "reward" &&
            both_places.participant.terms.condition.reference_level == "reward"
    end

    @testset "An intercept is included unless the formula has 0 or -1, also in varying effects" begin
        @test keys(regression_terms_of(@formula(learning_rate ~ age))) ===
            (:intercept, :age)
        @test keys(regression_terms_of(@formula(learning_rate ~ 1 + age))) ===
            (:intercept, :age)
        @test keys(regression_terms_of(@formula(learning_rate ~ 0 + age))) === (:age,)
        @test keys(regression_terms_of(@formula(learning_rate ~ -1 + age))) === (:age,)
        @test keys(regression_terms_of(@formula(learning_rate ~ 1))) === (:intercept,)
        @test keys(
            regression_terms_of(@formula(learning_rate ~ (age | participant))).participant.terms,
        ) === (:intercept, :age)
        @test keys(
            regression_terms_of(@formula(learning_rate ~ 0 + (0 + age | participant))).participant.terms,
        ) === (:age,)
    end

    @testset "Columns joined by & are an interaction, and varying effects can be over the combinations of several columns" begin
        @test keys(
            regression_terms_of(
                @formula(learning_rate ~ age * diagnosis); categorical_predictors=:diagnosis
            ),
        ) === (:intercept, :age, :diagnosis, :age_by_diagnosis)
        @test regression_terms_of(@formula(learning_rate ~ (1 | participant & block))).participant__block ===
            VaryingEffects((:participant, :block), Intercept(5); centered=false)
        varying_interaction = regression_terms_of(
            @formula(learning_rate ~ (1 + age + diagnosis + age & diagnosis | participant));
            categorical_predictors=:diagnosis,
        ).participant
        @test keys(varying_interaction.terms) ===
            (:intercept, :age, :diagnosis, :age_by_diagnosis)

        #A column of an interaction that isn't a term of its own in the sum is given to the interaction itself, which then comes without that main effect
        without_main_effects = regression_terms_of(
            @formula(learning_rate ~ age & diagnosis);
            categorical_predictors=CategoricalPredictor(:diagnosis; coding=:full_dummy),
        )
        @test keys(without_main_effects) === (:intercept, :age_by_diagnosis)
        @test without_main_effects.age_by_diagnosis === Interaction(
            (
                ContinuousPredictor(:age),
                CategoricalPredictor(:diagnosis; coding=:full_dummy),
            ),
            4,
        )
        with_one_main_effect = regression_terms_of(
            @formula(learning_rate ~ age + age & diagnosis);
            categorical_predictors=CategoricalPredictor(
                :diagnosis; reference_level="control"
            ),
        )
        @test keys(with_one_main_effect) === (:intercept, :age, :age_by_diagnosis)
        @test with_one_main_effect.age_by_diagnosis.predictors === (
            :age, CategoricalPredictor(:diagnosis; coding=:dummy, reference_level="control")
        )
        #The terms of their own of varying effects are those before the bar
        in_varying_effects = regression_terms_of(
            @formula(learning_rate ~ age + (1 + age & condition | participant));
            categorical_predictors=CategoricalPredictor(:condition; coding=:dummy),
        ).participant
        @test in_varying_effects.terms.age_by_condition.predictors ===
            (ContinuousPredictor(:age), CategoricalPredictor(:condition; coding=:dummy))
    end

    @testset "Priors are given by the names of the terms, with a NamedTuple for varying effects, and the other terms have the default priors" begin
        terms = regression_terms_of(
            @formula(learning_rate ~ age + iq + age & iq + (1 + age | participant));
            priors=(; iq=0.5, age_by_iq=0.25, participant=(; age=0.125)),
        )
        @test terms.intercept.prior === 1 &&
            terms.age.prior === 2 &&
            terms.iq.prior === 0.5 &&
            terms.age_by_iq.prior === 0.25
        @test terms.participant.terms.intercept.prior === 5 &&
            terms.participant.terms.age.prior === 0.125
    end

    @testset "The family distribution and its priors are given as for a regression written with terms" begin
        value_and_spread(value, spread) = (value, spread)
        from_formula = RegressionParameter(
            @formula(learning_rate ~ 1 + age);
            family_distribution=value_and_spread,
            family_distribution_priors=(; spread=2),
            default_priors=FORMULA_PRIORS,
        )
        @test from_formula === RegressionParameter(
            Intercept(),
            ContinuousPredictor(:age);
            family_distribution=value_and_spread,
            family_distribution_priors=(; spread=2),
            default_priors=FORMULA_PRIORS,
            name=:learning_rate,
        )
        @test from_formula.family_distribution === value_and_spread
    end

    @testset "Terms that a formula can't express are given after it" begin
        grouped = RegressionParameter(
            @formula(learning_rate ~ 1 + age),
            VaryingEffects(:participant, Intercept(); group_by=:diagnosis);
            default_priors=FORMULA_PRIORS,
        )
        @test keys(grouped.terms) === (:intercept, :age, :participant) &&
            grouped.terms.participant.group_by === (:diagnosis,)
    end

    @testset "The name of the parameter is checked against the name the regression is given under" begin
        named = RegressionParameter(
            @formula(learning_rate ~ 1); default_priors=FORMULA_PRIORS
        )
        @test named.name === :learning_rate
        @test HierarchicalRegressionParameters((; learning_rate=named)) isa
            HierarchicalRegressionParameters
        @test_throws "HierarchicalRegressionParameters was given the regression of learning_rate for the parameter inverse_temperature." HierarchicalRegressionParameters((;
            inverse_temperature=named
        ))
    end

    @testset "What a regression can't take is an error which says what it takes" begin
        #A name that isn't a predictor, as when it is misspelled, and a column that is given twice
        @test_throws "categorical_predictors names diagnosis, which isn't a predictor of the formula of learning_rate. Its predictors are (:age,)." regression_terms_of(
            @formula(learning_rate ~ age); categorical_predictors=:diagnosis
        )
        @test_throws "categorical_predictors of the formula of learning_rate gives the column diagnosis more than once." regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=(
                :diagnosis, CategoricalPredictor(:diagnosis; coding=:effects)
            ),
        )
        #A CategoricalPredictor of several columns, with another name than its column, or with a prior
        @test_throws "categorical_predictors of the formula of learning_rate has a CategoricalPredictor of the columns (:diagnosis, :condition), but a predictor of a formula reads one column. A predictor whose levels are the combinations of the values of several columns is given after the formula, as a term." regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor((:diagnosis, :condition)),
        )
        @test_throws "categorical_predictors of the formula of learning_rate has a CategoricalPredictor of the column diagnosis that is named group, but a term of a formula is named after its column. A predictor with another name is given after the formula, as a term." regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(:diagnosis; name=:group),
        )
        @test_throws "categorical_predictors of the formula of learning_rate has a CategoricalPredictor of the column diagnosis with the prior 0.5, but the priors of the terms of a formula are given in priors, under the names of the terms, such as priors = (; diagnosis = Normal(0, 1))." regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=CategoricalPredictor(:diagnosis, 0.5),
        )
        #Something that is neither a name nor a CategoricalPredictor, such as the codings by the names of the columns
        @test_throws "categorical_predictors of the formula of learning_rate takes the names of the columns that are categorical predictors, such as (:diagnosis, :site), with a CategoricalPredictor in the place of a name to give its coding or its reference level, such as CategoricalPredictor(:diagnosis; reference_level = \"control\"), but it was given (diagnosis = :effects,)." regression_terms_of(
            @formula(learning_rate ~ diagnosis);
            categorical_predictors=(; diagnosis=:effects),
        )
        @test_throws "but it was given \"diagnosis\"." regression_terms_of(
            @formula(learning_rate ~ diagnosis); categorical_predictors="diagnosis"
        )
        #Priors for names that aren't terms, and varying effects given one prior
        @test_throws "The priors of the formula of learning_rate name agee, which isn't one of its terms, (:intercept, :age)." regression_terms_of(
            @formula(learning_rate ~ age); priors=(; agee=0.5)
        )
        @test_throws "The priors of the formula of learning_rate give the varying effects participant 0.5, but varying effects take a NamedTuple with the priors of the standard deviations of their terms, such as (; intercept = Exponential(1))." regression_terms_of(
            @formula(learning_rate ~ (1 | participant)); priors=(; participant=0.5)
        )
        @test_throws "The priors of the varying effects participant of the formula of learning_rate name age, which isn't one of their terms, (:intercept,)." regression_terms_of(
            @formula(learning_rate ~ (1 | participant)); priors=(; participant=(; age=0.5))
        )
        #A left-hand side that isn't a name, constants other than 0, -1 and 1, and terms that aren't terms of a regression
        @test_throws "The left-hand side of the formula of a RegressionParameter is the name of the parameter, as in @formula(learning_rate ~ 1 + age), but it is 0." regression_terms_of(
            @formula(0 ~ age)
        )
        @test_throws "The sum of the formula of learning_rate has the constant 2, but a constant in a formula is 1 for an intercept, or 0 or -1 for none, as in @formula(learning_rate ~ 0 + age)." regression_terms_of(
            @formula(learning_rate ~ 2 + age)
        )
        @test_throws "The term :(log(age)) of the formula of learning_rate isn't a term of a RegressionParameter." regression_terms_of(
            @formula(learning_rate ~ log(age))
        )
        @test_throws "The varying effects of the formula of learning_rate are over :(participant / block), but varying effects are over a column, as in (1 | participant), or over the combinations of the values of several, as in (1 | participant & block)." regression_terms_of(
            @formula(learning_rate ~ (1 | participant / block))
        )
        #An interaction of something other than columns
        @test_throws "An interaction in the formula of learning_rate multiplies columns, as in diagnosis & condition, but it has" regression_terms_of(
            @formula(learning_rate ~ age & log(iq))
        )
    end
end
