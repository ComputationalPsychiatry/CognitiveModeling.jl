#The data of four sessions as a table, with two columns that group them, and one that changes within a session
const GROUPS_TABLE = (;
    id=["S1", "S1", "S2", "S2", "S3", "S4"],
    diagnosis=["clinical", "clinical", "control", "control", "clinical", "control"],
    site=[1, 1, 1, 1, 2, 2],
    wave=[1, 2, 1, 1, 1, 1],
    outcome=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    choice=[true, false, true, true, false, true],
)

#The sessions of a table with the columns of GROUPS_TABLE
function grouped_sessions(table=GROUPS_TABLE)
    return MultipleSessionsInfo(
        SessionCounter(),
        table;
        observation_columns=:outcome,
        action_columns=:choice,
        session_columns=:id,
        who_starts=EnvironmentStarts(),
    )
end

#The table of GROUPS_TABLE with the age and the participant of each session: S1 and S2 are of P1, and S3 and S4 of P2
const REGRESSION_TABLE = merge(
    GROUPS_TABLE,
    (; age=[30, 30, 40, 40, 50, 60], participant=["P1", "P1", "P1", "P1", "P2", "P2"]),
)

@testset "Session parameter models" begin
    @testset "The models hold their values or their priors in a NamedTuple" begin
        @test NoSessionParameters() isa AbstractSessionParametersModel &&
            isbitstype(NoSessionParameters)
        @test FixedSessionParameters((; scale=2.0)).parameters === (; scale=2.0)
        @test FixedSessionParametersBySession((; S1=(; scale=2.0))).parameters_by_session ===
            (; S1=(; scale=2.0))
        @test IdenticalSessionParameters((; probability=1)).priors === (; probability=1)
        @test IndependentSessionParameters((; probability=1)).priors === (; probability=1)
        @test all(
            parameters_model -> parameters_model isa AbstractSessionParametersModel,
            (
                FixedSessionParameters((;)),
                FixedSessionParametersBySession((;)),
                IdenticalSessionParameters((;)),
                IndependentSessionParameters((;)),
                GroupedHierarchicalSessionParameters((;)),
                HierarchicalRegressionSessionParameters((;)),
                CompositeSessionParameters(),
            ),
        )
        @test_throws MethodError FixedSessionParameters([2.0])
        @test_throws MethodError FixedSessionParametersBySession([(; scale=2.0)])
        @test_throws MethodError IdenticalSessionParameters([1, 2])
        @test_throws MethodError IndependentSessionParameters([1, 2])
    end

    @testset "Whether a model samples is decided by its type, and one of one's own does" begin
        @test !CognitiveModeling.estimates_parameters(NoSessionParameters())
        @test !CognitiveModeling.estimates_parameters(FixedSessionParameters((; scale=2.0)))
        @test !CognitiveModeling.estimates_parameters(FixedSessionParametersBySession((;)))
        @test CognitiveModeling.estimates_parameters(
            IdenticalSessionParameters((; probability=1))
        )
        @test CognitiveModeling.estimates_parameters(
            IndependentSessionParameters((; probability=1))
        )
        @test CognitiveModeling.estimates_parameters(
            GroupedHierarchicalSessionParameters((;
                probability=HierarchicalParameter((; mean=1, sd=2))
            )),
        )
        @test CognitiveModeling.estimates_parameters(ParametersModelOfOwn())
    end

    @testset "Each model tells which parameters it gives, and one of one's own doesn't unless it has a method" begin
        @test CognitiveModeling.parameter_names(NoSessionParameters()) === ()
        @test CognitiveModeling.parameter_names(
            FixedSessionParameters((; scale=2.0, offset=0.0))
        ) === (:scale, :offset)
        @test CognitiveModeling.parameter_names(
            FixedSessionParametersBySession((; S1=(; scale=2.0), S2=(; scale=3.0)))
        ) === (:scale,)
        @test CognitiveModeling.parameter_names(
            FixedSessionParametersBySession(Dict("S1" => (; scale=2.0)))
        ) === (:scale,)
        @test CognitiveModeling.parameter_names(FixedSessionParametersBySession((;))) === ()
        @test CognitiveModeling.parameter_names(
            FixedSessionParametersBySession((; scale=[2.0, 3.0], offset=[0.0, 0.1]))
        ) === (:scale, :offset)
        @test CognitiveModeling.parameter_names(
            IdenticalSessionParameters((; probability=1))
        ) === (:probability,)
        @test CognitiveModeling.parameter_names(
            IndependentSessionParameters((; probability=1, scale=2))
        ) === (:probability, :scale)
        @test CognitiveModeling.parameter_names(
            GroupedHierarchicalSessionParameters((;
                probability=HierarchicalParameter((; mean=1, sd=2))
            )),
        ) === (:probability,)
        @test CognitiveModeling.parameter_names(ParametersModelOfOwn()) === nothing
    end

    @testset "The parameters of each session are found by its symbol name or its identifier, and become one vector for each parameter in the order of the sessions" begin
        parameters_by_session = (;
            S2=(; scale=2.0, offset=0.1), S1=(; scale=1.0, offset=0.0)
        )
        @test CognitiveModeling.collect_parameters_in_session_order(
            parameters_by_session, ["S1", "S2"]
        ) == (; scale=[1.0, 2.0], offset=[0.0, 0.1])
        #A session from several columns is found by its symbol name, from the name and the value of each column
        @test CognitiveModeling.collect_parameters_in_session_order(
            (; id_S1__block_2=(; scale=3.0)), [(; id="S1", block=2)]
        ) == (; scale=[3.0])
        #A session without parameters, or with other parameters than the first, is an error
        @test_throws "FixedSessionParametersBySession has no parameters for the session S3. It has parameters for (:S2, :S1)." CognitiveModeling.collect_parameters_in_session_order(
            parameters_by_session, ["S1", "S3"]
        )
        @test_throws "FixedSessionParametersBySession gives the session \"S2\" the parameters (:offset,), and the first session the parameters (:scale,). Every session must have the same parameters." CognitiveModeling.collect_parameters_in_session_order(
            (; S1=(; scale=1.0), S2=(; offset=0.0)), ["S1", "S2"]
        )
        #The parameters can be in a dictionary from each session's identifier too, of one column or of several
        @test CognitiveModeling.collect_parameters_in_session_order(
            Dict("S2" => (; scale=2.0), "S1" => (; scale=1.0)), ["S1", "S2"]
        ) == (; scale=[1.0, 2.0])
        @test CognitiveModeling.collect_parameters_in_session_order(
            Dict((; id="S1", block=2) => (; scale=3.0)), [(; id="S1", block=2)]
        ) == (; scale=[3.0])
        @test_throws "FixedSessionParametersBySession has no parameters for the session \"S3\". It has parameters for the sessions [\"S1\"]." CognitiveModeling.collect_parameters_in_session_order(
            Dict("S1" => (; scale=1.0)), ["S1", "S3"]
        )
        #Parameters that are already one vector for each parameter are kept as they are
        in_order = (; scale=[1.0, 2.0], offset=[0.0, 0.1])
        @test CognitiveModeling.collect_parameters_in_session_order(
            in_order, ["S1", "S2"]
        ) === in_order
    end

    @testset "A HierarchicalParameter keeps its settings, with the defaults of those that aren't given" begin
        priors = (; mean=1, sd=2)
        #The default normal distribution is sampled as standardized values, in one group, without a transformation
        default = HierarchicalParameter(priors)
        @test default.priors === priors &&
            default.distribution === nothing &&
            default.transformation === identity &&
            default.group_by === () &&
            default.centered === false
        #The columns to group by are kept as a tuple of names
        grouped = HierarchicalParameter(
            priors; transformation=exp, group_by=:diagnosis, centered=true
        )
        @test grouped.transformation === exp &&
            grouped.group_by === (:diagnosis,) &&
            grouped.centered === true
        #Any other distribution is sampled from directly
        other = HierarchicalParameter(
            (; location=1); distribution=tuple, group_by=[:diagnosis, :site]
        )
        @test other.distribution === tuple &&
            other.group_by === (:diagnosis, :site) &&
            other.centered === true
        #A distribution given as a type is kept inside a function which makes it, and a function as it is
        of_type = HierarchicalParameter(priors; distribution=Pair)
        @test of_type.distribution === CognitiveModeling.ConstructorOf{Pair}() &&
            of_type.distribution(1, 2) === (1 => 2) &&
            of_type.centered === true
        #GroupedHierarchicalSessionParameters keeps one for each parameter, as they are given
        parameters = (; probability=default, scale=grouped, offset=other)
        @test GroupedHierarchicalSessionParameters(parameters).parameters === parameters
    end

    @testset "The settings of each parameter are checked" begin
        priors = (; mean=1, sd=2)
        #A setting that there isn't is an unsupported keyword argument
        @test_throws MethodError HierarchicalParameter(priors; transfomation=exp)
        @test_throws "A HierarchicalParameter takes the priors of the parameters of its group-level distribution in a NamedTuple, such as (; mean = Normal(0, 1), sd = Exponential(1)), but it was given 1." HierarchicalParameter(
            1
        )
        @test_throws "A HierarchicalParameter takes the priors of the parameters of its group-level distribution in a NamedTuple" HierarchicalParameter(
            (;); transformation=exp
        )
        @test_throws "The group-level distribution of a HierarchicalParameter is a normal distribution by default, which takes two priors, of its mean and its standard deviation, but it was given the priors (:mean,)." HierarchicalParameter((;
            mean=1
        ))
        #Whether it is centered can only be chosen for the default normal distribution
        for centered in (true, false)
            @test_throws "A HierarchicalParameter was given centered = $centered, but centered only applies to the default normal distribution" HierarchicalParameter(
                priors; distribution=tuple, centered
            )
        end
        #Each parameter must be given one
        @test_throws "GroupedHierarchicalSessionParameters takes a HierarchicalParameter for each parameter, such as HierarchicalParameter((; mean = Normal(0, 1), sd = Exponential(1)); transformation = exp), but for probability it was given (priors = (mean = 1, sd = 2),)." GroupedHierarchicalSessionParameters((;
            probability=(; priors)
        ))
    end

    @testset "The terms of a regression keep their priors and settings, and are named after their columns, or an interaction after its predictors, unless given a name" begin
        #An intercept and predictors, with the default settings or others
        @test Intercept(1).prior === 1 && Intercept(1).name === :intercept
        @test Intercept().prior === nothing &&
            Intercept(; name=:baseline).name === :baseline
        age = ContinuousPredictor(:age, 2)
        @test age.column === :age && age.prior === 2 && age.name === :age
        years = ContinuousPredictor(:age; name=:years)
        @test years.column === :age && years.prior === nothing && years.name === :years
        diagnosis = CategoricalPredictor(:diagnosis, 3)
        @test diagnosis.columns === (:diagnosis,) &&
            diagnosis.prior === 3 &&
            diagnosis.reference_level === nothing &&
            diagnosis.coding === :dummy &&
            diagnosis.name === :diagnosis
        @test CategoricalPredictor(:diagnosis, 3; coding=:full_dummy).coding === :full_dummy
        #Several columns give a name like the symbol name of a session, and an interaction is named after its predictors
        cell = CategoricalPredictor(
            [:diagnosis, :site],
            3;
            reference_level=(; diagnosis="control", site=1),
            coding=:effects,
        )
        @test cell.columns === (:diagnosis, :site) &&
            cell.name === :diagnosis__site &&
            cell.reference_level == (; diagnosis="control", site=1) &&
            cell.coding === :effects
        interaction = Interaction((:age, :diagnosis), 4)
        @test interaction.terms === (:age, :diagnosis) &&
            interaction.prior === 4 &&
            interaction.name === :age_by_diagnosis
        @test Interaction([:age, :diagnosis, :site]; name=:three_way).name === :three_way
        #Varying effects keep their terms under their names, and their columns and their groups as tuples
        by_participant = VaryingEffects(
            :participant,
            Intercept(5),
            ContinuousPredictor(:age, 6);
            group_by=[:diagnosis, :site],
            centered=true,
        )
        @test by_participant.columns === (:participant,) &&
            by_participant.name === :participant
        @test by_participant.terms ===
            (; intercept=Intercept(5), age=ContinuousPredictor(:age, 6))
        @test by_participant.group_by === (:diagnosis, :site) &&
            by_participant.centered === true
        by_session = VaryingEffects((:id, :block), Intercept(5))
        @test by_session.columns === (:id, :block) &&
            by_session.name === :id__block &&
            by_session.group_by === () &&
            by_session.centered === false
        @test VaryingEffects(:id, Intercept(5); name=:session).name === :session
        @test VaryingEffects(:id, Intercept()).terms.intercept.prior === nothing
        #A regression keeps its terms under their names, and gives those without a prior the default prior of their kind
        regression_parameter = RegressionParameter(
            Intercept(),
            years,
            diagnosis,
            Interaction((:years, :diagnosis)),
            by_participant;
            transformation=exp,
            default_priors=DefaultRegressionPriors(;
                intercept=7, continuous_predictor=8, interaction=9
            ),
        )
        terms = regression_parameter.terms
        @test keys(terms) ===
            (:intercept, :years, :diagnosis, :years_by_diagnosis, :participant)
        @test terms.intercept === Intercept(7)
        @test terms.years === ContinuousPredictor(:age, 8; name=:years)
        @test terms.diagnosis === diagnosis && terms.years_by_diagnosis.prior === 9
        @test terms.participant === by_participant
        @test regression_parameter.transformation === exp
        @test RegressionParameter(Intercept(1)).transformation === identity
        #The terms of varying effects without a prior get the default prior of the standard deviation of their kind, and keep their settings
        with_varying_defaults = RegressionParameter(
            VaryingEffects(
                :participant,
                Intercept(),
                ContinuousPredictor(:age),
                CategoricalPredictor(:diagnosis, 3),
                Interaction((:age, :diagnosis));
                group_by=:site,
                centered=true,
            );
            default_priors=DefaultRegressionPriors(; varying_intercept=5, varying_slope=6),
        )
        participant = with_varying_defaults.terms.participant
        @test map(term -> term.prior, participant.terms) ===
            (; intercept=5, age=6, diagnosis=3, age_by_diagnosis=6)
        @test participant.group_by === (:site,) &&
            participant.centered === true &&
            participant.name === :participant
        #Default priors that aren't given are nothing
        no_defaults = DefaultRegressionPriors()
        @test all(
            isnothing,
            (
                no_defaults.intercept,
                no_defaults.continuous_predictor,
                no_defaults.categorical_predictor,
                no_defaults.interaction,
                no_defaults.varying_intercept,
                no_defaults.varying_slope,
            ),
        )
        #A HierarchicalRegressionSessionParameters keeps one for each parameter, and has no designs until it is prepared
        parameters = (;
            probability=regression_parameter, scale=RegressionParameter(Intercept(1))
        )
        regression = HierarchicalRegressionSessionParameters(parameters)
        @test regression.parameters === parameters && regression.designs === nothing
        @test CognitiveModeling.parameter_names(regression) === (:probability, :scale)
        @test CognitiveModeling.estimates_parameters(regression)
    end

    @testset "The terms of a regression are checked" begin
        #A regression needs terms, and only terms
        @test_throws "A RegressionParameter takes the terms of the regression, such as RegressionParameter(Intercept(Normal(0, 1)), VaryingEffects(:participant, Intercept(Exponential(1)))), but it was given none." RegressionParameter()
        @test_throws "The terms given to a RegressionParameter include (intercept = 1,), which isn't a term of a regression. The terms are an Intercept, ContinuousPredictors, CategoricalPredictors, Interactions and VaryingEffects, such as ContinuousPredictor(:age, Normal(0, 1))." RegressionParameter((;
            intercept=1
        ))
        @test_throws "The terms given to a RegressionParameter include 2, which isn't a term of a regression." RegressionParameter(
            Intercept(1), 2
        )
        #Every coefficient needs a prior, of its own or the default prior of its kind, which the error names
        @test_throws "The term age of a RegressionParameter has no prior. Give it one, as in ContinuousPredictor(:age, Normal(0, 1)), or give the RegressionParameter default_priors = DefaultRegressionPriors(; continuous_predictor = Normal(0, 1)) for every term of its kind without one." RegressionParameter(
            Intercept(1), ContinuousPredictor(:age)
        )
        @test_throws "default_priors = DefaultRegressionPriors(; interaction = Normal(0, 1))" RegressionParameter(
            ContinuousPredictor(:age, 1),
            ContinuousPredictor(:site, 1),
            Interaction((:age, :site));
            default_priors=DefaultRegressionPriors(; continuous_predictor=2),
        )
        @test_throws "The term intercept of the varying effects participant has no prior. Give it the prior of the standard deviation of its deviations, as in Intercept(Exponential(1)), or give the RegressionParameter default_priors = DefaultRegressionPriors(; varying_intercept = Exponential(1)) for every term of its kind without one." RegressionParameter(
            VaryingEffects(:participant, Intercept())
        )
        @test_throws "DefaultRegressionPriors(; varying_slope = Exponential(1))" RegressionParameter(
            VaryingEffects(:participant, Intercept(1), ContinuousPredictor(:age));
            default_priors=DefaultRegressionPriors(; varying_intercept=2),
        )
        #Every term needs a name of its own, and there is one intercept at most
        @test_throws "The terms of a RegressionParameter must each have a name of their own, but more than one is named age. A term is named after its column or columns, and an interaction after its predictors, unless it is given another name, as in ContinuousPredictor(:age; name = :years)." RegressionParameter(
            ContinuousPredictor(:age, 1), CategoricalPredictor(:age, 2)
        )
        @test_throws "more than one is named participant." RegressionParameter(
            ContinuousPredictor(:age, 1; name=:participant),
            VaryingEffects(:participant, Intercept(2)),
        )
        @test_throws "The terms of a RegressionParameter include more than one Intercept, but a regression has one intercept at most." RegressionParameter(
            Intercept(1), Intercept(2; name=:baseline)
        )
        #An interaction multiplies two or more different predictors of the same regression
        @test_throws "An Interaction multiplies two or more different predictors, given by their names, such as Interaction((:diagnosis, :condition), Normal(0, 0.5)), but it was given :age." Interaction(
            :age, 1
        )
        @test_throws "but it was given (:age, :age)." Interaction((:age, :age), 1)
        @test_throws "The interaction age_by_iq of a RegressionParameter multiplies iq, which isn't one of its terms. Its terms are (:age, :age_by_iq)." RegressionParameter(
            ContinuousPredictor(:age, 1), Interaction((:age, :iq), 2)
        )
        @test_throws "The interaction intercept_by_age of a RegressionParameter multiplies intercept, which is of type Intercept, but an interaction multiplies ContinuousPredictors and CategoricalPredictors." RegressionParameter(
            Intercept(1), ContinuousPredictor(:age, 1), Interaction((:intercept, :age), 2)
        )
        #A continuous predictor reads one column, and a categorical one reads one or several and is coded in one of three ways, of which one has no reference level
        @test_throws "A ContinuousPredictor reads one column, given as a Symbol such as :age, but it was given (:age, :iq). The product of several predictors is an Interaction." ContinuousPredictor(
            (:age, :iq), 1
        )
        @test_throws "The coding of a CategoricalPredictor is :dummy, where each coefficient is the difference between a level and the reference level, :effects, where each is the difference between a level and the mean of the levels, or :full_dummy, where every level has a coefficient, but it was given :sum_to_zero." CategoricalPredictor(
            :diagnosis, 1; coding=:sum_to_zero
        )
        @test_throws "A CategoricalPredictor with the coding :full_dummy has a coefficient for every level and no reference level, but it was given the reference level \"control\"." CategoricalPredictor(
            :diagnosis, 1; coding=:full_dummy, reference_level="control"
        )
        @test_throws "A CategoricalPredictor reads one column or several, such as :diagnosis or (:diagnosis, :site), but it was given none." CategoricalPredictor(
            (), 1
        )
        @test_throws TypeError CategoricalPredictor("diagnosis", 1)
        #Varying effects need terms, which aren't varying effects themselves
        @test_throws "The varying effects participant have no terms. VaryingEffects takes the terms whose coefficients deviate for each level, such as VaryingEffects(:participant, Intercept(Exponential(1)))." VaryingEffects(
            :participant
        )
        @test_throws "The varying effects participant have VaryingEffects among their terms, but the terms of varying effects are an Intercept, ContinuousPredictors, CategoricalPredictors and Interactions." VaryingEffects(
            :participant, VaryingEffects(:site, Intercept(1))
        )
        @test_throws "The terms given to the varying effects participant include 1, which isn't a term of a regression." VaryingEffects(
            :participant, 1
        )
        @test_throws "The interaction age_by_diagnosis of the varying effects participant multiplies diagnosis, which isn't one of its terms. Its terms are (:age, :age_by_diagnosis)." VaryingEffects(
            :participant, ContinuousPredictor(:age, 1), Interaction((:age, :diagnosis), 2)
        )
        @test_throws "VaryingEffects reads one column or several" VaryingEffects(
            (), Intercept(1)
        )
        @test_throws TypeError VaryingEffects(
            :participant, Intercept(1); group_by="diagnosis"
        )
        #The coefficients of a regression are named coef, which varying effects can't be named
        @test_throws "VaryingEffects of a RegressionParameter can't be named coef, which is the name of the coefficients of the regression." RegressionParameter(
            VaryingEffects(:coef, Intercept(1))
        )
        @test_throws "HierarchicalRegressionSessionParameters takes a RegressionParameter for each parameter, such as RegressionParameter(Intercept(Normal(0, 1)), VaryingEffects(:participant, Intercept(Exponential(1)))), but for probability it was given 1." HierarchicalRegressionSessionParameters((;
            probability=1
        ))
        @test_throws MethodError HierarchicalRegressionSessionParameters([1])
    end

    @testset "The design of a regression has the prior and the values of each coefficient, and the levels and the values of the varying effects of each column" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        regression = HierarchicalRegressionSessionParameters((;
            probability=RegressionParameter(
                Intercept(1),
                ContinuousPredictor(:age, 2),
                CategoricalPredictor(:diagnosis, 3),
                VaryingEffects(
                    :participant,
                    Intercept(4),
                    CategoricalPredictor(:diagnosis, 5);
                    group_by=:site,
                ),
                VaryingEffects(:id, Intercept(6); name=:session),
            ),
            scale=RegressionParameter(Intercept(1)),
        ))
        prepared = CognitiveModeling.prepare_parameters_model(regression, sessions_info)
        @test prepared isa HierarchicalRegressionSessionParameters &&
            prepared.parameters === regression.parameters
        design = prepared.designs.probability
        #The intercept has 1 for every session, a continuous predictor its values, and a categorical one a column for each level other than the level of the first session
        @test design.coefficient_priors === (; intercept=1, age=2, diagnosis_control=3)
        @test design.design_columns == (;
            intercept=[1.0, 1.0, 1.0, 1.0],
            age=[30.0, 40.0, 50.0, 60.0],
            diagnosis_control=[0.0, 1.0, 0.0, 1.0],
        )
        @test design.reference_levels == (; diagnosis="clinical")
        #Varying effects have the levels of their column, each in the group of its sessions, and the prior of the standard deviation of each coefficient under the name of each group
        participant = design.varying_effects.participant
        @test participant.levels == (;
            level_of_each_session=[1, 1, 2, 2],
            n_levels=2,
            level_symbol_names=[:P1, :P2],
            groups=(;
                group_of_each_level=[1, 2],
                n_groups=2,
                group_symbol_names=[Symbol("1"), Symbol("2")],
            ),
        )
        @test participant.coefficient_sd_priors === (;
            intercept=NamedTuple{(Symbol("1"), Symbol("2"))}((4, 4)),
            diagnosis_control=NamedTuple{(Symbol("1"), Symbol("2"))}((5, 5)),
        )
        @test participant.design_columns ==
            (; intercept=[1.0, 1.0, 1.0, 1.0], diagnosis_control=[0.0, 1.0, 0.0, 1.0])
        @test participant.reference_levels == (; diagnosis="clinical") &&
            participant.centered === false
        #Varying effects over the column that identifies the sessions have every session as a level of its own
        session = design.varying_effects.session
        @test session.levels == (;
            level_of_each_session=[1, 2, 3, 4],
            n_levels=4,
            level_symbol_names=[:S1, :S2, :S3, :S4],
            groups=nothing,
        )
        @test session.coefficient_sd_priors === (; intercept=6)
        #A regression with an intercept alone has one column and no varying effects
        @test prepared.designs.scale == (;
            coefficient_priors=(; intercept=1),
            design_columns=(; intercept=[1.0, 1.0, 1.0, 1.0]),
            reference_levels=(;),
            varying_effects=(;),
        )
        #With an intercept alone, sessions that have no table can be prepared too
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test CognitiveModeling.prepare_parameters_model(
            HierarchicalRegressionSessionParameters((;
                scale=RegressionParameter(Intercept(1))
            )),
            without_table,
        ).designs.scale.design_columns == (; intercept=[1.0, 1.0, 1.0, 1.0])
        #A full model prepares it, and so does a composite, as one of its parts
        @test CognitiveModeling.prepare_full_model(
            HierarchicalRegressionSessionParameters((;
                probability=RegressionParameter(ContinuousPredictor(:age, 2))
            )),
            sessions_info,
        ).cognitive_parameters_model.designs.probability.design_columns ==
            (; age=[30.0, 40.0, 50.0, 60.0])
        @test CognitiveModeling.prepare_parameters_model(
            CompositeSessionParameters(regression), sessions_info
        ).parts[1].designs == prepared.designs
    end

    @testset "A categorical predictor takes a reference level, coefficients that sum to zero or a coefficient for every level, and terms can read several columns" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        function design_of(terms...)
            return CognitiveModeling.build_regression_design(
                sessions_info, RegressionParameter(terms...), :probability
            )
        end
        #A predictor that is given a name reads its column, and its coefficient takes the name
        renamed = design_of(ContinuousPredictor(:age, 2; name=:years))
        @test renamed.coefficient_priors === (; years=2) &&
            renamed.design_columns == (; years=[30.0, 40.0, 50.0, 60.0])
        #A categorical predictor of several columns has the combinations of their values as its levels, whose coefficients are named by their symbol names
        by_cell = design_of(CategoricalPredictor((:diagnosis, :site), 2))
        @test by_cell.coefficient_priors === (;
            diagnosis_control__site_1=2,
            diagnosis_clinical__site_2=2,
            diagnosis_control__site_2=2,
        )
        @test by_cell.design_columns == (;
            diagnosis_control__site_1=[0.0, 1.0, 0.0, 0.0],
            diagnosis_clinical__site_2=[0.0, 0.0, 1.0, 0.0],
            diagnosis_control__site_2=[0.0, 0.0, 0.0, 1.0],
        )
        @test by_cell.reference_levels ==
            (; diagnosis__site=(; diagnosis="clinical", site=1))
        #Its reference level is given as a NamedTuple of values, and a name of its own names the predictor but not its coefficients
        with_cell_reference = design_of(
            CategoricalPredictor(
                (:diagnosis, :site),
                2;
                reference_level=(; diagnosis="control", site=2),
                name=:cell,
            ),
        )
        @test keys(with_cell_reference.design_columns) == (
            :diagnosis_clinical__site_1,
            :diagnosis_control__site_1,
            :diagnosis_clinical__site_2,
        )
        @test with_cell_reference.reference_levels ==
            (; cell=(; diagnosis="control", site=2))
        #With a coefficient for every level there is no reference level, and of several columns this gives a coefficient for every combination of their values
        every_level = design_of(CategoricalPredictor(:diagnosis, 2; coding=:full_dummy))
        @test every_level.coefficient_priors ===
            (; diagnosis_clinical=2, diagnosis_control=2)
        @test every_level.design_columns == (;
            diagnosis_clinical=[1.0, 0.0, 1.0, 0.0], diagnosis_control=[0.0, 1.0, 0.0, 1.0]
        )
        @test every_level.reference_levels == (; diagnosis=nothing)
        every_cell = design_of(
            CategoricalPredictor((:diagnosis, :site), 3; coding=:full_dummy)
        )
        @test every_cell.coefficient_priors === (;
            diagnosis_clinical__site_1=3,
            diagnosis_control__site_1=3,
            diagnosis_clinical__site_2=3,
            diagnosis_control__site_2=3,
        )
        @test every_cell.design_columns.diagnosis_clinical__site_2 == [0.0, 0.0, 1.0, 0.0]
        #Varying effects of several columns have the combinations of their values as their levels, and are named after them
        by_visit = design_of(VaryingEffects((:participant, :site), Intercept(2)))
        @test by_visit.varying_effects.participant__site.levels == (;
            level_of_each_session=[1, 1, 2, 2],
            n_levels=2,
            level_symbol_names=[:participant_P1__site_1, :participant_P2__site_2],
            groups=nothing,
        )
        #Varying effects over the sessions can be grouped too, each session being a level
        grouped = design_of(VaryingEffects(:id, Intercept(5); group_by=:diagnosis))
        @test grouped.varying_effects.id.levels.groups == (;
            group_of_each_level=[1, 2, 1, 2],
            n_groups=2,
            group_symbol_names=[:clinical, :control],
        )
        @test grouped.varying_effects.id.coefficient_sd_priors ===
            (; intercept=(; clinical=5, control=5))
        #The level that is given as the reference level has no coefficient
        with_reference = design_of(
            CategoricalPredictor(:diagnosis, 2; reference_level="control")
        )
        @test with_reference.design_columns == (; diagnosis_clinical=[1.0, 0.0, 1.0, 0.0])
        @test with_reference.reference_levels == (; diagnosis="control")
        #With effects coding, whose coefficients sum to zero, the sessions of the reference level have -1 in every column, and each predictor is coded as it says
        effects_coded = design_of(
            CategoricalPredictor(:diagnosis, 2; coding=:effects),
            CategoricalPredictor(:id, 3; coding=:effects),
            CategoricalPredictor(:site, 4),
        )
        @test effects_coded.coefficient_priors ===
            (; diagnosis_control=2, id_S2=3, id_S3=3, id_S4=3, site_2=4)
        @test effects_coded.design_columns == (;
            diagnosis_control=[-1.0, 1.0, -1.0, 1.0],
            id_S2=[-1.0, 1.0, 0.0, 0.0],
            id_S3=[-1.0, 0.0, 1.0, 0.0],
            id_S4=[-1.0, 0.0, 0.0, 1.0],
            site_2=[0.0, 0.0, 1.0, 1.0],
        )
    end

    @testset "An interaction has a coefficient for each combination of its predictors' coefficients, multiplied by their values after the levels are coded" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        function design_of(terms...)
            return CognitiveModeling.build_regression_design(
                sessions_info, RegressionParameter(terms...), :probability
            )
        end
        #Two continuous predictors give one coefficient, multiplied by the product of their values
        continuous = design_of(
            ContinuousPredictor(:age, 1),
            ContinuousPredictor(:site, 2),
            Interaction((:age, :site), 3),
        )
        @test continuous.coefficient_priors === (; age=1, site=2, age_by_site=3)
        @test continuous.design_columns.age_by_site == [30.0, 40.0, 100.0, 120.0]
        #A continuous and a categorical predictor give one for each coefficient of the categorical predictor, coded as the predictor is
        mixed = design_of(
            ContinuousPredictor(:age, 1),
            CategoricalPredictor(:diagnosis, 2; coding=:effects),
            Interaction((:age, :diagnosis), 3),
        )
        @test mixed.design_columns.age_by_diagnosis_control == [-30.0, 40.0, -50.0, 60.0]
        every_level = design_of(
            ContinuousPredictor(:age, 1),
            CategoricalPredictor(:diagnosis, 2; coding=:full_dummy),
            Interaction((:age, :diagnosis), 3),
        )
        @test every_level.design_columns.age_by_diagnosis_clinical ==
              [30.0, 0.0, 50.0, 0.0] &&
            every_level.design_columns.age_by_diagnosis_control == [0.0, 40.0, 0.0, 60.0]
        #Three predictors are multiplied together
        three_way = design_of(
            ContinuousPredictor(:age, 1),
            ContinuousPredictor(:site, 1),
            CategoricalPredictor(:diagnosis, 1),
            Interaction((:age, :site, :diagnosis), 4),
        )
        @test three_way.design_columns.age_by_site_by_diagnosis_control ==
            [0.0, 40.0, 0.0, 120.0]
        #Two categorical predictors give one for each combination of levels, with the levels of the first predictor changing slowest, whatever the interaction is named
        crossed_sessions = grouped_sessions((;
            id=["S$i" for i in 1:9],
            condition=repeat(["a", "b", "c"]; inner=3),
            group=repeat(["x", "y", "z"], 3),
            outcome=fill(1.0, 9),
            choice=fill(true, 9),
        ))
        crossed = CognitiveModeling.build_regression_design(
            crossed_sessions,
            RegressionParameter(
                CategoricalPredictor(:condition, 1),
                CategoricalPredictor(:group, 2),
                Interaction((:condition, :group), 3; name=:cell),
            ),
            :probability,
        )
        @test keys(crossed.coefficient_priors) == (
            :condition_b,
            :condition_c,
            :group_y,
            :group_z,
            :condition_b_by_group_y,
            :condition_b_by_group_z,
            :condition_c_by_group_y,
            :condition_c_by_group_z,
        )
        @test crossed.design_columns.condition_c_by_group_y ==
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #In varying effects, an interaction multiplies predictors of the same varying effects, and gives varying slopes
        in_varying_effects = design_of(
            VaryingEffects(
                :participant,
                Intercept(1),
                ContinuousPredictor(:age, 2),
                CategoricalPredictor(:diagnosis, 3),
                Interaction((:age, :diagnosis), 4),
            ),
        )
        @test in_varying_effects.varying_effects.participant.coefficient_sd_priors ===
            (; intercept=1, age=2, diagnosis_control=3, age_by_diagnosis_control=4)
        @test in_varying_effects.varying_effects.participant.design_columns.age_by_diagnosis_control ==
            [0.0, 40.0, 0.0, 60.0]
    end

    @testset "The columns of a regression must have one value for each session, of the kind its term needs, and inform every coefficient" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        function design_of(terms...; sessions=sessions_info)
            return CognitiveModeling.build_regression_design(
                sessions, RegressionParameter(terms...), :probability
            )
        end
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column wave, which has the values [1, 2] in the session \"S1\". Each session must have one value of every column that a session parameter model reads." design_of(
            ContinuousPredictor(:wave, 2)
        )
        @test_throws "HierarchicalRegressionSessionParameters has varying effects of probability for each value of the column wave, which has the values [1, 2] in the session \"S1\"." design_of(
            VaryingEffects(:wave, Intercept(2))
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column wave, which has the values [1, 2] in the session \"S1\"." design_of(
            CategoricalPredictor((:diagnosis, :wave), 2)
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column diagnosis as a continuous predictor, but it has the value \"clinical\", which isn't a number. Give it as a CategoricalPredictor instead." design_of(
            ContinuousPredictor(:diagnosis, 2)
        )
        @test_throws "The table that the sessions were split from has no column iq." design_of(
            ContinuousPredictor(:iq, 2)
        )
        @test_throws "The table that the sessions were split from has no column block." design_of(
            VaryingEffects((:id, :block), Intercept(2))
        )
        #A categorical predictor needs two levels, and its reference level must be one of them
        one_diagnosis = grouped_sessions(
            merge(REGRESSION_TABLE, (; diagnosis=fill("clinical", 6)))
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column diagnosis as a categorical predictor, but every session has the value \"clinical\", so there is no difference between levels to estimate." design_of(
            CategoricalPredictor(:diagnosis, 2); sessions=one_diagnosis
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the columns (:diagnosis, :participant) as a categorical predictor, but every session has the value" design_of(
            CategoricalPredictor((:diagnosis, :participant), 2);
            sessions=grouped_sessions(
                merge(
                    REGRESSION_TABLE,
                    (; diagnosis=fill("clinical", 6), participant=fill("P1", 6)),
                ),
            ),
        )
        @test_throws "The reference level of the categorical predictor diagnosis of probability is \"healthy\", which isn't a value of the column diagnosis. Its values are [\"clinical\", \"control\"]." design_of(
            CategoricalPredictor(:diagnosis, 2; reference_level="healthy")
        )
        @test_throws "The reference level of the categorical predictor diagnosis__site of probability is (diagnosis = \"healthy\", site = 1), which isn't a value of the columns (:diagnosis, :site)." design_of(
            CategoricalPredictor(
                (:diagnosis, :site), 2; reference_level=(; diagnosis="healthy", site=1)
            ),
        )
        #P1 has a clinical session and a control session, so its level of the varying effects is in no one group
        @test_throws "HierarchicalRegressionSessionParameters groups the varying effects participant of probability by (:diagnosis,), but the sessions of P1 are in the groups clinical, control. The sessions of one level of varying effects must all be in one group." design_of(
            VaryingEffects(:participant, Intercept(2); group_by=:diagnosis)
        )
        #No session is a control at the second site, so nothing informs the coefficient of that combination
        no_control_at_site_2 = grouped_sessions(
            merge(
                REGRESSION_TABLE,
                (;
                    diagnosis=[
                        "clinical", "clinical", "control", "control", "clinical", "clinical"
                    ]
                ),
            ),
        )
        @test_throws "The coefficient diagnosis_control_by_site_2 of the interaction diagnosis_by_site of the regression of probability is multiplied by zero in every session, such as when no session has that combination of levels, so it can't be estimated." design_of(
            CategoricalPredictor(:diagnosis, 2),
            CategoricalPredictor(:site, 3),
            Interaction((:diagnosis, :site), 4);
            sessions=no_control_at_site_2,
        )
        #Each participant has one site, so a varying slope of the site can't be told apart from the varying intercept
        @test_throws "The varying slope site of the varying effects participant of probability has one value in all the sessions of each level, so it can't be told apart from a varying intercept. A varying slope needs values that differ between the sessions of a level, such as a condition that changes between the blocks of a participant." design_of(
            VaryingEffects(:participant, Intercept(2), ContinuousPredictor(:site, 3))
        )
        #The coefficient of a level can't take the name of another coefficient
        with_clashing_column = grouped_sessions(
            merge(REGRESSION_TABLE, (; diagnosis_control=[1, 1, 2, 2, 3, 4]))
        )
        @test_throws "The coefficients of the regression of probability must each have a name of their own, but more than one is named diagnosis_control. The coefficient of a level of a categorical predictor is named after the predictor and the level, as in diagnosis_clinical." design_of(
            ContinuousPredictor(:diagnosis_control, 2),
            CategoricalPredictor(:diagnosis, 3);
            sessions=with_clashing_column,
        )
        #Sessions that were given already prepared have no table to read a column from
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column age." design_of(
            ContinuousPredictor(:age, 2); sessions=without_table
        )
    end

    @testset "CompositeSessionParameters takes parts which give different parameters" begin
        sampled = IndependentSessionParameters((; probability=1))
        fixed = FixedSessionParameters((; scale=2.0))
        composite = CompositeSessionParameters(sampled, fixed)
        @test composite.parts === (sampled, fixed)
        @test CognitiveModeling.parameter_names(composite) === (:probability, :scale)
        #A part given by keyword has a name, and comes after the parts without one
        named = CompositeSessionParameters(sampled; noise=fixed)
        @test named.parts === (sampled, fixed)
        @test CognitiveModeling.part_names(named) === (nothing, :noise)
        @test CognitiveModeling.part_names(composite) === (nothing, nothing)
        @test CognitiveModeling.part_names(CompositeSessionParameters(; noise=fixed)) ===
            (:noise,)
        @test CognitiveModeling.parameter_names(named) === (:probability, :scale)
        @test_throws "The part noise of CompositeSessionParameters is of type Float64, but every part must be a session parameter model, a subtype of AbstractSessionParametersModel." CompositeSessionParameters(
            sampled; noise=2.0
        )
        @test_throws "Two parts of CompositeSessionParameters give the parameter probability" CompositeSessionParameters(
            sampled; noise=FixedSessionParameters((; probability=0.5))
        )
        #It samples when any part does
        @test CognitiveModeling.estimates_parameters(composite)
        @test !CognitiveModeling.estimates_parameters(
            CompositeSessionParameters(fixed, NoSessionParameters())
        )
        #Two parts can't give the same parameter, which is checked for the parts that tell their parameters
        @test_throws "Two parts of CompositeSessionParameters give the parameter probability, of the types IndependentSessionParameters and FixedSessionParameters. Each parameter must come from one part." CompositeSessionParameters(
            sampled, FixedSessionParameters((; probability=0.5))
        )
        with_own = CompositeSessionParameters(sampled, ParametersModelOfOwn())
        @test CognitiveModeling.parameter_names(with_own) === nothing
        #What the parts give is merged when it runs, where the others are checked
        @test CognitiveModeling.merge_parameters_of_parts(
            (; probability=[0.2, 0.3]), (; scale=[2.0, 2.0])
        ) == (; probability=[0.2, 0.3], scale=[2.0, 2.0])
        @test_throws "Two parts of CompositeSessionParameters give the parameters (:scale,). Each parameter must come from one part." CognitiveModeling.merge_parameters_of_parts(
            (; probability=[0.2], scale=[1.0]), (; scale=[2.0])
        )
        #A tuple of session parameter models is made into a composite of them, a NamedTuple into one with each part under its name, and anything else is left as it is
        @test CognitiveModeling.as_session_parameters_model(sampled) === sampled
        @test CognitiveModeling.as_session_parameters_model((sampled, fixed)) === composite
        from_named_tuple = CognitiveModeling.as_session_parameters_model((;
            learning=sampled, noise=fixed
        ))
        @test from_named_tuple ===
            CompositeSessionParameters(; learning=sampled, noise=fixed)
        @test CognitiveModeling.part_names(from_named_tuple) === (:learning, :noise)
        @test CognitiveModeling.as_session_parameters_model(2.0) === 2.0
        @test_throws "A session parameter model must be a subtype of AbstractSessionParametersModel, such as IndependentSessionParameters(priors) with the priors in a NamedTuple, or FixedSessionParameters(values) for given values, but it was given a tuple whose part 2 has type Float64. A tuple of session parameter models is made into a CompositeSessionParameters of them." CognitiveModeling.as_session_parameters_model((
            sampled, 2.0
        ))
        @test_throws "but it was given a NamedTuple whose noise has type Float64. A NamedTuple of session parameter models is made into a CompositeSessionParameters of them, each under its name." CognitiveModeling.as_session_parameters_model((;
            learning=sampled, noise=2.0
        ))
    end

    @testset "Preparing a model for the sessions looks up values, finds groups, and prepares the parts of a composite" begin
        sessions_info = grouped_sessions()
        by_session = FixedSessionParametersBySession(
            Dict(
                "S4" => (; probability=0.4),
                "S1" => (; probability=0.1),
                "S3" => (; probability=0.3),
                "S2" => (; probability=0.2),
            ),
        )
        #The values are then one vector for each parameter, in the order of the sessions, in the same kind of model
        looked_up = CognitiveModeling.prepare_parameters_model(by_session, sessions_info)
        @test looked_up isa FixedSessionParametersBySession
        @test looked_up.parameters_by_session == (; probability=[0.1, 0.2, 0.3, 0.4])
        @test CognitiveModeling.parameter_names(looked_up) === (:probability,)
        @test !CognitiveModeling.estimates_parameters(looked_up)
        #Preparing it again changes nothing, and the values can be given this way from the start
        @test CognitiveModeling.prepare_parameters_model(looked_up, sessions_info).parameters_by_session ===
            looked_up.parameters_by_session
        @test_throws "FixedSessionParametersBySession has 2 values of probability, but there are 4 sessions. A vector must have one value for every session, in the order of the sessions." CognitiveModeling.prepare_parameters_model(
            FixedSessionParametersBySession((; probability=[0.1, 0.2])), sessions_info
        )

        hierarchical = GroupedHierarchicalSessionParameters((;
            probability=HierarchicalParameter((; mean=1, sd=2); group_by=:diagnosis),
            scale=HierarchicalParameter((; mean=1, sd=2)),
        ))
        #The groups are kept in the same kind of model, in a field that is nothing until then
        @test hierarchical.groups === nothing
        grouped = CognitiveModeling.prepare_parameters_model(hierarchical, sessions_info)
        @test grouped isa GroupedHierarchicalSessionParameters
        @test grouped.parameters === hierarchical.parameters
        @test grouped.groups.probability.group_of_each_session == [1, 2, 1, 2]
        @test grouped.groups.scale.n_groups == 1
        #The priors of the group-level parameters are under the symbol name of each group, and are the priors themselves without groups
        @test grouped.groups.probability.group_priors ===
            (; mean=(; clinical=1, control=1), sd=(; clinical=2, control=2))
        @test grouped.groups.scale.group_priors === (; mean=1, sd=2)
        @test CognitiveModeling.parameter_names(grouped) === (:probability, :scale)

        #A model with nothing to prepare is returned as it is, and so is one of one's own
        sampled = IndependentSessionParameters((; scale=1))
        @test CognitiveModeling.prepare_parameters_model(sampled, sessions_info) === sampled
        @test CognitiveModeling.prepare_parameters_model(
            ParametersModelOfOwn(), sessions_info
        ) === ParametersModelOfOwn()
        composite = CognitiveModeling.prepare_parameters_model(
            CompositeSessionParameters(by_session, sampled), sessions_info
        )
        @test composite.parts[1].parameters_by_session ==
              (; probability=[0.1, 0.2, 0.3, 0.4]) && composite.parts[2] === sampled
        #The parts keep their names
        named = CognitiveModeling.prepare_parameters_model(
            CompositeSessionParameters(sampled; by_session), sessions_info
        )
        @test CognitiveModeling.part_names(named) === (nothing, :by_session) &&
            named.parts[2].parameters_by_session == (; probability=[0.1, 0.2, 0.3, 0.4])
    end
end
