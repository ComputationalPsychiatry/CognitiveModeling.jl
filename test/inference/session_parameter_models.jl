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

    @testset "A RegressionParameter keeps its terms, with a prior for every predictor and the settings of every random intercept" begin
        regression_parameter = RegressionParameter(;
            intercept=1,
            predictors=(; age=2),
            categorical_predictors=(; diagnosis=3),
            random_intercepts=(; participant=4),
            session_deviation=5,
        )
        @test regression_parameter.intercept === 1
        @test regression_parameter.predictors === (; age=2)
        @test regression_parameter.categorical_predictors === (; diagnosis=3)
        #A random intercept that is given as a prior has the default settings
        @test regression_parameter.random_intercepts.participant isa RandomIntercept
        @test regression_parameter.random_intercepts.participant.sd_prior === 4 &&
            regression_parameter.random_intercepts.participant.group_by === () &&
            regression_parameter.random_intercepts.participant.centered === false
        @test regression_parameter.session_deviation.sd_prior === 5
        @test regression_parameter.transformation === identity
        @test regression_parameter.center_predictors === false &&
            regression_parameter.reference_levels === (;) &&
            regression_parameter.sum_to_zero_coding === false
        #A RandomIntercept is kept with its settings, with the columns to group by as a tuple of names
        grouped = RandomIntercept(4; group_by=[:diagnosis, :site], centered=true)
        @test grouped.group_by === (:diagnosis, :site) && grouped.centered === true
        with_settings = RegressionParameter(;
            intercept=nothing,
            random_intercepts=(; participant=grouped),
            session_deviation=RandomIntercept(5; group_by=:diagnosis),
            transformation=exp,
            reference_levels=(;),
            center_predictors=true,
            sum_to_zero_coding=true,
        )
        @test with_settings.intercept === nothing
        @test with_settings.random_intercepts.participant === grouped
        @test with_settings.session_deviation.group_by === (:diagnosis,)
        @test with_settings.transformation === exp &&
            with_settings.center_predictors === true &&
            with_settings.sum_to_zero_coding === true
        #Predictors that are given without priors take the prior of every such coefficient
        without_priors = RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            predictors=(:age, :iq),
            categorical_predictors=:diagnosis,
            coefficient_prior=2,
        )
        @test without_priors.predictors === (; age=2, iq=2) &&
            without_priors.categorical_predictors === (; diagnosis=2)
        @test without_priors.session_deviation === nothing
        #A HierarchicalRegressionSessionParameters keeps one for each parameter, and has no designs until it is prepared
        parameters = (; probability=regression_parameter, scale=with_settings)
        regression = HierarchicalRegressionSessionParameters(parameters)
        @test regression.parameters === parameters && regression.designs === nothing
        @test CognitiveModeling.parameter_names(regression) === (:probability, :scale)
        @test CognitiveModeling.estimates_parameters(regression)
    end

    @testset "The terms of a regression are checked" begin
        #The intercept and the session deviation have no defaults
        @test_throws UndefKeywordError RegressionParameter(; intercept=1)
        @test_throws UndefKeywordError RegressionParameter(; session_deviation=1)
        @test_throws "A RegressionParameter needs an intercept, a predictor, a random intercept or a session deviation, but it was given none of them." RegressionParameter(;
            intercept=nothing, session_deviation=nothing
        )
        @test_throws "predictors of a RegressionParameter gives the columns (:age,) without priors, which needs coefficient_prior, the prior of every coefficient that has none of its own." RegressionParameter(;
            intercept=1, session_deviation=nothing, predictors=:age
        )
        @test_throws "A RegressionParameter was given coefficient_prior, but every predictor has a prior of its own." RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            predictors=(; age=2),
            coefficient_prior=3,
        )
        @test_throws "categorical_predictors of a RegressionParameter has nothing as the prior of (:diagnosis,), but the coefficient of every predictor needs a prior." RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            categorical_predictors=(; diagnosis=nothing),
        )
        #Every term needs a name of its own, and intercept and session_deviation are the names of those two
        @test_throws "The terms of a RegressionParameter must each have a name of their own, but more than one is named age." RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            predictors=(; age=2),
            categorical_predictors=(; age=3),
        )
        @test_throws "more than one is named intercept." RegressionParameter(;
            intercept=1, session_deviation=nothing, predictors=(; intercept=2)
        )
        @test_throws "more than one is named session_deviation." RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            random_intercepts=(; session_deviation=2),
        )
        @test_throws "reference_levels of a RegressionParameter gives a reference level for (:site,), but its categorical predictors are (:diagnosis,)." RegressionParameter(;
            intercept=1,
            session_deviation=nothing,
            categorical_predictors=(; diagnosis=2),
            reference_levels=(; site=1),
        )
        @test_throws "A RandomIntercept takes the prior of its standard deviation, such as RandomIntercept(Exponential(1)), but it was given nothing." RandomIntercept(
            nothing
        )
        @test_throws TypeError RandomIntercept(1; group_by="diagnosis")
        @test_throws TypeError RegressionParameter(;
            intercept=1, session_deviation=nothing, random_intercepts=:participant
        )
        @test_throws "HierarchicalRegressionSessionParameters takes a RegressionParameter for each parameter, such as RegressionParameter(; intercept = Normal(0, 1), session_deviation = Exponential(1)), but for probability it was given 1." HierarchicalRegressionSessionParameters((;
            probability=1
        ))
        @test_throws MethodError HierarchicalRegressionSessionParameters([1])
    end

    @testset "The design of a regression has a column for each coefficient, and the level of every session for each random intercept" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        regression = HierarchicalRegressionSessionParameters((;
            probability=RegressionParameter(;
                intercept=1,
                predictors=(; age=2),
                categorical_predictors=(; diagnosis=3),
                random_intercepts=(; participant=RandomIntercept(4; group_by=:site)),
                session_deviation=5,
            ),
            scale=RegressionParameter(; intercept=1, session_deviation=nothing),
        ))
        prepared = CognitiveModeling.prepare_parameters_model(regression, sessions_info)
        @test prepared isa HierarchicalRegressionSessionParameters &&
            prepared.parameters === regression.parameters
        design = prepared.designs.probability
        #A continuous predictor has its values, and a categorical one a column for each level other than the level of the first session
        @test design.coefficient_priors === (; age=2, diagnosis_control=3)
        @test design.design_columns ==
            (; age=[30.0, 40.0, 50.0, 60.0], diagnosis_control=[0.0, 1.0, 0.0, 1.0])
        @test design.predictor_centers === (;)
        @test design.reference_levels == (; diagnosis="clinical")
        #The levels of a random intercept are the values of its column, each in the group of its sessions
        @test design.random_intercept_levels.participant == (;
            level_of_each_session=[1, 1, 2, 2],
            n_levels=2,
            level_symbol_names=[:P1, :P2],
            groups=(;
                group_of_each_level=[1, 2],
                n_groups=2,
                group_symbol_names=[Symbol("1"), Symbol("2")],
            ),
        )
        #Every session is a level of the session deviations
        @test design.session_deviation_levels == (;
            level_of_each_session=[1, 2, 3, 4],
            n_levels=4,
            level_symbol_names=[:S1, :S2, :S3, :S4],
            groups=nothing,
        )
        #A regression with an intercept alone has no columns and no levels
        @test prepared.designs.scale == (;
            coefficient_priors=(;),
            design_columns=(;),
            predictor_centers=(;),
            reference_levels=(;),
            random_intercept_levels=(;),
            session_deviation_levels=nothing,
        )
        #Without predictors and random intercepts, sessions that have no table can be prepared too
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test CognitiveModeling.prepare_parameters_model(
            HierarchicalRegressionSessionParameters((;
                scale=RegressionParameter(; intercept=1, session_deviation=2)
            )),
            without_table,
        ).designs.scale.session_deviation_levels.n_levels == 4
        #A full model prepares it, and so does a composite, as one of its parts
        @test CognitiveModeling.prepare_full_model(
            HierarchicalRegressionSessionParameters((;
                probability=RegressionParameter(;
                    intercept=1, session_deviation=nothing, predictors=(; age=2)
                ),
            )),
            sessions_info,
        ).cognitive_parameters_model.designs.probability.design_columns ==
            (; age=[30.0, 40.0, 50.0, 60.0])
        @test CognitiveModeling.prepare_parameters_model(
            CompositeSessionParameters(regression), sessions_info
        ).parts[1].designs == prepared.designs
    end

    @testset "Predictors can be centered, and a categorical predictor takes a reference level, or coefficients that sum to zero" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        function design_of(; terms...)
            return CognitiveModeling.build_regression_design(
                sessions_info,
                RegressionParameter(; intercept=1, session_deviation=nothing, terms...),
                :probability,
            )
        end
        #The mean over the sessions is subtracted, and kept
        centered = design_of(; predictors=(; age=2, site=2), center_predictors=true)
        @test centered.design_columns ==
            (; age=[-15.0, -5.0, 5.0, 15.0], site=[-0.5, -0.5, 0.5, 0.5])
        @test centered.predictor_centers === (; age=45.0, site=1.5)
        #The level that is given as the reference level has no coefficient
        with_reference = design_of(;
            categorical_predictors=(; diagnosis=2), reference_levels=(; diagnosis="control")
        )
        @test with_reference.design_columns == (; diagnosis_clinical=[1.0, 0.0, 1.0, 0.0])
        @test with_reference.reference_levels == (; diagnosis="control")
        #With coefficients that sum to zero, the sessions of the reference level have -1 in every column
        summing_to_zero = design_of(;
            categorical_predictors=(; diagnosis=2, id=3), sum_to_zero_coding=true
        )
        @test summing_to_zero.coefficient_priors ===
            (; diagnosis_control=2, id_S2=3, id_S3=3, id_S4=3)
        @test summing_to_zero.design_columns == (;
            diagnosis_control=[-1.0, 1.0, -1.0, 1.0],
            id_S2=[-1.0, 1.0, 0.0, 0.0],
            id_S3=[-1.0, 0.0, 1.0, 0.0],
            id_S4=[-1.0, 0.0, 0.0, 1.0],
        )
        #The session deviations can be grouped too, each session being a level
        grouped = design_of(; session_deviation=RandomIntercept(5; group_by=:diagnosis))
        @test grouped.session_deviation_levels.groups == (;
            group_of_each_level=[1, 2, 1, 2],
            n_groups=2,
            group_symbol_names=[:clinical, :control],
        )
    end

    @testset "The columns of a regression must have one value for each session, of the kind its term needs" begin
        sessions_info = grouped_sessions(REGRESSION_TABLE)
        function design_of(sessions_info=sessions_info; terms...)
            return CognitiveModeling.build_regression_design(
                sessions_info,
                RegressionParameter(; intercept=1, session_deviation=nothing, terms...),
                :probability,
            )
        end
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column wave, which has the values [1, 2] in the session \"S1\". Each session must have one value of every column that a session parameter model reads." design_of(;
            predictors=(; wave=2)
        )
        @test_throws "HierarchicalRegressionSessionParameters has a random intercept of probability for each value of the column wave, which has the values [1, 2] in the session \"S1\"." design_of(;
            random_intercepts=(; wave=2)
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column diagnosis as a continuous predictor, but it has the value \"clinical\", which isn't a number. Give the column in categorical_predictors instead." design_of(;
            predictors=(; diagnosis=2)
        )
        @test_throws "The table that the sessions were split from has no column iq." design_of(;
            predictors=(; iq=2)
        )
        #A categorical predictor needs two levels, and its reference level must be one of them
        one_diagnosis = grouped_sessions(
            merge(REGRESSION_TABLE, (; diagnosis=fill("clinical", 6)))
        )
        @test_throws "HierarchicalRegressionSessionParameters predicts probability from the column diagnosis as a categorical predictor, but every session has the value \"clinical\", so there is no difference between levels to estimate." design_of(
            one_diagnosis; categorical_predictors=(; diagnosis=2)
        )
        @test_throws "The reference level of the categorical predictor diagnosis of probability is \"healthy\", which isn't a value of the column. Its values are [\"clinical\", \"control\"]." design_of(;
            categorical_predictors=(; diagnosis=2), reference_levels=(; diagnosis="healthy")
        )
        #P1 has a clinical session and a control session, so its random intercept is in no one group
        @test_throws "HierarchicalRegressionSessionParameters groups the random intercepts of probability for participant by (:diagnosis,), but the sessions of P1 are in the groups clinical, control. The sessions with one value of the random intercept's column must all be in one group." design_of(;
            random_intercepts=(; participant=RandomIntercept(2; group_by=:diagnosis))
        )
        #The coefficient of a level can't take the name of another term
        with_clashing_column = grouped_sessions(
            merge(REGRESSION_TABLE, (; diagnosis_control=[1, 1, 2, 2, 3, 4]))
        )
        @test_throws "The terms of a RegressionParameter must each have a name of their own, but more than one is named diagnosis_control." design_of(
            with_clashing_column;
            predictors=(; diagnosis_control=2),
            categorical_predictors=(; diagnosis=3),
        )
        #Sessions that were given already prepared have no table to read a column from
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column age." design_of(
            without_table; predictors=(; age=2)
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
