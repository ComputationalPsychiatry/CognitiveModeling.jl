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

@testset "Session parameter models" begin
    @testset "The models hold their values or their priors in a NamedTuple" begin
        @test NoParameters() isa AbstractSessionParametersModel && isbitstype(NoParameters)
        @test FixedSessionParameters((; scale=2.0)).parameters === (; scale=2.0)
        @test FixedParametersBySession((; S1=(; scale=2.0))).parameters_by_session ===
            (; S1=(; scale=2.0))
        @test IdenticalSessionParameters((; probability=1)).priors === (; probability=1)
        @test IndependentSessionParameters((; probability=1)).priors === (; probability=1)
        @test all(
            parameters_model -> parameters_model isa AbstractSessionParametersModel,
            (
                FixedSessionParameters((;)),
                FixedParametersBySession((;)),
                IdenticalSessionParameters((;)),
                IndependentSessionParameters((;)),
                GroupedHierarchicalSessionParameters((;)),
                CompositeSessionParameters(),
            ),
        )
        @test_throws MethodError FixedSessionParameters([2.0])
        @test_throws MethodError FixedParametersBySession([(; scale=2.0)])
        @test_throws MethodError IdenticalSessionParameters([1, 2])
        @test_throws MethodError IndependentSessionParameters([1, 2])
    end

    @testset "Whether a model samples is decided by its type, and one of one's own does" begin
        @test !CognitiveModeling.estimates_parameters(NoParameters())
        @test !CognitiveModeling.estimates_parameters(FixedSessionParameters((; scale=2.0)))
        @test !CognitiveModeling.estimates_parameters(FixedParametersBySession((;)))
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
        @test CognitiveModeling.parameter_names(NoParameters()) === ()
        @test CognitiveModeling.parameter_names(
            FixedSessionParameters((; scale=2.0, offset=0.0))
        ) === (:scale, :offset)
        @test CognitiveModeling.parameter_names(
            FixedParametersBySession((; S1=(; scale=2.0), S2=(; scale=3.0)))
        ) === (:scale,)
        @test CognitiveModeling.parameter_names(
            FixedParametersBySession(Dict("S1" => (; scale=2.0)))
        ) === (:scale,)
        @test CognitiveModeling.parameter_names(FixedParametersBySession((;))) === ()
        @test CognitiveModeling.parameter_names(
            FixedParametersBySession((; scale=[2.0, 3.0], offset=[0.0, 0.1]))
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

    @testset "The values of each session are found by its name or its identifier, and become one vector for each parameter in the order of the sessions" begin
        values_by_session = (; S2=(; scale=2.0, offset=0.1), S1=(; scale=1.0, offset=0.0))
        @test CognitiveModeling.parameters_by_session(values_by_session, ["S1", "S2"]) ==
            (; scale=[1.0, 2.0], offset=[0.0, 0.1])
        #A session from several columns is found by the name and the value of each column
        @test CognitiveModeling.parameters_by_session(
            (; id_S1__block_2=(; scale=3.0)), [(; id="S1", block=2)]
        ) == (; scale=[3.0])
        #A session without values, or with other parameters than the first, is an error
        @test_throws "FixedParametersBySession has no values for the session S3. It has values for (:S2, :S1)." CognitiveModeling.parameters_by_session(
            values_by_session, ["S1", "S3"]
        )
        @test_throws "FixedParametersBySession gives the session \"S2\" the parameters (:offset,), and the first session the parameters (:scale,). Every session must have values for the same parameters." CognitiveModeling.parameters_by_session(
            (; S1=(; scale=1.0), S2=(; offset=0.0)), ["S1", "S2"]
        )
        #The values can be in a dictionary from each session's identifier too, of one column or of several
        @test CognitiveModeling.parameters_by_session(
            Dict("S2" => (; scale=2.0), "S1" => (; scale=1.0)), ["S1", "S2"]
        ) == (; scale=[1.0, 2.0])
        @test CognitiveModeling.parameters_by_session(
            Dict((; id="S1", block=2) => (; scale=3.0)), [(; id="S1", block=2)]
        ) == (; scale=[3.0])
        @test_throws "FixedParametersBySession has no values for the session \"S3\". It has values for the sessions [\"S1\"]." CognitiveModeling.parameters_by_session(
            Dict("S1" => (; scale=1.0)), ["S1", "S3"]
        )
        #Values that are already one vector for each parameter are kept as they are
        in_order = (; scale=[1.0, 2.0], offset=[0.0, 0.1])
        @test CognitiveModeling.parameters_by_session(in_order, ["S1", "S2"]) === in_order
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

    @testset "The groups of the sessions are found from the table, in the order of their first session" begin
        sessions_info = grouped_sessions()
        @test CognitiveModeling.group_sessions(
            sessions_info, (:diagnosis,), :probability
        ) == (;
            group_of_each_session=[1, 2, 1, 2],
            n_groups=2,
            group_names=[:clinical, :control],
        )
        #With several columns, a group is a combination of their values
        @test CognitiveModeling.group_sessions(
            sessions_info, (:diagnosis, :site), :probability
        ) == (;
            group_of_each_session=[1, 2, 3, 4],
            n_groups=4,
            group_names=[
                :diagnosis_clinical__site_1,
                :diagnosis_control__site_1,
                :diagnosis_clinical__site_2,
                :diagnosis_control__site_2,
            ],
        )
        #Without columns, every session is in one group, also when the sessions have no table
        @test CognitiveModeling.group_sessions(sessions_info, (), :probability) == (;
            group_of_each_session=[1, 1, 1, 1], n_groups=1, group_names=[:all_sessions]
        )
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test CognitiveModeling.group_sessions(without_table, (), :probability).n_groups ==
            1
    end

    @testset "A session must have one value of each column it is grouped by, and the table must have the column" begin
        sessions_info = grouped_sessions()
        @test_throws "GroupedHierarchicalSessionParameters groups probability by the column wave, which has the values [1, 2] in the session \"S1\". Each session must have one value of every column its parameters are grouped by." CognitiveModeling.group_sessions(
            sessions_info, (:wave,), :probability
        )
        with_missing = grouped_sessions(
            merge(GROUPS_TABLE, (; site=[1, 1, 1, 1, missing, 2]))
        )
        @test_throws "GroupedHierarchicalSessionParameters groups probability by the column site, which has the values Union{Missing, Int64}[missing] in the session \"S3\"." CognitiveModeling.group_sessions(
            with_missing, (:site,), :probability
        )
        @test_throws "The table that the sessions were split from has no column age. Its columns are (:id, :diagnosis, :site, :wave, :outcome, :choice)." CognitiveModeling.group_sessions(
            sessions_info, (:age,), :probability
        )
        without_table = MultipleSessionsInfo(
            sessions_info.sessions, sessions_info.session_ids
        )
        @test_throws "The sessions were given already prepared, without the table they came from, so they have no column diagnosis." CognitiveModeling.group_sessions(
            without_table, (:diagnosis,), :probability
        )
    end

    @testset "CompositeSessionParameters takes parts which give different parameters" begin
        sampled = IndependentSessionParameters((; probability=1))
        fixed = FixedSessionParameters((; scale=2.0))
        composite = CompositeSessionParameters(sampled, fixed)
        @test composite.parts === (sampled, fixed)
        @test CognitiveModeling.parameter_names(composite) === (:probability, :scale)
        #It samples when any part does
        @test CognitiveModeling.estimates_parameters(composite)
        @test !CognitiveModeling.estimates_parameters(
            CompositeSessionParameters(fixed, NoParameters())
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
        by_session = FixedParametersBySession(
            Dict(
                "S4" => (; probability=0.4),
                "S1" => (; probability=0.1),
                "S3" => (; probability=0.3),
                "S2" => (; probability=0.2),
            ),
        )
        #The values are then one vector for each parameter, in the order of the sessions, in the same kind of model
        looked_up = CognitiveModeling.prepare_parameters_model(by_session, sessions_info)
        @test looked_up isa FixedParametersBySession
        @test looked_up.parameters_by_session == (; probability=[0.1, 0.2, 0.3, 0.4])
        @test CognitiveModeling.parameter_names(looked_up) === (:probability,)
        @test !CognitiveModeling.estimates_parameters(looked_up)
        #Preparing it again changes nothing, and the values can be given this way from the start
        @test CognitiveModeling.prepare_parameters_model(looked_up, sessions_info).parameters_by_session ===
            looked_up.parameters_by_session
        @test_throws "FixedParametersBySession has 2 values of probability, but there are 4 sessions. A vector must have one value for every session, in the order of the sessions." CognitiveModeling.prepare_parameters_model(
            FixedParametersBySession((; probability=[0.1, 0.2])), sessions_info
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
    end
end
