#####################################################
#  A => DemographicCharacteristicParams
#####################################################
using StaticArrays

"""
Return a DemographicCharacteristicParams depending on factor A's level.
Possible example levels: "L1", "L2", "L3".
"""
function getDemographicCharacteristicParams(levelA::String)
    # Here you set your custom logic
    # For example, maybe "L1" => 2 groups, "L2" => 4 groups, "L3" => 6 groups, etc.
    # Also you might choose characteristic_type, homogeneity
    if levelA == "L1"
        characteristic_type = @SVector [:ordinal, :ordinal, :ordinal, :nominal, :nominal]
        homogeneity = @SVector [:perfect, :perfect, :perfect, :perfect, :perfect]
    elseif levelA == "L2"
        characteristic_type = @SVector [:ordinal, :ordinal, :ordinal, :nominal, :nominal]
        homogeneity = @SVector [:perfect, :high, :moderate, :moderate, :low]
    elseif levelA == "L3"
        characteristic_type = @SVector [:ordinal, :ordinal, :ordinal, :nominal, :nominal]
        homogeneity = @SVector [:low, :low, :low, :low, :low]
    else
        throw(ArgumentError("unknown level"))
    end
    return DemographicCharacteristicParams(characteristic_type, homogeneity)
end

#####################################################
#  B => SpatialCharacteristicParams
#####################################################
function getSpatialCharacteristicParams(levelB::String)
    # Example: "L1" => fewer metros, "L2" => more metros, etc.
    if levelB == "L1"
        n_metros = 3
        urbanization = 0.3
        urban_sprawl = 0.2
        spatial_dispersion = 0.1
        a_vals = @SVector [1.0, 1.0, 1.0, 1.0, 1.0]  # Example of how to fill
    elseif levelB == "L2"
        n_metros = 5
        urbanization = 0.5
        urban_sprawl = 0.4
        spatial_dispersion = 0.2
        a_vals = @SVector [0.5, 0.5]
    elseif levelB == "L3"
        n_metros = 10
        urbanization = 0.7
        urban_sprawl = 0.6
        spatial_dispersion = 0.3
        a_vals = @SVector [2.0, 3.0]
    else
        throw(ArgumentError("unknown level"))
    end
    return SpatialCharacteristicParams(
        n_metros,
        urbanization,
        urban_sprawl,
        spatial_dispersion,
        a_vals
    )
end

#####################################################
#  C, D => SalienceStructure
#####################################################
function getSalienceStructure(levelC::String, levelD::String)
    # Suppose factor C picks from e.g. [:low, :medium, :high] salience
    # Suppose factor D modifies the cleavage matrix dimension.
    # This is very hypothetical; adjust as needed.

    #C_{e}_{level}

    if levelC == "C1_1"
        demographic_salience = @SVector [:none, :none, :none, :none, :none]
    elseif levelC == "C1_2"
        demographic_salience = @SVector [:none, :low, :moderate, :moderate, :high]
    elseif levelC == "C1_3"
        demographic_salience = @SVector [:high, :high, :high, :high, :high]
    elseif levelC == "C2_1"
        @SMatrix [:none, :none, :none, :none, :none]
    elseif levelC == "C2_2"
        @SMatrix [:none, :none, :none, :none, :none]
    elseif levelC == "C3_2"
        @SMatrix [:none, :none, :none, :none, :none]
    end

    if levelE == "L1"
        # Suppose factor D modifies the cleavage salience with a 2D or 3D structure
        if levelD == "L1"
            demographic_cleavage_salience = @SMatrix [0 0 0 0 0;]
        elseif levelD == "L2"
            demographic_cleavage_salience = @SMatrix [0 1 1 2 2;]
        elseif levelD == "L3"
            demographic_cleavage_salience = @SMatrix [3 3 3 3 3;]
        else
            error("Unknown D level $levelD")
        end

    elseif levelE == "L2"

        if levelD == "L1"
            demographic_cleavage_salience = @SMatrix [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0]
        elseif levelD == "L2"
            demographic_cleavage_salience = @SMatrix [0 1 1 2 2; 0 1 1 2 2; 0 0 0 0 0]
        elseif levelD == "L3"
            demographic_cleavage_salience = @SMatrix [3 3 3 3 3; 3 3 3 3; 3 3 3 3 3]
        else
            error("Unknown D level $levelD")
        end

    else

        if levelD == "L1"
            demographic_cleavage_salience = @SMatrix zeros(Int, 5, 5)
        elseif levelD == "L2"
            demographic_cleavage_salience = @SMatrix [
                0 1 1 2 2; 0 1 1 2 2; 2 2 1 1 0; 2 2 1 1 0; 0 0 0 0 0
            ]
        elseif levelD == "L3"
            demographic_cleavage_salience = @SMatrix [
                3 3 3 3 3; 3 3 3 3 3; 2 2 2 2 2; 3 3 3 3 3; 0 0 0 0 0
            ]
        else
            error("Unknown D level $levelD")
        end

    end

    return SalienceStructure(demographic_salience, demographic_cleavage_salience)
end

#####################################################
#  E => IssueStructure
#####################################################
function getIssueStructure(levelE::String)
    # Suppose "L1" => 2 issues, 1 dimension each
    #        "L2" => 3 issues, 2 dimension each
    #        "L3" => 5 issues, 2 dimension each
    if levelE == "L1"
        n_issues = 1
        issue_dimensions = @SVector [1]
    elseif levelE == "L2"
        n_issues = 3
        issue_dimensions = @SVector [2, 2, 2]
    elseif levelE == "L3"
        n_issues = 5
        issue_dimensions = @SVector [1, 2, 2, 3, 3]
    else
        error("Unknown E level $levelE")
    end
    return IssueStructure(n_issues, issue_dimensions)
end

#####################################################
#  F => QuestionStructure
#####################################################
function getQuestionStructure(levelF::String, levelE::String)

    # Suppose we vary the # questions and # positions:

    if levelE == "L1"
        if levelF == "L1"
            n_questions = @SVector [20]
            n_positions = @SVector [2]
        elseif levelF == "L2"
            n_questions = @SVector [20]
            n_positions = @SVector [3]
        elseif levelF == "L3"
            n_questions = @SVector [20]
            n_positions = @SVector [5]
        else
            error("Unknown F level $levelF")
        end
    elseif levelE == "L2"
        if levelF == "L1"
            n_questions = @SVector [10, 10, 5]
            n_positions = @SVector [2, 2, 2]
        elseif levelF == "L2"
            n_questions = @SVector [10, 10, 5]
            n_positions = @SVector [3, 3, 3]
        elseif levelF == "L3"
            n_questions = @SVector [10, 10, 5]
            n_positions = @SVector [5, 5, 5]
        else
            error("Unknown F level $levelF")
        end
    else
        levelE == "L3"
        if levelF == "L1"
            n_questions = @SVector [2, 3, 5, 5, 5]
            n_positions = @SVector [2]
        elseif levelF == "L2"
            n_questions = @SVector [2, 3, 5, 5, 5]
            n_positions = @SVector [3, 3, 3]
        elseif levelF == "L3"
            n_questions = @SVector [2, 3, 5, 5, 5]
            n_positions = @SVector [5, 5, 5]
        else
            error("Unknown F level $levelF")
        end
    end
    return QuestionStructure(n_questions, n_positions)
end

#####################################################
#  G => α_political_class, p_norm
#  H => party_threshold
#  I => α_candidate_entry
# => Combined in RepresentativesParams
#####################################################
function getRepresentativesParams(levelG::String, levelH::String, levelI::String)
    # G => alpha_political_class, p_norm
    if levelG == "L1"
        alpha_political_class = 1.0
        p_norm = 3.0
    elseif levelG == "L2"
        alpha_political_class = 0.5
        p_norm = 3.0
    elseif levelG == "L3"
        alpha_political_class = 0.25
        p_norm = 3.0
    else
        error("Unknown G level $levelG")
    end

    # H => party_threshold
    if levelH == "L1"
        party_threshold = 0.05
    elseif levelH == "L2"
        party_threshold = 0.075
    elseif levelH == "L3"
        party_threshold = 0.10
    else
        error("Unknown H level $levelH")
    end

    # I => alpha_candidate_entry
    if levelI == "L1"
        alpha_candidate_entry = 1.0
    elseif levelI == "L2"
        alpha_candidate_entry = 0.5
    elseif levelI == "L3"
        alpha_candidate_entry = 0.25
    else
        error("Unknown I level $levelI")
    end

    return RepresentativesParams(
        alpha_political_class,
        p_norm,
        alpha_candidate_entry,
        party_threshold
    )
end

#####################################################
#  J => n_parties
#  K => n_candidates
#  L => turnout_level
#  M => strategic_level
#  N => demographic_attitudes
# => Combined in BranchParams
#####################################################
function getBranchParams(X::Int, Y::Int, Z::Int,
    levelJ, levelK, levelL, levelM, levelN)
    # If X=1 => we use J, else ignore
    n_parties = (X == 1 && !ismissing(levelJ)) ? parse(Int, levelJ) : 0

    # If Y=1 => we use K, else ignore
    n_candidates = (Y == 1 && !ismissing(levelK)) ? parse(Int, levelK) : 0

    # If Z=1 => we use L, M, N, else ignore
    turnout_level = (Z == 1 && !ismissing(levelL)) ? parse(Float64, levelL) : 1.0
    strategic_level = (Z == 1 && !ismissing(levelM)) ? parse(Float64, levelM) : 0.0

    # N might be more complicated if it’s a matrix or vector of attitudes
    # For an example, we’ll store a random or single matrix if Z=1
    # In your real code, you might parse "L1", "L2", etc. differently:
    demographic_attitudes = nothing
    if Z == 1 && !ismissing(levelN)
        # Suppose we interpret "L1", "L2", "L3" etc. as different attitude matrices.
        # Or maybe the design table literally stores "somefile.csv" or "0.2" etc.
        # Example:
        if levelN == "L1"
            # 2×2 matrix
            demographic_attitudes = @SVector [Float64[0.1 0.2; 0.2 0.3],
                Float64[0.3 0.1; 0.1 0.1]]
        elseif levelN == "L2"
            demographic_attitudes = @SVector [Float64[0.5 0.1; 0.0 0.2]]
        elseif levelN == "L3"
            demographic_attitudes = @SVector [Float64[0.7 0.7; 0.7 0.7]]
        else
            # or parse JSON, etc.
            error("Unknown N level $levelN")
        end
    end

    return BranchParams(
        n_parties,
        n_candidates,
        turnout_level,
        strategic_level,
        demographic_attitudes
    )
end
