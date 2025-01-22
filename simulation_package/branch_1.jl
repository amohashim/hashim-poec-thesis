cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
include("SimulationPackage.jl")

using Random
using StaticArrays
using Parameters

using .HashimPoecThesisSimulationPackage
using ..SimulationParameters
using ..BranchAgnosticSequences
using ..Branch1Sequences
using ..TangianIndices

function define_parameters()

    rng = MersenneTwister(2024)

    n_characteristics = 5
    n_groups = SVector{5,Int}([3, 3, 5, 5, 2])
    salience_to_probs = Dict{Symbol,SVector{4,Float64}}(
        :none => SVector{4,Float64}([1.0, 0.0, 0.0, 0.0]),
        :low => SVector{4,Float64}([0.15, 0.70, 0.10, 0.05]),
        :moderate => SVector{4,Float64}([0.05, 0.10, 0.70, 0.15]),
        :high => SVector{4,Float64}([0.05, 0.10, 0.15, 0.7]),
    )

    n_seats = 50
    pop_per_seat = 5000

    σ_none = 20.0
    σ_low = 10.0
    σ_moderate = 5.0
    σ_high = 1.0

    gamma = 1.0

    n_iterations = 100
    sample_size = 400
    mantel_permutations = 1000

    characteristic_type = SVector{n_characteristics,Symbol}(
        [:ordinal, :ordinal, :nominal, :nominal, :nominal]
    )
    homogeneity = SVector{n_characteristics,Symbol}(
        [:high, :low, :moderate, :low, :high]
    )

    n_metros = 3
    urbanization = 0.9
    urban_sprawl = 0.05
    spatial_dispersion = 0.05
    a_vals = SVector{n_characteristics,Float64}([0.5, 0.5, 0.5, 0.5, 0.5])


    n_issues = 1
    issue_dimensions = SVector{n_issues,Int}([1])

    demographic_salience = SVector{n_characteristics,Symbol}(
        [:moderate, :moderate, :moderate, :moderate, :moderate]
    )
    demographic_cleavage_salience = SMatrix{n_characteristics,n_issues,Int}(
        [2; 2; 2; 2; 0]
    )

    n_questions = SVector{n_issues,Int}([20])
    n_positions = SVector{n_issues,Int}([2])

    α_political_class = 0.01
    α_candidate_entry = 1.0
    p_norm = 5.0
    party_threshold = 0.05

    n_parties = 5
    n_candidates = 5
    turnout_level = 0.0
    strategic_level = 1.0
    demographic_attitudes = nothing

    fixed_params = FixedParams(rng, n_characteristics, n_groups, salience_to_probs, n_seats,
        pop_per_seat, σ_none, σ_low, σ_moderate, σ_high, gamma, n_iterations, sample_size,
        mantel_permutations
    )

    dem_char_params = DemographicCharacteristicParams(characteristic_type, homogeneity)
    spatial_params = SpatialCharacteristicParams(
        n_metros, urbanization, urban_sprawl, spatial_dispersion, a_vals
    )
    issue_structure = IssueStructure(n_issues, issue_dimensions)
    salience_structure = SalienceStructure(demographic_salience, demographic_cleavage_salience)
    question_structure = QuestionStructure(n_questions, n_positions)
    representative_params = RepresentativesParams(
        α_political_class, p_norm, α_candidate_entry, party_threshold
    )
    branch_params = BranchParams{n_characteristics}(
        n_parties, n_candidates, turnout_level, strategic_level,
        demographic_attitudes
    )


    begin
        return fixed_params, dem_char_params, spatial_params, issue_structure, salience_structure,
        question_structure, representative_params, branch_params
    end

end

function main()

    # df_sub = filter(row -> (row.X == 1 && row.Y == 1 && row.Z == 0), df)

    begin
        fixed_params, dem_char_params, spatial_params, issue_structure, salience_structure,
        question_structure, representative_params, branch_params = define_parameters()
    end

    statewide_demographic_dists, district_dists, coords = run_spatial_dist_sequence(
        fixed_params, dem_char_params, spatial_params
    )

    begin
        voters, ideal_points, ideal_means, ideal_variances, voter_issue_weights,
        voter_question_positions = run_voter_information_sequence(
            fixed_params, salience_structure, issue_structure, question_structure,
            district_dists
        )
    end

    spatial_corr_measurements = run_endogeneous_param_measurement_sequence(
        fixed_params, district_dists, coords
    )

    prop_eval_metrics, tangian_inputs, preferred_parties = run_proportional_election_sequence(
        fixed_params, issue_structure, representative_params, branch_params, question_structure,
        ideal_points, ideal_means, ideal_variances, voter_question_positions, voter_issue_weights
    )

    majoritarian_eval_metrics, winning_candidates = run_majoritarian_sequence(
        fixed_params, issue_structure, branch_params, question_structure, representative_params,
        ideal_points, voter_question_positions, voter_issue_weights, preferred_parties
    )

    party_question_positions, winning_parties, n_parties = tangian_inputs

    tangian_indices = computeTangianIndices(fixed_params, issue_structure, question_structure,
        voter_question_positions, party_question_positions, winning_candidates, winning_parties,
        n_parties
    )

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end