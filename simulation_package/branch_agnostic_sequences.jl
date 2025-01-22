module BranchAgnosticSequences

using Parameters
using Random
using StaticArrays
using Statistics

using ..SimulationParameters
using ..HelpfulFunctions
using ..ExogeneousDemographicCharacteristics
using ..EndogeneousDemogprahicCharacteristics
using ..SpatialCharacteristics
using ..SimulateIssuePreferences
using ..SimulateQuestionPreferences

using ..ExperimentalDesign: RAW_NAME_TO_NICE_NAME

export run_spatial_dist_sequence, run_voter_information_sequence
export run_endogeneous_param_measurement_sequence

function run_spatial_dist_sequence(fixed_params::FixedParams{K},
    dem_char_params::DemographicCharacteristicParams{K},
    spatial_params::SpatialCharacteristicParams{K}) where {K}

    @unpack n_groups, n_seats, rng = fixed_params
    @unpack homogeneity, characteristic_type = dem_char_params
    @unpack n_metros, spatial_dispersion, urbanization, urban_sprawl, a_vals = spatial_params

    begin
        statewide_demographic_dists =
            ExogeneousDemographicCharacteristics.generate_statewide_distributions(
                homogeneity, characteristic_type, n_groups
            ) # also need this for finding entropy
    end

    district_dists, coords = SpatialCharacteristics.generate_spatial_characteristics(
        n_metros, spatial_dispersion, n_seats, urbanization, urban_sprawl, a_vals,
        statewide_demographic_dists, rng
    )

    return statewide_demographic_dists, district_dists, coords

end

function run_voter_information_sequence(fixed_params::FixedParams{K},
    salience_structure::SalienceStructure{K,J}, issue_structure::IssueStructure{J},
    question_structure::QuestionStructure{J}, district_dists::Array{Float64,3}
) where {K,J}

    @unpack n_characteristics, salience_to_probs, pop_per_seat, n_seats, rng = fixed_params
    @unpack n_groups, σ_none, σ_low, σ_moderate, σ_high, gamma = fixed_params
    @unpack demographic_salience, demographic_cleavage_salience = salience_structure
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_questions, n_positions = question_structure

    """
    N x A x C x 3 array

    where N is the number of districts, A is the numnber of people per district, C is the number of 
    charactersitics, and 3 corresponds to each of the following: 
    1 => what group in the characteristic you're in (can be [1,2,...,d_max])
    2 => how salient your identity is (can be [0,1,2,3])
    3 => whether your identity is salient or not (can be [0,1])

    """
    function generate_voters(demographic_salience::AbstractVector{Symbol},
        n_characteristics::Int, mapping::Dict{Symbol,SVector{4,Float64}},
        node_dists::Array{Float64,3}, pop_per_node::Int, rng::AbstractRNG
    )::Array{Int,4}

        salience_probs = Matrix{Float64}(undef, n_characteristics, 4)
        @inbounds for i in 1:n_characteristics
            salience_probs[i, :] = mapping[demographic_salience[i]]
        end

        salience_lookup = EndogeneousDemogprahicCharacteristics.precompute_salience_lookup_table(
            SMatrix{n_characteristics,4,Float64}(salience_probs)
        )

        voter_demographics = EndogeneousDemogprahicCharacteristics.simulate_agents_with_salience(
            node_dists, salience_lookup, pop_per_node, rng
        )

        return voter_demographics

    end


    voters = generate_voters(demographic_salience, n_characteristics,
        salience_to_probs, district_dists, pop_per_seat, rng
    )

    begin
        ideal_points, ideal_means, ideal_variances =
            SimulateIssuePreferences.generate_scaled_ideal_points(
                demographic_cleavage_salience, n_characteristics, n_groups, n_issues,
                issue_dimensions, voters, σ_none, σ_low, σ_moderate, σ_high, rng
            )
    end

    voter_issue_weights = HelpfulFunctions.find_issue_weights(ideal_points,
        n_issues, n_seats, pop_per_seat, issue_dimensions
    )

    ideal_points, ideal_means, ideal_variances = collect.(
        [ideal_points, ideal_means, ideal_variances]
    )

    ideal_points = SVector{n_issues,Array{Float64,3}}(ideal_points)

    voter_question_positions = SimulateQuestionPreferences.generate_question_positions(
        issue_dimensions, n_issues, n_questions, n_positions, ideal_points, gamma,
        n_seats, pop_per_seat
    )

    begin
        return voters, ideal_points, ideal_means, ideal_variances,
        voter_issue_weights, voter_question_positions
    end

end

function run_endogeneous_param_measurement_sequence(fixed_params::FixedParams,
    district_dists::Array{Float64,3}, coords::Matrix{Float64})

    @unpack n_groups, n_characteristics, mantel_permutations, rng, n_seats = fixed_params
    begin
        agg_dist_corr, agg_dist_corr_p, char_level_corr, char_level_corr_p =
            SpatialCharacteristics.compute_mantel_spatial_correlation(district_dists, n_groups,
                coords, mantel_permutations, rng)
    end

    entropies = Matrix{Float64}(undef, n_seats, n_characteristics)

    for seat in 1:n_seats

        district_distributions = @view district_dists[seat, :, :]
        entropies[seat, :] = SpatialCharacteristics.district_entropies(district_distributions)

    end

    avgs = mean.(eachcol(entropies))
    medians = median.(eachcol(entropies))
    stds = std.(eachcol(entropies))

    begin
        return SpatialAutocorrelationMeasurement(agg_dist_corr, agg_dist_corr_p,
            char_level_corr, char_level_corr_p, avgs, medians, stds)
    end
end

function run_compile_results_sequence(spatial_corr_measurements::SpatialAutocorrelationMeasurement,
    prop_eval_metrics::ProportionalEvaluation, majoritarian_eval_metrics::MajoritarianEvaluation,
    tangian_indices::TangianIndicesResults)

    measurements = HelpfulFunctions.flatten_into_dict(
        spatial_corr_measurements, prop_eval_metrics, majoritarian_eval_metrics, tangian_indices;
        remove_substring="main.hashimpoecthesissimulationpackage."
    )

    ordered_result = OrderedDict(
        RAW_NAME_TO_NICE_NAME[key] => measurements[key] for key in keys(RAW_NAME_TO_NICE_NAME3)
    )

end


end