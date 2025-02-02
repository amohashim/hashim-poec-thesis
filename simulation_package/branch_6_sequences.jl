module Branch6Sequences # Li_L2_L1_L1

# Endogoenous Parties: NO
# Dynamic Candidates: YES
# Imperfect voters: NO

export run_proportional_election_sequence, run_majoritarian_sequence

using DataStructures
using Distributions
using Random
using Parameters
using StatsBase

using ..SimulationParameters
using ..HelpfulFunctions
using ..ExogeneousDemographicCharacteristics
using ..EndogeneousDemogprahicCharacteristics
using ..SpatialCharacteristics
using ..PartySimulation
using ..CandidateSimulation
using ..ElectionSimulation
using ..StrategicExit
using ..SimulateIssuePreferences
using ..SimulateQuestionPreferences
using ..ProportionalEvaluationMetrics
using ..MajoritarianEvaluationMetrics

function run_proportional_election_sequence(fixed_params::FixedParams{K},
    issue_structure::IssueStructure{J}, representative_params::RepresentativesParams,
    branch_params::BranchParams{1}, question_structure::QuestionStructure{J},
    voter_ideal_points::AbstractVector{Array{Float64,3}},
    voter_ideal_means::AbstractVector{Vector{Float64}},
    voter_ideal_variances::AbstractVector{Vector{Float64}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
) where {K,J}


    @inline function generate_parties(ideal_point_means::AbstractVector{Vector{Float64}},
        ideal_point_variances::AbstractVector{Vector{Float64}},
        n_parties::Int, n_issues::Int, issue_dimensions::AbstractVector{Int}, rng::AbstractRNG)

        covariances = PartySimulation.build_covariances(ideal_point_variances, n_issues)
        party_ideal_points = PartySimulation.structured_noise_generate_party_ideal_points(n_parties,
            ideal_point_means, covariances, n_issues, issue_dimensions, rng
        )

        return party_ideal_points
    end

    function run_proportional_election(preferred_parties::Matrix{Int}, n_seats::Int,
        pop_per_seat::Int, percent_party_vote_threshold::Float64, n_parties::Int)

        min_vote_count = percent_party_vote_threshold * n_seats * pop_per_seat
        raw_vote_counts = Dict(counter(preferred_parties))

        proportional_results = HelpfulFunctions.filter_parties_below_threshold(
            raw_vote_counts, min_vote_count, n_parties, false
        )
        winning_parties = ProportionalEvaluationMetrics.allocate_by_dhondt(
            n_seats, proportional_results
        )

        return winning_parties, raw_vote_counts

    end

    @unpack n_seats, pop_per_seat, gamma, rng, n_iterations, sample_size = fixed_params
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_parties = branch_params
    @unpack n_positions, n_questions = question_structure
    @unpack party_threshold = representative_params

    party_ideal_points = generate_parties(voter_ideal_means, voter_ideal_variances, n_parties,
        n_issues, issue_dimensions, rng
    )

    preferred_parties, proportional_voter_utilities =
        ElectionSimulation.assign_voters_to_parties!(voter_ideal_points, party_ideal_points,
            voter_issue_weights, n_parties, n_issues, n_seats, pop_per_seat, issue_dimensions
        )

    proportional_voter_utilities = HelpfulFunctions.scale_utilities(
        proportional_voter_utilities, 0.0, 1000.0)

    party_ideal_points = convert_party_ideal_points_to_arrs(party_ideal_points, n_issues, n_parties,
        issue_dimensions
    )

    party_question_positions = SimulateQuestionPreferences.generate_question_positions(
        issue_dimensions, n_issues, n_questions, n_positions, party_ideal_points, gamma, 1,
        n_parties
    )

    winning_parties, raw_vote_counts = run_proportional_election(preferred_parties,
        n_seats, pop_per_seat, party_threshold, n_parties
    )

    begin
        prop_eval_metrics =
            ProportionalEvaluationMetrics.evaluate_proportional_election(
                party_ideal_points, voter_question_positions, party_question_positions,
                voter_issue_weights, winning_parties, raw_vote_counts, n_parties,
                n_issues, n_questions, issue_dimensions, n_seats, pop_per_seat,
                party_threshold, preferred_parties
            )
    end

    tangian_inputs = party_question_positions, winning_parties, n_parties

    begin
        return prop_eval_metrics, tangian_inputs, preferred_parties
    end
end

function run_majoritarian_sequence(fixed_params::FixedParams{K},
    issue_structure::IssueStructure{J}, branch_params::BranchParams{1},
    question_structure::QuestionStructure{J}, representative_params::RepresentativesParams,
    voter_ideal_points::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, preferred_parties::Matrix{Int},
    β::Float64
) where {K,J}

    @inline function enforce_n_candidates(entry_vector::BitVector, n_candidates::Int)

        count_ones = count(bit -> bit == 1, entry_vector)

        if count_ones > n_candidates
            # Indices of all 1's in the BitVector
            ones_indices = findall(x -> x == 1, entry_vector)
            # Randomly select indices to turn to 0
            to_flip = rand(ones_indices, count_ones - n_candidates)
            # Turn those indices to 0
            entry_vector[to_flip] .= 0
        elseif count_ones < n_candidates
            # Indices of all 0's in the BitVector
            zeros_indices = findall(x -> x == 0, entry_vector)
            # Randomly select indices to turn to 1
            to_flip = rand(zeros_indices, n_candidates - count_ones)
            # Turn those indices to 1
            entry_vector[to_flip] .= 1
        end

        return entry_vector

    end

    @inline function candidate_entry(voter_ideal_points::AbstractVector{Array{Float64,3}}, α::Float64,
        p_norm::Float64, n_candidates::Int, n_seats::Int, pop_per_seat::Int, α_entry::Float64,
        rng::AbstractRNG
    )

        engagement_matrix, is_political_class = CandidateSimulation.compute_engagement(
            voter_ideal_points, α, p_norm, rng
        )

        engagement_matrix = scale_utilities(engagement_matrix) # scale to [0,1]
        candidate_indices = Vector{Vector{Int}}(undef, n_seats)

        @inbounds for seat in 1:n_seats

            engagements = @view engagement_matrix[seat, :]
            political_class_indices = collect(1:pop_per_seat)[@view is_political_class[seat, :]]
            probs = α_entry .* engagements
            for (i, prob) in enumerate(probs)
                if prob > 1.0
                    probs[i] = 1.0
                end

                if !(i ∈ political_class_indices)
                    probs[i] = 0.0
                end
            end

            candidate_enters = rand.(Bernoulli.(probs))
            candidate_enters = enforce_n_candidates(candidate_enters, n_candidates)
            local_indices = collect(1:pop_per_seat)[candidate_enters]

            try
                candidate_indices[seat] = sample(rng, local_indices, n_candidates, replace=false)
            catch
                local_indices = collect(1:pop_per_seat)
                candidate_indices[seat] = sample(rng, local_indices, n_candidates)
            end

        end

        return candidate_indices, engagement_matrix
    end

    function run_majoritarian_election(voter_ideal_points::AbstractVector{Array{Float64,3}},
        voter_question_postitions::AbstractVector{Array{Float64,3}},
        voter_issue_weights::Array{Float64,3}, candidates::Vector{Vector{Int}}, n_seats::Int,
        n_issues::Int, pop_per_seat::Int, issue_dimensions::AbstractVector,
        n_questions::AbstractVector{Int},
        directional_utility::Bool, β::Float64, strategic_voters::Union{BitMatrix,Nothing},
        turnout_voters::Union{BitMatrix,Nothing}, rng::AbstractRNG)

        begin
            first_round_results, voter_utilities, first_round_candidate_choices =
                StrategicExit.run_single_round_election_with_exit(voter_ideal_points,
                    voter_question_postitions, candidates, voter_issue_weights, n_seats, n_issues,
                    pop_per_seat, issue_dimensions, n_questions, rng;
                    use_directional_utility=directional_utility, β=β,
                    strategic_voters=strategic_voters, turnout_voters=turnout_voters,
                    n_initial_candidates=10
                )
        end

        plurality_winners = ElectionSimulation.tally_top_1(first_round_results, n_seats)

        run_off_candidates = ElectionSimulation.tally_top_2(first_round_results, n_seats)

        second_round_results, _, _ = ElectionSimulation.run_single_round_election(voter_ideal_points,
            run_off_candidates, voter_issue_weights, n_seats, 2, n_issues, pop_per_seat,
            issue_dimensions
        )

        winning_candidates = ElectionSimulation.tally_top_1(second_round_results, n_seats)

        begin
            return winning_candidates, voter_utilities, first_round_candidate_choices,
            plurality_winners
        end
    end

    @unpack n_seats, pop_per_seat, sample_size, n_iterations, rng = fixed_params
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_questions = question_structure
    @unpack directional_utility, strategic_level, turnout_level = branch_params
    @unpack α_political_class, p_norm, α_candidate_entry = representative_params

    n_initial_candidates = 10

    candidates, engagement_matrix = candidate_entry(voter_ideal_points, α_political_class, p_norm,
        n_initial_candidates, n_seats, pop_per_seat, α_candidate_entry, rng)

    if strategic_level != 404.0
        strategic_voters = HelpfulFunctions.get_bitmatrix_from_matrix_quartiles(engagement_matrix,
            strategic_level, true)
    else
        strategic_voters = nothing
    end # NEED TO TEST THIS

    if turnout_level != 404.0
        turnout_voters = HelpfulFunctions.get_bitmatrix_from_matrix_quartiles(engagement_matrix,
            turnout_level, true)
    else
        turnout_voters = nothing
    end # NEED TO TEST THIS

    begin
        winning_candidates, voter_utilities_for_candidates, first_round_candidate_choices,
        plurality_winning_candidates = run_majoritarian_election(voter_ideal_points,
            voter_question_positions, voter_issue_weights,
            candidates, n_seats, n_issues, pop_per_seat, issue_dimensions, n_questions,
            directional_utility, β, strategic_voters, turnout_voters, rng
        )
    end

    voter_utilities_for_candidates = HelpfulFunctions.scale_utilities(
        voter_utilities_for_candidates, 0.0, 1000.0
    )

    maj_evaluation = MajoritarianEvaluationMetrics.evaluate_majoritarian_election(
        voter_question_positions, voter_issue_weights, candidates, winning_candidates,
        voter_utilities_for_candidates, preferred_parties, n_seats, pop_per_seat, n_issues,
        n_questions, n_initial_candidates
    )

    plurality_evaluation = MajoritarianEvaluationMetrics.evaluate_majoritarian_election(
        voter_question_positions, voter_issue_weights, candidates, plurality_winning_candidates,
        voter_utilities_for_candidates, preferred_parties, n_seats, pop_per_seat, n_issues,
        n_questions, 10; plurality=true)

    tangian_inputs = winning_candidates
    return maj_evaluation, plurality_evaluation, tangian_inputs

end

end