module Branch2Sequences # Li_L1_L2_L1

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
using ..SimulateIssuePreferences
using ..SimulateQuestionPreferences
using ..ProportionalEvaluationMetrics
using ..MajoritarianEvaluationMetrics

function run_proportional_election_sequence(fixed_params::FixedParams{K},
    issue_structure::IssueStructure{J}, representative_params::RepresentativesParams,
    question_structure::QuestionStructure{J}, ideal_points::AbstractVector{Array{Float64,3}}
) where {K,J}

    @inline function generate_parties(voter_ideal_points::AbstractVector{Array{Float64,3}},
        α_political_class::Float64, p_norm::Float64, n_issues::Int, issue_dimensions::AbstractVector,
        n_seats::Int, pop_per_seat::Int, rng::AbstractRNG)

        super_voter_ideal_points = PartySimulation.remodel_issue_space(
            voter_ideal_points, n_issues, n_seats, pop_per_seat, issue_dimensions
        )

        engagement, is_political_class = PartySimulation.compute_engagement(
            voter_ideal_points, α_political_class, p_norm, n_issues, n_seats, pop_per_seat, rng
        )

        reduced_points, party_ideal_points = PartySimulation.dimension_reduce_and_cluster_with_kmeans(
            super_voter_ideal_points, is_political_class, n_issues
        ) # parties is n_parties x n_issues

        n_parties = size(party_ideal_points)[1]

        voter_ideal_points = PartySimulation.recover_3d_after_pca(
            reduced_points, n_issues, n_seats, pop_per_seat
        )

        voter_ideal_points = [voter_ideal_points]
        party_ideal_points = [convert(Matrix{Float64}, party_ideal_points)]

        return voter_ideal_points, party_ideal_points, n_parties
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

    # lET'S MEASURE MEDIAN VOTER CONGRUENCE IT HERE

    @unpack n_seats, pop_per_seat, gamma, rng, n_iterations, sample_size = fixed_params
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_positions, n_questions = question_structure
    @unpack party_threshold, α_political_class, p_norm = representative_params

    voter_ideal_points, party_ideal_points, n_parties = generate_parties(ideal_points,
        α_political_class, p_norm, n_issues, issue_dimensions, n_seats, pop_per_seat, rng
    )

    n_questions = [sum(n_questions)]
    n_positions = [n_positions[1]]
    issue_dimensions = [n_issues]
    n_issues = 1

    voter_issue_weights = ones(Float64, n_issues, n_seats, pop_per_seat)

    voter_question_positions = SimulateQuestionPreferences.generate_question_positions(
        issue_dimensions, n_issues, n_questions, n_positions, voter_ideal_points, gamma, n_seats,
        pop_per_seat
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

    winning_parties, raw_vote_counts = run_proportional_election(preferred_parties, n_seats,
        pop_per_seat, party_threshold, n_parties)

    begin
        prop_eval_metrics =
            ProportionalEvaluationMetrics.evaluate_proportional_election(
                party_ideal_points, voter_question_positions, party_question_positions,
                voter_issue_weights, winning_parties, raw_vote_counts, n_parties,
                n_issues, n_questions, issue_dimensions, n_seats, pop_per_seat,
                party_threshold
            )
    end

    tangian_inputs = party_question_positions, winning_parties, n_parties

    begin
        return prop_eval_metrics, tangian_inputs, preferred_parties, voter_ideal_points,
        voter_question_positions, voter_issue_weights
    end
end

function run_majoritarian_sequence(fixed_params::FixedParams{K},
    issue_structure::IssueStructure{J}, branch_params::BranchParams{K},
    question_structure::QuestionStructure{J}, representative_params::RepresentativesParams,
    ideal_points::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, preferred_parties::Matrix{Int}
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

    @inline function candidate_entry(ideal_points::AbstractVector{Array{Float64,3}}, α::Float64,
        p_norm::Float64, n_candidates::Int, n_seats::Int, pop_per_seat::Int, α_entry::Float64,
        rng::AbstractRNG
    )

        engagement_matrix, is_political_class = CandidateSimulation.compute_engagement(
            ideal_points, α, p_norm, rng
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

        return candidate_indices
    end

    function run_majoritarian_election(ideal_points::AbstractVector{Array{Float64,3}},
        voter_issue_weights, candidates::Vector{Vector{Int}}, n_seats::Int, n_candidates::Int,
        n_issues::Int, pop_per_seat::Int, issue_dimensions::AbstractVector)

        begin
            first_round_results, voter_utilities, first_round_candidate_choices =
                ElectionSimulation.run_single_round_election(ideal_points, candidates,
                    voter_issue_weights, n_seats, n_candidates, n_issues, pop_per_seat,
                    issue_dimensions
                )
        end

        plurality_winners = ElectionSimulation.tally_top_1(first_round_results, n_seats)

        run_off_candidates = ElectionSimulation.tally_top_2(first_round_results, n_seats)

        second_round_results, _, _ = ElectionSimulation.run_single_round_election(ideal_points,
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
    @unpack n_candidates = branch_params
    @unpack α_political_class, p_norm, α_candidate_entry = representative_params

    n_questions = [sum(n_questions)]
    issue_dimensions = [n_issues]
    n_issues = 1

    candidates = candidate_entry(ideal_points, α_political_class, p_norm, n_candidates, n_seats,
        pop_per_seat, α_candidate_entry, rng)

    begin
        winning_candidates, voter_utilities_for_candidates, first_round_candidate_choices,
        plurality_winning_candidates = run_majoritarian_election(ideal_points, voter_issue_weights,
            candidates, n_seats, n_candidates, n_issues, pop_per_seat, issue_dimensions
        )
    end

    voter_utilities_for_candidates = HelpfulFunctions.scale_utilities(
        voter_utilities_for_candidates, 0.0, 1000.0
    )

    maj_evaluation = MajoritarianEvaluationMetrics.evaluate_majoritarian_election(
        voter_question_positions, voter_issue_weights, candidates, winning_candidates,
        voter_utilities_for_candidates, preferred_parties, n_seats, pop_per_seat, n_issues,
        n_questions, n_candidates
    )

    plurality_evaluation = MajoritarianEvaluationMetrics.evaluate_majoritarian_election(
        voter_question_positions, voter_issue_weights, candidates, plurality_winning_candidates,
        voter_utilities_for_candidates, preferred_parties, n_seats, pop_per_seat, n_issues,
        n_questions, n_candidates; plurality=true)


    tangian_inputs = winning_candidates
    return maj_evaluation, plurality_evaluation, tangian_inputs

end


end