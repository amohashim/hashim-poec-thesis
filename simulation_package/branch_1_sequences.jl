module Branch1Sequences

export run_proportional_election_sequence, run_majoritarian_sequence

using DataStructures
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
    branch_params::BranchParams{K}, question_structure::QuestionStructure{J},
    ideal_points::AbstractVector{Array{Float64,3}}, ideal_means::AbstractVector{Vector{Float64}},
    ideal_variances::AbstractVector{Vector{Float64}},
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
        # counts = counter(preferred_parties)
        # qualified_parties = filter_dict(Dict(counts), min_vote_count)
        # counting_votes = sum(values(qualified_parties))

        # proportional_results = Dict(
        #     party => count / counting_votes for (party, count) in counts
        # )

        # for party in 1:n_parties
        #     if !haskey(proportional_results, party)
        #         proportional_results[party] = 0.0
        #     end
        # end

        return winning_parties

    end

    @unpack n_seats, pop_per_seat, gamma, rng, n_iterations, sample_size = fixed_params
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_parties = branch_params
    @unpack n_positions, n_questions = question_structure
    @unpack party_threshold = representative_params

    party_ideal_points = generate_parties(ideal_means, ideal_variances, n_parties,
        n_issues, issue_dimensions, rng
    )

    preferred_parties, proportional_voter_utilities =
        ElectionSimulation.assign_voters_to_parties!(ideal_points, party_ideal_points,
            voter_issue_weights, n_parties, n_issues, n_seats, pop_per_seat, issue_dimensions
        )

    proportional_voter_utilities = HelpfulFunctions.scale_utilities(proportional_voter_utilities)

    party_ideal_points = convert_party_ideal_points_to_arrs(party_ideal_points, n_issues, n_parties,
        issue_dimensions
    )

    party_issue_weights = HelpfulFunctions.find_issue_weights(party_ideal_points,
        n_issues, 1, n_parties, issue_dimensions
    )

    party_question_positions = SimulateQuestionPreferences.generate_question_positions(
        issue_dimensions, n_issues, n_questions, n_positions, party_ideal_points, gamma, 1,
        n_parties
    )

    winning_parties = run_proportional_election(preferred_parties, n_seats,
        pop_per_seat, party_threshold, n_parties)

    begin
        prop_eval_metrics =
            ProportionalEvaluationMetrics.evaluate_proportional_election(
                party_ideal_points, voter_question_positions, party_question_positions,
                voter_issue_weights, winning_parties, n_parties,
                n_issues, n_questions, issue_dimensions, n_seats, pop_per_seat
            )
    end

    tangian_inputs = voter_question_positions, winning_parties

    return prop_eval_metrics, tangian_inputs
end

function run_majoritarian_sequence(fixed_params::FixedParams{K},
    issue_structure::IssueStructure{J}, branch_params::BranchParams{K},
    question_structure::QuestionStructure{J}, representative_params::RepresentativesParams,
    ideal_points::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}
) where {K,J}

    @inline function candidate_entry(ideal_points::AbstractVector{Array{Float64,3}}, α::Float64,
        p_norm::Float64, n_candidates::Int, n_seats::Int, pop_per_seat::Int, rng::AbstractRNG
    )

        _, is_political_class = CandidateSimulation.compute_engagement(
            ideal_points, α, p_norm, rng
        )
        candidate_indices = Vector{Vector{Int}}(undef, n_seats)

        @inbounds for seat in 1:n_seats

            local_indices = collect(1:pop_per_seat)[@view is_political_class[seat, :]]
            candidate_indices[seat] = sample(rng, local_indices, n_candidates)

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

        run_off_candidates = ElectionSimulation.tally_top_2(first_round_results, n_seats)

        second_round_results, _, _ = ElectionSimulation.run_single_round_election(ideal_points,
            run_off_candidates, voter_issue_weights, n_seats, 2, n_issues, pop_per_seat,
            issue_dimensions
        )

        winning_candidates = ElectionSimulation.tally_top_1(second_round_results, n_seats)

        return winning_candidates, voter_utilities, first_round_candidate_choices

    end

    @unpack n_seats, pop_per_seat, sample_size, n_iterations, rng = fixed_params
    @unpack n_issues, issue_dimensions = issue_structure
    @unpack n_questions = question_structure
    @unpack n_candidates = branch_params
    @unpack α_political_class, p_norm = representative_params

    candidates = candidate_entry(ideal_points, α_political_class, p_norm, n_candidates, n_seats,
        pop_per_seat, rng)

    winning_candidates, majoritarian_voter_utilities, first_round_candidate_choices =
        run_majoritarian_election(ideal_points, voter_issue_weights, candidates, n_seats,
            n_candidates, n_issues, pop_per_seat, issue_dimensions
        )

    majoritarian_voter_utilities = HelpfulFunctions.scale_utilities(majoritarian_voter_utilities)
    majoritarian_rankings = ElectionSimulation.compute_voter_rankings(majoritarian_voter_utilities,
        n_seats, pop_per_seat, n_candidates
    )

    # VSE MAJORITARIAN
    average_results = MajoritarianEvaluationMetrics.evaluate_majoritarian_election(
        voter_question_positions, voter_issue_weights, candidates, majoritarian_rankings,
        n_issues, n_seats, n_candidates, pop_per_seat, n_questions, winning_candidates,
        first_round_candidate_choices, sample_size, n_iterations, rng
    )

    return average_results

end

end
