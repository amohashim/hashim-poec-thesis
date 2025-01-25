module MajoritarianEvaluationMetrics

using DataStructures
using Distributions
using Random
using StatsBase
using Statistics
using StaticArrays

using ..HelpfulFunctions
using ..ElectionSimulation
using ..CondorcetSmithFunctions

export MajoritarianUtilitySummaryStats, MajoritarianUtilityMetrics
export MajoritarianVotingEfficiencyMetrics, MajoritarianDecisiveBodyMetrics
export MajoritarianEvaluation, PluratarianEvaluation

struct MajoritarianUtilitySummaryStats

    avg::Float64
    median_::Float64
    sd::Float64

    function MajoritarianUtilitySummaryStats(avg::Float64, median::Float64, sd::Float64)
        new(avg, median, sd)
    end

    function MajoritarianUtilitySummaryStats(utility_vector::AbstractArray)

        avg = mean(utility_vector)
        median_ = median(utility_vector)
        sd = std(utility_vector)

        new(avg, median_, sd)

    end

end

struct MajoritarianUtilityMetrics

    util_maximizer_election_rate::Float64
    utility_from_winner::MajoritarianUtilitySummaryStats
    utility_from_maximizer::MajoritarianUtilitySummaryStats
    utility_efficiency::MajoritarianUtilitySummaryStats

    function MajoritarianUtilityMetrics(util_maximizer_election_rate::Float64,
        utility_from_winner::MajoritarianUtilitySummaryStats,
        utility_from_maximizer::MajoritarianUtilitySummaryStats,
        utility_efficiency::MajoritarianUtilitySummaryStats)

        new(util_maximizer_election_rate, utility_from_winner, utility_from_maximizer,
            utility_efficiency)

    end

    function MajoritarianUtilityMetrics(winner_util_vector::AbstractArray,
        maxer_util_vector::AbstractArray, util_efficiency_vector::AbstractArray,
        util_maxer_elected_vector::AbstractArray)

        @assert all(i -> i ∈ [1.0, 0.0], util_maxer_elected_vector)

        utility_from_winner = MajoritarianUtilitySummaryStats(winner_util_vector)
        utility_from_maximizer = MajoritarianUtilitySummaryStats(maxer_util_vector)
        utility_efficiency = MajoritarianUtilitySummaryStats(util_efficiency_vector)

        util_maxer_elected_vector = convert(BitVector, util_maxer_elected_vector)
        begin
            util_maximizer_election_rate =
                count(util_maxer_elected_vector) / length(util_maxer_elected_vector)
        end

        new(util_maximizer_election_rate, utility_from_winner, utility_from_maximizer,
            utility_efficiency
        )

    end

end

struct MajoritarianVotingEfficiencyMetrics

    condorcet_efficiency::Float64
    smith_efficiency::Float64
    n_condorcet_paradox::Int

    function MajoritarianVotingEfficiencyMetrics(condorcet_efficiency::Float64,
        smith_efficiency::Float64, n_condorcet_paradox::Int)

        new(condorcet_efficiency, smith_efficiency, n_condorcet_paradox)

    end

    function MajoritarianVotingEfficiencyMetrics(condorcet_vector::AbstractArray,
        smith_vector::AbstractArray, paradox_vector::AbstractArray)

        n_seats = length(smith_vector)
        n_c, n_s = count(condorcet_vector), count(smith_vector)
        n_paradox = count(paradox_vector)

        c_eff = n_seats == n_paradox ? 2.0 : n_c / (n_seats - n_paradox)
        s_eff = n_s / n_seats

        new(c_eff, s_eff, n_paradox)

    end

end

struct MajoritarianDecisiveBodyMetrics

    social_utility_from_body::Float64
    non_zero_positions::Int
    median_position::Union{Int,Float64}

end

struct PluratarianEvaluation

    utility_metrics::MajoritarianUtilityMetrics
    voting_efficiency_stats::MajoritarianVotingEfficiencyMetrics
    strict_decisive_body_metrics::MajoritarianDecisiveBodyMetrics
    qualified_decisive_body_metrics::MajoritarianDecisiveBodyMetrics
    maj_gallagher_index::Float64
    vse::Float64

end

struct MajoritarianEvaluation

    utility_metrics::MajoritarianUtilityMetrics
    voting_efficiency_stats::MajoritarianVotingEfficiencyMetrics
    strict_decisive_body_metrics::MajoritarianDecisiveBodyMetrics
    qualified_decisive_body_metrics::MajoritarianDecisiveBodyMetrics
    maj_gallagher_index::Float64
    vse::Float64

end

function find_district_info(seat::Int, candidates::Vector{Vector{Int}},
    winning_candidates::Vector{Int}, voter_utilities_from_candidate_profiles::Array{Float64,3},
    smith_sets::Vector{Vector{Int}}, condorcet_winners::Vector{Union{Int,Nothing}})

    district_candidates = candidates[seat]

    # Find info about utilities for candidates
    utilities_by_candidate = @view voter_utilities_from_candidate_profiles[seat, :, :]
    district_social_utilities = sum(utilities_by_candidate, dims=1)[:]

    winning_candidate = winning_candidates[seat]
    winner_local_indx = findfirst(==(winning_candidate), district_candidates)

    district_smith_set = smith_sets[seat]
    district_condorcet_winner = condorcet_winners[seat]

    begin
        return winner_local_indx, district_social_utilities, district_smith_set,
        district_condorcet_winner
    end

end

function find_candidate_profiles_by_district(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    candidates::AbstractVector{Vector{Int}}, n_issues::Int, n_questions::AbstractVector{Int},
    n_seats::Int, n_candidates::Int)

    candidate_profiles = Vector{Array{Float64,3}}(undef, n_issues)

    @inbounds for issue in 1:n_issues

        n_q = n_questions[issue]
        issue_profiles = Array{Float64,3}(undef, n_q, n_seats, n_candidates)
        issue_question_positions = voter_question_positions[issue]

        @inbounds for seat in 1:n_seats
            district_candidates = candidates[seat]
            begin
                issue_profiles[:, seat, 1:n_candidates] =
                    issue_question_positions[:, seat, district_candidates]
            end
        end

        candidate_profiles[issue] = issue_profiles

    end

    return candidate_profiles

end

# function compute_utilities_for_candidate_profiles(
#     voter_question_positions::AbstractVector{Array{Float64,3}},
#     candidate_profiles::AbstractVector{Array{Float64,3}},
#     voter_issue_weights::Array{Float64,3},
#     n_candidates_per_district::Int, n_seats::Int, pop_per_seat::Int, n_issues::Int)

#     voter_utilities = Array{Float64,3}(undef, n_seats, pop_per_seat, n_candidates_per_district)

#     @inbounds for issue in 1:n_issues

#         @inbounds for seat in 1:n_seats

#             profile = @view candidate_profiles[issue][:, seat, :]

#             district_positions = @view voter_question_positions[issue][:, seat, :]
#             seat_issue_weights = @view voter_issue_weights[issue, seat, :]
#             utilities = [
#                 sum(abs.(district_positions[:, i] .- profile[:, j]))
#                 for i in 1:pop_per_seat, j in 1:n_candidates_per_district
#             ]

#             voter_utilities[seat, :, :] += -1 .* utilities .* @view seat_issue_weights[:, :]

#         end
#     end

#     return voter_utilities

# end

function compute_utilities_for_candidate_profiles(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    candidate_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    n_candidates_per_district::Int,
    n_seats::Int,
    pop_per_seat::Int,
    n_issues::Int
)
    # Pre-allocate the result and initialize to zero so we can accumulate.
    voter_utilities = zeros(Float64, n_seats, pop_per_seat, n_candidates_per_district)

    @inbounds @views for issue in 1:n_issues
        for seat in 1:n_seats
            # Extract the relevant slices:
            # candidate_profiles[issue][:, seat, :] is Qᵢ × n_candidates_per_district
            # voter_question_positions[issue][:, seat, :] is Qᵢ × pop_per_seat
            # voter_issue_weights[issue, seat, :] is pop_per_seat

            profile = candidate_profiles[issue][:, seat, :]
            district_positions = voter_question_positions[issue][:, seat, :]
            seat_issue_weights = voter_issue_weights[issue, seat, :]

            Qᵢ = size(profile, 1)

            @inbounds for i in 1:pop_per_seat
                for j in 1:n_candidates_per_district
                    s = 0.0
                    @inbounds for q in 1:Qᵢ
                        s += (district_positions[q, i] - profile[q, j])^2
                    end
                    voter_utilities[seat, i, j] += -s * seat_issue_weights[i]
                end
            end
        end
    end

    return voter_utilities
end


function evaluate_candidate_profiles(voter_question_positions::AbstractVector{Array{Float64,3}},
    candidates::AbstractVector{Vector{Int}}, voter_issue_weights::Array{Float64,3},
    n_issues::Int, n_questions::AbstractVector{Int}, pop_per_seat::Int,
    n_seats::Int, n_candidates::Int)

    candidate_profiles = find_candidate_profiles_by_district(voter_question_positions, candidates,
        n_issues, n_questions, n_seats, n_candidates)

    voter_utilities_from_candidate_profiles = compute_utilities_for_candidate_profiles(
        voter_question_positions, candidate_profiles, voter_issue_weights, n_candidates,
        n_seats, pop_per_seat, n_issues)

    voter_utilities_from_candidate_profiles = HelpfulFunctions.scale_utilities(
        voter_utilities_from_candidate_profiles, 0.0, 1000.0
    )

    return candidate_profiles, voter_utilities_from_candidate_profiles
end


function determine_district_utility_metrics(winning_candidate_local_index::Int,
    social_utilities::Vector{Float64})

    utility_maxer_elected = 0.0

    utility_from_winner = social_utilities[winning_candidate_local_index]
    util_from_maximizer, util_maxer = findmax(social_utilities)

    if (winning_candidate_local_index == util_maxer) || (utility_from_winner == util_from_maximizer)
        utility_maxer_elected = 1.0
    end

    utility_efficiency = utility_from_winner / util_from_maximizer

    return utility_from_winner, util_from_maximizer, utility_efficiency, utility_maxer_elected

end

function determine_condorcet_or_smith_for_district(winner_local_index::Int,
    district_smith_set::Vector{Int}, condorcet_winner::Union{Int,Nothing})

    winner_is_condorcet = false
    winner_in_smith_set = false
    condorcet_paradox = false

    if winner_local_index ∈ district_smith_set
        winner_in_smith_set = true
        if isnothing(condorcet_winner)
            condorcet_paradox = true
        elseif winner_local_index == condorcet_winner
            winner_is_condorcet = true
        end
    end

    return winner_is_condorcet, winner_in_smith_set, condorcet_paradox

end

function find_decisive_body_member_positions(candidates::Vector{Vector{Int}},
    winning_candidates::Vector{Int}, candidate_profiles::Vector{Array{Float64,3}},
    n_issues::Int, n_questions::AbstractVector{Int},
    n_seats::Int)

    candidate_local_indices = [
        findfirst(==(winning_candidates[n]), candidates[n]) for n in 1:n_seats
    ]

    member_positions = Vector{Matrix{Float64}}(undef, n_issues)

    @inbounds for issue in 1:n_issues

        n_q = n_questions[issue]
        all_candidate_profiles_on_issue = candidate_profiles[issue]

        profiles_for_issue = Matrix{Float64}(undef, n_q, n_seats)

        @inbounds for seat in 1:n_seats

            winner_local_idx = candidate_local_indices[seat]
            profiles_for_issue[:, seat] = all_candidate_profiles_on_issue[:, seat, winner_local_idx]

        end

        member_positions[issue] = profiles_for_issue

    end

    return member_positions

end

function find_decisive_body_positions(member_positions::AbstractVector{Matrix{Float64}},
    n_issues::Int, n_questions::AbstractVector{Int}, n_seats::Int)

    # Return the majority key from the accumulator, or 0.0 if no key
    # reaches at least ceil(n_seats/2) counts.
    function majority_key(acc::AbstractDict{Float64,Int}, n_seats::Int)
        threshold = ceil(Int, n_seats / 2)
        # Loop over key => count pairs.
        for (k, cnt) in acc
            if cnt >= threshold
                return k
            end
        end
        return 0.0
    end

    # Return the weighted median as described.
    function weighted_median(acc::AbstractDict{Float64,Int})
        pos_total = 0
        neg_total = 0
        # First pass: compute the total counts for nonnegative (k ≥ 0)
        # and nonpositive (k ≤ 0) subsets.
        for (k, cnt) in acc
            if k >= 0
                pos_total += cnt
            end
            if k <= 0
                neg_total += cnt
            end
        end
        # Choose subset: if totals are equal or nonnegative is larger, select keys k ≥ 0;
        # otherwise, select keys k ≤ 0.
        chosen = if pos_total >= neg_total
            # Build a vector of tuples (key, count) for keys ≥ 0.
            [(k, cnt) for (k, cnt) in acc if k >= 0]
        else
            [(k, cnt) for (k, cnt) in acc if k <= 0]
        end
        # If the chosen subset is empty, return 0.0.
        if isempty(chosen)
            return 0.0
        end
        # Sort by key in ascending order.
        sort!(chosen, by=x -> x[1])
        total = 0
        for (_, cnt) in chosen
            total += cnt
        end
        # The median index is (total + 1) // 2 (lower median).
        med_index = div(total + 1, 2)
        # Walk through the sorted chosen subset and return the key where we cross med_index.
        cum = 0
        for (k, cnt) in chosen
            cum += cnt
            if cum >= med_index
                return k
            end
        end
        return 0.0  # Fallback (should not be reached).
    end

    # Process a vector of accumulator objects and produce two vectors:
    # - The first contains the majority key for each accumulator,
    # - The second contains the weighted median for each accumulator.
    function process_accumulators(accumulators::Vector{<:AbstractDict{Float64,Int}}, n_seats::Int)
        n = length(accumulators)
        maj_keys = Vector{Float64}(undef, n)
        medians = Vector{Float64}(undef, n)
        for i in 1:n
            acc = accumulators[i]
            maj_keys[i] = majority_key(acc, n_seats)
            medians[i] = weighted_median(acc)
        end
        return maj_keys, medians
    end

    decisive_body_profile_strict = Vector{Array{Float64,3}}(undef, n_issues)
    decisive_body_profile_qualified = Vector{Array{Float64,3}}(undef, n_issues)

    @inbounds for issue in 1:n_issues

        n_q = n_questions[issue]
        issue_positions = member_positions[issue]

        strict_issue_profile = Array{Float64,3}(undef, n_q, 1, 1)
        qualified_issue_profile = Array{Float64,3}(undef, n_q, 1, 1)

        question_support_groups = counter.(eachrow(issue_positions))

        strict_positions, qualified_positions = process_accumulators(
            question_support_groups, n_seats
        )

        strict_issue_profile[1:n_q, 1, 1] = strict_positions
        qualified_issue_profile[1:n_q, 1, 1] = qualified_positions

        decisive_body_profile_strict[issue] = strict_issue_profile
        decisive_body_profile_qualified[issue] = qualified_issue_profile

    end

    return decisive_body_profile_strict, decisive_body_profile_qualified

end

function evaluate_decisive_body_profile(decisive_body_profile::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, n_seats::Int, pop_per_seat::Int, n_issues::Int)

    voter_utilities_for_decisive_body_profile = compute_utilities_for_candidate_profiles(
        voter_question_positions, decisive_body_profile, voter_issue_weights, 1, n_seats,
        pop_per_seat, n_issues)

    voter_utilities_for_decisive_body_profile = HelpfulFunctions.scale_utilities(
        voter_utilities_for_decisive_body_profile, 0.0, 1000.0
    )

    social_utility_from_body = sum(voter_utilities_for_decisive_body_profile)
    non_zero_positions = count(!iszero, vcat(decisive_body_profile...))
    median_positions = median(vcat(decisive_body_profile...))

    return MajoritarianDecisiveBodyMetrics(
        social_utility_from_body, non_zero_positions, median_positions
    )

end

function find_legislature_partisan_composition(winning_candidates::Vector{Int},
    preferred_parties::Matrix{Int}, n_seats::Int, pop_per_seat::Int)

    member_parties = Vector{Int}(undef, n_seats)
    for seat in 1:n_seats
        member_parties[seat] = preferred_parties[seat, winning_candidates[seat]]
    end

    legislature_party_proportions = Dict(
        party => n_members / n_seats for (party, n_members) in counter(member_parties)
    )

    voter_party_proportions = Dict(
        party => voters / (pop_per_seat * n_seats) for (party, voters) in counter(preferred_parties)
    )

    return legislature_party_proportions, voter_party_proportions

end

function find_gallagher_index(legislature_party_proportions::Dict{Int,Float64},
    voter_party_proportions::Dict{Int,Float64})

    diffs = Vector{Float64}(undef, length(voter_party_proportions))
    for (i, party) in enumerate(keys(voter_party_proportions))
        if !haskey(legislature_party_proportions, party)
            diffs[i] = voter_party_proportions[party]
        else
            diffs[i] = voter_party_proportions[party] - legislature_party_proportions[party]
        end
    end
    gallagher_index = sqrt(0.5 * sum((diffs) .^ 2))
    return gallagher_index

end

function evaluate_majoritarian_election(voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, candidates::Vector{Vector{Int}},
    winning_candidates::Vector{Int}, voter_utilities_for_candidates::Array{Float64,3},
    preferred_parties::Matrix{Int}, n_seats::Int, pop_per_seat::Int,
    n_issues::Int, n_questions::AbstractVector{Int},
    n_candidates::Int; plurality::Bool=false
)

    voter_rankings = ElectionSimulation.compute_voter_rankings(voter_utilities_for_candidates,
        n_seats, pop_per_seat, n_candidates)

    candidate_profiles, voter_utilities_from_candidate_profiles = evaluate_candidate_profiles(
        voter_question_positions, candidates, voter_issue_weights, n_issues, n_questions,
        pop_per_seat, n_seats, n_candidates
    )

    decisive_body_member_positions = find_decisive_body_member_positions(candidates,
        winning_candidates, candidate_profiles, n_issues, n_questions, n_seats
    )

    decisive_body_profile_strict, decisive_body_profile_qualified = find_decisive_body_positions(
        decisive_body_member_positions, n_issues, n_questions, n_seats
    )

    strict_maj_body_metrics = evaluate_decisive_body_profile(decisive_body_profile_strict,
        voter_question_positions, voter_issue_weights, 1, pop_per_seat, n_issues
    )

    qualified_maj_body_metrics = evaluate_decisive_body_profile(decisive_body_profile_qualified,
        voter_question_positions, voter_issue_weights, 1, pop_per_seat, n_issues
    )

    district_utility_metrics = Matrix{Float64}(undef, n_seats, 4)
    district_condorcet_metrics = Matrix{Bool}(undef, n_seats, 3)

    condorcet_winners, smith_sets, _ = find_condorcet_and_smith_sets_for_state(voter_rankings)

    @inbounds for seat in 1:n_seats

        begin
            winner_local_index, district_socail_utilities, district_smith_set,
            district_condorcet_winner =
                find_district_info(seat, candidates, winning_candidates,
                    voter_utilities_from_candidate_profiles, smith_sets, condorcet_winners
                )
        end
        district_utility_metrics[seat, :] .= determine_district_utility_metrics(
            winner_local_index, district_socail_utilities
        )
        district_condorcet_metrics[seat, :] .= determine_condorcet_or_smith_for_district(
            winner_local_index, district_smith_set, district_condorcet_winner
        )

    end

    maj_util_metrics = MajoritarianUtilityMetrics(eachcol(district_utility_metrics)...)
    maj_eff_metrics = MajoritarianVotingEfficiencyMetrics(eachcol(district_condorcet_metrics)...)

    legislature_party_proportions, voter_party_proportions = find_legislature_partisan_composition(
        winning_candidates, preferred_parties, n_seats, pop_per_seat
    )
    gallagher_index = find_gallagher_index(legislature_party_proportions, voter_party_proportions)

    if maj_util_metrics.utility_from_winner.avg / maj_util_metrics.utility_from_maximizer.avg != 0.0
        vse = maj_util_metrics.utility_from_winner.avg / maj_util_metrics.utility_from_maximizer.avg
    else
        vse = 2.0
    end

    if plurality
        return PluratarianEvaluation(
            maj_util_metrics, maj_eff_metrics, strict_maj_body_metrics, qualified_maj_body_metrics,
            gallagher_index, vse
        )
    else
        return MajoritarianEvaluation(
            maj_util_metrics, maj_eff_metrics, strict_maj_body_metrics, qualified_maj_body_metrics,
            gallagher_index, vse
        )
    end

end

end