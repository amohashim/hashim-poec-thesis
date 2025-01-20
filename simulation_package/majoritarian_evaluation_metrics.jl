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

struct MajoritarianUtilitySummaryStats

    avg::Float64
    median::Float64
    sd::Float64

end

struct MajoritarianEfficiencyMetrics

    util_maximizer_election_rate::Float64
    utility_efficiency::MajoritarianUtilitySummaryStats
    n_condorcet_winner_elected::Int
    n_winner_in_smith_set::Int
    n_condorcet_paradox::Int
    condorcet_efficiency::Float64
    smith_efficiency::Float64

    function MajoritarianEfficiencyStats(utility_efficiency::MajoritarianUtilitySummaryStats,
        n_condorcet_winner_elected::Int, n_winner_in_smith_set::Int, n_condorcet_paradox::Int,
        n_seats::Int)

        if n_seats != n_condorcet_paradox
            condorcet_efficiency = n_condorcet_winner_elected / (n_seats - n_condorcet_paradox)
        else
            condorcet_efficiency = 2.0
        end

        smith_efficiency = n_winner_in_smith_set / n_seats

        new(n_condorcet_winner_elected, n_winner_in_smith_set, n_condorcet_paradox,
            condorcet_efficiency, smith_efficiency)

    end

end

struct MajoritarianDecisiveBodyMetrics

    social_utility_from_body::Float64
    non_zero_positions::Int
    median_position::Int

end

struct MajoritarianEvaluation

    social_utility_from_winner::MajoritarianUtilitySummaryStats
    efficiency_stats::MajoritarianEfficiencyMetrics
    qualified_decisive_body_metrics::MajoritarianDecisiveBodyMetrics
    strict_decisive_body_metrics::MajoritarianDecisiveBodyMetrics

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

function compute_utilities_for_candidate_profiles(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    candidate_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    n_candidates_per_district::Int, n_seats::Int, pop_per_seat::Int, n_issues::Int)

    voter_utilities = Array{Float64,3}(undef, n_seats, pop_per_seat, n_candidates_per_district)

    @inbounds for issue in 1:n_issues

        @inbounds for seat in 1:n_seats

            profile = @view candidate_profiles[issue][:, seat, :]

            district_positions = @view voter_question_positions[issue][:, seat, :]
            seat_issue_weights = @view voter_issue_weights[issue, seat, :]
            utilities = [
                sum(abs.(district_positions[:, i] .- profile[:, j]))
                for i in 1:pop_per_seat, j in 1:n_candidates_per_district
            ]

            voter_utilities[seat, :, :] = -1 .* utilities .* @view seat_issue_weights[:, :]

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

    voter_utilities_from_candidate_profiles = scale_utilities(
        voter_utilities_from_candidate_profiles
    )

    return candidate_profiles, voter_utilities_from_candidate_profiles

end

function evaluate_condorcet_winner!(smith_counter::Ref{Int}, paradox_counter::Ref{Int},
    condorcet_counter::Ref{Int}, index_vector::Vector{Int}, smith_set::Vector{Int},
    condorcet_winner::Union{Int,Nothing}, winner_index::Int)


    if winner_index ∈ smith_set

        smith_counter[] += 1

        if isnothing(condorcet_winner)
            paradox_counter[] += 1

        elseif winner_index == condorcet_winner
            condorcet_counter[] += 1
        end

    end
end

function compute_vse_and_condorcet(voter_utilities_from_candidate_profiles::Array{Float64,3},
    voter_rankings::Array{Int,3}, candidates::Vector{Vector{Int}}, winning_candidates::Vector{Int},
    first_round_candidate_choices::Matrix{Int}, n_seats::Int, n_iterations::Int,
    sample_size::Int, pop_per_seat::Int, n_candidates::Int, rng::AbstractRNG)

    n_maximizer_as_winner = 0
    n_condorcet_winner = 0
    n_smith_winner = 0
    n_condorcet_paradox = 0
    utility_efficiency_by_district = Vector{Float64}(undef, n_seats)
    vse_by_district = Vector{Float64}(undef, n_seats)
    condorcet_efficiency_by_district = Vector{Float64}(undef, n_seats)
    smith_efficiency_by_district = Vector{Float64}(undef, n_seats)

    condorcet_winners, smith_sets, _ = find_condorcet_and_smith_sets_for_state(voter_rankings)

    @inbounds for seat in 1:n_seats

        # initialize district candidate info
        district_candidates = candidates[seat]
        prefered_candidates = @view first_round_candidate_choices[seat, :]

        # Find info about utilities for candidates
        utilities_by_candidate = @view voter_utilities_from_candidate_profiles[seat, :, :]
        social_utilities = sum(utilities_by_candidate, dims=1)[:]

        # Find the winning candidate
        winning_candidate = winning_candidates[seat]
        winner_local_indx = findfirst(==(winning_candidate), district_candidates)

        # Find the maximizing candidate
        utility_maximizer = argmax(social_utilities)
        if district_candidates[utility_maximizer] == winning_candidate
            n_maximizer_as_winner += 1
        end

        begin
            utility_efficiency_by_district[seat] =
                social_utilities[winner_local_indx] / social_utilities[utility_maximizer]
        end

        # Check if condorcet/smith winner was observed, and if there was a condorcet paradox
        district_smith_set = smith_sets[seat]
        condorcet_winner = condorcet_winners[seat]
        if winner_local_indx ∈ district_smith_set

            n_smith_winner += 1

            if isnothing(condorcet_winner)

                n_condorcet_paradox += 1

            elseif winner_local_indx == condorcet_winner

                n_condorcet_winner += 1

            end

        end

        # Dirichlet process for expected values
        iter_winners = Vector{Int}(undef, n_iterations)
        iter_n_smith_winner = Ref(0)
        iter_n_condorcet_winner = Ref(0)
        iter_n_condorcet_paradox = Ref(0)

        @inbounds for iteration in 1:n_iterations

            first_round_results = maj_sample_from_dirichlet(prefered_candidates, sample_size, rng,
                pop_per_seat, n_candidates)

            run_off_candidates_local_idx = ElectionSimulation.tally_top_2([first_round_results], 1)[1]
            local_i = run_off_candidates_local_idx
            ranks_for_run_off_candidates = @view voter_rankings[seat, :, run_off_candidates_local_idx]

            run_off_candidates_global_idx = global_i = district_candidates[run_off_candidates_local_idx]

            begin
                vote_for_first_cand =
                    ranks_for_run_off_candidates[:, 1] .> ranks_for_run_off_candidates[:, 2]
            end

            mask_winner = sum(vote_for_first_cand) / pop_per_seat > 0.5
            winner = mask_winner ? global_i[1] : global_i[2]
            winner_local = mask_winner ? local_i[1] : local_i[2]
            iter_winners[iteration] = winner

            evaluate_condorcet_winner!(iter_n_smith_winner, iter_n_condorcet_paradox,
                iter_n_condorcet_winner, local_i, district_smith_set, condorcet_winner,
                winner_local)

        end

        # Accumulate probabilities from samples
        vse_by_district[seat] = compute_vse(iter_winners, district_candidates, social_utilities,
            winner_local_indx, utility_maximizer, n_iterations
        )

        # Find Condorcet Efficiency
        condorcet_efficiency_by_district[seat] = compute_condorcet_efficiency(
            iter_n_condorcet_winner, n_iterations, iter_n_condorcet_paradox
        )

        # Find Smith Efficiency
        smith_efficiency_by_district[seat] = compute_condorcet_efficiency(
            iter_n_smith_winner, n_iterations, Ref(0))

    end

    statewide_condorcet_efficiency = compute_condorcet_efficiency(
        n_condorcet_winner, n_seats, n_condorcet_paradox
    )

    statewide_smith_efficiency = compute_condorcet_efficiency(n_smith_winner, n_seats, 0)

    prop_maximizer_as_winner = n_maximizer_as_winner / n_seats

    results = MajoritarianEvaluation(vse_by_district, condorcet_efficiency_by_district,
        smith_efficiency_by_district, utility_efficiency_by_district, prop_maximizer_as_winner,
        statewide_condorcet_efficiency, statewide_smith_efficiency)

    begin
        return results
    end
end


function determine_district_utility_metrics(winning_candidate_local_index::Int,
    social_utilities::Vector{Float64})

    utility_maxer_elected = 0

    utility_from_winner = social_utilities[winning_candidate_local_index]
    util_from_maximizer, util_maxer = findmax(social_utilities)

    if (winning_candidate_local_index == util_maxer) || (utility_from_winner == util_from_maximizer)
        utility_maxer_elected = 1
    end

    utility_efficiency = utility_from_winner / util_from_maximizer

    return utility_from_winner, utility_efficiency, utility_maxer_elected

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

function find_decisive_body_postiions(member_positions::AbstractVector{Matrix{Float64}},
    n_issues::Int, n_questions::AbstractVector{Int})

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

    for issue in 1:n_issues

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

function evaulate_decisive_body_profile(decisive_body_profile::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, n_seats::Int, pop_per_seat::Int, n_issues::Int)

    voter_utilities_for_decisive_body_profile = compute_utilities_for_candidate_profiles(
        voter_question_positions, decisive_body_profile, voter_issue_weights, 1, n_seats,
        pop_per_seat, n_issues)

    voter_utilities_for_decisive_body_profile = HelpfulFunctions.scale_utilities(
        voter_utilities_for_decisive_body_profile
    )

    social_utility_from_body = sum(voter_utilities_for_decisive_body_profile)
    non_zero_positions = count(!iszero, vcat(decisive_body_profile...))
    median_positions = median(vcat(decisive_body_profile...))

    return MajoritarianDecisiveBodyMetrics(
        social_utility_from_body, non_zero_positions, median_positions
    )

end

function evaluate_majoritarian_election()

    # refactor out because this will be for either qualified or strict 
    candidate_profiles, voter_utilities_from_candidate_profiles = evaluate_candidate_profiles(
        voter_question_positions, candidates, voter_issue_weights, n_issues, n_questions,
        pop_per_seat, n_seats, n_candidates
    )

    condorcet_winners, smith_sets, _ = find_condorcet_and_smith_sets_for_state(voter_rankings)

    decisive_body_member_positions = find_decisive_body_member_positions(candidates,
        winning_candidates, candidate_profiles,)

    # -------------------------------- YOU WERE HERE ------------------------------------ #

    begin decisive_body_profile_strict, decisive_body_profile_qualified = 
        find_decisive_body_member_positions()
    end

    @inbounds for seat in 1:n_seats

        # initialize district candidate info
        district_candidates = candidates[seat]
        prefered_candidates = @view first_round_candidate_choices[seat, :]

        # Find info about utilities for candidates
        utilities_by_candidate = @view voter_utilities_from_candidate_profiles[seat, :, :]
        social_utilities = sum(utilities_by_candidate, dims=1)[:]

        # Find the winning candidate
        winning_candidate = winning_candidates[seat]
        winner_local_indx = findfirst(==(winning_candidate), district_candidates)

        begin utility_from_winner, utility_efficiency, utility_maxer_elected = 
            determine_district_utility_metrics(winner_local_indx, social_utilities)
        end

        begin winner_is_condorcet, winner_in_smith_set, condorcet_paradox = 
            determine_condorcet_or_smith_for_district(winner_local_indx, 
            district_smith_set, condorcet_winner)
        end


end

function f()

    n_maximizer_as_winner = 0
    n_condorcet_winner = 0
    n_smith_winner = 0
    n_condorcet_paradox = 0
    utility_efficiency_by_district = Vector{Float64}(undef, n_seats)
    vse_by_district = Vector{Float64}(undef, n_seats)
    condorcet_efficiency_by_district = Vector{Float64}(undef, n_seats)
    smith_efficiency_by_district = Vector{Float64}(undef, n_seats)

    condorcet_winners, smith_sets, _ = find_condorcet_and_smith_sets_for_state(voter_rankings)

    @inbounds for seat in 1:n_seats

        # initialize district candidate info
        district_candidates = candidates[seat]
        prefered_candidates = @view first_round_candidate_choices[seat, :]

        # Find info about utilities for candidates
        utilities_by_candidate = @view voter_utilities_from_candidate_profiles[seat, :, :]
        social_utilities = sum(utilities_by_candidate, dims=1)[:]

        # Find the winning candidate
        winning_candidate = winning_candidates[seat]
        winner_local_indx = findfirst(==(winning_candidate), district_candidates)

        # Find the maximizing candidate
        utility_maximizer = argmax(social_utilities)
        if district_candidates[utility_maximizer] == winning_candidate
            n_maximizer_as_winner += 1
        end

        begin
            utility_efficiency_by_district[seat] =
                social_utilities[winner_local_indx] / social_utilities[utility_maximizer]
        end

        # Check if condorcet/smith winner was observed, and if there was a condorcet paradox
        district_smith_set = smith_sets[seat]
        condorcet_winner = condorcet_winners[seat]
        if winner_local_indx ∈ district_smith_set

            n_smith_winner += 1

            if isnothing(condorcet_winner)

                n_condorcet_paradox += 1

            elseif winner_local_indx == condorcet_winner

                n_condorcet_winner += 1

            end

        end

        # Dirichlet process for expected values
        iter_winners = Vector{Int}(undef, n_iterations)
        iter_n_smith_winner = Ref(0)
        iter_n_condorcet_winner = Ref(0)
        iter_n_condorcet_paradox = Ref(0)


    end



    function evaluate_majoritarian_election(voter_question_positions::AbstractVector{Array{Float64,3}},
        voter_issue_weights::Array{Float64,3}, candidates::AbstractVector{Vector{Int}},
        voter_rankings::Array{Int,3},
        n_issues::Int, n_seats::Int, n_candidates::Int, pop_per_seat::Int,
        n_questions::AbstractVector{Int}, winning_candidates::Vector{Int},
        first_round_candidate_choices::Matrix{Int}, sample_size::Int, n_iterations::Int,
        rng::AbstractRNG)

        candidate_profiles, voter_utilities_from_candidate_profiles = evaluate_candidate_profiles(
            voter_question_positions, candidates, voter_issue_weights, n_issues, n_questions,
            pop_per_seat, n_seats, n_candidates
        )

        # vse_by_district, condorcet_efficiency_by_district, smith_efficiency_by_district,
        # n_maximizer_as_winner, statewide_condorcet_efficiency, statewide_smith_efficiency
        results = compute_vse_and_condorcet(voter_utilities_from_candidate_profiles,
            voter_rankings, candidates, winning_candidates, first_round_candidate_choices,
            n_seats, n_iterations, sample_size, pop_per_seat, n_candidates, rng)

        return results

    end


end