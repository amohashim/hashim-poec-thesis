
module ProportionalEvaluationMetrics

using Combinatorics
using Distributions
using Distances
using Random
using LinearAlgebra
using DataStructures
using Statistics
using Base: @propagate_inbounds

using ..HelpfulFunctions
using ..CondorcetSmithFunctions
import ..ElectionSimulation: compute_voter_rankings

export evaluate_proportional_election, ProportionalEvalMinorityGovt, ProportionalEvalIndicators
export ProportionalEvalMeasures, ProportionalEvaluation

struct ProportionalEvalMinorityGovt

    elected::Bool
    unstable::Bool
    size::Int

end

struct ProportionalEvalIndicators

    minority_govt_indicators::ProportionalEvalMinorityGovt
    single_party_win::Bool

    condorcet_winner_elected::Bool
    winner_in_smith_set::Bool
    condorcet_paradox::Bool

    util_maxer_elected::Bool
    election_failed::Bool

    function ProportionalEvalIndicators(minority_govt_indicators::ProportionalEvalMinorityGovt,
        single_party_win, condorcet_winner_elected::Bool, winner_in_smith_set::Bool,
        condorcet_paradox::Bool, util_maxer_elected::Bool, election_failed::Bool)

        new(minority_govt_indicators, single_party_win, condorcet_winner_elected,
            winner_in_smith_set, condorcet_paradox, util_maxer_elected, election_failed)
    end

    function ProportionalEvalIndicators(none::Nothing)

        minority_govt_indicators = ProportionalEvalMinorityGovt(false, true, 0)
        new(minority_govt_indicators, false, false, false, false, false)

    end

end

struct ProportionalEvalMeasures

    utility_from_winner::Float64
    utility_from_maximizer::Float64
    utility_efficiency::Float64
    non_zero_positions::Int
    median_position::Float64
    unanimity::Int
    unanimity_rate::Float64

    size_of_coalition::Int
    n_parties::Int
    n_winning_parties::Int
    n_parties_effective::Float64

    function ProportionalEvalMeasures(utility_from_winner::Float64, utility_from_maximizer::Float64,
        utility_efficiency::Float64, non_zero_positions::Int, median_position::Float64,
        unanimity::Int, unanimity_rate::Float64, size_of_coalition::Int, n_parties::Int,
        n_winning_parties::Int, n_parties_effective::Float64)

        new(
            utility_from_winner, utility_from_maximizer, utility_efficiency, non_zero_positions,
            median_position, unanimity, unanimity_rate, size_of_coalition, n_parties,
            n_winning_parties, n_parties_effective
        )
    end

    function ProportionalEvalMeasures(none::Nothing)

        new(0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0, 0, 0, 0)

    end

    function ProportionalEvalMeasures(utility_from_winner::Float64, utility_from_maximizer::Float64,
        utility_efficiency::Float64, non_zero_positions::Int, median_position::Float64,
        unanimity::Int, n_seats::Int, size_of_coalition::Int, n_parties::Int,
        n_winning_parties::Int, n_parties_effective::Float64)

        new(
            utility_from_winner, utility_from_maximizer, utility_efficiency, non_zero_positions,
            median_position, unanimity, unanimity / n_seats, size_of_coalition, n_parties,
            n_winning_parties, n_parties_effective
        )

    end

end

struct ProportionalEvaluation

    qualified_indicators::ProportionalEvalIndicators
    qualified_measures::ProportionalEvalMeasures

    strict_indicators::ProportionalEvalIndicators
    strict_measures::ProportionalEvalMeasures

    prop_gallagher_index::Float64
    prop_true_gallagher_index::Float64

end

function allocate_by_dhondt(n_seats::Int, winner_vote_counts::Dict{Int,Int})
    # Initialize seat counts to zero
    seats = Dict(party => 0 for party in keys(winner_vote_counts))

    # Keep current "D'Hondt quotients" in a dictionary too:
    quotients = Dict(party => float(winner_vote_counts[party]) for party in keys(winner_vote_counts))

    # Allocate seats one at a time
    for _ in 1:n_seats
        # Find the party with the largest current quotient
        # findmax(...) returns (value, key), so use the second item for the party
        best_val, best_party = findmax(quotients)

        # Allocate a seat to that party
        seats[best_party] += 1

        # Update that party’s quotient = votes / (seats_already_won + 1)
        new_divisor = seats[best_party] + 1
        quotients[best_party] = winner_vote_counts[best_party] / new_divisor
    end

    # Convert seat counts to proportions
    seats_props = Dict{Int,Float64}(party => seats[party] / n_seats for party in keys(seats))
    return seats_props
end


function build_coalition_options(n_parties::Int)

    # worst case is 1/n_parties vote share for each party

    combos = collect(combinations(1:n_parties))
    i = length.(combos) .<= ceil(n_parties * 0.5)
    coalitions = combos[i]

    return coalitions

end

function build_party_profile_options(coalition_options::AbstractVector,
    party_question_positions::AbstractVector{Array{Float64,3}}, n_issues::Int,
    n_questions::AbstractVector{Int})

    strict_coalition_profiles = Vector{Array{Float64,3}}(undef, n_issues)
    qualified_coalition_profiles = Vector{Array{Float64,3}}(undef, n_issues)
    n_coalitions = length(coalition_options)

    coalition_unanimities = zeros(Int, n_coalitions)
    coalition_qualified_unanimities = zeros(Int, n_coalitions)

    for issue in 1:n_issues

        n_q = n_questions[issue]
        strict_profiles_for_issue = Array{Float64,3}(undef, n_q, 1, n_coalitions)
        qualified_profiles_for_issue = Array{Float64,3}(undef, n_q, 1, n_coalitions)
        q_holder = Vector{Float64}(undef, n_q)

        for (idx, coalition) in enumerate(coalition_options)

            fill!(q_holder, 0.0)

            coalition_positions = party_question_positions[issue][:, 1, coalition]

            unanimous_positions = [
                all(x -> x == row[1], row) for row in eachrow(coalition_positions)
            ]
            s_coalition_profiles = [unanimous_positions[q] == 1 ? coalition_positions[q] : 0.0
                                    for q in 1:n_q]

            same_direction = [
                all(x -> x == 0 || sign(x) == sign(row[findfirst(y -> y != 0, row)]), row)
                for row in eachrow(coalition_positions)
            ]

            for q in 1:n_q

                same_dir = same_direction[q]
                smallest_mag_idx = argmin(abs.(coalition_positions[q, :]))
                smallest_mag_pos = coalition_positions[q, smallest_mag_idx]
                q_holder[q] = same_dir ? smallest_mag_pos : 0.0

            end

            strict_profiles_for_issue[:, 1, idx] = s_coalition_profiles
            qualified_profiles_for_issue[:, 1, idx] = q_holder
            coalition_unanimities[idx] += length(findall(unanimous_positions))
            coalition_qualified_unanimities[idx] += length(findall(same_direction))

        end

        strict_coalition_profiles[issue] = strict_profiles_for_issue
        qualified_coalition_profiles[issue] = qualified_profiles_for_issue
    end

    begin
        return strict_coalition_profiles, qualified_coalition_profiles, coalition_unanimities,
        coalition_qualified_unanimities
    end
end

function compute_utilities_for_party_profiles(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    coalition_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    n_coalitions::Int,
    n_seats::Int,
    pop_per_seat::Int,
    n_issues::Int
)::Array{Float64,3}
    # Pre-allocate the output array: shape (n_seats, pop_per_seat, n_coalitions).
    # We'll accumulate across issues, so initialize to zero:
    voter_utilities = zeros(Float64, n_seats, pop_per_seat, n_coalitions)

    @inbounds @views for issue in 1:n_issues
        # Extract the "profile" for this issue (Q_ᵏ × 1 × C), then take a view (Q_ᵏ × C)
        profile = coalition_profiles[issue][:, 1, :]

        for seat in 1:n_seats
            # Get this seat's portion for voter positions and weights:
            # district_positions is Q_ᵏ × pop_per_seat
            district_positions = voter_question_positions[issue][:, seat, :]
            seat_issue_weights = voter_issue_weights[issue, seat, :]

            # For each voter i, each coalition j, compute sum of abs differences over Q_ᵏ
            @inbounds for voter in 1:pop_per_seat
                for coalition in 1:n_coalitions
                    s = 0.0
                    n_q = size(district_positions, 1)
                    @inbounds for q in 1:n_q
                        s += abs(district_positions[q, voter] - profile[q, coalition])^2
                    end
                    # Multiply by -1 and by seat_issue_weights[i], then accumulate
                    voter_utilities[seat, voter, coalition] += -s * seat_issue_weights[voter]
                end
            end
        end
    end

    return voter_utilities
end


function evaluate_party_profiles(voter_question_positions::AbstractVector{Array{Float64,3}},
    party_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}, n_parties::Int, n_issues::Int, n_seats::Int,
    pop_per_seat::Int, n_questions::AbstractVector{Int})

    coalition_options = build_coalition_options(n_parties)
    n_coalitions = length(coalition_options)

    begin
        strict_coalition_profiles, qualified_coalition_profiles, coalition_unanimities,
        coalition_qualified_unanimities =
            build_party_profile_options(coalition_options,
                party_question_positions, n_issues, n_questions)

    end

    begin
        strict_profile_utilities =
            compute_utilities_for_party_profiles(
                voter_question_positions, strict_coalition_profiles, voter_issue_weights,
                n_coalitions, n_seats, pop_per_seat, n_issues
            )
    end

    begin
        qualified_profile_utilities =
            compute_utilities_for_party_profiles(
                voter_question_positions, qualified_coalition_profiles, voter_issue_weights,
                n_coalitions, n_seats, pop_per_seat, n_issues
            )
    end

    begin
        return coalition_options, strict_coalition_profiles, qualified_coalition_profiles,
        strict_profile_utilities, qualified_profile_utilities, coalition_unanimities,
        coalition_qualified_unanimities
    end

end

function find_minimal_coalitions(winning_parties::Dict{Int,Float64},
    coalitions::Vector{Vector{Int}})

    minimal = Vector{Vector{Int}}()
    for coalition in coalitions
        total = sum(winning_parties[p] for p in coalition)
        # Check the winning condition
        if total >= 0.5
            # Check minimality: for every party in the coalition, if you remove it,
            # the remaining parties have less than 0.5 vote share.
            if all(total - winning_parties[p] < 0.5 for p in coalition)
                push!(minimal, coalition)
            end
        end
    end
    return minimal
end

@inline function find_eligible_coalitions(
    coalition_unanimities::AbstractVector{Int},
    n_questions::AbstractVector{Int})

    total_questions = sum(n_questions)
    eligible_coalitions = findall(
        i -> i / total_questions >= 0.5, coalition_unanimities
    )

    while length(eligible_coalitions) > 10

        shuffle!(eligible_coalitions)
        pop!(eligible_coalitions)

    end

    return sort(eligible_coalitions)
end


function find_winning_coalition(coalition_options::Vector{Vector{Int}},
    party_utilities_for_coalitions::Array{Float64,3}, coalition_unanimities::Vector{Int},
    winning_parties::Dict{Int,Float64}, n_questions::AbstractVector{Int})

    single_party_win = false

    total_questions = sum(n_questions)
    min_coalitions = find_minimal_coalitions(winning_parties, coalition_options)
    min_coalitions
    min_coalition_indices = [findfirst(==(x), coalition_options) for x in min_coalitions]
    min_coalition_indices
    min_eligible_coalition_indices = findall(
        i -> coalition_unanimities[i] / total_questions >= 0.5, min_coalition_indices
    )
    min_eligible_coalition_indices

    mec = min_coalitions[min_eligible_coalition_indices]
    mec_global_idx = min_coalition_indices[min_eligible_coalition_indices]
    mec_global_idx

    if length(mec_global_idx) < 1

        return 0, single_party_win # minority government

    elseif length(mec_global_idx) == 1

        single_party_win = true
        return vec(mec_global_idx)[1], single_party_win # the unique minimal eligible coalition

    else

        # if there are more than 1 minimal eligible coalition, choose the one that maximizes the 
        # coalitions's utility as a group

        party_utils_for_mec = @view party_utilities_for_coalitions[1, :, mec_global_idx]

        coalition_utilities = Vector{Float64}(undef, length(mec))

        for (index, parties) in enumerate(mec)

            coalition_utilities[index] = sum(party_utils_for_mec[parties, index]) / sqrt(length(parties))

        end

        return mec_global_idx[argmin(coalition_utilities)], single_party_win

    end

end

@inline function subset_coalitions_by_party(parties::Union{Vector{Int},Base.KeySet},
    coalition_options::Vector{Vector{Int}})

    coalitions = Vector{Vector{Int}}()
    @inbounds for coalition in coalition_options
        if any(x -> x in coalition, parties)
            push!(coalitions, coalition)
        end
    end

    return coalitions

end

# requires S ⊂ L
@inline function find_subset_indices(S::Vector, L::Vector)
    # Create a dictionary for fast lookup of indices in L
    index_map = Dict(v => i for (i, v) in enumerate(L))

    # Find indices in L corresponding to elements of S
    indices = [index_map[s] for s in S if haskey(index_map, s)]
    return indices
end

function select_minority_coalition(party_utils_for_min_co::AbstractMatrix,
    minority_coalitions::Vector{Vector{Int}}, minority_govt_party::Int)
    ncols = size(party_utils_for_min_co, 2)
    P = size(party_utils_for_min_co, 1)

    # Precompute the row `p` values.
    row_p = party_utils_for_min_co[minority_govt_party, :]

    # Precompute the maximum value for each row.
    row_max = [maximum(party_utils_for_min_co[r, :]) for r in 1:P]

    # First, filter columns where U[p, c] is non-negative (needed for both primary & fallback).
    candidate_cols = findall(x -> x >= 0, row_p)

    primary_best = nothing
    best_val = -Inf

    # Primary selection: check additional Cs condition.
    # For every r in Cs[c], we require that:
    #    U[r, c] is non-negative, OR U[r,c] equals the max in that row.
    for c in candidate_cols
        valid = true
        for r in minority_coalitions[c]
            val = party_utils_for_min_co[r, c]
            # If the value is negative, then it must be the row's maximum.
            if val < 0 && val != row_max[r]
                valid = false
                break
            end
        end
        if valid && row_p[c] > best_val
            best_val = row_p[c]
            primary_best = c
        end
    end

    # If a column passed the primary selection, return it with the negative flag.
    if primary_best !== nothing
        neg_flag = any(
            r -> party_utils_for_min_co[r, primary_best] < 0, minority_coalitions[primary_best]
        )
        return (primary_best, neg_flag)
    end

    # Fallback selection: choose the candidate (U[p,c] >= 0) with the highest column sum.
    # Precompute column sums.
    col_sums = vec(sum(party_utils_for_min_co, dims=1))
    best_fallback = nothing
    best_sum = -Inf

    for c in candidate_cols
        if col_sums[c] > best_sum
            best_sum = col_sums[c]
            best_fallback = c
        end
    end

    if best_fallback !== nothing
        neg_flag = any(
            r -> party_utils_for_min_co[r, best_fallback] < 0, minority_coalitions[best_fallback]
        )
        return (best_fallback, neg_flag)
    end

    return nothing, true

end

function find_minority_government_party(winning_parties::Dict{Int,Float64})

    max_vote_share = maximum(values(winning_parties))
    max_parties = [party for (party, vote_share) in winning_parties if vote_share == max_vote_share]
    minority_govt_party = rand(max_parties)

    return minority_govt_party

end

function find_minority_coalitions(minority_govt_party::Int,
    winning_parties::Dict{Int,Float64}, coalition_options::Vector{Vector{Int}})

    minority_coalitions = subset_coalitions_by_party([minority_govt_party], coalition_options)
    min_co_with_winners = subset_coalitions_by_party(keys(winning_parties), minority_coalitions)
    minimal_coalitions = find_minimal_coalitions(winning_parties, min_co_with_winners)
    minimal_coalition_indices = find_subset_indices(minimal_coalitions, coalition_options)

    return minimal_coalitions, minimal_coalition_indices

end

# requires scaled utilities
function find_minority_government(minority_govt_party::Int,
    minimal_coalitions::Vector{Vector{Int}}, minimal_coalition_indices::Vector{Int},
    party_utilities_for_coalition_profiles::Array{Float64,3}
)

    election_failed = false

    # reuiqres utilities scaled to [0,1]
    party_utils_for_min_co = party_utilities_for_coalition_profiles[
        :, :, minimal_coalition_indices
    ][1, :, :]

    party_utils_for_min_co = 2 .* party_utils_for_min_co .- 1
    minority_coalition_local, unstable = select_minority_coalition(
        party_utils_for_min_co, minimal_coalitions, minority_govt_party
    )
    if isnothing(minority_coalition_local)

        election_failed = true
        return nothing, nothing, 0, true, election_failed

    end

    minority_government = minimal_coalition_indices[minority_coalition_local]
    size_minority_government = length(minimal_coalitions[minority_coalition_local])

    return minority_government, size_minority_government, unstable, election_failed

end

function find_eligible_condorcet_winner(voter_utilities_for_coalition_profiles::Array{Float64,3},
    coalition_unanimities::AbstractVector{Int}, n_questions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int)

    eligible_coalitions = find_eligible_coalitions(coalition_unanimities, n_questions)

    if isempty(eligible_coalitions)
        return nothing, nothing
    end

    eligible_coalition_profile_utilities = voter_utilities_for_coalition_profiles[
        :, :, eligible_coalitions
    ]
    n_eligible_coalitions = size(eligible_coalition_profile_utilities)[3]

    eligible_coalition_rankings = compute_voter_rankings(eligible_coalition_profile_utilities,
        n_seats, pop_per_seat, n_eligible_coalitions)

    eligible_coalition_rankings = reshape(
        permutedims(eligible_coalition_rankings, (2, 1, 3)), n_seats * pop_per_seat,
        n_eligible_coalitions
    )

    begin
        condorcet_winner, smith_set, _ =
            find_condorcet_and_smith_sets_for_district(eligible_coalition_rankings)
    end

    return condorcet_winner, smith_set

end

using Statistics

function compute_absolute_ideological_congruence(
    voter_ideal_points::AbstractVector{Array{Float64,3}},
    party_ideal_points::AbstractVector{Array{Float64,3}}, n_parties::Int, n_issues::Int,
    issue_dimensions::AbstractVector{Int}, pop_per_seat::Int, n_seats::Int)

    city_block_diffs_by_issue = Matrix{Float64}(undef, n_issues, n_parties)
    @inbounds for issue in 1:n_issues

        issue_dims = issue_dimensions[issue]
        voter_points_issue = voter_ideal_points[issue]
        party_points_issue = party_ideal_points[issue]

        voter_points = reshape(voter_points_issue, (n_seats * pop_per_seat, issue_dims))
        party_points = reshape(party_points_issue, (n_parties, issue_dims))

        @inbounds for party in 1:n_parties

            city_block_diffs_by_issue[issue, party] = sum(
                cityblock.(Ref(party_points[party]), voter_points)
            )

        end

    end

    # arithmetic mean over the dimensions, arithmetic mean over the population
    # so, it's the congruences on each issue averaged over the number of issues
    return mean(city_block_diffs_by_issue, dims=1) ./ (pop_per_seat * n_seats)

end

function determine_condorcet_or_smith(observed_winner::Int,
    voter_utilities_for_coalition_profiles::Array{Float64,3},
    coalition_unanimities::AbstractVector{Int}, n_questions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int
)

    winner_is_condorcet = false
    winner_in_smith_set = false
    condorcet_paradox = false
    condorcet_winner, smith_set = find_eligible_condorcet_winner(
        voter_utilities_for_coalition_profiles, coalition_unanimities, n_questions, n_seats,
        pop_per_seat
    )

    if observed_winner != 0 # if not minority govt
        if observed_winner ∈ smith_set
            winner_in_smith_set = true
            if isnothing(condorcet_winner)
                condorcet_paradox = true
            elseif observed_winner == condorcet_winner
                winner_is_condorcet = true
            end
        end
    else
        if isnothing(condorcet_winner)
            condorcet_paradox = true
        end
    end

    return winner_is_condorcet, winner_in_smith_set, condorcet_paradox

end

function determine_utility_metrics(observed_winner::Int, social_utilities::Vector{Float64})

    @assert observed_winner != 0

    utility_maxer_elected = false

    utility_of_winner = social_utilities[observed_winner]
    utility_of_util_maxer, util_maxer = findmax(social_utilities)

    if (observed_winner == util_maxer) || (utility_of_winner == utility_of_util_maxer)

        utility_maxer_elected = true

    end

    utility_efficiency = utility_of_winner / utility_of_util_maxer

    return utility_of_winner, utility_of_util_maxer, utility_efficiency, utility_maxer_elected

end

function find_gallagher_index(winning_parties::Dict{Int,Float64},
    raw_vote_counts::Dict{Int,Int}, n_parties::Int)

    raw_vote_shares = HelpfulFunctions.filter_parties_below_threshold(
        raw_vote_counts, 0.0, n_parties, true
    )
    diffs = Vector{Float64}(undef, n_parties) # NOT necessarily ordered

    for (i, party) in enumerate(keys(raw_vote_shares))
        if !haskey(winning_parties, party)
            diffs[i] = raw_vote_shares[party]
        else
            diffs[i] = raw_vote_shares[party] - winning_parties[party]
        end
    end
    gallagher_index = sqrt(0.5 * sum(diffs .^ 2))
    return gallagher_index

end

function determine_proportional_metrics(coalition_options::Vector{Vector{Int}},
    coalition_profiles::AbstractVector{Array{Float64,3}},
    party_utilities_for_coalition_profiles::Array{Float64,3},
    voter_utilities_for_coalition_profiles::Array{Float64,3},
    social_utilities::Vector{Float64},
    coalition_unanimities::Vector{Int},
    winning_parties::Dict{Int,Float64}, n_parties::Int, party_threshold::Float64,
    n_questions::AbstractVector{Int}, n_seats::Int, pop_per_seat::Int)

    minority_govt_elected = false
    size_minority_govt = 0
    minority_govt_unstable = false
    election_failed = false

    observed_winner, single_party_win = find_winning_coalition(coalition_options,
        party_utilities_for_coalition_profiles, coalition_unanimities, winning_parties, n_questions
    )

    if observed_winner == 0
        minority_govt_party = find_minority_government_party(winning_parties)
        minority_govt_coalitions, minority_govt_indices = find_minority_coalitions(
            minority_govt_party, winning_parties, coalition_options
        )
        begin
            minority_govt, size_minority_govt, minority_govt_unstable, election_failed =
                find_minority_government(minority_govt_party, minority_govt_coalitions,
                    minority_govt_indices, party_utilities_for_coalition_profiles
                )
        end

        if isnothing(minority_govt)

            return ProportionalEvalMeasures(nothing), ProportionalEvalIndicators(nothing)

        end

        observed_winner = minority_govt
        minority_govt_elected = true # for all Le_L2_L2_L1, this was incorrect

    end

    unanimity = coalition_unanimities[observed_winner]
    size_of_coalition = length(coalition_options[observed_winner])

    if !minority_govt_elected
        winner_is_condorcet, winner_in_smith_set, condorcet_paradox =
            determine_condorcet_or_smith(
                observed_winner, voter_utilities_for_coalition_profiles,
                coalition_unanimities, n_questions, n_seats, pop_per_seat
            )
    else
        winner_is_condorcet, winner_in_smith_set, condorcet_paradox = false, true, true
    end

    utility_of_winner, utility_from_maxer, utility_efficiency, utility_maxer_elected =
        determine_utility_metrics(
            observed_winner, social_utilities
        )

    non_zero_positions = count(!iszero, vcat(coalition_profiles...)[:, :, observed_winner])
    median_positions = median(vcat(coalition_profiles...)[:, :, observed_winner])

    prop_minco = ProportionalEvalMinorityGovt(minority_govt_elected,
        minority_govt_unstable, size_minority_govt
    )

    indicators = [
        prop_minco, single_party_win, winner_is_condorcet, winner_in_smith_set, condorcet_paradox,
        utility_maxer_elected, election_failed
    ]

    n_winning_parties = count(values(winning_parties) .> party_threshold)
    n_parties_effective = 1.0 / sum(values(winning_parties) .^ 2) # Laasko and Taagepra/Herfindahl-Hirschman index

    prop_eval_measures = ProportionalEvalMeasures(utility_of_winner, utility_from_maxer,
        utility_efficiency, non_zero_positions, median_positions, unanimity, n_seats,
        size_of_coalition, n_parties, n_winning_parties, n_parties_effective)

    pop_eval_indicators = ProportionalEvalIndicators(indicators...)

    return prop_eval_measures, pop_eval_indicators

end

function evaluate_proportional_election(
    party_ideal_points::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    party_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    winning_parties::Dict{Int,Float64}, raw_vote_counts::AbstractDict{Int,Int},
    n_parties::Int, n_issues::Int, n_questions::AbstractVector{Int},
    issue_dimensions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int, party_threshold::Float64,
    preferred_parties::Matrix{Int}) # winning parties should be true, over-threshold winning parties

    party_issue_weights = find_issue_weights(party_ideal_points, n_issues, 1, n_parties,
        issue_dimensions)

    begin
        coalition_options, strict_coalition_profiles, qualified_coalition_profiles,
        voter_utilities_for_strict_coalition_profiles,
        voter_utilities_for_qualified_coalition_profiles, coalition_unanimities,
        coalition_qualified_unanimities = evaluate_party_profiles(
            voter_question_positions, party_question_positions, voter_issue_weights, n_parties,
            n_issues, n_seats, pop_per_seat, n_questions
        )
    end

    strict_coalition_profiles = convert.(Array, strict_coalition_profiles) # profiles of each coalition on each issue
    qualified_coalition_profiles = convert.(Array, qualified_coalition_profiles)

    voter_utilities_for_strict_coalition_profiles = scale_utilities(
        voter_utilities_for_strict_coalition_profiles, 0.0, 1000.0
    )
    strict_social_utilities = sum(voter_utilities_for_strict_coalition_profiles, dims=(1, 2))[:]

    voter_utilities_for_qualified_coalition_profiles = scale_utilities(
        voter_utilities_for_qualified_coalition_profiles, 0.0, 1000.0
    )
    qualified_social_utilities = sum(
        voter_utilities_for_qualified_coalition_profiles, dims=(1, 2)
    )[:]

    n_coalitions = length(coalition_options)

    party_utilities_for_strict_coalition_profiles = compute_utilities_for_party_profiles(
        party_question_positions, strict_coalition_profiles, party_issue_weights, n_coalitions, 1,
        n_parties, n_issues
    ) # each party's utility from a given coalition
    party_utilities_for_strict_coalition_profiles = scale_utilities(
        party_utilities_for_strict_coalition_profiles, 0.0, 1000.0
    )

    party_utilities_for_qualified_coalition_profiles = compute_utilities_for_party_profiles(
        party_question_positions, qualified_coalition_profiles, party_issue_weights, n_coalitions, 1,
        n_parties, n_issues
    ) # each party's utility from a given coalition
    party_utilities_for_qualified_coalition_profiles = scale_utilities(
        party_utilities_for_qualified_coalition_profiles, 0.0, 1000.0)

    utils_strict, indicators_strict = determine_proportional_metrics(
        coalition_options, strict_coalition_profiles, party_utilities_for_strict_coalition_profiles,
        voter_utilities_for_strict_coalition_profiles, strict_social_utilities,
        coalition_unanimities, winning_parties, n_parties, party_threshold,
        n_questions, n_seats, pop_per_seat
    )

    utils_qualified, indicators_qualified = determine_proportional_metrics(
        coalition_options, qualified_coalition_profiles,
        party_utilities_for_qualified_coalition_profiles,
        voter_utilities_for_qualified_coalition_profiles, qualified_social_utilities,
        coalition_qualified_unanimities, winning_parties, n_parties, party_threshold,
        n_questions, n_seats, pop_per_seat
    )

    true_party_support = convert(Dict{Int,Int}, counter(preferred_parties))
    gallagher_index = find_gallagher_index(winning_parties, raw_vote_counts, n_parties)
    true_gallagher_index = find_gallagher_index(winning_parties, true_party_support, n_parties)

    return ProportionalEvaluation(
        indicators_qualified, utils_qualified, indicators_strict, utils_strict,
        gallagher_index, true_gallagher_index
    )

end

end