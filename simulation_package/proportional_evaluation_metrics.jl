
module ProportionalEvaluationMetrics

using Combinatorics
using Distributions
using Random
using LinearAlgebra
using DataStructures
using Statistics

using ..HelpfulFunctions
using ..CondorcetSmithFunctions
import ..ElectionSimulation: compute_voter_rankings

export evaluate_proportional_election

struct ProportionalEvalIndicators

    condorcet_winner_elected::Bool
    winner_in_smith_set::Bool
    condorcet_paradox::Bool

    util_maxer_elected::Bool
    minority_govt_elected::Bool

end

struct ProportionalEvalMeasures

    utility_from_winner::Float64
    utility_efficiency::Float64
    mon_zero_positions::Int
    median_position::Int
    unanimity::Int
    unanimity_rate::Float64

    function ProportionalEvalMeasures(utility_from_winner::Float64, utility_efficiency::Float64,
        mon_zero_positions::Int, median_position::Int, unanimity::Int, n_seats::Int)

        new(utility_from_winner, utility_efficiency, mon_zero_positions, median_position,
            unanimity, unanimity / n_seats)

    end

end

struct ProportionalEvaluation

    qualified_indicators::ProportionalEvalIndicators
    qualified_measures::ProportionalEvalMeasures

    strict_indicators::ProportionalEvalIndicators
    strict_measures::ProportionalEvalMeasures

end

function allocate_by_dhondt(n_seats::Int, winner_vote_counts::AbstractDict{Int,Int})
    # N: Total number of seats to allocate
    # votes: Dictionary of party (key) to vote total (value)


    # Initialize a dictionary to track allocated seats
    seats = Dict(party => 0 for party in keys(winner_vote_counts))

    # Generate a priority queue of (quotient, party) pairs
    quotients = [(winner_vote_counts[party] / 1, party) for party in keys(winner_vote_counts)]

    # Repeat for each seat
    for _ in 1:n_seats
        # Find the party with the largest quotient
        max_quotient, winning_party = findmax(quotients)

        # Allocate a seat to the winning party
        seats[winning_party] += 1

        # Update the quotient for the winning party
        updated_divisor = seats[winning_party] + 1
        quotients[winning_party] = (winner_vote_counts[winning_party] / updated_divisor,
            winning_party)
    end

    seats_props = Dict{Int,Float64}(party => seats_won / n_seats for (party, seats_won) in seats)

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

"""
N x A x n_coalitions
"""
function compute_utilities_for_party_profiles(
    voter_question_positions::AbstractVector{Array{Float64,3}},
    coalition_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    n_coalitions::Int, n_seats::Int, pop_per_seat::Int, n_issues::Int)

    voter_utilities = Array{Float64,3}(undef, n_seats, pop_per_seat, n_coalitions)

    @inbounds for issue in 1:n_issues

        profile = @view coalition_profiles[issue][:, 1, :]

        @inbounds for seat in 1:n_seats

            district_positions = @view voter_question_positions[issue][:, seat, :]
            seat_issue_weights = @view voter_issue_weights[issue, seat, :]
            utilities = [
                sum(abs.(district_positions[:, i] .- profile[:, j]))
                for i in 1:pop_per_seat, j in 1:n_coalitions
            ]

            voter_utilities[seat, :, :] = -1 .* utilities .* @view seat_issue_weights[:, :]

        end
    end

    return voter_utilities

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

        return 0 # minority government

    elseif length(mec_global_idx) == 1

        return vec(mec_global_idx)[1] # the unique minimal eligible coalition

    else

        # if there are more than 1 minimal eligible coalition, choose the one that maximizes the 
        # coalitions's utility as a group

        party_utils_for_mec = @view party_utilities_for_coalitions[1, :, mec_global_idx]

        coalition_utilities = Vector{Float64}(undef, length(mec))

        for (index, parties) in enumerate(mec)

            coalition_utilities[index] = sum(party_utils_for_mec[parties, index]) / sqrt(length(parties))

        end

        return mec_global_idx[argmin(coalition_utilities)]

    end

end

function find_minority_government_profile(winning_parties::Dict{Int,Float64},
    coalition_profiles::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    n_issues::Int, n_questions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int, n_parties::Int, strict::Bool)

    # Find the maximum value in the dictionary
    max_vote_share = maximum(values(winning_parties))
    max_parties = [party for (party, vote_share) in winning_parties if vote_share == max_vote_share]
    minority_govt_party = rand(max_parties)
    minority_govt_profile = Vector{Array{Float64,3}}(undef, n_issues)

    for issue in 1:n_issues

        n_q = n_questions[issue]
        issue_profile = Array{Float64,3}(undef, n_q, 1, 1)

        profiles = @view coalition_profiles[issue][:, 1, 1:n_parties]

        for question in 1:n_q

            min_party_pos = profiles[question, minority_govt_party]
            if strict

                if length(findall(i -> profiles[question, i] == min_party_pos,
                    1:n_parties)) / n_parties >= 0.5

                    issue_profile[question, 1, 1] = profiles[question, minority_govt_party]

                else

                    issue_profile[question, 1, 1] = 0.0

                end

            else

                if length(
                    findall(i -> sign(profiles[question, i]) == sign(min_party_pos), 1:n_parties)
                ) / n_parties >= 0.5

                    issue_profile[question, 1, 1] = profiles[question, minority_govt_party]

                else

                    issue_profile[question, 1, 1] = 0.0

                end
            end
        end

        minority_govt_profile[issue] = issue_profile

    end

    voter_utilities_for_minority_govt = compute_utilities_for_party_profiles(
        voter_question_positions, minority_govt_profile, voter_issue_weights, 1, n_seats,
        pop_per_seat, n_issues)

    voter_utilities_for_minority_govt = HelpfulFunctions.scale_utilities(
        voter_utilities_for_minority_govt
    )

    return minority_govt_party, minority_govt_profile, voter_utilities_for_minority_govt

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

function find_eligible_condorcet_winner(voter_utilities_for_coalition_profiles::Array{Float64,3},
    coalition_unanimities::AbstractVector{Int}, n_questions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int)

    eligible_coalitions = find_eligible_coalitions(coalition_unanimities, n_questions)
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
            elseif observed_winner == condorce_winner
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

    return utility_of_winner, utility_efficiency, utility_maxer_elected

end

function determine_proportional_metrics(coalition_options::Vector{Vector{Int}},
    coalition_profiles::AbstractVector{Array{Float64,3}},
    party_utilities_for_coalitions::Array{Float64,3},
    voter_utilities_for_coalition_profiles::Array{Float64,3},
    social_utilities::Vector{Float64},
    coalition_unanimities::Vector{Int},
    voter_issue_weights::Array{Float64,3},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    winning_parties::Dict{Int,Float64},
    n_issues::Int, n_questions::AbstractVector{Int}, n_seats::Int, pop_per_seat::Int,
    n_parties::Int, strict::Bool)

    minority_govt_elected = false

    observed_winner = find_winning_coalition(coalition_options, party_utilities_for_coalitions,
        coalition_unanimities, winning_parties, n_questions)
    unanimity = observed_winner == 0 ? 0 : coalition_unanimities[observed_winner]

    begin
        winner_is_condorcet, winner_in_smith_set, condorcet_paradox =
            determine_condorcet_or_smith(
                observed_winner, voter_utilities_for_coalition_profiles,
                coalition_unanimities, n_questions, n_seats, pop_per_seat
            )
    end

    if observed_winner == 0

        begin
            _, minority_govt_profile, voter_utilities_for_minority_govt_profile =
                find_minority_government_profile(
                    winning_parties, coalition_profiles,
                    voter_issue_weights, voter_question_positions, n_issues, n_questions, n_seats,
                    pop_per_seat, n_parties, strict
                )
        end

        social_utility = sum(voter_utilities_for_minority_govt_profile)
        minority_utils = [social_utilities; social_utility]
        non_zero_positions = count(!iszero, vcat(minority_govt_profile...))
        median_position = median(vcat(minority_govt_profile...))

        begin
            utility_of_winner, utility_efficiency, utility_maxer_elected =
                determine_utility_metrics(length(social_utilities) + 1, minority_utils)
        end

        minority_govt_elected = true

    else

        utility_of_winner, utility_efficiency, utility_maxer_elected = determine_utility_metrics(
            observed_winner, social_utilities
        )

        non_zero_positions = count(!iszero, vcat(minority_govt_profile...)[:, :, observed_winner])
        median_positions = median(vcat(coalition_profiles...)[:, :, observed_winner])

    end

    indicators = [
        winner_is_condorcet, winner_in_smith_set, condorcet_paradox,
        utility_maxer_elected, minority_govt_elected
    ]

    prop_eval_measures = ProportionalEvalMeasures(utility_of_winner, utility_efficiency,
        non_zero_positions, median_positions, unanimity, n_seats)
    pop_eval_indicators = ProportionalEvalIndicators(indicators...)

    return prop_eval_measures, pop_eval_indicators

end

function evaluate_proportional_election(
    party_ideal_points::AbstractVector{Array{Float64,3}},
    voter_question_positions::AbstractVector{Array{Float64,3}},
    party_question_positions::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    winning_parties::Dict{Int,Float64},
    n_parties::Int, n_issues::Int, n_questions::AbstractVector{Int},
    issue_dimensions::AbstractVector{Int},
    n_seats::Int, pop_per_seat::Int) # winning parties should be true, over-threshold winning parties

    party_issue_weights = find_issue_weights(party_ideal_points, n_issues, 1, n_parties,
        issue_dimensions)

    begin
        coalition_options, strict_coalition_profiles, qualified_coalition_profiles,
        voter_utilities_for_strict_coalition_profile,
        voter_utilities_for_qualified_coalition_profile, coalition_unanimities,
        coalition_qualified_unanimities = evaluate_party_profiles(
            voter_question_positions, party_question_positions, voter_issue_weights, n_parties,
            n_issues, n_seats, pop_per_seat, n_questions
        )
    end

    strict_coalition_profiles = convert.(Array, strict_coalition_profiles) # profiles of each coalition on each issue
    qualified_coalition_profiles = convert.(Array, qualified_coalition_profiles)

    voter_utilities_for_strict_coalition_profile = scale_utilities(
        voter_utilities_for_strict_coalition_profile
    )
    strict_social_utilities = sum(voter_utilities_for_strict_coalition_profile, dims=(1, 2))[:]

    voter_utilities_for_qualified_coalition_profile = scale_utilities(
        voter_utilities_for_qualified_coalition_profile
    )
    qualified_social_utilities = sum(
        voter_utilities_for_qualified_coalition_profile, dims=(1, 2)
    )[:]

    n_coalitions = length(coalition_options)

    party_utilities_for_strict_coalition_profiles = compute_utilities_for_party_profiles(
        party_question_positions, strict_coalition_profiles, party_issue_weights, n_coalitions, 1,
        n_parties, n_issues
    ) # each party's utility from a given coalition
    party_utilities_for_qualified_coalition_profiles = compute_utilities_for_party_profiles(
        party_question_positions, qualified_coalition_profiles, party_issue_weights, n_coalitions, 1,
        n_parties, n_issues
    ) # each party's utility from a given coalition

    utils_strict, indicators_strict = determine_proportional_metrics(
        coalition_options, strict_coalition_profiles, party_utilities_for_strict_coalition_profiles,
        voter_utilities_for_strict_coalition_profile, strict_social_utilities,
        coalition_unanimities, voter_issue_weights, voter_question_positions, winning_parties,
        n_issues, n_questions, n_seats, pop_per_seat, n_parties, true
    )

    utils_qualified, indicators_qualified = determine_proportional_metrics(
        coalition_options, qualified_coalition_profiles,
        party_utilities_for_qualified_coalition_profiles,
        voter_utilities_for_qualified_coalition_profile, qualified_social_utilities,
        coalition_qualified_unanimities, voter_issue_weights, voter_question_positions,
        winning_parties, n_issues, n_questions, n_seats, pop_per_seat, n_parties, false
    )

    return ProportionalEvaluation(
        indicators_qualified, utils_qualified, indicators_strict, utils_strict
    )

end


end

