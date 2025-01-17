
module ProportionalEvaluationMetrics

using Combinatorics
using Distributions
using Random
using LinearAlgebra
using DataStructures

@inline function build_coalition_options(n_parties::Int)

    combos = collect(combinations(1:n_parties))
    i = length.(combos) .<= ceil(n_parties * 0.5)
    coalitions = combos[i]

    return coalitions

end

function build_party_profile_options(coalition_options::AbstractVector,
    party_question_positions::AbstractVector{Array{Float64,3}}, n_issues::Int,
    n_questions::AbstractVector{Int})

    coalition_profiles = Vector{Array{Float64,3}}(undef, n_issues)
    n_coalitions = length(coalition_options)

    coalition_unanimities = zeros(Int, n_coalitions)
    coalition_qualified_unanimities = zeros(Int, n_coalitions)

    @inbounds for issue in 1:n_issues

        n_q = n_questions[issue]
        profiles_for_issue = Array{Float64,3}(undef, n_q, 1, n_coalitions)

        @inbounds for (idx, coalition) in enumerate(coalition_options)

            coalition_positions = party_question_positions[issue][:, 1, coalition]
            unanimous_positions = [
                all(x -> x == row[1], row) for row in eachrow(coalition_positions)
            ]
            same_direction = [
                all(x -> x == 0 || sign(x) == sign(row[findfirst(y -> y != 0, row)]), row)
                for row in eachrow(coalition_positions)
            ]

            coalition_profile = [unanimous_positions[q] == 1 ? coalition_positions[q] : 0.0
                                 for q in 1:n_q]

            profiles_for_issue[:, 1, idx] = coalition_profile

            coalition_unanimities[idx] += length(findall(unanimous_positions))
            coalition_qualified_unanimities[idx] += length(findall(same_direction))

        end

        coalition_profiles[issue] = profiles_for_issue
    end

    return coalition_profiles, coalition_unanimities, coalition_qualified_unanimities

end

function find_issue_weights(ideal_points::AbstractVector{Array{Float64,3}}, n_issues::Int,
    n_seats::Int, pop_per_seat::Int, issue_dimensions::AbstractVector{Int})

    voter_magnitudes = Array{Float64,3}(undef, n_issues, n_seats, pop_per_seat)

    @inbounds for issue in 1:n_issues

        issue_dimension = issue_dimensions[issue]
        points_for_issue = ideal_points[issue]
        norms = similar(@view points_for_issue[:, :, 1])

        mats = [@view points_for_issue[:, :, d] for d in 1:issue_dimension]
        @inbounds for i in eachindex(norms)
            sum_of_squares = 0.0
            @inbounds for M in mats
                sum_of_squares += M[i]^2
            end
            norms[i] = sqrt(sum_of_squares)
        end

        voter_magnitudes[issue, :, :] = norms ./ Ref(sqrt(issue_dimension))

    end

    return voter_magnitudes

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


function find_winning_coalition(n_questions::AbstractVector{Int},
    winning_parties::Dict{Int,Float64}, coalition_options::Vector{Vector{Int}},
    coalition_qualified_unanimities::Vector{Int}, party_coalition_utilities::Array{Float64,3})

    total_questions = sum(n_questions)
    min_coalitions = find_minimal_coalitions(winning_parties, coalition_options)
    min_coalitions
    min_coalition_indices = [findfirst(==(x), coalition_options) for x in min_coalitions]
    min_coalition_indices
    min_eligible_coalition_indices = findall(
        i -> coalition_qualified_unanimities[i] / total_questions >= 0.5, min_coalition_indices
    )
    min_eligible_coalition_indices

    mec = min_coalitions[min_eligible_coalition_indices]
    mec_global_idx = min_coalition_indices[min_eligible_coalition_indices]
    mec_global_idx

    if length(mec_global_idx) < 1

        return 0

    elseif length(mec_global_idx) == 1

        return vec(mec_global_idx)[1]

    else

        party_utils_for_mec = @view party_coalition_utilities[1, :, mec_global_idx]

        coalition_utilities = Vector{Float64}(undef, length(mec))

        for (index, parties) in enumerate(mec)

            coalition_utilities[index] = sum(party_utils_for_mec[parties, index]) / sqrt(length(parties))

        end

        return mec_global_idx[argmin(coalition_utilities)]

    end

end

@inline function sample_from_dirichlet(preferred_parties::Matrix{Int}, sample_size::Int,
    rng::AbstractRNG, n_seats::Int, pop_per_seat::Int, n_parties::Int)

    sampled_voters = sample(rng, 1:(n_seats*pop_per_seat), sample_size, replace=false)
    party_counts = Dict(counter(preferred_parties[sampled_voters]))

    for party in 1:n_parties

        if !haskey(party_counts, party)
            party_counts[party] = 0

        end

    end

    party_counts = OrderedDict(k => party_counts[k] for k in sort(collect(keys(party_counts))))

    alpha = [1 + party_counts[party] for party in 1:n_parties]
    dirichlet_sample = rand(rng, Dirichlet(alpha))

    return Dict{Int,Float64}(p => dirichlet_sample[p] for p in 1:n_parties)

end

function compute_vse(n_iterations::Int, preferred_parties::Matrix{Int}, rng::AbstractRNG,
    n_seats::Int, pop_per_seat::Int, n_parties::Int, sample_size::Int,
    n_questions::AbstractVector{Int}, coalition_options::AbstractVector,
    coalition_qualified_unanimities::AbstractVector{Int},
    party_coalition_utilities::Array{Float64,3}, social_utilities::Vector{Float64},
    observed_election_winner::Int)

    n_coalitions = length(coalition_options)
    minority_govt_count = 0
    coalition_wins = zeros(Int, n_coalitions)

    for _ in 1:n_iterations

        winning_parties = sample_from_dirichlet(preferred_parties, sample_size, rng, n_seats,
            pop_per_seat, n_parties
        )

        winning_coalition = find_winning_coalition(n_questions,
            winning_parties, coalition_options, coalition_qualified_unanimities,
            party_coalition_utilities)

        if winning_coalition == 0
            minority_govt_count += 1
        end
        coalition_wins[winning_coalition] += 1

    end

    prob_of_win = coalition_wins ./ n_iterations
    expected_utilities = prob_of_win .* social_utilities

    e_winner = expected_utilities[observed_election_winner]

    e_maximizer = expected_utilities[argmax(social_utilities)]

    e_average = mean(social_utilities)

    return (e_winner - e_average) / (e_maximizer - e_average)

end

end
