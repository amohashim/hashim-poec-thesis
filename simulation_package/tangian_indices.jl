module TangianIndices

export TangianIndicesResults

using StaticArrays
using Statistics
using Parameters

using ..SimulationParameters

struct TangianIndicesResults

    raw_pop_body::Float64
    adj_pop_body::Float64
    raw_uni_body::Float64
    adj_uni_body::Float64

    raw_pop_parties::Float64
    adj_pop_parties::Float64
    raw_uni_parties::Float64
    adj_uni_parties::Float64

end

export computeTangianIndices

"""
    computeTangianIndices(
        question_positions,      # Vector{Array{Float64,3}} of length K
        party_question_positions,# Vector{Array{Float64,3}} of length K
        election_winners,        # Vector{Int} of length N
        proportional_election_results::Dict{Int,Float64},
        N_QUESTIONS::SVector{K,Int},
        N_POSITIONS::SVector{K,Int},
        K::Int,
        N::Int,
        A::Int,
        N_PARTIES::Int
    )

Returns a NamedTuple with fields:
  :raw_pop_candidates  => Vector{Float64} (length N)
  :adj_pop_candidates  => Vector{Float64} (length N)
  :raw_uni_candidates  => Vector{Float64} (length N)
  :adj_uni_candidates  => Vector{Float64} (length N)

  :raw_pop_body  => Float64
  :adj_pop_body  => Float64
  :raw_uni_body  => Float64
  :adj_uni_body  => Float64

  :raw_pop_parties  => Float64
  :adj_pop_parties  => Float64
  :raw_uni_parties  => Float64
  :adj_uni_parties  => Float64

The function calculates:
1) Raw & adjusted popularity for each district’s winning candidate
2) Raw & adjusted universality for each district’s winning candidate
3) Raw & adjusted popularity (and universality) of the decisive body of winners (equal weight)
4) Raw & adjusted popularity (and universality) of the decisive body of parties (weighted by vote share)
"""
function computeTangianIndices(
    fixed_params::FixedParams, issue_structure::IssueStructure,
    question_structure::QuestionStructure,
    question_positions::Vector{Array{Float64,3}},
    party_question_positions::Vector{Array{Float64,3}},
    election_winners::Vector{Int},
    proportional_election_results::Dict{Int,Float64},
    n_parties::Int
)

    @unpack n_seats, pop_per_seat = fixed_params
    @unpack n_questions, n_positions = question_structure
    @unpack n_issues = issue_structure

    ############################################################################
    ## 1) Precompute totals for normalizations
    ############################################################################
    # Total count of "decomposed" sub-questions:
    # Summation of (Q_k * N_POSITIONS[k])
    total_decomposed = 0
    # Summation of Q_k (the original question count)
    total_questions = 0
    @inbounds for k_ in 1:n_issues
        total_decomposed += n_questions[k_] * n_positions[k_]
        total_questions += n_questions[k_]
    end

    ############################################################################
    ## 2) Prepare output arrays for district-level results
    ############################################################################
    raw_pop_candidates = Array{Float64}(undef, n_seats)
    adj_pop_candidates = Array{Float64}(undef, n_seats)
    raw_uni_candidates = Array{Float64}(undef, n_seats)
    adj_uni_candidates = Array{Float64}(undef, n_seats)

    ############################################################################
    ## 3) District-by-district loop for winning candidates
    ############################################################################
    # We'll accumulate the results for each district's winning candidate.
    # Then we can average them for the "decisive body of winners".
    @inbounds for n_ in 1:n_seats
        c = election_winners[n_]  # index of the winning candidate in district n_

        # We'll keep track of:
        #   sum_of_fractions_raw      -> sum_{over all questions} r_q^D * (N_POSITIONS for that question)
        #   sum_of_fractions_adjusted -> sum_{over all questions} r_q^D
        #   sum_of_univ_raw          -> count of questions (weighted by N_POSITIONS) where r_q^D >= 0.5
        #   sum_of_univ_adjusted     -> count of questions (weighted by 1) where r_q^D >= 0.5
        sum_of_fractions_raw = 0.0
        sum_of_fractions_adjusted = 0.0
        sum_of_univ_raw = 0.0
        sum_of_univ_adjusted = 0.0

        # Loop across issues, then questions
        for k_ in 1:n_issues
            Qk = n_questions[k_]
            Pk = n_positions[k_]  # number of positions in decomposition for issue k
            arr_k = question_positions[k_]  # 3D array: Qk x N x A

            for q_ in 1:Qk
                # Count how many voters match the candidate's position
                pos_candidate = @inbounds arr_k[q_, n_, c]
                match_count::Int = 0
                @inbounds @simd for i_ in 1:pop_per_seat
                    if arr_k[q_, n_, i_] == pos_candidate
                        match_count += 1
                    end
                end

                fraction = match_count / pop_per_seat

                # Raw weighting:
                sum_of_fractions_raw += fraction * Pk
                # Adjusted weighting:
                sum_of_fractions_adjusted += fraction

                # Universality step:
                if fraction >= 0.5
                    sum_of_univ_raw += Pk
                    sum_of_univ_adjusted += 1
                end
            end
        end

        # Now compute final district-level indices:
        raw_pop_candidates[n_] = sum_of_fractions_raw / total_decomposed
        adj_pop_candidates[n_] = sum_of_fractions_adjusted / total_questions
        raw_uni_candidates[n_] = sum_of_univ_raw / total_decomposed
        adj_uni_candidates[n_] = sum_of_univ_adjusted / total_questions
    end

    ############################################################################
    ## 4) Compute body-level (winning candidates) results
    ############################################################################
    # The problem statement says: “Calculate the raw & adjusted popularity (and
    # universality) index of the decisive body composed of all winning candidates,
    # each with equal weight.” Because each district is the same size (A voters),
    # taking the mean of the district-level indices is equivalent to weighting
    # each district equally by its electorate. 
    raw_pop_body = mean(raw_pop_candidates)
    adj_pop_body = mean(adj_pop_candidates)
    raw_uni_body = mean(raw_uni_candidates)
    adj_uni_body = mean(adj_uni_candidates)

    ############################################################################
    ## 5) Calculate raw & adjusted popularity/universality for the parties
    ##    (weighted by vote share)
    ############################################################################
    # We'll do a single pass to accumulate total "popularity" for each party,
    # both raw & adjusted, plus total "universality" for each party (raw & adjusted).
    # Then we combine them by party's vote share.
    #
    # For popularity and universality, we do effectively the same logic as for
    # a candidate, but now "the representative position" = party_question_positions[k][q,1,p].
    #
    # We'll accumulate results across the entire electorate (N*A voters).
    # Then convert to an index on the scale [0,1] the same way (divide by total_decomposed
    # or total_questions). Then weight by the party’s vote share.

    # Pre-allocate accumulators for each party:
    pop_raw_by_party = zeros(Float64, n_parties)
    pop_adjusted_by_party = zeros(Float64, n_parties)
    uni_raw_by_party = zeros(Float64, n_parties)
    uni_adjusted_by_party = zeros(Float64, n_parties)

    # We'll iterate party by party:
    @inbounds for p_ in 1:n_parties
        # We'll accumulate over all issues & questions, counting how many
        # voters share the party's position. Then do the raw vs adjusted weighting.
        sum_of_fractions_raw = 0.0
        sum_of_fractions_adjusted = 0.0
        sum_of_univ_raw = 0.0
        sum_of_univ_adjusted = 0.0

        for k_ in 1:n_issues
            Qk = n_questions[k_]
            Pk = n_positions[k_]
            arr_k_voters = question_positions[k_]           # Qk x N x A
            arr_k_party = party_question_positions[k_]     # Qk x 1 x N_PARTIES

            for q_ in 1:Qk
                pos_party = @inbounds arr_k_party[q_, 1, p_]
                match_count = 0
                # We will count matches for *all* voters across all N districts
                @inbounds @simd for n_ in 1:n_seats
                    @inbounds @simd for i_ in 1:pop_per_seat
                        if arr_k_voters[q_, n_, i_] == pos_party
                            match_count += 1
                        end
                    end
                end
                # fraction of entire electorate that matches:
                fraction = match_count / (n_seats * pop_per_seat)

                # Raw weighting (treat sub-positions as separate)
                sum_of_fractions_raw += fraction * Pk
                # Adjusted weighting (treat question as 1 unit)
                sum_of_fractions_adjusted += fraction

                if fraction >= 0.5
                    sum_of_univ_raw += Pk
                    sum_of_univ_adjusted += 1
                end
            end
        end

        # Now convert sums to an index
        raw_pop = sum_of_fractions_raw / total_decomposed
        adj_pop = sum_of_fractions_adjusted / total_questions
        raw_uni = sum_of_univ_raw / total_decomposed
        adj_uni = sum_of_univ_adjusted / total_questions

        # Store these partial results for this party
        pop_raw_by_party[p_] = raw_pop
        pop_adjusted_by_party[p_] = adj_pop
        uni_raw_by_party[p_] = raw_uni
        uni_adjusted_by_party[p_] = adj_uni
    end

    # Combine by vote share
    raw_pop_parties = 0.0
    adj_pop_parties = 0.0
    raw_uni_parties = 0.0
    adj_uni_parties = 0.0
    for p_ in keys(proportional_election_results)
        w = proportional_election_results[p_]
        raw_pop_parties += w * pop_raw_by_party[p_]
        adj_pop_parties += w * pop_adjusted_by_party[p_]
        raw_uni_parties += w * uni_raw_by_party[p_]
        adj_uni_parties += w * uni_adjusted_by_party[p_]
    end

    return TangianIndicesResults(
        raw_pop_body, adj_pop_body, raw_uni_body, adj_uni_body,
        raw_pop_parties, adj_pop_parties, raw_uni_parties, adj_uni_parties
    )


    ############################################################################
    ## 6) Return all results in a convenient container
    ############################################################################
    # return (
    #     raw_pop_candidates=raw_pop_candidates,
    #     adj_pop_candidates=adj_pop_candidates,
    #     raw_uni_candidates=raw_uni_candidates,
    #     adj_uni_candidates=adj_uni_candidates, raw_pop_body=raw_pop_body,
    #     adj_pop_body=adj_pop_body,
    #     raw_uni_body=raw_uni_body,
    #     adj_uni_body=adj_uni_body, raw_pop_parties=raw_pop_parties,
    #     adj_pop_parties=adj_pop_parties,
    #     raw_uni_parties=raw_uni_parties,
    #     adj_uni_parties=adj_uni_parties
    # )
end

end # module

module TangianIndicesRedone

using StaticArrays
using Statistics

function computeTangianIndicesFixed(
    question_positions::Vector{Array{Float64,3}},
    party_question_positions::Vector{Array{Float64,3}},
    election_winners::Vector{Int},
    proportional_election_results::Dict{Int,Float64},
    N_QUESTIONS::SVector{n_issues,Int},
    N_POSITIONS::SVector{n_issues,Int},
    K::Int,
    N::Int,
    A::Int,
    N_PARTIES::Int
) where {n_issues}
    # Precompute denominators
    total_raw_questions = sum(N_QUESTIONS[k_] * N_POSITIONS[k_] for k_ in 1:K)
    total_adj_questions = sum(N_QUESTIONS[k_] for k_ in 1:K)

    # Preallocate outputs
    raw_pop_candidates = zeros(Float64, N)
    adj_pop_candidates = zeros(Float64, N)
    raw_uni_candidates = zeros(Float64, N)
    adj_uni_candidates = zeros(Float64, N)

    # Helper for district-level calculations
    function compute_district_indices(c::Int, n_::Int)
        raw_numer_pop = 0.0
        raw_numer_uni = 0.0
        adj_numer_pop = 0.0
        adj_numer_uni = 0.0

        for k_ in 1:K
            Qk, Pk = N_QUESTIONS[k_], N_POSITIONS[k_]
            arr_k = question_positions[k_]

            for q_ in 1:Qk
                chosen_pos = arr_k[q_, n_, c]
                match_count = sum(arr_k[q_, n_, i_] == chosen_pos for i_ in 1:A)
                fraction_local = match_count / A

                # Raw indices
                raw_numer_pop += fraction_local * Pk
                if fraction_local >= 0.5
                    raw_numer_uni += Pk
                end

                # Adjusted indices
                adj_numer_pop += fraction_local
                if fraction_local >= 0.5
                    adj_numer_uni += 1
                end
            end
        end

        return (
            raw_numer_pop / total_raw_questions,
            adj_numer_pop / total_adj_questions,
            raw_numer_uni / total_raw_questions,
            adj_numer_uni / total_adj_questions
        )
    end

    # Compute candidate indices
    for n_ in 1:N
        c = election_winners[n_]
        raw_pop_candidates[n_], adj_pop_candidates[n_], raw_uni_candidates[n_], adj_uni_candidates[n_] =
            compute_district_indices(c, n_)
    end

    # Aggregate for decisive body
    raw_pop_body = mean(raw_pop_candidates)
    adj_pop_body = mean(adj_pop_candidates)
    raw_uni_body = mean(raw_uni_candidates)
    adj_uni_body = mean(adj_uni_candidates)

    # Helper for party-level calculations
    function compute_party_indices(p::Int)
        raw_numer_pop = 0.0
        raw_numer_uni = 0.0
        adj_numer_pop = 0.0
        adj_numer_uni = 0.0

        for k_ in 1:K
            Qk, Pk = N_QUESTIONS[k_], N_POSITIONS[k_]
            arr_voters, arr_party = question_positions[k_], party_question_positions[k_]

            for q_ in 1:Qk
                chosen_pos = arr_party[q_, 1, p]
                match_count = sum(arr_voters[q_, n_, i_] == chosen_pos for n_ in 1:N, i_ in 1:A)
                fraction_whole = match_count / (N * A)

                # Raw indices
                raw_numer_pop += fraction_whole * Pk
                if fraction_whole >= 0.5
                    raw_numer_uni += Pk
                end

                # Adjusted indices
                adj_numer_pop += fraction_whole
                if fraction_whole >= 0.5
                    adj_numer_uni += 1
                end
            end
        end

        return (
            raw_numer_pop / total_raw_questions,
            adj_numer_pop / total_adj_questions,
            raw_numer_uni / total_raw_questions,
            adj_numer_uni / total_adj_questions
        )
    end

    # Compute party indices
    raw_pop_parties = 0.0
    adj_pop_parties = 0.0
    raw_uni_parties = 0.0
    adj_uni_parties = 0.0

    for (p, vote_share) in proportional_election_results
        raw_pop, adj_pop, raw_uni, adj_uni = compute_party_indices(p)
        raw_pop_parties += vote_share * raw_pop
        adj_pop_parties += vote_share * adj_pop
        raw_uni_parties += vote_share * raw_uni
        adj_uni_parties += vote_share * adj_uni
    end

    return TangianIndicesResults(
        raw_pop_body, adj_pop_body, raw_uni_body, adj_uni_body,
        raw_pop_parties, adj_pop_parties, raw_uni_parties, adj_uni_parties
    )

end

end # module
