module ElectionSimulation

using Distributions
using LinearAlgebra
using Random
using StaticArrays
using LinearAlgebra: norm
using DataStructures

export poll_district, expected_policy_utility

"""

DISTANCE UTILITY

    find_closest_party_for_one_voter(
        voter_ideals::Vector{Vector{Float64}},
        parties::Vector{Matrix{Float64}}
    ) -> Int
Given:
- voter_ideals[i]: the d_i-dimensional ideal point of this voter on issue i
- parties[i]: a P x d_i matrix of party positions for issue i
Compute w_i = norm(voter_ideals[i]) / sqrt(d_i) and then the Euclidean distance
d(voter, party) for each issue i, sum across i with the weight w_i,
and return the party index (1..P) that yields the minimum sum.
This function processes a single voter’s assignment.
"""
function find_closest_representative_for_one_voter(
    dist_sums::Vector{Float64},
    voter_ideal_points::Vector{Vector{Float64}},
    representative_ideal_points::Vector{Matrix{Float64}},
    voter_issue_weights::Vector{Float64},
    n_representatives::Int,
    n_issues::Int,
    issue_dimensions::AbstractVector{Int}
)
    # We'll accumulate total distances in a local vector
    fill!(dist_sums, 0.0)   # set all party sums to zero
    @inbounds for issue in 1:n_issues
        v_i = voter_ideal_points[issue]     # d_i-vector (the voter)
        d_i = issue_dimensions[issue]
        w_i = voter_issue_weights[issue]
        representative_mat = representative_ideal_points[issue]    # P x d_i
        # for each party p, compute the Euclidean distance in dimension i
        @inbounds for rep in 1:n_representatives
            # distance on issue i: sum of squared differences in each dimension, then sqrt
            ssd = 0.0
            # ssd = sum((v_i - representative_mat[rep, :]) .^ 2)
            @inbounds for dim in 1:d_i
                diff = v_i[dim] - representative_mat[rep, dim]
                ssd += diff * diff
            end
            dist_sums[rep] += w_i * sqrt(ssd)
        end
    end

    utilities = -1 .* (dist_sums .^ 2)
    # now find the party with the min distance sum
    _, idx = findmax(utilities)
    return idx, utilities
end

"""

DIRECTIONAL UTILITY

    find_closest_party_for_one_voter(
        voter_ideals::Vector{Vector{Float64}},
        parties::Vector{Matrix{Float64}}
    ) -> Int
Given:
- voter_ideals[i]: the d_i-dimensional ideal point of this voter on issue i
- parties[i]: a P x d_i matrix of party positions for issue i
Compute w_i = norm(voter_ideals[i]) / sqrt(d_i) and then the Euclidean distance
d(voter, party) for each issue i, sum across i with the weight w_i,
and return the party index (1..P) that yields the minimum sum.
This function processes a single voter’s assignment.
"""
function find_closest_representative_for_one_voter(
    voter_super_ideal_points::Vector{Vector{Float64}},
    representative_super_ideal_points::Vector{Matrix{Float64}},
    β::Float64
)
    v_i = voter_super_ideal_points[1]  # d_i-vector (the voter)
    representative_mat = representative_super_ideal_points[1]  # P x d_i

    @views voter_mag_sq = sum(v_i .^ 2)  # Precompute voter magnitude squared
    @views candidate_mags_sq = sum(representative_mat .^ 2, dims=2)[:]  # Vectorized computation

    # Compute utilities
    @views utilities = 2.0 .* (representative_mat * v_i) .- β .* (voter_mag_sq .+ candidate_mags_sq)

    idx = argmax(utilities)
    return idx, utilities
end


"""

For static number of parties

    assign_voters_to_parties!(
        agents_ideals::Vector{Array{Float64,3}},
        parties::Vector{Matrix{Float64}}
    ) -> Matrix{Int}
Given:
- agents_ideals[i] is an N x A x d_max array of voter ideal points for issue i
- parties[i] is a P x d_i matrix of party positions for issue i
For each district n in 1..N, and each agent a in 1..A, gather that voter’s
d_i-dimensional vector in each issue i, compute the weighted distance to
each party, and assign the voter to the closest party. Return an N x A
matrix containing the assigned party index for each (n,a).
"""
function assign_voters_to_parties!(
    voter_ideal_points::AbstractVector{Array{Float64,3}}, # all ideal points
    party_ideal_points::AbstractVector{Matrix{Float64}},
    voter_issue_weights::Array{Float64,3},
    n_parties::Int,
    n_issues::Int,
    n_seats::Int,
    pop_per_seat::Int,
    issue_dimensions::AbstractVector{Int};
    use_directional_utility::Bool=false,
    β::Float64=1.0
)

    # We'll produce an N x A result
    result = Matrix{Int}(undef, n_seats, pop_per_seat)
    # To avoid repeated allocations, we can store each voter's I-dimensional vector-of-vectors
    # in a pre-existing structure. We'll do a small trick: create a container to hold references
    # to dimension slices. Something like: voter_ideals[i] = some d_i-vector.
    # We'll reuse this container for each (n,a).
    # Pre-allocate container for voter_i for each i
    # We'll store them as Vectors, but we do not want to re-allocate the underlying arrays each time.
    # We'll just "view" the slice of the 3D array for dimension handling.
    voter_ideals = Vector{Vector{Float64}}(undef, n_issues)
    voter_utilities = Array{Float64,3}(undef, n_seats, pop_per_seat, n_parties)
    for issue in 1:n_issues
        # We'll allocate a zero-length Vector for each i initially
        voter_ideals[issue] = Float64[]
    end

    dist_sums = Vector{Float64}(undef, n_parties)

    @inbounds for seat in 1:n_seats

        seat_issue_weights = @view voter_issue_weights[:, seat, :]

        @inbounds for voter in 1:pop_per_seat
            # Step 1: gather the voter's ideal for each issue i
            @inbounds for issue in 1:n_issues

                d_i = issue_dimensions[issue] # the relevant dimension for this issue
                # We'll just take a *view* into agents_ideals[i][n, a, 1:d_i]
                # For performance, we can do something like:
                voter_ideals[issue] = @views voter_ideal_points[issue][seat, voter, 1:d_i]
            end

            # Step 2: find the closest party

            if use_directional_utility
                party_idx, utilities = find_closest_representative_for_one_voter(
                    voter_ideals, party_ideal_points, β)
            else

                issue_weights = seat_issue_weights[:, voter]
                party_idx, utilities = find_closest_representative_for_one_voter(dist_sums,
                    voter_ideals, party_ideal_points, issue_weights, n_parties, n_issues,
                    issue_dimensions)

            end

            # Step 3: store
            result[seat, voter] = party_idx
            voter_utilities[seat, voter, :] = utilities
        end
    end
    return result, voter_utilities
end


"""

For fixed number of candidates in the district

"""
function assign_voters_to_candidates!(
    dist_sums::Vector{Float64},
    voter_ideal_points::AbstractVector{Array{Float64,3}},
    candidate_ideal_points::AbstractVector{Matrix{Float64}},
    voter_issue_weights::Array{Float64,3},
    n_candidates::Int,
    n_issues::Int,
    seat::Int,
    pop_per_seat::Int,
    issue_dimensions::AbstractVector{Int};
    use_directional_utility::Bool=false,
    β::Float64=1.0
)
    # We'll produce an N x A result
    result = Matrix{Int}(undef, 1, pop_per_seat)
    voter_utilities = Matrix{Float64}(undef, pop_per_seat, n_candidates)
    # To avoid repeated allocations, we can store each voter's I-dimensional vector-of-vectors
    # in a pre-existing structure. We'll do a small trick: create a container to hold references
    # to dimension slices. Something like: voter_ideals[i] = some d_i-vector.
    # We'll reuse this container for each (n,a).
    # Pre-allocate container for voter_i for each i
    # We'll store them as Vectors, but we do not want to re-allocate the underlying arrays each time.
    # We'll just "view" the slice of the 3D array for dimension handling.
    voter_ideals = Vector{Vector{Float64}}(undef, n_issues)
    @inbounds for issue in 1:n_issues
        # We'll allocate a zero-length Vector for each i initially
        voter_ideals[issue] = Float64[]
    end

    @inbounds for voter in 1:pop_per_seat
        # Step 1: gather the voter's ideal for each issue i
        @inbounds for issue in 1:n_issues
            d_i = issue_dimensions[issue]  # the relevant dimension for this issue
            # We'll just take a *view* into agents_ideals[i][n, a, 1:d_i]
            # For performance, we can do something like:
            voter_ideals[issue] = @views voter_ideal_points[issue][seat, voter, 1:d_i]
        end
        # Step 2: find the closest party

        if use_directional_utility
            party_idx, utilities = find_closest_representative_for_one_voter(
                voter_ideals, candidate_ideal_points, β
            )
        else
            issue_weights = voter_issue_weights[:, seat, voter]
            party_idx, utilities = find_closest_representative_for_one_voter(dist_sums,
                voter_ideals, candidate_ideal_points, issue_weights, n_candidates, n_issues,
                issue_dimensions
            )
        end
        # Step 3: store
        result[1, voter] = party_idx
        voter_utilities[voter, :] = utilities
    end

    return result, voter_utilities
end

"""
For dynamic number of candidates in the district

"""
function assign_voters_to_candidates!(
    voter_ideal_points::AbstractVector{Array{Float64,3}},
    candidate_ideal_points::AbstractVector{Matrix{Float64}},
    voter_issue_weights::Array{Float64,3},
    n_candidates::Int,
    n_issues::Int,
    seat::Int,
    pop_per_seat::Int,
    issue_dimensions::AbstractVector{Int};
    use_directional_utility::Bool=false,
    β::Float64=1.0
)
    # We'll produce an N x A result
    result = Matrix{Int}(undef, 1, pop_per_seat)
    voter_utilities = Matrix{Float64}(undef, pop_per_seat, n_candidates)
    # To avoid repeated allocations, we can store each voter's I-dimensional vector-of-vectors
    # in a pre-existing structure. We'll do a small trick: create a container to hold references
    # to dimension slices. Something like: voter_ideals[i] = some d_i-vector.
    # We'll reuse this container for each (n,a).
    # Pre-allocate container for voter_i for each i
    # We'll store them as Vectors, but we do not want to re-allocate the underlying arrays each time.
    # We'll just "view" the slice of the 3D array for dimension handling.
    voter_ideals = Vector{Vector{Float64}}(undef, n_issues)
    @inbounds for issue in 1:n_issues
        # We'll allocate a zero-length Vector for each i initially
        voter_ideals[issue] = Float64[]
    end

    dist_sums = Vector{Float64}(undef, n_candidates)
    @inbounds for voter in 1:pop_per_seat
        # Step 1: gather the voter's ideal for each issue i
        @inbounds for issue in 1:n_issues
            d_i = issue_dimensions[issue]  # the relevant dimension for this issue
            # We'll just take a *view* into agents_ideals[i][n, a, 1:d_i]
            # For performance, we can do something like:
            voter_ideals[issue] = @views voter_ideal_points[issue][seat, voter, 1:d_i]
        end
        # Step 2: find the closest party

        if use_directional_utility
            party_idx, utilities = find_closest_representative_for_one_voter(
                voter_ideal_points, candidate_ideal_points, β
            )
        else
            issue_weights = voter_issue_weights[:, seat, voter]
            party_idx, utilities = find_closest_representative_for_one_voter(dist_sums,
                voter_ideal_points, candidate_ideal_points, issue_weights, n_candidates,
                n_issues, issue_dimensions)
        end

        # Step 3: store
        result[1, voter] = party_idx
        voter_utilities[voter, :] = utilities
    end

    return result, voter_utilities
end

@inline function map_global_indx_to_props(local_to_global::Dict{Int,Int},
    local_to_props::Dict{Int,Float64})

    global_to_local = Dict(value => key for (key, value) in local_to_global)
    global_to_props = Dict{Int,Float64}()

    @inbounds for (global_idx, local_idx) in global_to_local

        if haskey(local_to_props, local_idx)

            global_to_props[global_idx] = local_to_props[local_idx]

        end
    end

    return global_to_props

end

"""

for a given district

"""
function find_candidate_ideal_points(ideal_points::AbstractVector{Array{Float64,3}},
    seat_candidates::AbstractVector{Int}, n_candidates::Int, seat_number::Int, n_issues::Int)

    candidate_matrices = Vector{Matrix{Float64}}(undef, n_issues)
    row_to_candidate_map = Dict{Int,Int}()

    for issue in 1:n_issues
        # Get the 3D array for issue `k`
        arr_issue = ideal_points[issue]

        # Collect the ideal points for the candidates
        candidate_matrices[issue] = hcat(
            [arr_issue[seat_number, c, :] for c in seat_candidates]...
        )'

        # Update the mapping
        for (local_idx, global_idx) in enumerate(seat_candidates)
            row_to_candidate_map[local_idx] = global_idx
        end
    end

    return candidate_matrices, row_to_candidate_map

end

"""
N x A x C array where each entry in rankings[n,:,:] is what rank voter a gives candidate C

So, rows are voters and columns are candidates; this is NOT rows are voters and columsn are rankings

"""
function compute_voter_rankings(candidate_utilities::Array{Float64,3}, n_seats::Int,
    pop_per_seat::Int, n_candidates::Int; use_specific_seat::Bool=false, seat_number::Int=0)

    # Prepare an output array for the rankings.
    rankings = Array{Int}(undef, n_seats, pop_per_seat, n_candidates)

    # Preallocate arrays used inside loops
    rank = Vector{Int}(undef, n_candidates)
    sorted_indices = Vector{Int}(undef, n_candidates)

    for seat in 1:n_seats

        if use_specific_seat

            seat = seat_number

        end

        for voter in 1:pop_per_seat
            # Extract the utilities for the voter
            utilities = @view candidate_utilities[seat, voter, :]

            # Sort indices by descending utilities
            sortperm!(sorted_indices, utilities, rev=true)

            # Compute ranks directly
            for candidate in 1:n_candidates
                rank[sorted_indices[candidate]] = candidate
            end

            # Assign computed ranks to the output
            @views rankings[seat, voter, :] = rank
        end
    end

    return rankings
end

function compute_voter_rankings(candidate_utilities::AbstractMatrix,
    pop_per_seat::Int, n_candidates::Int)

    # Prepare an output array for the rankings.
    rankings = Array{Int}(undef, pop_per_seat, n_candidates)

    # Preallocate arrays used inside loops
    rank = Vector{Int}(undef, n_candidates)
    sorted_indices = Vector{Int}(undef, n_candidates)

    for voter in 1:pop_per_seat
        # Extract the utilities for the voter
        utilities = @view candidate_utilities[voter, :]

        # Sort indices by descending utilities
        sortperm!(sorted_indices, utilities, rev=true)

        # Compute ranks directly
        for candidate in 1:n_candidates
            rank[sorted_indices[candidate]] = candidate
        end

        # Assign computed ranks to the output
        @views rankings[voter, :] = rank
    end

    return rankings
end

function sample_voters(A::Int, sample_size::Int, rng::AbstractRNG)
    # If A < sample_size, we just take all
    n_samp = min(A, sample_size)
    # create a permutation of 1..A
    perm = randperm(rng, A)
    return @view perm[1:n_samp]
end


"""
    poll_district(
        d::Int,
        sample_size::Int,
        voter_rankings::Vector{Matrix{Int}},
        is_active::Vector{Bool};
        rng=MersenneTwister(42)
    ) -> Vector{Int}

Randomly samples `sample_size` voters (or all if sample_size > A_d),
from district d, and returns a length-(numCands_d) integer vector of counts.
"""
function poll_district!(
    counts::Vector{Int},
    rankings_n::Matrix{Int},
    is_active::Union{Vector{Bool},BitVector},
    rng::AbstractRNG;
    sample_size::Int=400
)
    # Wipe counts
    fill!(counts, 0)
    A, numCands = size(rankings_n)
    n_samp = min(A, sample_size)

    # We'll produce a random subset of voters
    voters = randperm(rng, A)
    # take the first n_samp as the chosen voters
    chosen = @view voters[1:n_samp]

    @inbounds for v in chosen
        # rankings_n[v, :] is the best->worst order of local cands
        @inbounds for r in 1:numCands
            c_local = rankings_n[v, r]
            if is_active[c_local]
                counts[c_local] += 1
                break
            end
        end
    end
end

function make_preferred_candidates_strategic(
    R_n::Matrix{Int},
    top_two::AbstractVector{Int},
    is_strategic::BitVector,
    preferred_candidates::Vector{Int},
)
    # Precompute each strategic voter's best choice among the top-two
    A, C = size(R_n)  # A: number of voters, C: number of candidates
    c1, c2 = top_two

    # For each voter a, figure out which top-2 candidate they prefer
    # We'll store that in an Int vector of length A
    best_among_top_two = Vector{Int}(undef, A)
    for a in 1:A
        if is_strategic[a]
            if R_n[a, c1] < R_n[a, c2]
                best_among_top_two[a] = c1
            else
                best_among_top_two[a] = c2
            end
        else
            # For non-strategic, store a sentinel (e.g. -1) or 0
            best_among_top_two[a] = -1
        end
    end

    # Now make a copy of the original matrix so we don't mutate it
    preferred_candidates_strategic = copy(preferred_candidates)

    # For each strategic voter, replace the entire column in one shot
    for (a, strategic) in enumerate(is_strategic)
        if strategic
            preferred_candidates_strategic[a] = best_among_top_two[a]
        end
    end

    return preferred_candidates_strategic
end

"""
counting votes for a given district, with given turnout

"""
function count_votes(
    strategic_preferred_candidates_n::Vector{Int},
    turnout_voters_n::BitVector,
    n_candidates::Int
)
    # C is the total number of candidates
    counts = zeros(Int, n_candidates)  # counts[i] will be how many votes candidate i gets

    for i in 1:length(strategic_preferred_candidates_n)
        if turnout_voters_n[i]
            candidate = strategic_preferred_candidates_n[i]
            counts[candidate] += 1
        end
    end
    return counts
end

function run_single_round_election(ideal_points::AbstractVector{Array{Float64,3}},
    candidates::Vector{Vector{Int}}, voter_issue_weights::Array{Float64,3}, n_seats::Int,
    n_candidates::Int, n_issues::Int, pop_per_seat::Int, issue_dimensions::AbstractVector{Int};
    use_directional_utility::Bool=false, β::Float64=1.0,
    strategic_voters::Union{Matrix{Bool},BitMatrix,Nothing}=nothing,
    turnout_voters::Union{Matrix{Bool},BitMatrix,Nothing}=nothing,
    rng::Union{AbstractRNG,Nothing}=nothing)

    election_proportions = Vector{Dict{Int,Float64}}(undef, n_seats)
    voter_utilites = Array{Float64,3}(undef, n_seats, pop_per_seat, n_candidates)
    candidate_choices = Matrix{Int}(undef, n_seats, pop_per_seat)

    @inbounds for seat in 1:n_seats
        candidate_points, candidate_map = find_candidate_ideal_points(
            ideal_points, candidates[seat], n_candidates, seat, n_issues
        )

        dist_sums = Vector{Float64}(undef, n_candidates)

        results, utilities = assign_voters_to_candidates!(dist_sums,
            ideal_points, candidate_points, voter_issue_weights, n_candidates, n_issues, seat,
            pop_per_seat, issue_dimensions; use_directional_utility=use_directional_utility, β=β
        )

        if isnothing(strategic_voters) || n_candidates <= 3
            candidate_choices[seat, :] = results
        else
            is_strategic = strategic_voters[seat, :]
            rankings = compute_voter_rankings(utilities, pop_per_seat, n_candidates)
            poll_counts = zeros(Int, n_candidates)
            poll_district!(poll_counts, rankings, [true for _ in 1:n_candidates], rng)
            top_3_cands = partialsortperm(poll_counts, 1:3, rev=true)
            results = make_preferred_candidates_strategic(
                rankings, top_3_cands, is_strategic, vec(results)
            )
            candidate_choices[seat, :] = results
        end

        if isnothing(turnout_voters)
            proportions = Dict(idx => ct / pop_per_seat for (idx, ct) in counter(results))
        else
            turns_out = turnout_voters[seat, :]
            raw_vote_count = count_votes(vec(results), turns_out, n_candidates)
            proportions = Dict{Int,Float64}(
                candidate => vote_count / count(turnout_voters[seat, :])
                for (candidate, vote_count) in enumerate(raw_vote_count)
            )
        end

        election_proportions[seat] = map_global_indx_to_props(candidate_map, proportions)

        voter_utilites[seat, :, :] = utilities

    end

    return election_proportions, voter_utilites, candidate_choices

end

"""
1) Collapse the 3D array R to a 2D matrix
   R is size (N, A, P).
   The result is size (N*A, P).

    collapse_rankings(R) -> Matrix{Int}

Given a 3D array `R` of size `(N, A, P)`, returns a new 2D array `(N*A, P)` 
where rows 1:A correspond to R[1, :, :], rows (A+1):2A correspond to R[2, :, :], etc.
"""
function collapse_rankings(R::Array{Int,3})::Matrix{Int}
    @assert ndims(R) == 3 "R must be a 3D array"
    N, A, P = size(R)
    out = Matrix{Int}(undef, N * A, P)

    idx = 1
    @inbounds for n in 1:N
        for a in 1:A
            for p in 1:P
                out[idx, p] = R[n, a, p]
            end
            idx += 1
        end
    end
    return out
end
# ----------------------------------------------
# 2) For each voter, determine which party they
#    will vote for, given:
#      - R[n,a,p] : rank of party p for voter (n,a)
#      - strategic_voters[n,a] : if true => pick the top-ranked
#        among parties_above_threshold, otherwise pick top choice
#      - preferred_parties[n,a] : the voter’s normal top choice
#      - parties_above_threshold[p] : a BitVector marking viable parties
#
#    Returns an N x A matrix of chosen parties.
# ----------------------------------------------
function compute_strategic_preferred_parties(
    party_rankings::AbstractArray{Int,3},
    strategic_voters::BitMatrix,
    preferred_parties::Matrix{Int},
    parties_above_threshold::BitVector
)::Matrix{Int}
    @assert size(party_rankings, 1) == size(strategic_voters, 1) == size(preferred_parties, 1)
    @assert size(party_rankings, 2) == size(strategic_voters, 2) == size(preferred_parties, 2)
    N, A, P = size(party_rankings)

    # Allocate an N x A matrix to hold each voter's final chosen party
    strategic_preferred_parties = similar(preferred_parties)

    @inbounds for n in 1:N
        for a in 1:A
            if strategic_voters[n, a]
                # This voter is strategic: pick the "best" among parties that are above threshold
                best_party = 0
                best_rank = typemax(Int)  # large placeholder
                @inbounds for p in 1:P
                    if parties_above_threshold[p]
                        rnk = party_rankings[n, a, p]
                        if rnk < best_rank
                            best_rank = rnk
                            best_party = p
                        end
                    end
                end
                strategic_preferred_parties[n, a] = best_party
            else
                # Non-strategic: keep their normal #1 choice
                strategic_preferred_parties[n, a] = preferred_parties[n, a]
            end
        end
    end

    return strategic_preferred_parties
end

# ----------------------------------------------
# 3) Count how many voters (among those who turn out)
#    prefer each party. 
#    - chosen_parties[n,a] : the final chosen party for each voter
#    - turnout_voters[n,a] : whether the voter shows up
#    - P : total number of parties
# ----------------------------------------------
function count_party_support(
    chosen_parties::Matrix{Int},
    turnout_voters::BitMatrix,
    n_parties::Int
)::Vector{Int}
    @assert size(chosen_parties) == size(turnout_voters)
    N, A = size(chosen_parties)
    party_support = zeros(Int, n_parties)

    @inbounds for n in 1:N
        for a in 1:A
            if turnout_voters[n, a]
                p_chosen = chosen_parties[n, a]
                party_support[p_chosen] += 1
            end
        end
    end

    return party_support
end

function tally_top_2(election_proportions::Vector{Dict{Int,Float64}}, n_seats::Int
)

    top_two_candidates = Vector{Vector{Int}}(undef, n_seats)
    for seat in 1:n_seats

        # Initialize trackers for the top two values and their associated keys
        top_value = -Inf
        second_value = -Inf
        top_keys = Int[]
        second_keys = Int[]

        # Iterate over dictionary to determine top two values and their keys
        for (candidate, proportion) in election_proportions[seat]
            if proportion > top_value
                # New top value found; demote current top to second
                second_value, second_keys = top_value, top_keys
                top_value, top_keys = proportion, [candidate]
            elseif proportion == top_value
                # Tie for top value
                push!(top_keys, candidate)
            elseif proportion > second_value
                # New second value found
                second_value, second_keys = proportion, [candidate]
            elseif proportion == second_value
                # Tie for second value
                push!(second_keys, candidate)
            end
        end

        # Handle tie-breaking scenarios
        if length(top_keys) > 1
            # Randomly select two from tied top keys
            top_two_candidates[seat] = rand(top_keys, 2)
        elseif length(second_keys) > 1
            # One from top, one randomly from tied second keys
            top_two_candidates[seat] = [top_keys[1], rand(second_keys)]
        else
            # One from top, one from second
            top_two_candidates[seat] = [top_keys[1], second_keys[1]]
        end
    end

    return top_two_candidates

end

@inline function tally_top_1(election_proportions::Vector{Dict{Int,Float64}}, n_seats::Int)

    top_1_candidates = Vector{Int}(undef, n_seats)

    for seat in 1:n_seats

        results = election_proportions[seat]
        # Find the maximum value in the dictionary
        # Find the maximum value in the dictionary
        max_val = maximum(values(results))
        # Collect all keys corresponding to the maximum value
        max_keys = [k for (k, v) in results if v == max_val]
        # Randomly select one key if there are ties
        top_1_candidates[seat] = rand(max_keys)

    end

    return top_1_candidates
end

function expected_policy_utility(
    c::Int,
    ptt::Vector{Float64},
    U1_n::Vector{Float64},
    U2_n::Matrix{Float64}
)
    val = ptt[c] * U1_n[c]
    @inbounds for j in eachindex(ptt)
        if j != c
            val += ptt[j] * U2_n[c, j]
        end
    end
    return val
end


function run_strategic_exit_and_top2_for_district(
    n::Int,
    rankings_n::Matrix{Int},
    U1_n::Vector{Float64},
    U2_n::Matrix{Float64},
    rng::AbstractRNG;
    sample_size::Int=400,
    Nsamples_dirichlet::Int=100,
    max_rounds::Int=3
)
    numCands = size(U2_n, 1)
    is_active = trues(numCands)

    run_strategic_exit_for_district!(
        n,
        rankings_n,
        U1_n,
        U2_n,
        is_active,
        rng;
        sample_size=sample_size,
        Nsamples_dirichlet=Nsamples_dirichlet,
        max_rounds=max_rounds
    )

    # Then pick final top2
    top2_locals = final_top_two(rankings_n, is_active)
    return top2_locals, is_active
end


"""
    run_entire_sim_for_issue(
        pop_issues_ideal_i::Array{Float64,3},
        pop_issues_positions_i::Array{Float64,3},
        candidate_list_by_district::Vector{Vector{Int}},
        dims_issue::Int,
        Q::Int,
        rng::AbstractRNG
    ) -> Vector{Vector{Int}}

For each district n in 1..N:
  1) Build (U1[n], U2[n]) for that district's candidates.
  2) Build voter_rankings[n].
  3) run_strategic_exit_for_district(n, ...).
Return a vector of length N, 
where each entry is the top2 local indices for that district.
"""
function run_entire_sim_for_issue(
    pop_issues_ideal_i::Array{Float64,3},
    pop_issues_positions_i::Array{Float64,3},
    candidate_list_by_district::Vector{Vector{Int}},
    dims_issue::Int,
    Q::Int,
    rng::AbstractRNG=MersenneTwister(42);
    sample_size::Int=400,
    Nsamples_dirichlet::Int=100
)
    N = size(pop_issues_ideal_i, 1)  # number of districts
    U1 = Vector{Vector{Float64}}(undef, N)
    U2 = Vector{Matrix{Float64}}(undef, N)
    district_rankings = Vector{Matrix{Int}}(undef, N)

    for n in 1:N
        # Build the candidate utilities
        U1[n], U2[n] = build_candidate_utilities_for_issue(
            pop_issues_ideal_i,
            pop_issues_positions_i,
            candidate_list_by_district,
            n,
            dims_issue,
            Q
        )
        # Build the voter rankings
        district_rankings[n] = build_voter_rankings_for_district(
            pop_issues_positions_i,
            candidate_list_by_district,
            n
        )
    end

    # Now run strategic exit for each district
    top2_local_indices = Vector{Vector{Int}}(undef, N)
    for n in 1:N
        top2_local_indices[n] = run_strategic_exit_for_district(
            n,
            district_rankings,
            U1,
            U2,
            rng;
            sample_size=sample_size,
            Nsamples_dirichlet=Nsamples_dirichlet
        )
    end

    return top2_local_indices
end



end

module StrategicExit

using Distributions
using LinearAlgebra
using Random
using StaticArrays
using LinearAlgebra: norm
using DataStructures

using ..ElectionSimulation: poll_district, expected_policy_utility
using ..HelpfulFunctions: scale_utilities
using ..CandidatesSimulation: build_candidate_utilities_multi_issue

"""
    build_candidate_ideal_points_for_issue(
       all_voters_ideal_points_k::Array{Float64,3},  # shape (N, A, d_k)
       candidate_list_n::Vector{Int},               # e.g. [5,10,12]
       n::Int
    ) -> Matrix{Float64}

Returns a (C_n x d_k) matrix, where row c_local 
corresponds to agent = candidate_list_n[c_local], i.e. 
all_voters_ideal_points_k[n, a_global, :].
"""
function build_candidate_ideal_points_for_issue(
    all_voters_ideal_points_k::Array{Float64,3},
    candidate_list_n::Vector{Int},
    n::Int
)
    C_n = length(candidate_list_n)
    d_k = size(all_voters_ideal_points_k, 3)
    rep_mat = Matrix{Float64}(undef, C_n, d_k)
    @inbounds for c_local in 1:C_n
        a_global = candidate_list_n[c_local]
        @views rep_mat[c_local, :] = all_voters_ideal_points_k[n, a_global, :]
    end
    return rep_mat
end

"""
    compute_voter_utilities_for_one_district(
       n::Int, 
       v::Int,
       candidate_list_n::Vector{Int},
       all_voters_ideal_points::Vector{Array{Float64,3}},
       voter_issue_weights::Array{Float64,3}
    ) -> Vector{Float64}

Given that:
 - we have K = length(all_voters_ideal_points) issues
 - each all_voters_ideal_points[k] is NxAxd_k
 - voter_issue_weights is shape (K, N, A)
 - candidate_list_n is length C_n
Compute the utility for voter (n,v) for each candidate in candidate_list_n, 
using "sum of w_k * Eucl. distances across issues => then negative squared."

Returns a Vector{Float64} of length C_n with the final utilities.
"""
function compute_voter_utilities_for_one_district(
    n::Int,
    v::Int,
    candidate_list_n::Vector{Int},
    all_voters_ideal_points::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3}
)
    C_n = length(candidate_list_n)
    K = length(all_voters_ideal_points)

    # We'll do "distance_sums[c_local]" then convert to utility
    distance_sums = zeros(C_n)

    # Loop issues
    for k in 1:K
        # weight
        w_k = voter_issue_weights[k, n, v]
        # voter ideal point in R^{d_k}
        voter_vec = @view all_voters_ideal_points[k][n, v, :]

        # for each candidate c_local
        for c_local in 1:C_n
            a_c = candidate_list_n[c_local]
            candidate_vec = @view all_voters_ideal_points[k][n, a_c, :]

            # Eucl distance in d_k dimension
            ssd = 0.0
            @inbounds for dd in eachindex(candidate_vec)
                diff = voter_vec[dd] - candidate_vec[dd]
                ssd += diff * diff
            end
            distance_sums[c_local] += w_k * sqrt(ssd)
        end
    end

    # Convert distances => utilities
    utilities = similar(distance_sums)
    @inbounds for c_local in 1:C_n
        utilities[c_local] = -distance_sums[c_local]^2
    end

    return scale_utilities(utilities)
end

"""
    build_voter_rankings_for_district(
       n::Int,
       all_voters_ideal_points::Vector{Array{Float64,3}}, 
       voter_issue_weights::Array{Float64,3}, 
       candidate_list_by_district::Vector{Vector{Int}}
    ) -> Matrix{Int}

Construct an (A x C_n) ranking matrix for district n, 
where row v is the preference order of local candidates in candidate_list_by_district[n].
We assume A is the # of agents in district n (1..A) are all voters. 
We do "Specification One" distance-based utility.

We return a matrix `rankings_n` of size (A x C_n).
"""
function build_voter_rankings_for_district(
    n::Int,
    all_voters_ideal_points::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    candidate_list_by_district::Vector{Vector{Int}}
)
    # gather the candidate list for district n
    candidate_list_n = candidate_list_by_district[n]
    C_n = length(candidate_list_n)

    # We assume the shape of all_voters_ideal_points[1] is (N, A, d_1),
    # so the second dimension is the # of agents A. We'll take that from the data
    A = size(all_voters_ideal_points[1], 2)  # # of agents in district n

    # We'll produce an (A x C_n) matrix of local candidate indices in descending utility order
    rankings_n = Matrix{Int}(undef, A, C_n)

    # We'll do a workspace for (utility, cand_local) pairs
    utility_pairs = Vector{Tuple{Float64,Int}}(undef, C_n)

    for v in 1:A
        # compute utilities for voter (n,v)
        utilities_v = compute_voter_utilities_for_one_district(
            n, v, candidate_list_n, all_voters_ideal_points, voter_issue_weights
        )

        # we want to sort in descending order => candidate with highest utility first
        for c_local in 1:C_n
            utility_pairs[c_local] = (utilities_v[c_local], c_local)
        end
        sort!(utility_pairs, by=x -> x[1], rev=true)

        # fill rankings_n[v, :]
        for j in 1:C_n
            # the j-th best candidate in local index space
            c_local_best = utility_pairs[j][2]
            rankings_n[v, j] = c_local_best
        end
    end

    return rankings_n
end

function build_voter_rankings_for_all_districts(
    N::Int,
    all_voters_ideal_points::AbstractVector{Array{Float64,3}},
    voter_issue_weights::Array{Float64,3},
    candidate_list_by_district::Vector{Vector{Int}}
)
    district_rankings = Vector{Matrix{Int}}(undef, N)
    for n in 1:N
        district_rankings[n] = build_voter_rankings_for_district(
            n,
            all_voters_ideal_points,
            voter_issue_weights,
            candidate_list_by_district
        )
    end
    return district_rankings
end

function prob_top_two_dirichlet(counts::Vector{Int}, rng::AbstractRNG, Nsamples::Int=100)
    numCands = length(counts)
    alpha = [1.0 + c for c in counts]
    dpost = Dirichlet(alpha)
    top_two_counts = zeros(Int, numCands)

    @inbounds for s in 1:Nsamples
        p = rand(rng, dpost)  # a Vector{Float64}
        idxs = sortperm(p, rev=true)
        if numCands >= 1
            top_two_counts[idxs[1]] += 1
        end
        if numCands >= 2
            top_two_counts[idxs[2]] += 1
        end
    end

    ptt = similar(counts, Float64)
    for c in 1:numCands
        ptt[c] = top_two_counts[c] / Nsamples
    end
    return ptt
end

"""
    run_strategic_exit_for_district!(
        n::Int,
        rankings_n::Matrix{Int},
        U1_n::Vector{Float64},
        U2_n::Matrix{Float64},
        is_active::Vector{Bool},
        rng::AbstractRNG;
        sample_size::Int=400,
        Nsamples_dirichlet::Int=100,
        max_rounds::Int=3
    ) -> nothing

Modifies `is_active` in place. 
After up to `max_rounds` (3) rounds, some candidates remain active, 
others dropped out.
Each round:
  - poll 400 voters (store them, so if a cand drops, we re-poll for redistribution)
  - top3 remain safe
  - from lowest poll to highest poll, each non-top3 decides dropping or not, 
    with stay vs drop utility. 
  - up to 3 dropouts in that round. 
Stop if <=3 remain.

We do *not* return top2 here; we just end with an updated `is_active`.
"""
function run_strategic_exit_for_district!(
    n::Int,
    rankings_n::Matrix{Int},
    U1_n::Vector{Float64},
    U2_n::Matrix{Float64},
    is_active::Vector{Bool},
    rng::AbstractRNG;
    sample_size::Int=400,
    Nsamples_dirichlet::Int=100,
    max_rounds::Int=3
)
    numCands = length(is_active)
    counts = zeros(Int, numCands)
    counts_drop = similar(counts)

    for round_i in 1:max_rounds
        active_cands = findall(is_active)
        if length(active_cands) <= 3
            return
        end

        # (1) poll 400 voters
        poll_district!(counts, rankings_n, is_active, rng; sample_size=sample_size)

        # (2) find top2 probs
        ptt = prob_top_two_dirichlet(counts, rng, Nsamples_dirichlet)

        # (3) identify top3 by raw counts
        sorted_active_desc = sort(active_cands, by=c -> counts[c], rev=true)
        top3 = sorted_active_desc[1:3]

        # gather the "non-top3" in ascending poll
        non_top3 = setdiff(active_cands, top3)
        non_top3_sorted_asc = sort(non_top3, by=c -> counts[c], rev=false)

        dropouts_this_round = 0

        # (4) each non-top3 candidate tries to drop
        for c_local in non_top3_sorted_asc
            if dropouts_this_round >= 3
                break
            end
            if length(findall(is_active)) <= 3
                return
            end

            stay_utility = expected_policy_utility(c_local, ptt, U1_n, U2_n)

            # For "drop," we remove c_local from is_active, re-poll to redistribute
            is_active[c_local] = false
            poll_district!(counts_drop, rankings_n, is_active, rng; sample_size=sample_size)
            ptt_drop = prob_top_two_dirichlet(counts_drop, rng, Nsamples_dirichlet)

            drop_utility = 0.0
            @inbounds for j in 1:numCands
                if is_active[j]
                    drop_utility += ptt_drop[j] * U2_n[c_local, j]
                end
            end

            if drop_utility > stay_utility
                # remain dropped
                dropouts_this_round += 1
            else
                # revert
                is_active[c_local] = true
            end
        end
    end
end

"""
    final_top_two(
        rankings_n::Matrix{Int},
        is_active::Vector{Bool}
    ) -> Vector{Int}

Once we finish the 3 rounds, we do a "full-district" run-off:
Map *all* voters (no sampling) to their top active choice, 
tally raw counts, pick top2.

Returns the *local indices* of the top2.
"""
function final_top_two(
    rankings_n::Matrix{Int},
    is_active::Vector{Bool}
)
    A, Cn = size(rankings_n)
    counts = zeros(Int, Cn)
    @inbounds for v in 1:A
        for r in 1:Cn
            c_local = rankings_n[v, r]
            if is_active[c_local]
                counts[c_local] += 1
                break
            end
        end
    end
    active_cands = findall(is_active)
    if length(active_cands) <= 2
        return active_cands
    else
        sorted_active = sort(active_cands, by=c -> counts[c], rev=true)
        return sorted_active[1:2]
    end
end

function example_script()

    rankings = build_voter_rankings_for_all_districts(50, voter_ideal_points, voter_issue_weights,
        candidates
    )

    seat = 7
    # for a given district
    U1_seat, U2_seat = CandidatesSimulation.build_candidate_utilities_multi_issue(
        voter_ideal_points, voter_question_positions, candidates, seat, issue_dimensions,
        n_questions
    )

    is_active = [true for _ in 1:size(rankings)[2]]
    run_strategic_exit_for_district!(seat, rankings[seat], U1_seat, U2_seat, is_active, rng)

    top_2_locals = final_top_two(rankings[seat], is_active)

    # we can then index candidates[seat] by the top_2_locals the get the runoff candidates,
    # and run a simple majoritarian election off that

end

end