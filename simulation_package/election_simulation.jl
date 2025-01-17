module ElectionSimulation

using Distributions
using LinearAlgebra
using Random
using StaticArrays
using LinearAlgebra: norm
using DataStructures

"""
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
    voter_ideals::Vector{Vector{Float64}},
    parties::Vector{Matrix{Float64}},
    n_parties::Int,
    n_issues::Int
)
    # We'll accumulate total distances in a local vector
    zeros(Float64, n_parties)
    dist_sums = Vector{Float64}(undef, n_parties)
    fill!(dist_sums, 0.0)   # set all party sums to zero
    @inbounds for i in 1:n_issues
        v_i = voter_ideals[i]     # d_i-vector (the voter)
        d_i = length(v_i)
        w_i = norm(v_i) / sqrt(d_i)
        party_mat = parties[i]    # P x d_i
        # for each party p, compute the Euclidean distance in dimension i
        @inbounds for p in 1:n_parties
            # distance on issue i: sum of squared differences in each dimension, then sqrt
            ssd = 0.0
            @inbounds for dim in 1:d_i
                diff = v_i[dim] - party_mat[p, dim]
                ssd += diff * diff
            end
            dist_sums[p] += w_i * sqrt(ssd)
        end
    end

    utilities = -1 .* (dist_sums .^ 2)
    # now find the party with the min distance sum
    _, idx = findmax(utilities)
    return idx, utilities
end
"""
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
    agents_ideals::AbstractVector{Array{Float64,3}},
    parties::AbstractVector{Matrix{Float64}},
    n_parties::Int,
    n_issues::Int,
    n_seats::Int,
    pop_per_seat::Int
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
    @inbounds for seat in 1:n_seats
        @inbounds for voter in 1:pop_per_seat
            # Step 1: gather the voter's ideal for each issue i
            @inbounds for issue in 1:n_issues
                d_i = size(parties[issue], 2)  # the relevant dimension for this issue
                # We'll just take a *view* into agents_ideals[i][n, a, 1:d_i]
                # For performance, we can do something like:
                voter_ideals[issue] = @views agents_ideals[issue][seat, voter, 1:d_i]
            end
            # Step 2: find the closest party
            party_idx, utilities = find_closest_representative_for_one_voter(voter_ideals, parties,
                n_parties, n_issues)
            # Step 3: store
            result[seat, voter] = party_idx
            voter_utilities[seat, voter, :] = utilities
        end
    end
    return result, voter_utilities
end

"""

mememememememe 

"""
function assign_voters_to_candidates!(
    agents_ideals::AbstractVector{Array{Float64,3}},
    candidate_points::AbstractVector{Matrix{Float64}},
    n_candidates::Int,
    n_issues::Int,
    pop_per_seat::Int,
    issue_dimensions::AbstractVector{Int},
    seat::Int
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
            voter_ideals[issue] = @views agents_ideals[issue][seat, voter, 1:d_i]
        end
        # Step 2: find the closest party
        party_idx, utilities = find_closest_representative_for_one_voter(
            voter_ideals, candidate_points, n_candidates, n_issues
        )
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

function run_single_round_election(ideal_points::AbstractVector{Array{Float64,3}},
    candidates::Vector{Vector{Int}}, n_seats::Int, n_candidates::Int, n_issues::Int,
    pop_per_seat::Int, issue_dimensions::AbstractVector{Int})

    election_proportions = Vector{Dict{Int,Float64}}(undef, n_seats)
    voter_utilites = Array{Float64,3}(undef, n_seats, pop_per_seat, n_candidates)
    candidate_choices = Matrix{Int}(undef, n_seats, pop_per_seat)

    @inbounds for seat in 1:n_seats
        candidate_points, candidate_map = find_candidate_ideal_points(
            ideal_points, candidates[seat], n_candidates, seat, n_issues
        )
        results, utilities = assign_voters_to_candidates!(
            ideal_points, candidate_points, n_candidates, n_issues, pop_per_seat,
            issue_dimensions, seat
        )

        candidate_choices[seat, :] = results
        proportions = Dict(idx => ct / pop_per_seat for (idx, ct) in counter(results))

        election_proportions[seat] = map_global_indx_to_props(candidate_map, proportions)
        voter_utilites[seat, :, :] = utilities

    end

    return election_proportions, voter_utilites, candidate_choices

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



"""
    build_voter_rankings_dot!(
        rankings_n::Matrix{Int},
        ideal_points::Array{Float64,3},  # size: (N, A, d_i)
        candidate_list::Vector{Int},
        n::Int
    )

Fills `rankings_n` (size A x #cands) with local-candidate indices 
sorted by the dot product between voter v's ideal point 
and each candidate's ideal point. 
 - n = district index
 - candidate_list = candidate_list_by_district[n], e.g. [5,10,12]
 - local index i_local => candidate_list[i_local] = a_global
"""
function build_voter_rankings_dot!(
    rankings_n::Matrix{Int},
    ideal_points::Array{Float64,3},
    candidate_list::Vector{Int},
    n::Int
)
    A = size(ideal_points, 2)
    numCands = length(candidate_list)
    d_i = size(ideal_points, 3)

    # We do: for each voter v in 1..A:
    #   For each candidate c_local => a_global = candidate_list[c_local]:
    #       score = dot( ideal_points[n, v, :], ideal_points[n, a_global, :] )
    #   Sort c_local in descending order of score.
    # Place them in rankings_n[v, :].
    temp_scores = Vector{Tuple{Float64,Int}}(undef, numCands)
    @inbounds for v in 1:A
        # compute dot product for each candidate
        for (i_local, a_cand) in enumerate(candidate_list)
            sc = 0.0
            @inbounds @simd for dd in 1:d_i # we can accept small errors here
                sc += ideal_points[n, v, dd] * ideal_points[n, a_cand, dd]
            end
            temp_scores[i_local] = (sc, i_local)
        end
        # sort by sc descending
        sort!(temp_scores, by=x -> x[1], rev=true)
        # fill rankings
        for (rank_idx, (_, c_local)) in pairs(temp_scores)
            rankings_n[v, rank_idx] = c_local
        end
    end
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
    is_active::Vector{Bool},
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


using Distributions

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