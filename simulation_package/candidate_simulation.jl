module CandidateSimulation

using Random, LinearAlgebra


"""
    compute_engagement(pop_issues::Vector{Array{Float64,3}}, alpha::Float64, p::Float64;
                       rng=MersenneTwister(42))

Given a vector of 3D arrays, `pop_issues`, where each array is of size 
`(N, A, d_I)` for one issue I of dimensionality `d_I`, compute two 
things for each (district, agent):

1. `engagement[n,a]`, which is

   Engagement(v) = ( sum_{I in Issues} ( norm(v_I) / sqrt(d_I) )^p )^(1/p)

2. `is_political_class[n,a]` ~ Bernoulli( alpha * engagement[v] )

   i.e. whether each voter is in the "political class."  

Returns
-------
- `engagement::Matrix{Float64}` of size `(N, A)`
- `is_political_class::BitMatrix` of size `(N, A)`

The Bernoulli draw uses the provided `rng` (defaulting to MersenneTwister(42)).
Any probability above 1.0 is capped at 1.0.
"""
function compute_engagement(pop_issues::AbstractVector{Array{Float64,3}},
    alpha::Float64,
    p::Float64;
    rng=MersenneTwister(42))

    # Number of issues
    M = length(pop_issues)

    # All issues have the same number of districts N,
    # the same number of agents A, but possibly different d_I
    (N, A, _) = size(pop_issues[1])

    # Store engagement for each (district n, agent a)
    engagement = zeros(Float64, N, A)

    # 1) Compute \ell_p norm across all issues
    @inbounds for n in 1:N
        @inbounds for a in 1:A
            sum_ = 0.0
            @inbounds for i in 1:M
                arr_i = pop_issues[i]
                d_i = size(arr_i, 3)
                # magnitude of the agent's vector on issue i
                mag = norm(view(arr_i, n, a, :))

                # ( mag / sqrt(d_i) )^p
                sum_ += (mag / sqrt(d_i))^p
            end
            # ( sum_ )^(1/p)
            E_na = sum_^(1 / p)
            engagement[n, a] = E_na
        end
    end

    # 2) Bernoulli( alpha * engagement ) for Political Class
    is_political_class = falses(N, A)
    @inbounds for n in 1:N
        @inbounds for a in 1:A
            prob = alpha * engagement[n, a]
            # Cap the probability at 1.0
            prob = prob > 1.0 ? 1.0 : prob
            if rand(rng) < prob
                is_political_class[n, a] = true
            end
        end
    end

    return engagement, is_political_class
end

"""
    entry_candidates(pop_issues::Vector{Array{Float64,3}}, 
                     alpha::Float64, 
                     p::Float64; 
                     rng=MersenneTwister(42))

Implements the mechanism of candidate entry.  For each (district n, agent a),
the probability of running is Bernoulli(alpha * Engagement(v)).

Returns
-------
- `candidate_list_by_district`: Vector{Vector{Int}} of length N
   with the indices of the agents (1..A) who decided to run in each district.
   If a district has more than 10 who decide, randomly drop until 10 remain.
"""
function entry_candidates(pop_issues::Vector{Array{Float64,3}},
    alpha::Float64,
    p::Float64;
    rng=MersenneTwister(42))

    # 1) Compute engagement
    engagement, _ = compute_engagement(pop_issues, alpha, p; rng=rng)

    (N, A) = size(engagement)

    candidate_list_by_district = Vector{Vector{Int}}(undef, N)

    for n in 1:N
        # collect the set of agents in district n who run
        running_in_district = Int[]
        for a in 1:A
            prob = alpha * engagement[n, a]
            prob = prob > 1.0 ? 1.0 : prob
            if rand(rng) < prob
                push!(running_in_district, a)
            end
        end

        # if length(running_in_district) > 10, drop randomly
        if length(running_in_district) > 10
            shuffle!(running_in_district, rng)
            while length(running_in_district) > 10
                pop!(running_in_district)   # drop last until we have 10
            end
        end

        candidate_list_by_district[n] = running_in_district
    end

    return candidate_list_by_district
end

function build_global_local_maps(
    candidate_list_by_district::Vector{Vector{Int}},
    N::Int
)
    global2local = Vector{Dict{Int,Int}}(undef, N)
    local2global = Vector{Vector{Int}}(undef, N)

    for d in 1:N
        cand_ids = candidate_list_by_district[d]  # e.g. [5, 10, 12]
        numCands_d = length(cand_ids)
        map_d = Dict{Int,Int}()
        for (i, global_id) in enumerate(cand_ids)
            map_d[global_id] = i
        end
        global2local[d] = map_d
        local2global[d] = cand_ids
    end

    return global2local, local2global
end

function norm_of_ideal_point(
    pop_issues_ideal::Array{Float64,3},  # e.g. N x A x d_i
    n::Int,
    a::Int
)
    # returns Euclidian norm of agent a's ideal point in district n
    # This is the vector in R^{d_i}
    v = @view pop_issues_ideal[n, a, :]
    return LinearAlgebra.norm(v)
end

"""
    build_candidate_utilities_for_issue(
        pop_issues_ideal_i::Array{Float64,3},
        pop_issues_positions_i::Array{Float64,3},
        candidate_list_by_district::Vector{Vector{Int}},
        n::Int,
        dims_issue::Int,
        Q::Int
    ) -> (U1_n, U2_n)

Builds U1 and U2 for district n on a single issue i. 
 - `pop_issues_ideal_i` is size (N, A, d_i).
 - `pop_issues_positions_i` is size (N, A, Q).
 - `candidate_list_by_district[n]` is e.g. [5,10,12].
 - `dims_issue` = d_i
 - `Q` = number of questions in this issue.

We return two arrays: 
 - U1_n of size #cands
 - U2_n of size #cands x #cands
"""
function build_candidate_utilities_for_issue(
    pop_issues_ideal_i::Array{Float64,3},
    pop_issues_positions_i::Array{Float64,3},
    candidate_list_by_district::Vector{Vector{Int}},
    n::Int,
    dims_issue::Int,
    Q::Int
)
    cands = candidate_list_by_district[n]
    numCands = length(cands)

    # Precompute M_c = (norm_of_ideal_point / sqrt(dims_issue)) for each candidate
    M = zeros(Float64, numCands)
    for (i_local, a_global) in enumerate(cands)
        mag = norm_of_ideal_point(pop_issues_ideal_i, n, a_global)
        M[i_local] = mag / sqrt(dims_issue)
    end

    # Build arrays for same_count, diff_sum
    same_count = zeros(Int, numCands, numCands)
    diff_sum = zeros(Float64, numCands, numCands)

    # For each candidate c, extract question positions: 
    # pos_c[q] = pop_issues_positions_i[n, a_global, q]
    # Then compare c vs j
    for c_local in 1:numCands
        a_c = cands[c_local]
        for j_local in c_local+1:numCands

            a_j = cands[j_local]

            s_cnt = 0
            d_sum = 0.0

            for q in 1:Q

                c_q = pop_issues_positions_i[n, a_c, q]
                j_q = pop_issues_positions_i[n, a_j, q]

                if c_q == j_q
                    s_cnt += 1
                else
                    d_sum += abs(c_q - j_q)
                end
            end
            same_count[c_local, j_local] = s_cnt
            same_count[j_local, c_local] = s_cnt
            diff_sum[c_local, j_local] = d_sum
            diff_sum[j_local, c_local] = d_sum
        end
    end

    # Now build U1, U2
    U1_n = zeros(Float64, numCands)
    U2_n = zeros(Float64, numCands, numCands)

    # U1(c) = ( M[c]* Q ) * sqrt(1 + Q )
    for c_local in 1:numCands
        U1_n[c_local] = (M[c_local] * Q) * sqrt(1 + Q)
    end

    # U2(c,j) = M[c]* same_count[c,j]*sqrt(1 + same_count[c,j]) - M[c]* diff_sum[c,j]
    for c_local in 1:numCands
        for j_local in 1:numCands
            if c_local == j_local
                U2_n[c_local, j_local] = 0.0
            else
                sc = same_count[c_local, j_local]
                begin
                    U2_n[c_local, j_local] =
                        M[c_local] * (sc * sqrt(1 + sc)) - M[c_local] * diff_sum[c_local, j_local]
                end
            end
        end
    end

    return U1_n, U2_n
end

end