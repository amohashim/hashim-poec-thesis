
module SimulateIssuePreferences

using Test
using Random
using Distributions
using StaticArrays
using Statistics

using ..HelpfulFunctions

export sample_theta, generate_theta, generate_ideal_points, generate_scaled_ideal_points

"""
    map_salience_to_interval(salience_level::Int) => (Float64, Float64)

Maps an integer salience level in {0,1,2,3} to the interval:
    0 => (0.0, 0.0)
    1 => (0.0, 2.0)
    2 => (1.0, 3.0)
    3 => (2.0, 4.0)
"""
@inline function map_salience_to_interval(salience_level::Int)
    if salience_level == 0
        return (0.0, 0.0)
    elseif salience_level == 1
        return (0.0, 20.0)
    elseif salience_level == 2
        return (10.0, 30.0)
    else
        return (20.0, 40.0)
    end
end

"""
    sample_theta(salience_level, rng) => Float64

Samples from:
  - 0.0, if salience_level == 0
  - Bernoulli{±1}(0.5) * Beta_{[a,b]}(7,7), otherwise,
    where [a,b] = map_salience_to_interval(salience_level).
"""
@inline function sample_theta(salience_level::Int, rng::AbstractRNG)
    if salience_level == 0
        return 0.0
    end
    # random ±1
    sign_ = ifelse(rand(rng) < 0.5, +1.0, -1.0)
    # Beta(7,7) distributed
    val = rand(rng, Beta(7.0, 7.0))
    # scale
    (a, b) = map_salience_to_interval(salience_level)
    scaled_val = a + (b - a) * val
    return sign_ * scaled_val
end

"""
    generate_theta(Delta_array::Vector{Int}, J::Int, K::Int, d::Int, rng::AbstractRNG)
        => Matrix{Float64} of size (J*K, d)

Given J demographic characteristics, each with cleavage-salience in {0,1,2,3},
and each characteristic having K groups, produce a coefficient matrix of size
(J*K) x d. The row ( (j-1)*K + g ) for g=1..K, j=1..J corresponds to group g of
characteristic j, and each column k=1..d is the dimension in R^d.

If Delta_array[j] == 0, all groups in characteristic j get a 0.0 coefficient.
Otherwise, each group's coefficient is ±Beta_{[a,b]}(7,7).

** For a given issue **
"""
function generate_theta(Delta_array::AbstractVector{Int}, J::Int, K::Int, d::Int, rng::AbstractRNG)
    Theta = Matrix{Float64}(undef, J * K, d)

    # Fill row by row
    @inbounds for j in 1:J
        sal_level = Delta_array[j]  # e.g. 0..3
        for g in 1:K
            row_idx = (j - 1) * K + g
            for kdim in 1:d
                Theta[row_idx, kdim] = sample_theta(sal_level, rng)
            end
        end
    end
    return Theta
end

"""
    generate_points_masked!(
        agents::Array{Int,4},
        Theta::Matrix{Float64},
        sigma_low::Float64,
        sigma_mod::Float64,
        sigma_high::Float64,
        d::Int;
        rng::AbstractRNG = MersenneTwister(0)
    ) -> Array{Float64,3}

Optimized version of `generate_points` that uses masked assignments
and single-pass summation with minimal branching.

Inputs:
  - `agents` : 4D array of size (N, num_agents, J, 3)
     * `agents[n,a,j,1]` => group index in [1..K]
     * `agents[n,a,j,2]` => salience level in [0..3]
     * `agents[n,a,j,3]` => salience indicator in {0,1}
  - `Theta` : (J*K, d) matrix of demographic coefficients
  - `sigma_*` : noise scales
  - `d` : dimension of the output space
  - `rng` : random number generator

Returns:
  - `points` : (N, num_agents, d) array of the final coordinates.

Key optimizations:
  1. We fill random noise in a **vectorized** manner using masks for each salience level.
  2. We gather all valid positions (where `sindicator==1`) in one pass, eliminating repeated `if/else` checks.
  3. No function definitions are nested inside.

This function handles **one issue** (one set of cleavage-salience values).
"""
function generate_points_masked!(
    agents::Array{Int,4},
    Theta::AbstractMatrix{Float64},
    sigma_none::Float64,
    sigma_low::Float64,
    sigma_mod::Float64,
    sigma_high::Float64,
    d::Int,
    rng::AbstractRNG
)
    @assert ndims(agents) == 4 "agents must be 4D: (N, num_agents, J, 3)"
    @assert size(Theta, 2) == d "Theta must have 'd' columns"

    # Dimensions
    N, A, J, _ = size(agents)
    K_ = size(Theta, 1) ÷ J  # K groups per characteristic

    # Output array
    out = zeros(Float64, N, A, d)

    # Precompute distributions
    dist_none = Normal(0, sigma_none)
    dist_low = Normal(0, sigma_low)
    dist_mod = Normal(0, sigma_mod)
    dist_high = Normal(0, sigma_high)

    # Loop over characteristics
    @inbounds for j in 1:J
        row_offset = (j - 1) * K_

        # 2D slices (views) for group, salience, indicator
        group_slice = @view agents[:, :, j, 1]
        slevel_slice = @view agents[:, :, j, 2]
        sindicator_slc = @view agents[:, :, j, 3]

        # row_idx_mat[n,a] = row_offset + group_slice[n,a]
        row_idx_mat = row_offset .+ group_slice

        # We'll store random noise in a (N x A) matrix
        noise_mat = zeros(Float64, N, A)

        # == 1 => dist_low
        mask1 = (slevel_slice .== 1) .& (sindicator_slc .== 1)
        if any(mask1)
            noise_mat[mask1] .= rand(rng, dist_low, count(mask1))
        end

        # == 2 => dist_mod
        mask2 = (slevel_slice .== 2) .& (sindicator_slc .== 1)
        if any(mask2)
            noise_mat[mask2] .= rand(rng, dist_mod, count(mask2))
        end

        # == 3 => dist_high
        mask3 = (slevel_slice .== 3) .& (sindicator_slc .== 1)
        if any(mask3)
            noise_mat[mask3] .= rand(rng, dist_high, count(mask3))
        end

        mask4 = (slevel_slice .== 0) .& (sindicator_slc .== 0)
        if any(mask4)
            noise_mat[mask4] .= rand(rng, dist_none, count(mask4))
        end

        # gather all valid positions in a single pass
        valid_positions = findall(x -> x == 1 || x == 0, sindicator_slc) # legacy
        # each element of valid_positions is a CartesianIndex (n,a)

        @inbounds for pos in valid_positions
            n, a = pos[1], pos[2]  # Unpacks the CartesianIndex to a tuple (n, a)
            row_ = row_idx_mat[n, a]
            noise_ = noise_mat[n, a]

            for kdim in 1:d
                out[n, a, kdim] += (Theta[row_, kdim] * sindicator_slc[n, a]) + noise_
            end
        end

    end

    return out
end

generate_ideal_points = generate_points_masked!

"""
Returns: 
1. Vector of 3D arrays with dimensions N x A x d_I, where d_I is the dimension of the issue I
2. Vector of means of ideal points in each dimension in each issue space
3. Vector of variances of ideal points in each dimension of each issue space

"""
function generate_scaled_ideal_points(demographic_cleavage_salience::AbstractMatrix{Int},
    n_characteristics::Int, n_groups::AbstractVector{Int}, n_issues::Int,
    issue_dimensions::AbstractVector{Int}, voters::Array{Int,4}, σ_none::Float64, σ_low::Float64,
    σ_moderate::Float64, σ_high::Float64, rng::AbstractRNG
)

    ideal_points_per_issue = Vector{Array{Float64,3}}(undef, n_issues)
    max_n_groups = maximum(n_groups)


    for issue in 1:n_issues

        θ = SimulateIssuePreferences.generate_theta(
            demographic_cleavage_salience[:, issue], n_characteristics, max_n_groups,
            issue_dimensions[issue], rng
        )

        ideal_points_per_issue[issue] = SimulateIssuePreferences.generate_ideal_points(
            voters, θ, σ_none, σ_low, σ_moderate, σ_high, issue_dimensions[issue], rng
        )

    end

    results = HelpfulFunctions.z_scale_points.(ideal_points_per_issue)

    ideal_points_scaled, scaled_means, scaled_variances = map(x -> getindex.(results, x), 1:3)

    return ideal_points_scaled, scaled_means, scaled_variances

end


end

module TestPointGeneration

using Test
using Random
using Distributions
using StaticArrays
using Statistics

using ..SimulateIssuePreferences

# A naive version just for cross-checking correctness on a small example
function generate_points_naive(
    agents::Array{Int,4},
    Theta::Matrix{Float64},
    sigma_none::Float64,
    sigma_low::Float64,
    sigma_mod::Float64,
    sigma_high::Float64,
    d::Int;
    rng::AbstractRNG=MersenneTwister(0)
)
    dist_low = Normal(0.0, sigma_low)
    dist_mod = Normal(0.0, sigma_mod)
    dist_high = Normal(0.0, sigma_high)

    N, A, J, _ = size(agents)
    K_ = size(Theta, 1) ÷ J
    out = zeros(Float64, N, A, d)

    @inbounds for n in 1:N
        for a in 1:A
            local_y = fill(0.0, d)
            for j in 1:J
                g_j = agents[n, a, j, 1]
                slevel = agents[n, a, j, 2]
                sind = agents[n, a, j, 3]
                if sind == 1
                    row_ = (j - 1) * K_ + g_j
                    noise_ = 0.0
                    if slevel == 1
                        noise_ = rand(rng, dist_low)
                    elseif slevel == 2
                        noise_ = rand(rng, dist_mod)
                    elseif slevel == 3
                        noise_ = rand(rng, dist_high)
                    end
                    @simd for kdim in 1:d
                        local_y[kdim] += Theta[row_, kdim] + noise_
                    end
                end
            end
            @simd for kdim in 1:d
                out[n, a, kdim] = local_y[kdim]
            end
        end
    end
    return out
end

function run_tests()

    @testset "generate_points_masked! Tests" begin

        @testset "1) Small By-Hand Check" begin
            # We'll do an extremely small scenario:
            # N=1, A=2, J=2, K=2, d=2 => so we can do the math ourselves.

            # We'll define a Theta that we can handle easily by hand:
            # J=2, K=2 => (J*K=4) x d=2
            # Suppose:
            #   Theta(row=1, :) = (1.0, 1.0)
            #   Theta(row=2, :) = (2.0, 2.0)
            #   Theta(row=3, :) = (10.0, -1.0)
            #   Theta(row=4, :) = (0.5, 0.5)
            Theta_small = [
                1.0 1.0;   # row1
                2.0 2.0;   # row2
                10.0 -1.0;  # row3
                0.5 0.5    # row4
            ]

            # We'll define a single node (N=1) with A=2 agents, J=2, K=2 => 4D array
            # shape => (1, 2, 2, 3)
            # agent #1 => belongs to group=1 in char1, slevel=1 => low, indicator=1
            #            belongs to group=2 in char2, slevel=0 => no,  indicator=0
            # agent #2 => belongs to group=2 in char1, slevel=3 => high, indicator=1
            #            belongs to group=1 in char2, slevel=3 => high, indicator=1
            agents_small = Array{Int}(undef, 1, 2, 2, 3)

            # Fill agent #1
            agents_small[1, 1, 1, 1] = 1  # group=1
            agents_small[1, 1, 1, 2] = 1  # slevel=1 => low
            agents_small[1, 1, 1, 3] = 1  # indicator=1
            agents_small[1, 1, 2, 1] = 2
            agents_small[1, 1, 2, 2] = 0
            agents_small[1, 1, 2, 3] = 0

            # Fill agent #2
            agents_small[1, 2, 1, 1] = 2  # char1 => group=2
            agents_small[1, 2, 1, 2] = 3  # slevel=3 => high
            agents_small[1, 2, 1, 3] = 1  # indicator=1
            agents_small[1, 2, 2, 1] = 1  # char2 => group=1
            agents_small[1, 2, 2, 2] = 3  # slevel=3 => high
            agents_small[1, 2, 2, 3] = 1  # indicator=1

            # Let's define sigmas:
            sigma_low = 0.1
            sigma_mod = 0.2
            sigma_high = 0.5
            d = 2

            # We'll do a naive approach with a fixed RNG seed so we know the noise draws
            rng_naive = MersenneTwister(1234)
            out_naive = generate_points_naive(
                agents_small, Theta_small,
                sigma_low, sigma_mod, sigma_high, d; rng=rng_naive
            )

            # We'll do the masked approach with the *same* RNG seed => to match noise draws exactly
            rng_masked = MersenneTwister(1234)
            out_masked = generate_points_masked!(
                agents_small, Theta_small,
                sigma_low, sigma_mod, sigma_high, d; rng=rng_masked
            )

            @test size(out_naive) == (1, 2, 2)
            @test size(out_masked) == (1, 2, 2)

            # They should match exactly because we used identical seeds
            @test isapprox(out_naive, out_masked; atol=1e-14)

            # For a "by-hand" deterministic check ignoring noise:
            #   agent #1 => char1 => group=1 => Theta row=1 => (1,1). slevel=1 => low noise => ~N(0,0.1^2)
            #   agent #1 => char2 => group=2 => indicator=0 => no addition
            # => so deterministic part is (1,1). We'll just check that out_naive[1,1,:] is near (1,1) plus some small random offset.

            #   agent #2 => char1 => group=2 => row=2 => (2,2). slevel=3 => ~N(0,0.5^2) noise
            #            => char2 => group=1 => row=3 => (10,-1). slevel=3 => also ~N(0,0.5^2)
            # => total deterministic = (2,2) + (10,-1) = (12,1), plus sum of two ~N(0,0.5^2). 
            # => we see the result. The test above ensures it matches the naive approach.
        end


        @testset "2) Large Scenario - Your Setup" begin
            rng = MersenneTwister(42)

            # As specified:
            N, J, K = 100, 5, 4
            d = 4
            num_agents = 5_000

            # Salience probabilities
            salience_probs = SMatrix{5,4,Float64}([
                0.1 0.2 0.5 0.2;   # Characteristic 1
                0.3 0.3 0.3 0.1;   # Characteristic 2
                0.2 0.3 0.4 0.1;   # Characteristic 3
                0.25 0.25 0.25 0.25; # Characteristic 4
                0.05 0.15 0.60 0.20  # Characteristic 5
            ])
            salience_lookup_static = precompute_salience_lookup_table(salience_probs)

            # Build random node distributions
            node_dists = rand(rng, Float64, N, J, K)
            for n in 1:N, j in 1:J
                node_dists[n, j, :] ./= sum(node_dists[n, j, :])
            end

            # simulate agents
            agents = simulate_agents_with_salience(node_dists, salience_lookup_static, num_agents; rng=rng)
            @test size(agents) == (N, num_agents, J, 3)

            # define cleavage-salience array
            Delta_array = [1, 3, 2, 2, 0]  # e.g. char1=low, char2=high, char3=moderate, char4=moderate, char5=no
            Theta = generate_theta(Delta_array, J, K, d, rng)
            @test size(Theta) == (J * K, d)  # => (5*4, 4) => (20,4)

            # define noise scales
            sigma_low, sigma_mod, sigma_high = 0.1, 0.25, 0.4

            # We'll run the new masked approach
            points_masked = generate_points_masked!(
                agents, Theta,
                sigma_low, sigma_mod, sigma_high,
                d; rng=rng
            )
            @test size(points_masked) == (N, num_agents, d)

            # 2.1) Check no NaNs or Inf
            @test !any(isnan.(points_masked))
            @test !any(isinf.(points_masked))

            # 2.2) Check there's at least some non-zero content
            @test any(points_masked .!= 0.0)

            # 2.3) Optional: Check distribution of final values in one dimension
            # For instance, if we pick dimension k=1:
            values_k1 = vec(view(points_masked, :, :, 1))  # flatten
            @test length(values_k1) == N * num_agents

            # Because some characteristic has no salience => partial zero,
            # but we expect a broad spread due to the high-salience characteristic(s).
            # We can do a quick check that std is not too small or too large:
            s_approx = std(values_k1)
            @test s_approx > 0.01  # expect > 0.0
            @test s_approx < 100.0 # just some upper bound sanity check

            # 2.4) (Optional) Benchmark
            println("Benchmarking generate_points_masked! with large scenario:")
            generate_points_masked!(
                agents, Theta,
                sigma_low, sigma_mod, sigma_high,
                d; rng=rng
            )
        end

    end

end

end

module SimulateQuestionPreferences

using Random, Statistics, LinearAlgebra, StaticArrays
using Base: @propagate_inbounds
using StatsBase

export z_scale_points_for_tangian,
    build_tangian_questions_oneissue,
    build_tangian_questions_multiissue,
    simulate_tangian_questions_chunked!,
    generate_question_positions,
    map_voters_to_positions!

################################################################################

################################################################################
@propagate_inbounds function _oneD_coords(m::Int)
    if m == 2
        return [-2.0, 2.0]
    elseif m == 3
        # [-2, 0, 2]
        return [-2.0, 0.0, 2.0]
    elseif m == 5
        # [-2, -1, 0, 1, 2], for example
        return [-2.0, -1.0, 0.0, 1.0, 2.0]
    else
        return collect(range(-3, 3, length=m))
    end
end

################################################################################
"""
    build_tangian_questions_oneissue(d::Int, which_dim::Int, m::Int)
        -> Matrix{Float64} (d x m)

Place `m` columns in a d-dim space, all zeros except the `which_dim` dimension. 
Use _oneD_coords(m) to fill that dimension with positions in [-2,2].
"""
function build_tangian_questions_oneissue(d::Int, which_dim::Int, m::Int)
    coords = _oneD_coords(m)
    mat = zeros(d, m)
    @inbounds for col in 1:m
        mat[which_dim, col] = coords[col]
    end
    return mat
end

"""
build_tangian_questions_multiissue(d::Int, question_specs::Vector{Tuple{Int,Int}})
 -> Vector{Matrix{Float64}}

Given a total dimension d, and a list of (which_dim, m) pairs,
returns a vector of length Q, each a (d x m) matrix of "yes-positions".

So, (d x m) means (dimensions in the issue space * number of yes-positions)
Each column is the coordinates of a yes-position
Each row is a dimension in the issue-space

For Tangian questions that are only related to a single dimension, then, we should expect
to have a matrix of all zeros except for one row. In that row, there should be mostly non-zero 
    values corresponding to where in ℝ that yes-position falls. 

So "multi-issue" is a little misleading here, it means an issue with multiple-dimensions
"""
function build_tangian_questions_multiissue(d::Int, question_specs::Vector{Tuple{Int,Int}})
    l = length(question_specs)
    out = Vector{Matrix{Float64}}(undef, l)
    for i in 1:length(question_specs)
        (dim_i, m_i) = question_specs[i]
        coords = _oneD_coords(m_i)
        mat = zeros(d, m_i)
        for col in 1:m_i
            mat[dim_i, col] = coords[col]
        end
        out[i] = mat
    end
    return SVector{l,Matrix{Float64}}(out)
end

# Helper to sample an index from weights
@inline function pick_index!(weights::Vector{Float64}, total_weight::Float64, rng::AbstractRNG)::Int
    r = rand(rng) * total_weight
    cume = 0.0
    @inbounds for idx in 1:length(weights)
        cume += weights[idx]
        if r <= cume
            return idx
        end
    end
    return length(weights)
end

"""
outputs N x A x Q array, where:

N = number of nodes (districts)
A = number of agents (voters)
Q = number of Tangian questions (typically, numnber of Tangian questions in a given issue)

"""
function simulate_tangian_questions_chunked!(
    points::Array{Float64,3},
    tangian_positions::SVector{Q,AbstractMatrix{Float64}} where {Q},
    gamma::Float64,
    rng::AbstractRNG,
    chunk_size::Int64=5000
)::Array{Int64,3}
    @assert ndims(points) == 3 "Input `points` must be a 3D array (N, A, d)"
    N, A, d = size(points)
    Q = length(tangian_positions)
    points_flat = reshape(points, :, d)  # Flatten to (N*A, d)

    chosen = Array{Int}(undef, N, A, Q)  # Result array

    # Precompute constant
    neg_gamma = -gamma

    # Preallocate buffers
    weights_row = Vector{Float64}(undef, 5)  # Adjust size as needed
    d2_values = Vector{Float64}(undef, 5)   # Adjust size as needed

    # Process points in chunks
    start_idx = 1
    while start_idx <= N * A
        end_idx = min(start_idx + chunk_size - 1, N * A)
        chunk_size_actual = end_idx - start_idx + 1
        chunk_view = @view points_flat[start_idx:end_idx, :]

        # Process each Tangian question
        for q_idx in 1:Q
            pos_mat = tangian_positions[q_idx]
            d2, m = size(pos_mat)
            @assert d2 == d "Dimension mismatch between points and Tangian positions"

            # Process each point in the chunk
            @inbounds for voter_idx in 1:chunk_size_actual
                max_acc = -Inf

                # Compute weights directly using pairwise distances
                for col in 1:m
                    # Calculate squared distances
                    d2 = 0.0
                    @inbounds for dim in 1:d
                        diff = chunk_view[voter_idx, dim] - pos_mat[dim, col]
                        d2 += diff^2
                    end
                    d2_values[col] = d2
                    acc = neg_gamma * d2
                    weights_row[col] = acc
                    max_acc = max(max_acc, acc)
                end

                # Stabilize weights to prevent underflow/overflow
                total_weight = 0.0
                for col in 1:m
                    weights_row[col] = exp(weights_row[col] - max_acc)
                    total_weight += weights_row[col]
                end

                # Pick index
                idx_picked = pick_index!(weights_row, total_weight, rng)

                # Map global voter index back to (N, A)
                n_global = start_idx + voter_idx - 1
                node_idx = (n_global - 1) ÷ A + 1
                agent_idx = (n_global - 1) % A + 1
                chosen[node_idx, agent_idx, q_idx] = idx_picked
            end
        end

        start_idx += chunk_size_actual
    end

    return chosen
end


"""
    map_voters_to_positions!(
        ideal_points::AbstractVector{Array{Float64,3}},
        available_positions::AbstractVector{AbstractVector{AbstractMatrix{Float64}}},
        gamma::Float64,
        n_issues::Int,
        n_questions::AbstractVector{Int},
        n_positions::AbstractVector{Int},
        n_seats::Int,
        pop_per_seat::Int,
        issue_dimensions::AbstractVector{Int};
        skip_sqrt::Bool = false
    ) -> Vector{Array{Float64,3}}

For each issue `issue` in `1:n_issues`:
  - We have ideal points `arr_k = ideal_points[issue]` of size (n_seats, pop_per_seat, d_k).
  - We have `pos_kv = available_positions[issue]` which is a length `Q_k` vector,
    each an (d_k, p_k) matrix. Exactly one row is non-zero.
  - We want to produce a `Q_k × n_seats × pop_per_seat` result array where
    `result_k[q, seat, voter]` = the *active coordinate* of the position chosen by
    that voter from the softmax distribution over all `p_k` positions.

We do stable softmax with "logit" = -γ * dist (or -γ * sqrt(dist_sq)).  
Then sample from that distribution rather than pick the argmax.

Options:
  - `skip_sqrt = true`: Uses the squared distance instead of Euclidean distance
    for the logit, which can be faster. The scale changes, but sampling is
    still valid if the rest of your model is consistent with that.

Returns a `Vector{Array{Float64,3}}` of length `n_issues`.
Each element has size `(Q_k, n_seats, pop_per_seat)`.
"""
function map_voters_to_positions!(
    ideal_points::AbstractVector{Array{Float64,3}},
    available_positions::AbstractVector{AbstractVector{AbstractMatrix{Float64}}},
    gamma::Float64,
    n_issues::Int,
    n_questions::AbstractVector{Int},
    n_positions::AbstractVector{Int},
    n_seats::Int,
    pop_per_seat::Int,
    issue_dimensions::AbstractVector{Int};
    skip_sqrt::Bool=false
)
    # We'll store the results for each issue
    results = Vector{Array{Float64,3}}(undef, n_issues)

    @inline @fastmath function distance_penalty(
        arr_k::Array{Float64,3},
        seat::Int,
        voter::Int,
        pos_q::AbstractMatrix{Float64},
        p::Int,
        d_k::Int
    )::Float64
        # Compute squared distance between voter ideal [seat,voter,:]
        # and position col p in pos_q
        # Then either return sqrt(...) or keep dist_sq, depending on skip_sqrt.
        @inbounds @fastmath begin
            dist_sq = 0.0
            @inbounds for i in 1:d_k
                diff = arr_k[seat, voter, i] - pos_q[i, p]
                dist_sq = muladd(diff, diff, dist_sq)   # dist_sq += diff^2
            end
            return skip_sqrt ? (gamma * dist_sq) : (gamma * sqrt(dist_sq))
        end
    end

    for issue in 1:n_issues
        # For the k-th issue:
        arr_k = ideal_points[issue]         # size (n_seats, pop_per_seat, d_k)
        pos_kv = available_positions[issue]  # Vector of length Q_k
        d_k = issue_dimensions[issue]
        Q_k = n_questions[issue]
        p_k = n_positions[issue]

        # Allocate the result for this issue:
        # shape is (Q_k, n_seats, pop_per_seat)
        result_k = Array{Float64,3}(undef, Q_k, n_seats, pop_per_seat)

        # For each question q, we have a d_k x p_k matrix pos_q
        for q in 1:Q_k
            pos_q = pos_kv[q]   # shape (d_k, p_k)

            # Find the single "active" row.  We'll break on the first non-zero entry:

            active_dim = 0
            @inbounds for row_idx in 1:d_k
                for col_idx in 1:p_k
                    if pos_q[row_idx, col_idx] != 0.0
                        active_dim = row_idx
                        break
                    end
                end
                # Exit the outer loop early if we've already set active_dim
                if active_dim != 0
                    break
                end
            end

            @assert active_dim != 0 "No active dimension found. The matrix may be all zeros."

            # We pre-allocate logits array once per question to reuse
            # for each voter in each seat:
            logits = Vector{Float64}(undef, p_k)

            # Outer loops over seats & voters
            @inbounds for seat in 1:n_seats
                @inbounds for voter in 1:pop_per_seat
                    # 1) Compute raw logits = -(distance), but let's do stable softmax
                    # We'll do two passes: one for max, one for sum-of-exps + sampling
                    max_logit = -Inf
                    @inbounds @fastmath for p in 1:p_k
                        # Dist or Dist^2 times gamma
                        # The "logit" is -( gamma * distance ), so we store:
                        #   logits[p] = - distance_penalty(...)
                        # Because distance_penalty returns +gamma * dist (or dist_sq),
                        # we need the negative of that for the "utility".
                        logits[p] = -distance_penalty(arr_k, seat, voter, pos_q, p, d_k)
                        if logits[p] > max_logit
                            max_logit = logits[p]
                        end
                    end

                    # 2) sum-of-exps for normalization
                    sum_expshift = 0.0
                    @inbounds @fastmath for p in 1:p_k
                        sum_expshift += exp(logits[p] - max_logit)
                    end

                    # 3) Sample from that distribution.
                    #    We'll do a single pass: 
                    #    pick r ~ Uniform(0, sum_expshift)
                    #    accumulate partial sums of "exp(logit - max_logit)" until we exceed r.
                    r = rand() * sum_expshift
                    cumsum = 0.0
                    chosen_p = 1
                    @inbounds @fastmath for p in 1:p_k
                        cumsum += exp(logits[p] - max_logit)
                        if cumsum >= r
                            chosen_p = p
                            break
                        end
                    end

                    # 4) Store the chosen position's active coordinate
                    @inbounds result_k[q, seat, voter] = pos_q[active_dim, chosen_p]
                end
            end
        end
        results[issue] = result_k
    end

    return results
end

"""
Vector of N x A x Q_i arrays, where Q_i is the number of questions in a given issue.
So, question_positions[1][1,1,:] gives us the positions of agent 1 on 

"""
function generate_question_positions(issue_dimensions::AbstractVector{Int}, n_issues::Int,
    n_questions::AbstractVector{Int64}, n_positions::AbstractVector{Int64},
    ideal_points::AbstractVector{Array{Float64,3}},
    gamma::Float64, n_seats::Int, pop_per_seat::Int)::Vector

    available_positions = Vector{AbstractVector{AbstractMatrix{Float64}}}(undef, n_issues)

    for issue in 1:n_issues

        number_of_questions = n_questions[issue]
        number_of_positions = n_positions[issue]
        issue_dimension = issue_dimensions[issue]

        q_specs = map(
            random_dim -> (random_dim, number_of_positions),
            rand(1:issue_dimension, number_of_questions)
        )

        available_positions[issue] = build_tangian_questions_multiissue(
            issue_dimension, q_specs
        )

    end

    available_positions = SVector{n_issues,AbstractVector{AbstractMatrix{Float64}}}(
        available_positions
    )

    question_positions = map_voters_to_positions!(ideal_points,
        available_positions, gamma, n_issues, n_questions, n_positions, n_seats, pop_per_seat,
        issue_dimensions)

    return question_positions

end

end # module

module TestQuestionPreferences

using Test
using Random
using StaticArrays
using ..SimulateQuestionPreferences  # adjust if needed

function run_tests()
    @testset "TangianQuestionsRefactored Tests" begin

        @testset "Toy Example (hand check)" begin
            # Suppose N=1, A=3 => total 3 voters, d=1
            # points => shape(1,3,1) => e.g. x-coords [ -1.5, 0.2, 1.0 ]
            points_3d = reshape([-1.5, 0.2, 1.0], 1, 3, 1)
            # We'll define question_specs => dimension=1, m=3 => [-2,0,2]
            question_specs = Vector{Tuple{Int,Int}}()
            push!(question_specs, (1, 3))
            tangian_pos = build_tangian_questions_multiissue(1, question_specs)
            # large gamma => pick nearest
            chosen = simulate_tangian_questions_chunked!(points_3d, tangian_pos, 1.0e5; rng=MersenneTwister(1234), chunk_size=2)
            @test size(chosen) == (1, 3, 1)

            # Distances to [-2,0,2]
            #   -1.5 => nearest -2 => index=1
            #    0.2 => nearest 0 => index=2
            #    1.0 => nearest 0 => index=2 if purely distance 
            # Actually 1.0 is equidistant from 0 and 2 => dist=1.0 and 1.0
            # let's see which is "strictly" smaller => 1 => they are same => random tie
            # So let's see what the code picks with that seed => we can do a check.
            # We'll see which index it yields:
            result_array = reshape(chosen, 3)  # flatten
            println("Toy chosen indexes = ", result_array)
            # If the random tie picks 2 or 3, let's just check we get either 2 or 3:
            @test result_array[1] == 1   # for -1.5
            @test result_array[2] == 2   # for 0.2
            @test result_array[3] in (2, 3)  # for 1.0
        end

        @testset "Large Scenario (100 x 10000 x 5)" begin
            # We'll do the real scenario: 100 districts, 10k people => 1 million voters, d=5
            # That might be huge for a test. We'll do a smaller scale for demonstration:
            # let's do N=10, A=1000 => 10k total, d=5
            rng = MersenneTwister(999)
            N, A, d = 100, 10000, 5
            points_3ds = SVector{5,Array{Float64,3}}([randn(rng, N, A, d) for _ in 1:5])

            # question_specs => maybe 5 questions for a single issue: each dimension with m=3
            # 4 questions, one on each dimension; each one has 3 possible positions
            qspecs = Vector{Tuple{Int,Int}}()
            for dim_i in 1:d
                push!(qspecs, (dim_i, 3))
            end

            # This generates questions for a single, multi-dimensional issue
            tangian_pos = build_tangian_questions_multiissue(d, qspecs)
            Q = length(tangian_pos)
            @test Q == d

            # let's do chunk_size=5000 => means we process 5000 voters at a time
            chosen = @time simulate_tangian_questions_chunked!.(points_3ds, Ref(tangian_pos), Ref(1.0))
            @test size(chosen) == (N, A, Q)

            # Just do some basic checks
            for q in 1:Q
                # m=3 for each
                for n in 1:N, a in 1:A
                    cval = chosen[n, a, q]
                    @test 1 <= cval <= 3
                end
            end
        end

    end
end
end