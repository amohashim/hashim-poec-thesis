#!/usr/bin/env julia

# =======================================================
# SingleMapSimulation.jl
#
# This script:
# 1) Generates a single "map" (graph) representing a state.
#    - Node coordinates in [0,1]^2
#    - Some nodes are "urban" (in clusters), some "rural"
#    - An adjacency list
# 2) Builds node-level conditional demographic distributions
#    for a set of characteristics {C1,...,C_J},
#    each of which has a statewide marginal distribution
#    plus a spatial autocorrelation factor.
# 3) Provides thorough tests to ensure correctness.
#
# No multi-threading or parallelization is used.
# =======================================================

module SingleMapSimulation

using Random
using Distributions
using LinearAlgebra
using Distances
using Statistics
using StatsBase
using NearestNeighbors
using StaticArrays
using SparseArrays
using Test  # for our test suite

# -------------------------------------------------------
# == Data Structures ==
# -------------------------------------------------------

"""
`MapData` holds all relevant information about the generated map.

Fields:
- `coords`:   (N x 2) array of node coordinates
- `edges`:    Vector{Tuple{Int,Int}} adjacency
- `is_urban`: Bool array of length N; is_urban[i] = true if node i is in a cluster
- `node_dists`:
    A Vector of length N, where each entry is a Vector of length J 
    (J = # of characteristics).
    - node_dists[i][j] is a vector of probabilities of shape (k_j),
      i.e. the local distribution of characteristic j at node i.
    Example:
      node_dists[i][j][r] = probability that node i belongs to group r
      of characteristic j.
"""
struct MapData
    coords::Array{Float64,2}         # (N x 2)
    edges::Vector{Tuple{Int,Int}}   # adjacency
    is_urban::Vector{Bool}            # length N
    node_dists::Vector{Vector{Vector{Float64}}}
end

# -------------------------------------------------------
# == Parameter Defaults ==
# -------------------------------------------------------
const DEFAULT_N = 200
const DEFAULT_K = 3
const DEFAULT_U = 0.6
const DEFAULT_SIGMA_C = 0.05
const DEFAULT_SIGMA_C_PRIME = 0.15
const DEFAULT_DELTA = 0.1

# For the latent-field-based autocorrelation, we define
# an example "degree of autocorrelation" A_j for each characteristic
# here. In real usage, the user might pass an array of length J.
const DEFAULT_A = [5.0, 10.0]  # Example for 2 characteristics

# Example statewide (marginal) distributions for 2 characteristics:
#   1) Characteristic #1 has 3 groups, e.g. Race
#   2) Characteristic #2 has 2 groups, e.g. Gender
const DEFAULT_STATEWIDE = [[0.4, 0.3, 0.3],  # 3 groups
    [0.5, 0.5]]     # 2 groups


# -------------------------------------------------------
# 1. Generate cluster centers
# -------------------------------------------------------
"""
Generate K cluster centers in [0,1]^2, 
ensuring they are not too close to each other 
(using sigma_C_prime for minimal separation).
If we cannot place all K with the given constraints 
within max_tries, we fill the remainder randomly.

Returns a (K x 2) array of centers.
"""
function generate_cluster_centers(K::Int, sigma_C_prime::Float64;
    max_tries::Int=10000,
    rng=MersenneTwister(0))::Array{Float64,2}
    result = Array{Float64}(undef, K, 2)
    found = 0
    tries_count = 0

    while (found < K) && (tries_count < max_tries)
        candidate = rand(rng, 2)
        ok = true
        for i in 1:found
            distval = norm(candidate .- result[i, :])
            if distval < sigma_C_prime
                ok = false
                break
            end
        end
        if ok
            found += 1
            result[found, :] = candidate
        end
        tries_count += 1
    end

    # If we didn't place all centers, fill remainder ignoring separation
    if found < K
        for i in (found+1):K
            result[i, 1] = rand(rng)
            result[i, 2] = rand(rng)
        end
    end

    return result
end

# -------------------------------------------------------
# 2. Place nodes (urban vs rural)
# -------------------------------------------------------
"""
Places N nodes in [0,1]^2:
- fraction U*N are assigned to clusters (urban),
- rest are rural (uniform).
We also return an is_urban array indicating which nodes are urban.

coords[i,:] = (x,y) location in [0,1]^2
is_urban[i] = true if node i is in a cluster
"""
function place_nodes(N::Int, U::Float64, centers::Array{Float64,2}, sigma_C::Float64;
    rng=MersenneTwister(0))
    coords = zeros(Float64, N, 2)
    is_urban = fill(false, N)

    K = size(centers, 1)
    n_urban = floor(Int, U * N)
    n_rural = N - n_urban

    # Shuffle indices so we pick n_urban distinct nodes
    all_indices = collect(1:N)
    shuffle!(rng, all_indices)

    urban_idxs = all_indices[1:n_urban]
    rural_idxs = all_indices[(n_urban+1):end]

    # Mark them
    for idx in urban_idxs
        is_urban[idx] = true
    end

    # Place urban
    for idx in urban_idxs
        while true
            c_idx = rand(rng, 1:K)
            meanx, meany = centers[c_idx, 1], centers[c_idx, 2]
            x = rand(rng, Normal(meanx, sigma_C))
            y = rand(rng, Normal(meany, sigma_C))
            if 0.0 <= x <= 1.0 && 0.0 <= y <= 1.0
                coords[idx, 1] = x
                coords[idx, 2] = y
                break
            end
        end
    end

    # Place rural
    for idx in rural_idxs
        coords[idx, 1] = rand(rng)
        coords[idx, 2] = rand(rng)
    end

    return coords, is_urban
end

# -------------------------------------------------------
# 3. Build adjacency (no duplicates, no self-loops)
# -------------------------------------------------------
"""
Build adjacency from coords using either:
- a distance threshold delta, or
- a k-nearest-neighbors approach (k not nothing).

We return a Vector{(Int,Int)} of edges, 
where each edge is (i, j) with i != j, 
and i < j => we also store the symmetric (j, i) 
*only if* you want an undirected adjacency repeated in both directions.
To avoid duplicates, we systematically do i<j in the main loop. 
Then we can optionally add (j, i) if you want symmetrical listing.

If no pairs satisfy the threshold or k, you might get an empty edge list, 
which is perfectly valid.
"""
function build_adjacency(coords::Array{Float64,2};
    delta::Union{Nothing,Float64}=DEFAULT_DELTA,
    k::Union{Nothing,Int}=nothing,
    rng=MersenneTwister(0),
    undirected::Bool=true)
    N = size(coords, 1)
    edges = Vector{Tuple{Int,Int}}()

    if delta !== nothing
        # distance-based
        dmat = pairwise(Euclidean(), coords')
        for i in 1:(N-1)
            for j in (i+1):N
                distval = dmat[i, j]
                if distval <= delta
                    push!(edges, (i, j))
                    if undirected
                        push!(edges, (j, i))
                    end
                end
            end
        end
    elseif k !== nothing
        # naive kNN
        dmat = pairwise(Euclidean(), coords')
        for i in 1:N
            # sort the row
            # Note: enumerates all j, but sort by distance
            rowdist = dmat[i, :]
            sorted_idx = sortperm(rowdist)
            # first one is i itself, so skip it
            neighbors = sorted_idx[2:(k+1)]  # possible out-of-bounds if k> N-1, check if needed
            for j in neighbors
                if i != j
                    if i < j
                        push!(edges, (i, j))
                        if undirected
                            push!(edges, (j, i))
                        end
                    else
                        push!(edges, (j, i))
                        if undirected
                            push!(edges, (i, j))
                        end
                    end
                end
            end
        end
    else
        error("build_adjacency: Must provide either delta or k.")
    end

    return edges
end

# -------------------------------------------------------
# 4. Build node-level distributions
# -------------------------------------------------------
"""
We want a node-level distribution for each characteristic j at each node i.

We have:
  - `statewide_probs`: a Vector{Vector{Float64}} of length J,
      where statewide_probs[j] is the baseline distribution
      (e.g. [0.4, 0.3, 0.3] for 3 groups).
  - `A`: a Vector{Float64} of length J (degree of autocorrelation).
  - `coords`: the (N x 2) node positions in [0,1]^2

We produce `node_dists[i][j]`: 
  the local (conditional) distribution for characteristic j at node i, 
  computed by a "latent field" approach:

    1) For each j, we create (k_j - 1) Gaussian random fields with 
       Cov(Z_i, Z_m) = exp(- A[j] * dist(i,m)^2).
    2) We shift log-odds from statewide_probs[j].
    3) Apply a "softmax" to get local probabilities.

Return: 
  A Vector of length N, 
  where node_dists[i] is a Vector of length J. 
  node_dists[i][j] is the local distribution for characteristic j.

Implementation detail: We'll do them all at once if we like,
but for clarity we do one characteristic at a time.
"""

function build_node_distributions(
    coords::Array{Float64,2},
    statewide_probs::Vector{Vector{Float64}},
    A::Vector{Float64};
    rng=MersenneTwister(0),
    α=0.5
)
    N = size(coords, 1)
    J = length(statewide_probs)

    dmat = pairwise(Euclidean(), coords')
    node_dists = Vector{Vector{Vector{Float64}}}(undef, N)
    for i in 1:N
        node_dists[i] = Vector{Vector{Float64}}(undef, J)
    end

    for j_idx in 1:J
        p_j = statewide_probs[j_idx]
        k_j = length(p_j)
        A_j = A[j_idx]

        Cov_mat = exp.(-A_j .* dmat .^ 2) + 1e-8 * I
        L = cholesky(Symmetric(Cov_mat))
        Z_fields = zeros(N, k_j - 1)

        for r in 1:(k_j-1)
            standard_normal = randn(rng, N)
            Z_fields[:, r] = L \ standard_normal
        end

        for i in 1:N
            extended = Vector{Float64}(undef, k_j)
            for r in 1:(k_j-1)
                extended[r] = Z_fields[i, r]
            end
            extended[k_j] = 0.0  # log-odds for last group
            extended .= extended .+ α * log.(p_j .+ 1e-8)


            max_val = maximum(extended)
            exps = exp.(extended .- max_val)
            denom = sum(exps)
            if denom <= 0
                error("Invalid probabilities: sum of exps is non-positive")
            end
            local_probs = exps / denom
            node_dists[i][j_idx] = local_probs
        end
    end

    return node_dists
end

function build_node_distributions_fixed(
    coords::Array{Float64,2},
    statewide_probs::Vector{Vector{Float64}},
    A::Vector{Float64};
    rng=MersenneTwister(0),
    α=1.0,
    β=1.0
)
    N = size(coords, 1)
    J = length(statewide_probs)

    # Precompute pairwise distances
    dmat = pairwise(Euclidean(), coords')

    # Initialize node-level distributions
    node_dists = Vector{Vector{Vector{Float64}}}(undef, N)
    for i in 1:N
        node_dists[i] = Vector{Vector{Float64}}(undef, J)
    end

    for j_idx in 1:J
        p_j = statewide_probs[j_idx]  # Statewide distribution
        k_j = length(p_j)
        A_j = A[j_idx]

        # Build covariance matrix and sample latent fields
        Cov_mat = exp.(-A_j .* dmat .^ 2) + 1e-8 * I
        L = cholesky(Symmetric(Cov_mat))

        # Sample (k_j - 1) latent fields
        Z_fields = zeros(N, k_j - 1)
        for r in 1:(k_j-1)
            standard_normal = randn(rng, N)
            Z_fields[:, r] = L \ standard_normal
        end

        # Center and scale latent fields to prevent extreme values
        for r in 1:(k_j-1)
            Z_fields[:, r] .= (Z_fields[:, r] .- mean(Z_fields[:, r])) ./ std(Z_fields[:, r] .+ 1e-6)
        end

        # Compute local probabilities for each node
        for i in 1:N
            # Build extended log-odds array
            extended = Vector{Float64}(undef, k_j)
            for r in 1:(k_j-1)
                extended[r] = Z_fields[i, r]
            end
            extended[k_j] = 0.0  # Log-odds for the last group

            # Adjust log-odds using statewide probabilities as priors
            extended .= β * extended .+ α * log.(p_j .+ 1e-8)

            # Apply softmax to obtain probabilities
            max_val = maximum(extended)
            exps = exp.(extended .- max_val)
            denom = sum(exps)
            if denom <= 0
                error("Invalid probabilities: sum of exps is non-positive")
            end
            local_probs = exps / denom

            # Store the local distribution
            node_dists[i][j_idx] = local_probs
        end
    end

    return node_dists
end



# -------------------------------------------------------
# 5. Main function to build the entire map
# -------------------------------------------------------
"""
Generate a single map with the given parameters:
1) K cluster centers,
2) Place N nodes with fraction U as urban,
3) Build adjacency with distance threshold = delta,
4) Build node-level distributions for each characteristic j 
   given statewide_probs and A.

Returns a `MapData` struct containing everything.
"""
function generate_single_map(
    N::Int,
    K::Int,
    U::Float64,
    sigma_C::Float64,
    sigma_C_prime::Float64,
    delta::Float64,
    statewide_probs::Vector{Vector{Float64}},  # J length
    A::Vector{Float64},                        # J length
    rng=MersenneTwister(0),
    α=1.0,
    β=1.0
)::MapData

    # 1) Cluster centers
    centers = generate_cluster_centers(K, sigma_C_prime; rng=rng)

    # 2) Place nodes
    coords, is_urban = place_nodes(N, U, centers, sigma_C; rng=rng)

    # 3) Build adjacency
    edges = build_adjacency(coords;
        delta=delta,
        k=nothing,
        rng=rng,
        undirected=true)
    # 4) Build node-level distributions for each characteristic
    node_dists = build_node_distributions_fixed(coords, statewide_probs, A; rng=rng, α=α, β=β)

    # Package it all up
    return MapData(coords, edges, is_urban, node_dists)
end

# -------------------------------------------------------
# 6. Comprehensive Tests
# -------------------------------------------------------
@testset "Single Map Simulation Tests" begin

    # We'll do a small test run
    RNG_SEED = MersenneTwister(1234)
    N_test = 20
    K_test = 2
    U_test = 0.5
    sigmaC_test = 0.01
    sigmaCprime_test = 0.1
    delta_test = 0.2

    statewide_test = [
        [0.4, 0.3, 0.3],  # characteristic j=1 => 3 groups
        [0.5, 0.5]        # j=2 => 2 groups
    ]
    A_test = [2.0, 5.0]  # degrees of autocorr

    map_test = generate_single_map(N_test, K_test, U_test,
        sigmaC_test, sigmaCprime_test,
        delta_test,
        statewide_test, A_test,
        RNG_SEED, 1.0, 5.0)

    @test size(map_test.coords, 1) == N_test
    @test length(map_test.is_urban) == N_test
    @test length(map_test.node_dists) == N_test

    # Check adjacency correctness
    #  - no duplicates
    #  - no self-loops
    seen_edges = Set{Tuple{Int,Int}}()
    for e in map_test.edges
        @test e[1] != e[2]   # no self-loop
        @test !(e in seen_edges)  # no duplicates
        push!(seen_edges, e)
    end

    # Because we used a distance threshold, it's possible we have 0 edges
    # for small N or certain parameters. Let's just confirm it runs:
    @test isa(map_test.edges, Vector{Tuple{Int,Int}})

    # Check node-level distributions
    #  - For each node i, for each characteristic j, the distribution sums to ~1.0
    #  - No negative probabilities
    for i in 1:N_test
        for j in 1:length(statewide_test)
            local_probs = map_test.node_dists[i][j]
            @test all(x -> x >= 0.0, local_probs)
            s = sum(local_probs)
            @test isapprox(s, 1.0, atol=1e-7)
        end
    end

    # Check is_urban fraction
    #  - Because it's random, we just check if #urban is near 10 for N=20, U=0.5
    num_urban = count(x -> x, map_test.is_urban)
    @test 5 <= num_urban <= 15  # just a range check

end

# -------------------------------------------------------
# 7. Example main usage
# -------------------------------------------------------
function main(α, β)# Example `generate_single_map` and MapData must be defined before running
    # Replace `generate_single_map` with your actual map generation function.

    # Define domain sizes and A_vals
    domain_sizes = [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]
    A_vals = [0.01, 0.1, 0.5, 1.0]

    # Run the debugging function
    results = diagnose_domain_size_debug_fixed(generate_single_map, domain_sizes, A_vals)

    # Print results
    for (domain, metrics) in results
        println("==== Domain: $domain ====")
        for (A_j, result) in metrics
            println("    A_j = $A_j -> $result")
        end
    end

    rng = MersenneTwister(42)

    # For a production run, we might do:
    N = 200
    K = 3
    U = 0.60
    sigmaC = 0.05
    sigmaCprime = 0.15
    delta = 0.1

    # 2 characteristics
    statewide = [
        [0.4, 0.3, 0.3],
        [0.5, 0.5]
    ]
    A = [5.0, 10.0]

    mapdata = generate_single_map(N, K, U, sigmaC, sigmaCprime, delta,
        statewide, A, rng, α, β)

    # println("Map generation complete.")
    # println("Number of nodes: ", size(mapdata.coords, 1))
    # println("Number of edges: ", length(mapdata.edges))
    # println("Urban count: ", count(x -> x, mapdata.is_urban))
    # println("Sample distribution for node 1, characteristic 1: ",
    #     mapdata.node_dists[1][1])

    return mapdata
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end  # module SingleMapSimulation

# -------------------------------------------------------
# 8. Spatial Autocorrelation Analysis
# -------------------------------------------------------

"""
Computes the spatial autocorrelation of node-level distributions for each characteristic.

For each characteristic j:
- Calculate pairwise Euclidean distances between nodes.
- Calculate pairwise "distribution distances" for each characteristic
  using L2 norm or cosine similarity.
- Compute correlation between spatial distances and distribution distances.

Returns:
- A vector of correlations, one for each characteristic.
"""
function analyze_spatial_autocorrelation(
    mapdata::MapData,
    statewide_probs::Vector{Vector{Float64}}
)::Vector{Float64}
    coords = mapdata.coords
    node_dists = mapdata.node_dists
    N = size(coords, 1)
    J = length(statewide_probs)

    # Precompute pairwise Euclidean distances
    spatial_dists = [norm(coords[i, :] - coords[m, :]) for i in 1:N, m in (i+1):N]
    spatial_dists = vec(spatial_dists)  # Ensure it's a flat vector

    results = Float64[]

    for j_idx in 1:J
        # Compute distribution distances (e.g., L2 norm)
        dist_diffs = [
            norm(node_dists[i][j_idx] - node_dists[m][j_idx])
            for i in 1:N, m in (i+1):N
        ]
        dist_diffs = vec(dist_diffs)  # Ensure it's a flat vector

        # Compute correlation between spatial distances and distribution distances
        correlation = cor(spatial_dists, dist_diffs)
        push!(results, correlation)
    end

    return results
end

function analyze_spatial_autocorrelation(
    mapdata::MapData,
    j_idx::Int,
    delta::Float64
)
    N = size(mapdata.coords, 1)
    node_dists = mapdata.node_dists

    # Use a single category (e.g., first category) instead of mean probability
    values = [node_dists[i][j_idx][1] for i in 1:N]

    # Build a binary adjacency matrix based on the distance threshold
    coords = mapdata.coords
    dmat = pairwise(Euclidean(), coords')
    adj_matrix = dmat .<= delta
    adj_matrix[diagind(adj_matrix)] .= 0  # No self-loops
    wmat = adj_matrix ./ sum(adj_matrix, dims=2)  # Normalize rows

    # Moran's I calculation
    mean_value = mean(values)
    W = sum(adj_matrix)
    numerator = sum(
        (values[i] - mean_value) * (values[j] - mean_value) * wmat[i, j]
        for i in 1:N, j in 1:N
    )
    denominator = sum((values[i] - mean_value)^2 for i in 1:N)
    morans_I = (N / W) * (numerator / denominator)

    # Geary's C calculation
    gearys_C = (N - 1) / (2 * W) * sum(
        (values[i] - values[j])^2 * wmat[i, j]
        for i in 1:N, j in 1:N
    ) / denominator

    return morans_I, gearys_C
end

function compute_morans_i(values::Vector{Float64}, wmat::Matrix{Float64})
    N = length(values)
    mean_value = mean(values)
    numerator = 0.0
    denominator = sum((v - mean_value)^2 for v in values)

    # Handle zero variance case
    if denominator == 0
        println("Variance of values is zero. Moran's I is undefined.")
        return NaN
    end

    for i in 1:N
        for j in 1:N
            numerator += (values[i] - mean_value) * (values[j] - mean_value) * wmat[i, j]
        end
    end

    W = sum(wmat)
    if W == 0
        println("Sum of weights is zero. Moran's I is undefined.")
        return NaN
    end

    return (N / W) * (numerator / denominator)
end

function compute_gearys_c(values::Vector{Float64}, wmat::Matrix{Float64})
    N = length(values)
    mean_value = mean(values)
    numerator = 0.0
    denominator = sum((v - mean_value)^2 for v in values)

    # Handle zero variance case
    if denominator == 0
        println("Variance of values is zero. Geary's C is undefined.")
        return NaN
    end

    for i in 1:N
        for j in 1:N
            numerator += wmat[i, j] * (values[i] - values[j])^2
        end
    end

    W = sum(wmat)
    if W == 0
        println("Sum of weights is zero. Geary's C is undefined.")
        return NaN
    end

    return ((N - 1) / (2 * W)) * (numerator / denominator)
end

function analyze_spatial_autocorrelation_improved(mapdata::MapData, statewide_probs::Vector{Vector{Float64}})
    N = size(mapdata.coords, 1)
    dmat = pairwise(Euclidean(), mapdata.coords')
    dmat[dmat.==0] .= Inf  # Avoid self-loops in inverse distances

    # Use a simpler adjacency approach (threshold or k-NN)
    dist_threshold = 0.2
    wmat = sparse((dmat .< dist_threshold) .+ 0.0)  # Binary adjacency
    wmat[diagind(wmat)] .= 0.0  # Remove self-loops
    W = sum(wmat)  # Total weight

    for j_idx in 1:length(statewide_probs)
        # Extract the probabilities for a specific category or derived measure
        # Example: Analyze the second category
        values = [mapdata.node_dists[i][j_idx][2] for i in 1:N]

        # Optional: Analyze a derived measure
        # values = [mapdata.node_dists[i][j_idx][2] - mapdata.node_dists[i][j_idx][1] for i in 1:N]

        mean_value = mean(values)

        # Moran's I calculation
        numerator = sum((values[i] - mean_value) * (values[m] - mean_value) * wmat[i, m]
                        for i in 1:N for m in 1:N)
        denominator = sum((values[i] - mean_value)^2 for i in 1:N)
        morans_I = (N / W) * (numerator / denominator)

        # Geary's C calculation
        gearys_C_numerator = sum((values[i] - values[m])^2 * wmat[i, m]
                                 for i in 1:N for m in 1:N)
        gearys_C_denominator = 2 * sum((values[i] - mean_value)^2 for i in 1:N)
        gearys_C = (N - 1) / W * (gearys_C_numerator / gearys_C_denominator)

        # Inspect value range and variability
        println("Characteristic $j_idx:")
        println("  Moran's I: $morans_I")
        println("  Geary's C: $gearys_C")
        println("  Value range: ", minimum(values), " to ", maximum(values))
        println("  Value variance: ", var(values))
        println()
    end
end

using LinearAlgebra, SparseArrays, Statistics, Distances

"""
    analyze_spatial_autocorrelation_simple!(mapdata, value_extractor; dist_threshold=0.2, k=nothing)

Analyzes the spatial autocorrelation of a numeric value associated with each node in a `MapData` structure.

Arguments:
- `mapdata::MapData`: The graph and node-level data.
- `value_extractor::Function`: A function `i -> value` that extracts a single numeric value for each node `i`.
- `dist_threshold::Float64`: The distance threshold for adjacency (default: 0.2).
- `k::Union{Nothing, Int}`: If provided, uses k-nearest neighbors instead of a distance threshold.

Returns:
- Prints Moran's I and Geary's C, along with the range and variance of the extracted values.
"""
function analyze_spatial_autocorrelation_simple!(
    mapdata::MapData,
    value_extractor::Function;
    dist_threshold::Float64=0.2,
    k::Union{Nothing,Int}=nothing
)
    coords = mapdata.coords
    N = size(coords, 1)

    # 1. Extract numeric values for analysis
    values = [value_extractor(i) for i in 1:N]

    # Debugging: Inspect value range and variance
    println("Value range: [", minimum(values), ", ", maximum(values), "]")
    println("Value mean: ", mean(values), ", variance: ", var(values))

    if var(values) < 1e-12
        println("WARNING: Very little variation in node values => Moran's I or Geary's C may be degenerate.")
    end

    # 2. Build adjacency matrix
    dmat = pairwise(Euclidean(), coords')

    if k === nothing
        # Distance-threshold adjacency
        adjacency = spzeros(Float64, N, N)
        for i in 1:N
            for j in (i+1):N
                if dmat[i, j] < dist_threshold
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
                end
            end
        end
    else
        # k-Nearest neighbors adjacency
        adjacency = spzeros(Float64, N, N)
        for i in 1:N
            # Sort distances and get k nearest neighbors (excluding self)
            drow = dmat[i, :]
            idx_sorted = sortperm(drow)
            neighbors = idx_sorted[2:(k+1)]  # Skip the first (self)
            for nn in neighbors
                adjacency[i, nn] = 1.0
                adjacency[nn, i] = 1.0
            end
        end
    end

    # Remove self-loops
    for i in 1:N
        adjacency[i, i] = 0.0
    end

    W = sum(adjacency)  # Total weight

    # 3. Moran's I Calculation
    mean_v = mean(values)
    numerator = sum((values[i] - mean_v) * (values[j] - mean_v) * adjacency[i, j]
                    for i in 1:N for j in 1:N)
    denominator = sum((values[i] - mean_v)^2 for i in 1:N)

    if denominator == 0.0 || W == 0.0
        println("Moran's I: Degenerate (zero variance or no edges).")
    else
        morans_I = (N / W) * (numerator / denominator)
        println("Moran's I = ", morans_I)
    end

    # 4. Geary's C Calculation
    numerator_C = sum(adjacency[i, j] * (values[i] - values[j])^2
                      for i in 1:N for j in 1:N)
    denominator_C = 2 * denominator

    if denominator_C == 0.0 || W == 0.0
        println("Geary's C: Degenerate (zero variance or no edges).")
    else
        gearys_C = ((N - 1) / W) * (numerator_C / denominator_C)
        println("Geary's C = ", gearys_C)
    end
end



# -------------------------------------------------------
# 9. Main Function for Spatial Autocorrelation Analysis
# -------------------------------------------------------

"""
Runs the map generation and spatial autocorrelation analysis.
"""
function main_2(α, β)
    rng = MersenneTwister(rand(1:100))

    # Map generation parameters
    N = 200
    K = 3
    U = 0.60
    sigmaC = 0.05
    sigmaCprime = 0.15
    delta = 0.1

    # 2 characteristics
    statewide = [
        [0.1, 0.6, 0.3],
        [0.1, 0.6, 0.3]
    ]
    A = [50.0, 1.0]

    # Generate map
    mapdata = generate_single_map(N, K, U, sigmaC, sigmaCprime, delta,
        statewide, A, rng, α, β)

    # Analyze spatial autocorrelation
    correlations = analyze_spatial_autocorrelation(mapdata, statewide)

    # Print results

end

function main_2(α::Float64, β::Float64)
    rng = MersenneTwister(rand(1:100))

    # Map generation parameters
    N = 200
    K = 3
    U = 0.6
    sigmaC = 0.05
    sigmaCprime = 0.15
    delta = 0.1

    # Characteristics and autocorrelation
    statewide = [
        [0.1, 0.6, 0.3],
        [0.1, 0.6, 0.3],
        [0.1, 0.6, 0.3],
        [0.1, 0.6, 0.3]
    ]
    A = [0.0, 25.0, 50.0, 75.0]

    try
        # Generate the map
        mapdata = generate_single_map(N, K, U, sigmaC, sigmaCprime, delta, statewide, A, rng, α, β)

        # Perform spatial autocorrelation analysis
        println("\n=== Spatial Autocorrelation Analysis ===")
        analyze_spatial_autocorrelation(mapdata, statewide)

    catch e
        # Handle unexpected issues gracefully
        println("Error encountered during spatial autocorrelation analysis: ", e)
        println("Stacktrace:\n", stacktrace(e))
    end
end


