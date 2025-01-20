module SpatialCharacteristics

using Random
using LinearAlgebra
using Statistics
using Distances

export generate_spatial_characteristics

function generate_cluster_centers(K::Int, sigma_c_prime::Float64, rng::AbstractRNG; max_tries::Int=10000)
    centers = Matrix{Float64}(undef, K, 2)
    found = 0
    tries_count = 0
    while (found < K) && (tries_count < max_tries)
        candidate = rand(rng, 2)  # Generate a candidate point in the [0, 1]x[0, 1] space
        ok = true
        for i in 1:found
            if norm(candidate .- centers[i, :]) < sigma_c_prime
                ok = false
                break
            end
        end
        if ok
            found += 1
            centers[found, :] = candidate
        end
        tries_count += 1
    end
    if found < K
        for i in (found+1):K
            centers[i, :] = rand(rng, 2)
        end
    end
    return centers
end

function place_nodes(N::Int, U::Float64, centers::Matrix{Float64}, sigma_c::Float64, rng::AbstractRNG)
    coords = zeros(Float64, N, 2)
    is_urban = fill(false, N)
    K = size(centers, 1)
    n_urban = floor(Int, U * N)
    idx = collect(1:N)
    shuffle!(rng, idx)
    urban_idx = idx[1:n_urban]
    rural_idx = idx[n_urban+1:end]

    for i in urban_idx
        is_urban[i] = true
        attempts = 0
        while true
            c_idx = rand(rng, 1:K)
            meanx, meany = centers[c_idx, :]
            candx = randn(rng) * sigma_c + meanx
            candy = randn(rng) * sigma_c + meany
            if (0.0 <= candx <= 1.0) && (0.0 <= candy <= 1.0)
                coords[i, 1] = candx
                coords[i, 2] = candy
                break
            end
            attempts += 1
            if attempts > 1000
                coords[i, 1] = rand(rng)
                coords[i, 2] = rand(rng)
                break
            end
        end
    end
    for i in rural_idx
        coords[i, 1] = rand(rng)
        coords[i, 2] = rand(rng)
    end
    return coords, is_urban
end

function build_adjacency(coords::Matrix{Float64}; delta::Union{Nothing,Float64}=0.1, k::Union{Nothing,Int}=nothing)
    N = size(coords, 1)
    adjacency = zeros(Float64, N, N)
    # vectorized distance
    dmat = pairwise(Euclidean(), coords')
    if delta !== nothing
        @inbounds for i in 1:N
            for j in i+1:N
                if dmat[i, j] <= delta
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
                end
            end
        end
    elseif k !== nothing
        @inbounds for i in 1:N
            row = dmat[i, :]
            idx_s = sortperm(row)
            nn = idx_s[2:(k+1)]
            for j in nn
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
            end
        end
    else
        error("build_adjacency: Must provide either delta or k.")
    end
    return adjacency
end

function build_node_distributions(
    coords::AbstractMatrix{Float64},
    statewide_probs::AbstractVector{Vector{Float64}},
    A_values::AbstractVector{Float64},
    rng::AbstractRNG;
    alpha::Float64=1.0,
    beta::Float64=1.0,
    center_and_scale::Bool=true
)
    N = size(coords, 1)
    J = length(statewide_probs)
    max_k = maximum(length.(statewide_probs))
    node_dists = fill(0.0, N, J, max_k)
    dmat = pairwise(Euclidean(), coords')

    @inbounds for j_idx in 1:J
        p_j = statewide_probs[j_idx]
        k_j = length(p_j)
        A_j = A_values[j_idx]
        phi_j = (A_j == 0) ? 1e12 : 1.0 / A_j

        # Build covariance matrix
        cov_mat = Array{Float64}(undef, N, N)
        @inbounds for i in 1:N
            for m in 1:N
                cov_mat[i, m] = (A_j == 0) ? ((i == m) ? 1e-9 : 0.0) : exp(-dmat[i, m] / phi_j)
            end
        end
        for i in 1:N
            cov_mat[i, i] += 1e-9
        end

        # Cholesky decomposition
        L = cholesky(Symmetric(cov_mat)).L

        # Sample latent fields
        Z_fields = zeros(Float64, N, k_j)
        @inbounds for r in 1:k_j
            z_r = randn(rng, N)
            Z_fields[:, r] .= L * z_r
        end

        # Center and scale if requested
        if center_and_scale
            @inbounds for r in 1:k_j
                col = Z_fields[:, r]
                m_c = mean(col)
                s_c = std(col) + 1e-12
                @inbounds for i in 1:N
                    col[i] = (col[i] - m_c) / s_c
                end
            end
        end

        # Compute log-baseline
        log_baseline = [alpha * log(x + 1e-12) for x in p_j]

        # Compute final distribution
        @inbounds for i in 1:N
            extended = Vector{Float64}(undef, k_j)
            @inbounds for r in 1:k_j
                extended[r] = beta * Z_fields[i, r] + log_baseline[r]
            end
            max_val = maximum(extended)
            exps = [exp(x - max_val) for x in extended]
            denom = sum(exps)
            @inbounds for r in 1:k_j
                node_dists[i, j_idx, r] = exps[r] / denom
            end
        end
    end

    return node_dists
end

"""
Array with the following dimensions:
N x C x d_max

where N is the number of districts, C is the number of charactersitics, and d_max is the maximum
number of groups

"""
function generate_spatial_characteristics(n_metros::Int, spatial_dispersion::Float64,
    n_seats::Int, urbanization::Float64, urban_sprawl::Float64, a_vals::AbstractVector{Float64},
    statewide_distributions::Vector{Vector{Float64}}, rng::AbstractRNG
)::Array{Float64,3}

    centers = SpatialCharacteristics.generate_cluster_centers(n_metros, spatial_dispersion, rng)
    coords, is_urban = SpatialCharacteristics.place_nodes(n_seats, urbanization, centers,
        urban_sprawl, rng)
    node_dists = SpatialCharacteristics.build_node_distributions(coords, statewide_distributions,
        a_vals, rng)

    return node_dists

end


function test_spatial_fourthpass()
    rng = MersenneTwister(42)
    K = 3
    sigma_c_prime = 0.1
    centers = generate_cluster_centers(K, sigma_c_prime, rng)
    coords, is_urban = place_nodes(100, 0.5, centers, 0.01, rng)
    adjacency = build_adjacency(coords; delta=0.2)
    statewide_probs = [[0.4, 0.3, 0.1, 0.2], [0.4, 0.4, 0.2], [0.5, 0.5]]
    A_values = [2.0, 2.0, 5.0]
    node_dists = build_node_distributions(coords, statewide_probs, A_values, rng)
    # println("Test done. adjacency size: ", size(adjacency))
    # println("Node dist example: ", node_dists[1,1,:])
end

test_spatial_fourthpass()

end

module TestSpatialCharacteristics

using Random
using LinearAlgebra
using Statistics
using Distances
using DataFrames
using CSV

# Compute Moran's I
function compute_spatial_autocorrelation(values::Vector{Float64}, adjacency::Matrix{Float64})
    """
    Computes Moran's I based on an adjacency matrix.
    """
    N = size(adjacency, 1)
    w = adjacency ./ sum(adjacency, dims=2)  # Normalize row sums
    mean_val = mean(values)
    dev = values .- mean_val

    # Numerator and denominator
    num = sum(adjacency[i, j] * dev[i] * dev[j] for i in 1:N, j in 1:N)
    denom = sum(dev .^ 2)

    return (N / sum(adjacency)) * (num / denom)
end

# Generate a single map
function generate_single_map(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng, alpha, beta)
    centers = generate_cluster_centers(K, sigma_c_prime, rng)
    coords, is_urban = place_nodes(N, U, centers, sigma_c, rng)
    adjacency = build_adjacency(coords; delta=delta)
    node_dists = build_node_distributions(coords, statewide_probs, A_values, rng; alpha=alpha, beta=beta)
    return adjacency, node_dists
end

function compute_entropy(probabilities::Vector{Float64})
    # Compute entropy: H = -sum(p * log(p)) for p > 0
    return -sum(p * log(p + 1e-12) for p in probabilities)
end

function compute_spatial_autocorrelation(values::Vector{Float64}, adjacency::Matrix{Float64})
    """
    Computes Moran's I based on an adjacency matrix.
    """
    N = size(adjacency, 1)
    w = adjacency ./ sum(adjacency, dims=2)  # Normalize row sums
    mean_val = mean(values)
    dev = values .- mean_val

    # Numerator and denominator
    num = sum(adjacency[i, j] * dev[i] * dev[j] for i in 1:N, j in 1:N)
    denom = sum(dev .^ 2)

    return (N / sum(adjacency)) * (num / denom)
end

# Bootstrap Moran's I
function bootstrap_morans_I(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng; alpha=1.0, beta=1.0, n_bootstrap=100)
    morans_I_samples = Float64[]

    for _ in 1:n_bootstrap
        adjacency, node_dists = generate_single_map(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng, alpha, beta)
        values = node_dists[:, 1, 2]  # Extract values for Moran's I computation
        push!(morans_I_samples, compute_spatial_autocorrelation(values, adjacency))
    end

    mean_I = mean(morans_I_samples)
    median_I = median(morans_I_samples)
    std_I = std(morans_I_samples)

    return mean_I, median_I, std_I
end

function bootstrap_entropy_morans_I(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng; alpha=1.0, beta=1.0, n_bootstrap=100)
    morans_I_samples = Float64[]

    for _ in 1:n_bootstrap
        adjacency, node_dists = generate_single_map(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng, alpha, beta)

        # Compute entropy for each node
        entropies = [compute_entropy(node_dists[i, 1, :]) for i in 1:N]

        # Compute Moran's I for the entropies
        push!(morans_I_samples, compute_spatial_autocorrelation(entropies, adjacency))
    end

    mean_I = mean(morans_I_samples)
    median_I = median(morans_I_samples)
    std_I = std(morans_I_samples)

    return mean_I, median_I, std_I
end

function run_combinations_with_entropy()
    # Parameters
    N = 200
    K = 3
    sigma_c = 0.05
    sigma_c_prime = 0.15
    statewide_probs = [[0.4, 0.3, 0.3]]
    n_bootstrap = 100
    rng = MersenneTwister(123)

    U_values = [0.9, 0.6, 0.3]
    A_j_values = [0.0, 0.5, 1.0]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    threshold_values = [0.05]

    combinations = [(U, A_j, alpha, threshold) for U in U_values, A_j in A_j_values, alpha in alpha_values, threshold in threshold_values]

    results = DataFrame(U=Float64[], A_j=Float64[], alpha=Float64[], threshold=Float64[], mean_I=Float64[], median_I=Float64[], std_I=Float64[])

    for (U, A_j, alpha, threshold) in combinations
        println("Running combination U=$U, A_j=$A_j, alpha=$alpha, threshold=$threshold")
        mean_I, median_I, std_I = bootstrap_entropy_morans_I(
            N, K, U, sigma_c, sigma_c_prime, threshold, statewide_probs, [A_j], rng; alpha=alpha, beta=1.0, n_bootstrap=n_bootstrap
        )
        push!(results, (U, A_j, alpha, threshold, mean_I, median_I, std_I))
    end

    # Save results
    CSV.write("bootstrap_entropy_morans_I_results.csv", results)
    println("Results saved to 'bootstrap_entropy_morans_I_results.csv'")
end

function run_combinations()
    # Parameters
    N = 100
    K = 3
    sigma_c = 0.05
    sigma_c_prime = 0.15
    statewide_probs = [[0.4, 0.3, 0.3]]
    n_bootstrap = 100
    rng = MersenneTwister(123)

    U_values = [0.9, 0.6, 0.3]
    A_j_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    threshold_values = [0.05, 0.1, 0.2, 0.3]

    combinations = [(U, A_j, alpha, threshold) for U in U_values, A_j in A_j_values, alpha in alpha_values, threshold in threshold_values]

    results = DataFrame(U=Float64[], A_j=Float64[], alpha=Float64[], threshold=Float64[], mean_I=Float64[], median_I=Float64[], std_I=Float64[])

    for (U, A_j, alpha, threshold) in combinations
        println("Running combination U=$U, A_j=$A_j, alpha=$alpha, threshold=$threshold")
        mean_I, median_I, std_I = bootstrap_morans_I(
            N, K, U, sigma_c, sigma_c_prime, threshold, statewide_probs, [A_j], rng; alpha=alpha, beta=1.0, n_bootstrap=n_bootstrap
        )
        push!(results, (U, A_j, alpha, threshold, mean_I, median_I, std_I))
    end

    # Save results
    CSV.write("bootstrap_morans_I_results.csv", results)
    println("Results saved to 'bootstrap_morans_I_julia_results.csv'")
end

function jensen_shannon_divergence(p::Vector{Float64}, q::Vector{Float64})
    m = 0.5 * (p .+ q)
    return 0.5 * sum(p .* log.(p ./ (m .+ 1e-12) .+ 1e-12)) +
           0.5 * sum(q .* log.(q ./ (m .+ 1e-12) .+ 1e-12))
end

function compute_distance_correlation(spatial_distances::Matrix{Float64}, distributional_distances::Matrix{Float64})
    N = size(spatial_distances, 1)
    spatial_vector = []
    distributional_vector = []

    for i in 1:N
        for j in i+1:N
            push!(spatial_vector, spatial_distances[i, j])
            push!(distributional_vector, distributional_distances[i, j])
        end
    end

    return cor(spatial_vector, distributional_vector)
end

function generate_distance_matrices(coords::Matrix{Float64}, node_dists::Array{Float64,3}, m::Vector{Float64})
    N = size(coords, 1)

    # Spatial distances
    spatial_distances = pairwise(Euclidean(), coords')

    # Distributional distances (JSD)
    distributional_distances = Matrix{Float64}(undef, N, N)
    @inbounds for i in 1:N
        distributional_distances[i, i] = 0.0  # JSD for same node
        for j in i+1:N
            p_i, p_j = node_dists[i, 1, :], node_dists[j, 1, :]
            @. m = 0.5 * (p_i + p_j)
            distance = 0.5 * sum(p_i .* log.(p_i ./ m .+ 1e-12)) +
                       0.5 * sum(p_j .* log.(p_j ./ m .+ 1e-12))
            distributional_distances[i, j] = distance
            distributional_distances[j, i] = distance  # Symmetric
        end
    end

    return spatial_distances, distributional_distances
end

function bootstrap_distributional_correlation(N, K, U, sigma_c, sigma_c_prime, delta, statewide_probs, A_values, rng; alpha=1.0, beta=1.0, n_bootstrap=100)
    correlations = Float64[]
    m = zeros(Float64, size(statewide_probs[1]))  # Preallocate JSD intermediate result

    for _ in 1:n_bootstrap
        # Generate map
        centers = generate_cluster_centers(K, sigma_c_prime, rng)
        coords, is_urban = place_nodes(N, U, centers, sigma_c, rng)
        adjacency = build_adjacency(coords; delta=delta)
        node_dists = build_node_distributions(coords, statewide_probs, A_values, rng; alpha=alpha, beta=beta)

        # Generate distance matrices
        spatial_distances, distributional_distances = generate_distance_matrices(coords, node_dists, m)

        # Compute correlation
        correlation = compute_distance_correlation(spatial_distances, distributional_distances)
        push!(correlations, correlation)
    end

    mean_correlation = mean(correlations)
    median_correlation = median(correlations)
    std_correlation = std(correlations)

    return mean_correlation, median_correlation, std_correlation
end


# Run combinations with JSD-based spatial correlation
function run_combinations_with_jsd()
    # Parameters
    N = 200
    K = 3
    sigma_c = 0.05
    sigma_c_prime = 0.15
    statewide_probs = [[0.4, 0.3, 0.3]]
    n_bootstrap = 100
    rng = MersenneTwister(123)

    U_values = [0.9, 0.6, 0.3]
    A_j_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    threshold_values = [0.1, 0.2, 0.3]

    combinations = [(U, A_j, alpha, threshold) for U in U_values, A_j in A_j_values, alpha in alpha_values, threshold in threshold_values]

    results = DataFrame(U=Float64[], A_j=Float64[], alpha=Float64[], threshold=Float64[], mean_corr=Float64[], median_corr=Float64[], std_corr=Float64[])

    for (U, A_j, alpha, threshold) in combinations
        println("Running combination U=$U, A_j=$A_j, alpha=$alpha, threshold=$threshold")
        mean_corr, median_corr, std_corr = bootstrap_distributional_correlation(
            N, K, U, sigma_c, sigma_c_prime, threshold, statewide_probs, [A_j], rng; alpha=alpha, beta=1.0, n_bootstrap=n_bootstrap
        )
        push!(results, (U, A_j, alpha, threshold, mean_corr, median_corr, std_corr))
    end

    # Save results
    CSV.write("bootstrap_jsd_correlation_results.csv", results)
    println("Results saved to 'bootstrap_jsd_correlation_results.csv'")
end

end
